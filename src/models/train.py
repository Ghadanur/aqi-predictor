# src/models/train.py
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import traceback
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

try:
    from src.features.process import AQI3DayForecastProcessor
except ImportError:
    from features.process import AQI3DayForecastProcessor

# Define the output directory
MODEL_DIR = Path(__file__).parent
os.makedirs(MODEL_DIR, exist_ok=True)

def plot_predictions(y_true, y_pred, horizon, save_path=None):
    """Visual comparison of predictions vs actuals"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values, label='Actual', marker='o', linestyle='-', alpha=0.7)
    plt.plot(y_pred, label='Predicted', marker='x', linestyle='--', alpha=0.7)
    plt.title(f'AQI Forecast Validation ({horizon} horizon)')
    plt.xlabel('Time Steps')
    plt.ylabel('AQI Category')
    plt.yticks(sorted(np.unique(np.concatenate([y_true, y_pred]))))
    plt.legend()
    if save_path:
        output_path = os.path.join(save_path, f'validation_{horizon}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved validation plot to: {output_path}")
    plt.close()

def evaluate_forecast(y_true, y_pred, horizon):
    """Classification evaluation with baseline comparison"""
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    baseline_pred = [y_true.mode()[0]] * len(y_true)
    baseline_acc = accuracy_score(y_true, baseline_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix ({horizon}):")
    print(cm)
    
    return {
        'horizon': horizon,
        'Accuracy': acc,
        'Baseline_Accuracy': baseline_acc,
        'Improvement': acc - baseline_acc,
        'samples': len(y_true),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }

def train_3day_forecaster():
    try:
        # 1. Get processed data
        processor = AQI3DayForecastProcessor()
        print("Fetching and processing data...")
        features, targets = processor.get_3day_forecast_data(lookback_days=120)
        
        # Convert targets to integer categories
        targets = targets.round().astype(int)
        
        # Data validation checks
        print("\n=== Data Validation ===")
        print("Features shape:", features.shape)
        print("Targets shape:", targets.shape)
        print("\nTarget value counts:")
        print(targets.apply(lambda x: x.value_counts()))
        
        # 2. Add temporal features
        print("\nAdding temporal features...")
        available_columns = features.columns.tolist()

# Add time-based features only if we have datetime information
        if 'datetime' in available_columns:
            features['hour'] = features['datetime'].dt.hour
            features['day_of_week'] = features['datetime'].dt.dayofweek
            features['hour_sin'] = np.sin(2*np.pi*features['hour']/24)
            features['hour_cos'] = np.cos(2*np.pi*features['hour']/24)
        elif 'timestamp' in available_columns:
            features['datetime'] = pd.to_datetime(features['timestamp'])
            features['hour'] = features['datetime'].dt.hour
            features['day_of_week'] = features['datetime'].dt.dayofweek
            features['hour_sin'] = np.sin(2*np.pi*features['hour']/24)
            features['hour_cos'] = np.cos(2*np.pi*features['hour']/24)
        else:
            print("Warning: No datetime column found - skipping temporal features")

        
        # 3. Time-based split
        split_idx = int(0.8 * len(features))
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = targets.iloc[:split_idx], targets.iloc[split_idx:]
        
        # 4. Verify alignment
        assert X_train.index.equals(y_train.index), "Train index mismatch"
        assert X_test.index.equals(y_test.index), "Test index mismatch"
        
        # 5. Time Series Cross Validation
        print("\n=== Time Series Cross Validation ===")
        horizon_map = {'24h': 0, '48h': 1, '72h': 2}
        tscv = TimeSeriesSplit(n_splits=3)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = make_pipeline(
                StandardScaler(),
                ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            )
            
            for h_name, h_idx in horizon_map.items():
                model.fit(X_fold_train, y_fold_train.iloc[:, h_idx])
                fold_preds = model.predict(X_fold_val)
                fold_acc = accuracy_score(y_fold_val.iloc[:, h_idx], fold_preds)
                baseline_acc = accuracy_score(y_fold_val.iloc[:, h_idx], 
                                           [y_fold_val.iloc[:, h_idx].mode()[0]]*len(y_fold_val))
                print(f"Fold {fold+1} {h_name} Accuracy: {fold_acc:.2f} (Baseline: {baseline_acc:.2f})")
        
        # 6. Train final models (one per horizon)
        print("\nTraining final models...")
        models = {}
        for h_name, h_idx in horizon_map.items():
            model = make_pipeline(
                StandardScaler(),
                ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            )
            model.fit(X_train, y_train.iloc[:, h_idx])
            models[h_name] = model
        
        # 7. Save models
        for h_name, model in models.items():
            model_path = os.path.join(MODEL_DIR, f'3day_forecaster_{h_name}.pkl')
            joblib.dump(model, model_path)
            print(f"Saved {h_name} model to: {model_path}")
        
        # 8. Generate predictions and evaluate
        validation_results = []
        for h_name, h_idx in horizon_map.items():
            try:
                print(f"\n=== Evaluating {h_name} forecast ===")
                y_true = y_test.iloc[:, h_idx]
                y_pred = models[h_name].predict(X_test)
                
                metrics = evaluate_forecast(y_true, y_pred, h_name)
                validation_results.append(metrics)
                
                print(f"\n{h_name} Forecast Performance:")
                print(f"- Accuracy: {metrics['Accuracy']:.2f} (Baseline: {metrics['Baseline_Accuracy']:.2f})")
                print(f"- Improvement: {metrics['Improvement']:.2f}")
                print("\nClassification Report:")
                print(classification_report(y_true, y_pred))
                
                plot_predictions(y_true, y_pred, h_name, MODEL_DIR)
                
            except Exception as e:
                print(f"Error evaluating {h_name}: {str(e)}")
                continue
        
        # 9. Save reports
        pd.DataFrame(validation_results).to_csv(
            os.path.join(MODEL_DIR, 'validation_report.csv'), 
            index=False
        )
        print("\nSaved validation report")
        
        return models

    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        print("Starting AQI 3-day forecast model training...")
        trained_models = train_3day_forecaster()
        
        print("\nGenerated files in model directory:")
        for f in sorted(os.listdir(MODEL_DIR)):
            if f.startswith(('3day_forecaster', 'validation_')):
                size = os.path.getsize(os.path.join(MODEL_DIR, f))
                print(f"- {f} ({size} bytes)")
        
        print("\nModel training and validation completed successfully")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {str(e)}")
        print("Traceback:", traceback.format_exc())
        sys.exit(1)



