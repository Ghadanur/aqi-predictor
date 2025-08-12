# src/models/train.py
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import traceback
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

def plot_predictions(y_true, y_pred, horizon, baseline_pred=None, save_path=None):
    """Enhanced visual comparison of predictions vs actuals"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Model Predicted', linestyle='--', alpha=0.8)
    if baseline_pred is not None:
        plt.plot(baseline_pred, label='Baseline (Mean)', linestyle=':', alpha=0.6)
    plt.title(f'AQI Forecast Validation ({horizon} horizon)')
    plt.xlabel('Time Steps')
    plt.ylabel('AQI Value')
    plt.legend()
    if save_path:
        output_path = os.path.join(save_path, f'validation_{horizon}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved validation plot to: {output_path}")
    plt.close()

def evaluate_forecast(y_true, y_pred, horizon):
    """Enhanced forecast evaluation with baseline comparison"""
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Baseline comparison
    baseline_pred = np.full_like(y_true, y_true.mean())
    baseline_r2 = r2_score(y_true, baseline_pred)
    
    # Robust accuracy calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        perc_errors = np.nanmean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1, None))) * 100
    
    return {
        'horizon': horizon,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Baseline_R2': baseline_r2,
        'Accuracy (%)': 100 - perc_errors,
        'samples': len(y_true),
        'mean_error': np.mean(y_pred - y_true),
        'std_error': np.std(y_pred - y_true),
        'max_error': np.max(np.abs(y_pred - y_true))
    }

def train_3day_forecaster():
    try:
        # 1. Get processed data
        processor = AQI3DayForecastProcessor()
        print("Fetching and processing data...")
        features, targets = processor.get_3day_forecast_data(lookback_days=120)
        
        # Data validation checks
        print("\n=== Data Validation ===")
        print("Features shape:", features.shape)
        print("Targets shape:", targets.shape)
        print("\nTarget summary:")
        print(targets.describe())
        print("\nNaN counts:")
        print("Features:", features.isna().sum().sum())
        print("Targets:", targets.isna().sum().sum())
        
        # 2. Reset indices
        features = features.reset_index(drop=True)
        targets = targets.reset_index(drop=True)
        
        # 3. Time-based split
        split_idx = int(0.8 * len(features))
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = targets.iloc[:split_idx], targets.iloc[split_idx:]
        
        # 4. Verify alignment
        assert X_train.index.equals(y_train.index), "Train index mismatch"
        assert X_test.index.equals(y_test.index), "Test index mismatch"
        
        # Feature correlation analysis
        print("\n=== Feature-Target Correlation ===")
        for i, col in enumerate(targets.columns):
            corr = features.corrwith(targets[col]).mean()
            print(f"{col}: {corr:.3f}")
        
        # 5. Time Series Cross Validation
        print("\n=== Time Series Cross Validation ===")
        horizon_map = {'24h': 0, '48h': 1, '72h': 2}
        tscv = TimeSeriesSplit(n_splits=3)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = make_pipeline(
                StandardScaler(),
                ExtraTreesRegressor(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1
                )
            )
            model.fit(X_fold_train, y_fold_train)
            fold_preds = model.predict(X_fold_val)
            
            for h_name, h_idx in horizon_map.items():
                val_mask = ~y_fold_val.iloc[:, h_idx].isna()
                if sum(val_mask) > 0:  # Only evaluate if we have valid samples
                    fold_metrics = evaluate_forecast(
                        y_fold_val.loc[val_mask].iloc[:, h_idx],
                        fold_preds[val_mask, h_idx],
                        h_name
                    )
                    print(f"Fold {fold+1} {h_name} R²: {fold_metrics['R2']:.2f} (Baseline: {fold_metrics['Baseline_R2']:.2f})")
        
        # 6. Train final model
        print("\nTraining final model...")
        model = make_pipeline(
            StandardScaler(),
            ExtraTreesRegressor(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
        )
        model.fit(X_train, y_train)
        
        # 7. Save model
        model_path = os.path.join(MODEL_DIR, '3day_forecaster.pkl')
        joblib.dump(model, model_path)
        print(f"Saved model to: {model_path}")
        
        # 8. Generate predictions
        print("\nGenerating predictions...")
        preds = model.predict(X_test)
        
        # 9. Enhanced Validation
        validation_results = []
        baseline_comparison = []

        for horizon_name, horizon_idx in horizon_map.items():
            try:
                print(f"\n=== Evaluating {horizon_name} forecast ===")
                valid_mask = ~y_test.iloc[:, horizon_idx].isna()
                y_true = y_test.loc[valid_mask].iloc[:, horizon_idx]
                y_pred = preds[valid_mask, horizon_idx]
                y_baseline = np.full_like(y_true, y_true.mean())
                
                print(f"Samples: {len(y_true)}")
                print(f"True mean: {y_true.mean():.2f}")
                
                # Calculate metrics
                model_metrics = evaluate_forecast(y_true, y_pred, horizon_name)
                baseline_metrics = evaluate_forecast(y_true, y_baseline, horizon_name)
                
                validation_results.append(model_metrics)
                baseline_comparison.append({
                    'horizon': horizon_name,
                    'Baseline_R2': baseline_metrics['R2'],
                    'Model_R2': model_metrics['R2'],
                    'Improvement': model_metrics['R2'] - baseline_metrics['R2']
                })
                
                # Print performance
                print(f"\n{horizon_name} Forecast Performance:")
                print(f"- R²: {model_metrics['R2']:.2f} (Baseline: {baseline_metrics['R2']:.2f})")
                print(f"- RMSE: {model_metrics['RMSE']:.2f}")
                print(f"- MAE: {model_metrics['MAE']:.2f}")
                print(f"- Accuracy: {model_metrics['Accuracy (%)']:.2f}%")
                
                # Plot comparison
                plot_predictions(y_true, y_pred, horizon_name, y_baseline, MODEL_DIR)
                
            except Exception as e:
                print(f"Error evaluating {horizon_name}: {str(e)}")
                continue
        
        # 10. Save reports
        pd.DataFrame(validation_results).to_csv(
            os.path.join(MODEL_DIR, 'validation_report.csv'), 
            index=False
        )
        pd.DataFrame(baseline_comparison).to_csv(
            os.path.join(MODEL_DIR, 'baseline_comparison.csv'),
            index=False
        )
        print("\nSaved validation reports")
        
        return model

    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        print("Starting AQI 3-day forecast model training...")
        trained_model = train_3day_forecaster()
        
        print("\nGenerated files in model directory:")
        for f in sorted(os.listdir(MODEL_DIR)):
            if f.startswith(('3day_forecaster', 'validation_', 'baseline_')):
                size = os.path.getsize(os.path.join(MODEL_DIR, f))
                print(f"- {f} ({size} bytes)")
        
        print("\nModel training and validation completed successfully")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {str(e)}")
        print("Traceback:", traceback.format_exc())
        sys.exit(1)
