import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import traceback
import os
import logging
import shap
from shap import Explanation
from shap.plots import beeswarm, bar
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                           classification_report, balanced_accuracy_score,
                           cohen_kappa_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime

try:
    from src.features.process import AQI3DayForecastProcessor
except ImportError:
    from features.process import AQI3DayForecastProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

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
        logging.info(f"Saved validation plot to: {output_path}")
    plt.close()

def evaluate_forecast(y_true, y_pred, horizon):
    """Comprehensive evaluation with multiple metrics"""
    # Calculate metrics
    metrics = {
        'horizon': horizon,
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'samples': len(y_true),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    
    # Baseline comparison
    baseline_pred = [y_true.mode()[0]] * len(y_true)
    metrics.update({
        'baseline_accuracy': accuracy_score(y_true, baseline_pred),
        'improvement': metrics['accuracy'] - accuracy_score(y_true, baseline_pred)
    })
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logging.info(f"\nConfusion Matrix ({horizon}):\n{cm}")
    
    return metrics

def analyze_with_shap(model, X, horizon, sample_size=100):
    """Comprehensive SHAP analysis with visualization"""
    try:
        # Sample for efficiency
        if len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=42)
        else:
            X_sample = X
        
        # Create explainer
        explainer = shap.TreeExplainer(model.named_steps['extratreesclassifier'])
        shap_values = explainer.shap_values(X_sample)
        
        # Generate visualizations
        plt.figure()
        beeswarm(Explanation(shap_values, data=X_sample.values, feature_names=X.columns))
        plt.title(f'SHAP Summary - {horizon} Forecast')
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, f'shap_summary_{horizon}.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # Feature importance plot
        plt.figure()
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {horizon} Forecast')
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, f'shap_importance_{horizon}.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # Return mean absolute SHAP values
        return pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=False)
        
    except Exception as e:
        logging.error(f"SHAP analysis failed: {str(e)}")
        return None

def temporal_train_test_split(features, targets, test_size=0.2, gap_days=7):
    """Time-based split with temporal gap to prevent leakage"""
    split_idx = int(len(features) * (1 - test_size))
    X_train = features.iloc[:split_idx - gap_days]
    X_test = features.iloc[split_idx:]
    y_train = targets.iloc[:split_idx - gap_days]
    y_test = targets.iloc[split_idx:]
    return X_train, X_test, y_train, y_test

def train_3day_forecaster():
    try:
        # 1. Get processed data
        processor = AQI3DayForecastProcessor()
        logging.info("Fetching and processing data...")
        features, targets = processor.get_3day_forecast_data(lookback_days=120)
        
        # Convert targets to integer categories
        targets = targets.round().astype(int)
        
        # Data validation checks
        logging.info("\n=== Data Validation ===")
        logging.info(f"Features shape: {features.shape}")
        logging.info(f"Targets shape: {targets.shape}")
        logging.info("\nTarget value counts:")
        logging.info(targets.apply(lambda x: x.value_counts()))
        
        # 2. Temporal train-test split with gap
        X_train, X_test, y_train, y_test = temporal_train_test_split(
            features, targets, test_size=0.2, gap_days=7
        )
        
        # 3. Verify alignment
        logging.info("\n=== Index Verification ===")
        logging.info(f"X_train index range: {X_train.index[0]} to {X_train.index[-1]}")
        logging.info(f"y_train index range: {y_train.index[0]} to {y_train.index[-1]}")
        assert len(X_train) == len(y_train), "Train length mismatch"
        assert len(X_test) == len(y_test), "Test length mismatch"
        
        # 4. Time Series Cross Validation
        logging.info("\n=== Time Series Cross Validation ===")
        horizon_map = {'24h': 0, '48h': 1, '72h': 2}
        tscv = TimeSeriesSplit(n_splits=3)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            for h_name, h_idx in horizon_map.items():
                # Calculate class weights
                classes = np.unique(y_fold_train.iloc[:, h_idx])
                weights = compute_class_weight('balanced', classes=classes, 
                                             y=y_fold_train.iloc[:, h_idx])
                class_weights = dict(zip(classes, weights))
                
                model = make_pipeline(
                    StandardScaler(),
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                        class_weight=class_weights
                    )
                )
                
                model.fit(X_fold_train, y_fold_train.iloc[:, h_idx])
                fold_preds = model.predict(X_fold_val)
                
                # Calculate multiple metrics
                acc = accuracy_score(y_fold_val.iloc[:, h_idx], fold_preds)
                bal_acc = balanced_accuracy_score(y_fold_val.iloc[:, h_idx], fold_preds)
                baseline = accuracy_score(y_fold_val.iloc[:, h_idx], 
                                        [y_fold_val.iloc[:, h_idx].mode()[0]]*len(y_fold_val))
                
                logging.info(
                    f"Fold {fold+1} {h_name} - "
                    f"Accuracy: {acc:.2f} (Baseline: {baseline:.2f}), "
                    f"Balanced Acc: {bal_acc:.2f}"
                )
        
        # 5. Train final models with versioning
        logging.info("\nTraining final models...")
        models = {}
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for h_name, h_idx in horizon_map.items():
            # Calculate class weights
            classes = np.unique(y_train.iloc[:, h_idx])
            weights = compute_class_weight('balanced', classes=classes, 
                                         y=y_train.iloc[:, h_idx])
            class_weights = dict(zip(classes, weights))
            
            model = make_pipeline(
                StandardScaler(),
                CalibratedClassifierCV(
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                        class_weight=class_weights
                    ),
                    cv=3
                )
            )
            
            model.fit(X_train, y_train.iloc[:, h_idx])
            models[h_name] = model
            
            # Save model with versioning
            model_path = os.path.join(MODEL_DIR, f'model_v{model_version}_{h_name}.pkl')
            joblib.dump(model, model_path)
            logging.info(f"Saved {h_name} model to: {model_path}")
            
            # SHAP analysis
            logging.info(f"Running SHAP analysis for {h_name} forecast...")
            shap_results = analyze_with_shap(model, X_train, h_name)
            if shap_results is not None:
                logging.info(f"\nTop 10 features by SHAP importance ({h_name}):")
                logging.info(shap_results.head(10))
                shap_results.to_csv(
                    os.path.join(MODEL_DIR, f'shap_importance_{h_name}.csv'), 
                    index=False
                )
        
        # 6. Evaluate on test set
        validation_results = []
        for h_name, h_idx in horizon_map.items():
            try:
                logging.info(f"\n=== Evaluating {h_name} forecast ===")
                y_true = y_test.iloc[:, h_idx]
                y_pred = models[h_name].predict(X_test)
                
                metrics = evaluate_forecast(y_true, y_pred, h_name)
                validation_results.append(metrics)
                
                logging.info(f"\n{h_name} Forecast Performance:")
                logging.info(f"- Accuracy: {metrics['accuracy']:.2f} (Baseline: {metrics['baseline_accuracy']:.2f})")
                logging.info(f"- Balanced Accuracy: {metrics['balanced_accuracy']:.2f}")
                logging.info(f"- Cohen's Kappa: {metrics['kappa']:.2f}")
                logging.info("\nClassification Report:")
                logging.info(classification_report(y_true, y_pred))
                
                plot_predictions(y_true, y_pred, h_name, MODEL_DIR)
                
            except Exception as e:
                logging.error(f"Error evaluating {h_name}: {str(e)}")
                continue
        
        # 7. Save comprehensive reports
        report_df = pd.DataFrame(validation_results)
        report_df.to_csv(
            os.path.join(MODEL_DIR, f'validation_report_v{model_version}.csv'), 
            index=False
        )
        logging.info("\nSaved validation report")
        
        # Save SHAP values for all horizons
        shap_all = pd.concat([
            pd.read_csv(os.path.join(MODEL_DIR, f'shap_importance_{h}.csv'))
            .assign(horizon=h)
            for h in horizon_map.keys()
        ])
        shap_all.to_csv(os.path.join(MODEL_DIR, 'combined_shap_importance.csv'), index=False)
        
        return models

    except Exception as e:
        logging.error(f"\nTraining failed: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        logging.info("Starting AQI 3-day forecast model training...")
        trained_models = train_3day_forecaster()
        
        logging.info("\nGenerated files in model directory:")
        for f in sorted(os.listdir(MODEL_DIR)):
            if any(f.startswith(prefix) for prefix in ['model_v', 'validation_', 'shap_']):
                size = os.path.getsize(os.path.join(MODEL_DIR, f))
                logging.info(f"- {f} ({size} bytes)")
        
        logging.info("\nModel training and validation completed successfully")
    except Exception as e:
        logging.error(f"\nCRITICAL ERROR: {str(e)}")
        logging.error("Traceback: " + traceback.format_exc())
        sys.exit(1)
