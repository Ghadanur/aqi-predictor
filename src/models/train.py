import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                           balanced_accuracy_score, confusion_matrix)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
import logging
from pathlib import Path
from datetime import datetime

# Import processor with robust error handling
try:
    from src.features.process import AQI3DayForecastProcessor
except ImportError:
    try:
        from features.process import AQI3DayForecastProcessor
    except ImportError as e:
        logging.error("Failed to import AQI3DayForecastProcessor: %s", str(e))
        raise

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aqi_training.log'),
        logging.StreamHandler()
    ]
)
MODEL_DIR = Path(__file__).parent
os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess_data(features, targets):
    """Handle feature engineering and target processing with index alignment"""
    # Create copies to avoid SettingWithCopyWarning
    features = features[['pm2_5', 'pm10', 'co', 'no2', 'o3']].copy()
    targets = targets.copy()
    
    # Add rolling averages (creates NaNs at start)
    for col in ['pm2_5', 'pm10']:
        features[f'{col}_24h_avg'] = features[col].rolling(24, min_periods=1).mean()
        features[f'{col}_48h_avg'] = features[col].rolling(48, min_periods=1).mean()
    
    # Convert to binary classification
    binary_targets = targets.apply(lambda x: np.where(x >= 4, 2, 1))
    
    # Reset indices to ensure alignment
    features = features.reset_index(drop=True)
    binary_targets = binary_targets.reset_index(drop=True)
    
    # Drop rows where we couldn't calculate rolling features
    # (now unnecessary since we used min_periods=1, but kept for safety)
    valid_idx = features.dropna().index
    return features.loc[valid_idx].reset_index(drop=True), \
           binary_targets.loc[valid_idx].reset_index(drop=True)

def evaluate_model(y_true, y_pred, labels=[1, 2]):
    """Robust evaluation with proper label handling"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    present_labels = np.unique(np.concatenate([y_true, y_pred]))
    eval_labels = [l for l in labels if l in present_labels]
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'report': classification_report(y_true, y_pred, labels=eval_labels, 
                                      zero_division=0, output_dict=True)
    }
    return metrics

def train_aqi_model():
    try:
        # 1. Get data from Hopsworks
        logging.info("Fetching data from Hopsworks...")
        processor = AQI3DayForecastProcessor()
        features, targets = processor.get_3day_forecast_data(lookback_days=180)
        
        # 2. Preprocess data with index alignment
        features, binary_targets = preprocess_data(features, targets.round().astype(int))
        
        # 3. Verify alignment
        assert len(features) == len(binary_targets), "Feature-target length mismatch"
        assert (features.index == binary_targets.index).all(), "Index mismatch"
        
        # 4. Time-based split with temporal gap
        split_idx = int(0.7 * len(features))
        X_train, X_test = features.iloc[:split_idx].copy(), features.iloc[split_idx:].copy()
        y_train, y_test = binary_targets.iloc[:split_idx].copy(), binary_targets.iloc[split_idx:].copy()
        
        # Reset indices again to be safe
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        # 5. Verify we have both classes in training data
        for col in y_train.columns:
            unique_classes = y_train[col].nunique()
            if unique_classes < 2:
                raise ValueError(f"Not enough classes ({unique_classes}) in training data for {col}")
        
        # 6. Train models for each horizon
        horizons = {'24h': 0, '48h': 1, '72h': 2}
        models = {}
        
        for horizon_name, horizon_idx in horizons.items():
            logging.info(f"\n=== Training {horizon_name} model ===")
            
            y_train_h = y_train.iloc[:, horizon_idx]
            y_test_h = y_test.iloc[:, horizon_idx]
            
            # Check class distribution
            class_counts = y_train_h.value_counts()
            logging.info(f"Class distribution:\n{class_counts.to_string()}")
            
            # Build pipeline
            model = make_pipeline(
                StandardScaler(),
                ExtraTreesClassifier(
                    n_estimators=200,
                    max_features='sqrt',
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    bootstrap=True,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                )
            )
            
            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            best_score = 0
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train_h.iloc[train_idx], y_train_h.iloc[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                fold_preds = model.predict(X_fold_val)
                fold_score = balanced_accuracy_score(y_fold_val, fold_preds)
                
                if fold_score > best_score:
                    best_score = fold_score
                    best_model = clone(model)
                    best_model.fit(
                        pd.concat([X_fold_train, X_fold_val]),
                        pd.concat([y_fold_train, y_fold_val])
                    )
            
            model = best_model
            models[horizon_name] = model
            
            # Evaluate
            preds = model.predict(X_test)
            metrics = evaluate_model(y_test_h, preds)
            
            logging.info(f"\nPerformance Metrics for {horizon_name}:")
            logging.info(f"Accuracy: {metrics['accuracy']:.2f}")
            logging.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.2f}")
            logging.info("Confusion Matrix:")
            logging.info(np.array2string(np.array(metrics['confusion_matrix']), 
                       formatter={'int': lambda x: f"{x:4d}"}))
            
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test_h, preds, zero_division=0))
            
            # Save model
            model_path = MODEL_DIR / f"aqi_forecaster_{horizon_name}.pkl"
            joblib.dump(model, model_path)
            logging.info(f"Saved model to: {model_path}")
            
            # Save metrics
            metrics_path = MODEL_DIR / f"metrics_{horizon_name}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
        return models

    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    train_aqi_model()
