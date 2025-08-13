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
import logging
from pathlib import Path
from datetime import datetime

# Import Hopsworks processor
try:
    from src.features.process import AQI3DayForecastProcessor
except ImportError:
    from features.process import AQI3DayForecastProcessor

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

def evaluate_model(y_true, y_pred, horizon):
    """Enhanced evaluation with proper metric handling"""
    # Convert to numpy arrays to avoid pandas index issues
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Get unique classes present in both y_true and y_pred
    present_classes = np.union1d(np.unique(y_true), np.unique(y_pred))
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=present_classes).tolist(),
        'horizon': horizon,
        'timestamp': datetime.now().isoformat()
    }
    
    # Handle classification report carefully
    report = classification_report(
        y_true, y_pred,
        labels=present_classes,
        zero_division=0,
        output_dict=True
    )
    metrics.update(report)
    
    # Flatten nested dictionary structure
    flat_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):  # Handle nested class metrics
                    for class_key, class_value in subvalue.items():
                        flat_metrics[f"{key}_{subkey}_{class_key}"] = class_value
                else:
                    flat_metrics[f"{key}_{subkey}"] = subvalue
        else:
            flat_metrics[key] = value
    
    return flat_metrics

def train_aqi_model():
    try:
        # 1. Get data from Hopsworks
        logging.info("Fetching data from Hopsworks...")
        processor = AQI3DayForecastProcessor()
        features, targets = processor.get_3day_forecast_data(lookback_days=120)
        
        # 2. Select key features and ensure integer targets
        key_features = features[['pm2_5', 'pm10', 'co']].copy()
        targets = targets.round().astype(int).copy()
        
        # 3. Handle extreme class imbalance by merging rare classes
        # Merge class 2 into class 3 if there are too few samples
        for col in targets.columns:
            class_counts = targets[col].value_counts()
            if class_counts.get(2, 0) < 5:  # If fewer than 5 samples of class 2
                targets[col] = targets[col].replace(2, 3)
        
        # 4. Time-based train-test split with gap
        split_idx = int(0.8 * len(features))
        X_train, X_test = key_features.iloc[:split_idx], key_features.iloc[split_idx:]
        y_train, y_test = targets.iloc[:split_idx], targets.iloc[split_idx:]
        
        # 5. Train models for each horizon
        horizons = {'24h': 0, '48h': 1, '72h': 2}
        models = {}
        
        for horizon_name, horizon_idx in horizons.items():
            logging.info(f"\n=== Training {horizon_name} model ===")
            
            y_train_h = y_train.iloc[:, horizon_idx]
            y_test_h = y_test.iloc[:, horizon_idx]
            
            # Check final class distribution
            class_counts = y_train_h.value_counts()
            logging.info(f"Final class distribution: {class_counts.to_dict()}")
            
            # Build pipeline with class weighting
            model = make_pipeline(
                StandardScaler(),
                ExtraTreesClassifier(
                    n_estimators=300,
                    max_features='sqrt',
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    bootstrap=True,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                )
            )
            
            model.fit(X_train, y_train_h)
            models[horizon_name] = model
            
            # Evaluate
            preds = model.predict(X_test)
            metrics = evaluate_model(y_test_h, preds, horizon_name)
            
            logging.info(f"\nPerformance Metrics for {horizon_name}:")
            logging.info(f"- Accuracy: {metrics['accuracy']:.2f}")
            logging.info(f"- Balanced Accuracy: {metrics['balanced_accuracy']:.2f}")
            logging.info("\nClassification Report:")
            logging.info(classification_report(
                y_test_h, preds,
                zero_division=0,
                labels=np.unique(np.concatenate([y_test_h, preds]))
            ))
            
            # Save model
            model_path = MODEL_DIR / f"3day_forecaster_{horizon_name}.pkl"
            joblib.dump(model, model_path)
            logging.info(f"Saved model to: {model_path}")
            
            # Save metrics as JSON (properly serialized)
            metrics_path = MODEL_DIR / f"metrics_{horizon_name}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logging.info(f"Saved metrics to: {metrics_path}")
        
        return models

    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    train_aqi_model()
