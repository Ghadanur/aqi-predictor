import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                           balanced_accuracy_score, confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
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
    """Enhanced evaluation with more metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    # Handle classification report with zero_division parameter
    report = classification_report(
        y_true, y_pred, 
        zero_division=0,  # Silences warnings
        output_dict=True
    )
    metrics.update(report)
    
    return metrics

def train_aqi_model():
    try:
        # 1. Get data from Hopsworks
        logging.info("Fetching data from Hopsworks...")
        processor = AQI3DayForecastProcessor()
        features, targets = processor.get_3day_forecast_data(lookback_days=120)
        
        # 2. Select key features and ensure integer targets
        key_features = features[['pm2_5', 'pm10', 'co']]
        targets = targets.round().astype(int)
        
        # 3. Time-based train-test split with gap
        split_idx = int(0.8 * len(features))
        X_train, X_test = key_features.iloc[:split_idx], key_features.iloc[split_idx:]
        y_train, y_test = targets.iloc[:split_idx], targets.iloc[split_idx:]
        
        # 4. Train models for each horizon
        horizons = {'24h': 0, '48h': 1, '72h': 2}
        models = {}
        
        for horizon_name, horizon_idx in horizons.items():
            logging.info(f"\n=== Training {horizon_name} model ===")
            
            # Get current horizon targets
            y_train_h = y_train.iloc[:, horizon_idx]
            y_test_h = y_test.iloc[:, horizon_idx]
            
            # Check class distribution
            class_counts = pd.Series(y_train_h).value_counts()
            logging.info(f"Class distribution: {class_counts.to_dict()}")
            
            # Only use SMOTE if we have samples for all classes
            use_smote = all(count > 5 for count in class_counts)
            
            # Build appropriate pipeline
            if use_smote:
                model = make_imb_pipeline(
                    SMOTE(sampling_strategy='not majority', random_state=42),
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        min_samples_leaf=5,
                        random_state=42,
                        class_weight='balanced_subsample',
                        n_jobs=-1
                    )
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=5,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                )
            
            # Train model
            model.fit(X_train, y_train_h)
            models[horizon_name] = model
            
            # Evaluate
            preds = model.predict(X_test)
            metrics = evaluate_model(y_test_h, preds, horizon_name)
            
            logging.info(f"\nPerformance Metrics for {horizon_name}:")
            logging.info(f"- Accuracy: {metrics['accuracy']:.2f}")
            logging.info(f"- Balanced Accuracy: {metrics['balanced_accuracy']:.2f}")
            logging.info("\nConfusion Matrix:")
            logging.info(metrics['confusion_matrix'])
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test_h, preds, zero_division=0))
            
            # Save model
            model_path = MODEL_DIR / f"3day_forecaster_{horizon_name}.pkl"
            joblib.dump(model, model_path)
            logging.info(f"Saved model to: {model_path}")
            
            # Save evaluation metrics
            metrics_path = MODEL_DIR / f"metrics_{horizon_name}.json"
            pd.DataFrame(metrics).to_json(metrics_path)
            logging.info(f"Saved metrics to: {metrics_path}")
        
        return models

    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    train_aqi_model()
