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

def preprocess_targets(targets):
    """Convert to binary classification (good vs bad air quality)"""
    # Merge classes 2 and 3 into "moderate" (1), keep 4 as "poor" (2)
    return targets.apply(lambda x: np.where(x >= 4, 2, 1))

def evaluate_model(y_true, y_pred):
    """Simplified binary evaluation"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'report': classification_report(y_true, y_pred, output_dict=True)
    }
    return metrics

def train_aqi_model():
    try:
        # 1. Get data from Hopsworks
        logging.info("Fetching data from Hopsworks...")
        processor = AQI3DayForecastProcessor()
        features, targets = processor.get_3day_forecast_data(lookback_days=180)  # Increased lookback
        
        # 2. Select key features and convert to binary targets
        key_features = features[['pm2_5', 'pm10', 'co', 'no2', 'o3']].copy()  # Added more features
        binary_targets = preprocess_targets(targets.round().astype(int))
        
        # 3. Time-based split with larger test set
        split_idx = int(0.7 * len(features))  # 70-30 split
        X_train, X_test = key_features.iloc[:split_idx], key_features.iloc[split_idx:]
        y_train, y_test = binary_targets.iloc[:split_idx], binary_targets.iloc[split_idx:]
        
        # 4. Train models for each horizon
        horizons = {'24h': 0, '48h': 1, '72h': 2}
        models = {}
        
        for horizon_name, horizon_idx in horizons.items():
            logging.info(f"\n=== Training {horizon_name} model ===")
            
            y_train_h = y_train.iloc[:, horizon_idx]
            y_test_h = y_test.iloc[:, horizon_idx]
            
            # Check class distribution
            class_counts = pd.Series(y_train_h).value_counts()
            logging.info(f"Class distribution: {class_counts.to_dict()}")
            
            # Build pipeline with optimized parameters
            model = make_pipeline(
                StandardScaler(),
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_features=0.8,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
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
            metrics = evaluate_model(y_test_h, preds)
            
            logging.info(f"\nPerformance Metrics for {horizon_name}:")
            logging.info(f"Accuracy: {metrics['accuracy']:.2f}")
            logging.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.2f}")
            logging.info("Confusion Matrix:")
            logging.info(metrics['confusion_matrix'])
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test_h, preds))
            
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
