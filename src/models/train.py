import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import logging
from pathlib import Path
from datetime import datetime

# Import Hopsworks processor (adjust import as needed)
try:
    from src.features.process import AQI3DayForecastProcessor
except ImportError:
    from features.process import AQI3DayForecastProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_DIR = Path(__file__).parent
os.makedirs(MODEL_DIR, exist_ok=True)

def train_aqi_model():
    try:
        # 1. Get data from Hopsworks
        logging.info("Fetching data from Hopsworks...")
        processor = AQI3DayForecastProcessor()
        features, targets = processor.get_3day_forecast_data(lookback_days=120)
        
        # 2. Select only the key features we care about
        key_features = features[['pm2_5', 'pm10', 'co']]  # Add other confirmed important features
        targets = targets.round().astype(int)  # Ensure AQI is integer
        
        # 3. Time-based train-test split (no future leakage)
        split_idx = int(0.8 * len(features))
        X_train, X_test = key_features.iloc[:split_idx], key_features.iloc[split_idx:]
        y_train, y_test = targets.iloc[:split_idx], targets.iloc[split_idx:]
        
        # 4. Train separate models for each horizon
        models = {}
        horizons = {
            '24h': 0,
            '48h': 1, 
            '72h': 2
        }
        
        for horizon_name, horizon_idx in horizons.items():
            logging.info(f"\n=== Training {horizon_name} model ===")
            
            # Simple Random Forest with balanced classes
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
            
            # Train on the specific horizon target
            model.fit(X_train, y_train.iloc[:, horizon_idx])
            models[horizon_name] = model
            
            # Evaluate
            preds = model.predict(X_test)
            true_values = y_test.iloc[:, horizon_idx]
            
            logging.info(f"Accuracy: {accuracy_score(true_values, preds):.2f}")
            logging.info("\nClassification Report:")
            logging.info(classification_report(true_values, preds))
            
            # Save model with timestamp
            model_path = MODEL_DIR / f"3day_forecaster_{horizon_name}.pkl"
            joblib.dump(model, model_path)
            logging.info(f"Saved model to: {model_path}")
        
        return models

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    train_aqi_model()


