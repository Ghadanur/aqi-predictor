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

def preprocess_data(features, targets):
    """Handle feature engineering and target processing"""
    # Feature engineering
    features = features[['pm2_5', 'pm10', 'co', 'no2', 'o3']].copy()
    
    # Add rolling averages
    for col in ['pm2_5', 'pm10']:
        features[f'{col}_24h_avg'] = features[col].rolling(24).mean()
        features[f'{col}_48h_avg'] = features[col].rolling(48).mean()
    
    # Convert to binary classification (1 = good/moderate, 2 = poor)
    binary_targets = targets.apply(lambda x: np.where(x >= 4, 2, 1))
    
    # Drop rows with NaN values from rolling averages
    valid_idx = features.dropna().index
    return features.loc[valid_idx], binary_targets.loc[valid_idx]

def evaluate_model(y_true, y_pred, labels=[1, 2]):
    """Robust evaluation with proper label handling"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Ensure we have both classes represented
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
        
        # 2. Preprocess data
        features, binary_targets = preprocess_data(features, targets.round().astype(int))
        
        # 3. Time-based split with temporal gap
        split_idx = int(0.7 * len(features))
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = binary_targets.iloc[:split_idx], binary_targets.iloc[split_idx:]
        
        # Verify we have both classes in training data
        for col in y_train.columns:
            if len(y_train[col].unique()) < 2:
                raise ValueError(f"Not enough classes in training data for {col}")
        
        # 4. Train models for each horizon
        horizons = {'24h': 0, '48h': 1, '72h': 2}
        models = {}
        
        for horizon_name, horizon_idx in horizons.items():
            logging.info(f"\n=== Training {horizon_name} model ===")
            
            y_train_h = y_train.iloc[:, horizon_idx]
            y_test_h = y_test.iloc[:, horizon_idx]
            
            # Check class distribution
            class_counts = y_train_h.value_counts()
            logging.info(f"Class distribution:\n{class_counts.to_string()}")
            
            # Build pipeline with careful parameter tuning
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
            
            # Add cross-validation
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
                    best_model.fit(np.concatenate([X_fold_train, X_fold_val]),
                                  np.concatenate([y_fold_train, y_fold_val]))
            
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
