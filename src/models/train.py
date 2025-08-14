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

# Import processor with absolute import only
from src.features.process import AQI3DayForecastProcessor

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
    """Enhanced feature engineering with proper validation"""
    # Select core features
    required_features = ['pm2_5', 'pm10', 'co', 'no2', 'o3']
    missing = [f for f in required_features if f not in features.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    
    features = features[required_features].copy()
    targets = targets.copy()
    
    # Add meaningful temporal features
    for col in ['pm2_5', 'pm10']:
        features[f'{col}_24h_avg'] = features[col].rolling(24, min_periods=12).mean()
        features[f'{col}_48h_avg'] = features[col].rolling(48, min_periods=24).mean()
        features[f'{col}_24h_diff'] = features[col] - features[col].shift(24)
    
    # Convert to binary classification (1 = good/moderate, 2 = poor)
    binary_targets = targets.apply(lambda x: np.where(x >= 4, 2, 1))
    
    # Drop NA rows and ensure alignment
    valid_idx = features.dropna().index.intersection(binary_targets.index)
    features = features.loc[valid_idx].reset_index(drop=True)
    binary_targets = binary_targets.loc[valid_idx].reset_index(drop=True)
    
    return features, binary_targets

def evaluate_model(y_true, y_pred, labels=[1, 2]):
    """Robust evaluation with explicit label handling"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Get actually present labels
    present_labels = np.unique(np.concatenate([y_true, y_pred]))
    eval_labels = [l for l in labels if l in present_labels]
    
    if len(present_labels) < 2:
        logging.warning(f"Only one class present in evaluation: {present_labels}")
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'class_distribution': {
            'actual': dict(zip(*np.unique(y_true, return_counts=True)),
            'predicted': dict(zip(*np.unique(y_pred, return_counts=True)))
        }
    }
    
    # Only include classification report if we have both classes
    if len(eval_labels) > 1:
        metrics['report'] = classification_report(
            y_true, y_pred, labels=eval_labels, zero_division=0, output_dict=True)
    
    return metrics

def train_aqi_model():
    try:
        # 1. Get data from Hopsworks
        logging.info("Fetching data from Hopsworks...")
        processor = AQI3DayForecastProcessor()
        features, targets = processor.get_3day_forecast_data(lookback_days=180)
        
        # 2. Preprocess data with strict validation
        features, binary_targets = preprocess_data(features, targets.round().astype(int))
        
        # 3. Verify we have sufficient data
        if len(features) < 100:
            raise ValueError(f"Insufficient data samples: {len(features)}")
        
        # 4. Time-based split with temporal gap
        split_idx = int(0.7 * len(features))
        X_train, X_test = features.iloc[:split_idx].copy(), features.iloc[split_idx:].copy()
        y_train, y_test = binary_targets.iloc[:split_idx].copy(), binary_targets.iloc[split_idx:].copy()
        
        # 5. Verify both classes exist in training data
        for col in y_train.columns:
            class_counts = y_train[col].value_counts()
            logging.info(f"\nTraining class distribution ({col}):\n{class_counts.to_string()}")
            
            if len(class_counts) < 2:
                raise ValueError(f"Only one class present in training data for {col}")
            if min(class_counts) < 10:
                logging.warning(f"Very few samples ({min(class_counts)}) for one class in {col}")

        # 6. Train models for each horizon
        horizons = {'24h': 0, '48h': 1, '72h': 2}
        models = {}
        
        for horizon_name, horizon_idx in horizons.items():
            logging.info(f"\n=== Training {horizon_name} model ===")
            
            y_train_h = y_train.iloc[:, horizon_idx]
            y_test_h = y_test.iloc[:, horizon_idx]
            
            # Build pipeline with careful regularization
            model = make_pipeline(
                StandardScaler(),
                ExtraTreesClassifier(
                    n_estimators=150,  # Reduced for faster training
                    max_features=0.7,  # More randomness
                    max_depth=8,       # Shallower trees
                    min_samples_split=15,
                    min_samples_leaf=8,
                    bootstrap=True,
                    random_state=42,
                    class_weight='balanced_subsample',  # Better for imbalanced data
                    n_jobs=-1
                )
            )
            
            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train_h.iloc[train_idx], y_train_h.iloc[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                preds = model.predict(X_fold_val)
                score = balanced_accuracy_score(y_fold_val, preds)
                cv_scores.append(score)
                logging.info(f"Fold {len(cv_scores)} CV Balanced Accuracy: {score:.3f}")
            
            # Train final model on all training data
            model.fit(X_train, y_train_h)
            models[horizon_name] = model
            
            # Evaluate on test set
            preds = model.predict(X_test)
            metrics = evaluate_model(y_test_h, preds)
            
            logging.info(f"\nTest Performance - {horizon_name}:")
            logging.info(f"Accuracy: {metrics['accuracy']:.3f}")
            logging.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
            logging.info("Confusion Matrix:")
            logging.info(np.array2string(np.array(metrics['confusion_matrix']), 
                       formatter={'int': lambda x: f"{x:4d}"}))
            
            if 'report' in metrics:
                logging.info("\nClassification Report:")
                logging.info(classification_report(y_test_h, preds, zero_division=0))
            
            # Save model and metrics
            model_path = MODEL_DIR / f"aqi_forecaster_{horizon_name}.pkl"
            joblib.dump(model, model_path)
            
            metrics_path = MODEL_DIR / f"metrics_{horizon_name}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logging.info(f"Saved model and metrics for {horizon_name}")
        
        return models

    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    train_aqi_model()
