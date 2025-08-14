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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
import logging
from pathlib import Path
from datetime import datetime

# Import processor
try:
    from src.features.process import AQI3DayForecastProcessor
except ImportError as e:
    logging.error(f"Failed to import AQI3DayForecastProcessor: {str(e)}")
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

def stratified_time_split(features, targets, test_size=0.3):
    """Time-based split ensuring all classes in test set"""
    split_idx = int(len(features) * (1 - test_size))
    
    # Find split where test set has both classes
    while split_idx > 0:
        test_classes = targets.iloc[split_idx:].nunique()
        if test_classes == targets.nunique():
            break
        split_idx -= 24  # Move back one day at a time
    else:
        raise ValueError("Cannot create balanced test set")
    
    return (features.iloc[:split_idx], features.iloc[split_idx:],
            targets.iloc[:split_idx], targets.iloc[split_idx:])

def preprocess_data(features, targets):
    """Enhanced feature engineering with temporal validation"""
    # Convert targets to categorical bins
    bins = [0, 50, 100, 150, 200, 300, 500]
    labels = [1, 2, 3, 4, 5, 6]
    
    binary_targets = pd.DataFrame()
    for col in targets.columns:
        binary_targets[col] = pd.cut(targets[col], bins=bins, labels=labels)
    
    # Drop NA and align indices
    valid_idx = features.dropna().index.intersection(binary_targets.index)
    return features.loc[valid_idx], binary_targets.loc[valid_idx]

def evaluate_model(y_true, y_pred, labels):
    """Robust evaluation with class validation"""
    if len(np.unique(y_true)) < 2:
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': 0.5,
            'confusion_matrix': [[len(y_true), 0], [0, 0]],
            'note': 'single_class_evaluation'
        }
    
    # Rest of evaluation logic...
    return metrics

def train_aqi_model():
    try:
        # 1. Get enhanced data
        processor = AQI3DayForecastProcessor()
        features, targets = processor.get_3day_forecast_data(lookback_days=365)  # 1 year data
        
        # 2. Preprocess with 6-class targets
        features, targets = preprocess_data(features, targets)
        
        # 3. Validate dataset
        class_dist = targets.iloc[:, 0].value_counts(normalize=True)
        if class_dist.min() < 0.1:  # Any class <10%
            logging.warning(f"Severe class imbalance: {class_dist.to_dict()}")
        
        # 4. Stratified time split
        X_train, X_test, y_train, y_test = stratified_time_split(
            features, targets.iloc[:, 0]  # Use first horizon for split
        )
        
        # 5. Enhanced model pipeline with SMOTE
        model = make_imb_pipeline(
            StandardScaler(),
            SMOTE(sampling_strategy='not majority', random_state=42),
            ExtraTreesClassifier(
                n_estimators=200,
                class_weight='balanced',
                max_depth=10,
                min_samples_leaf=5,
                n_jobs=-1
            )
        )
        
        # 6. Time-series cross validation
        tscv = TimeSeriesSplit(n_splits=5)
        for i, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            fold_scores = []
            
            # Train separate models for each horizon
            for horizon in ['24h', '48h', '72h']:
                h_idx = ['aqi_current', 'aqi_24h', 'aqi_48h', 'aqi_72h'].index(f'aqi_{horizon}')
                model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx, h_idx])
                preds = model.predict(X_train.iloc[val_idx])
                score = balanced_accuracy_score(y_train.iloc[val_idx, h_idx], preds)
                fold_scores.append(score)
            
            logging.info(f"Fold {i+1} CV Scores - 24h:{fold_scores[0]:.2f} "
                       f"48h:{fold_scores[1]:.2f} 72h:{fold_scores[2]:.2f}")
        
        # Final training and evaluation
        for horizon in ['24h', '48h', '72h']:
            h_idx = ['aqi_current', 'aqi_24h', 'aqi_48h', 'aqi_72h'].index(f'aqi_{horizon}')
            model.fit(X_train, y_train.iloc[:, h_idx])
            
            # Save model and metrics
            joblib.dump(model, MODEL_DIR / f"aqi_{horizon}_model.pkl")
            
            # Evaluate
            preds = model.predict(X_test)
            metrics = evaluate_model(y_test.iloc[:, h_idx], preds, labels=range(1,7))
            
            with open(MODEL_DIR / f"aqi_{horizon}_metrics.json", 'w') as f:
                json.dump(metrics, f)
                
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise
