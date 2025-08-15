import logging
from pathlib import Path
import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                           balanced_accuracy_score, confusion_matrix,
                           mean_absolute_error, mean_squared_error, r2_score)
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

# AQI bins configuration
AQI_BINS = [0, 50, 100, 150, 200, 300, 500]
AQI_LABELS = [1, 2, 3, 4, 5, 6]

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
    binary_targets = pd.DataFrame()
    for col in targets.columns:
        binary_targets[col] = pd.cut(targets[col], bins=AQI_BINS, labels=AQI_LABELS)
    
    # Drop NA and align indices
    valid_idx = features.dropna().index.intersection(binary_targets.index)
    return features.loc[valid_idx], binary_targets.loc[valid_idx]

def aqi_category_accuracy(y_true, y_pred):
    """Check if predictions fall in correct EPA category"""
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

def high_aqi_recall(y_true, y_pred, threshold=4):
    """Detection rate for unhealthy days (category 4+)"""
    unhealthy = y_true >= threshold
    if np.sum(unhealthy) == 0:
        return np.nan
    return np.sum(y_pred[unhealthy] >= threshold) / np.sum(unhealthy)

def evaluate_model(y_true, y_pred, return_continuous=False):
    """
    Comprehensive evaluation with both classification and regression metrics
    
    Args:
        return_continuous: If True, returns metrics for continuous AQI values
    """
    # Convert from categorical to continuous AQI
    true_aqi = np.interp(y_true, [1, 6], [0, 500])
    pred_aqi = np.interp(y_pred, [1, 6], [0, 500])
    
    metrics = {
        'classification': {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'category_accuracy': aqi_category_accuracy(y_true, y_pred),
            'high_aqi_recall': high_aqi_recall(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=AQI_LABELS).tolist()
        },
        'regression': {
            'mae': mean_absolute_error(true_aqi, pred_aqi),
            'rmse': np.sqrt(mean_squared_error(true_aqi, pred_aqi)),
            'r2': r2_score(true_aqi, pred_aqi)
        }
    }
    
    if return_continuous:
        metrics['continuous_values'] = {
            'true_aqi': true_aqi.tolist(),
            'pred_aqi': pred_aqi.tolist()
        }
    
    return metrics

def train_aqi_model():
    try:
        # 1. Get enhanced data
        processor = AQI3DayForecastProcessor()
        features, targets = processor.get_3day_forecast_data(lookback_days=365)
        
        # 2. Preprocess with 6-class targets
        features, targets = preprocess_data(features, targets)
        
        # 3. Validate dataset - handle single-class case
        class_dist = targets.iloc[:, 0].value_counts(normalize=True)
        if len(class_dist) < 2:
            logging.error("Insufficient class diversity - only one AQI category present")
            return None
        
        # 4. Stratified time split with fallback
        try:
            X_train, X_test, y_train, y_test = stratified_time_split(
                features, targets.iloc[:, 0]
            )
        except ValueError as e:
            logging.warning(f"Could not create balanced split: {str(e)}")
            # Fallback to simple time split
            split_idx = int(len(features) * 0.7)
            X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_test = targets.iloc[:split_idx], targets.iloc[split_idx:]
        
        # 5. Enhanced model pipeline with class weighting
        model = make_imb_pipeline(
            StandardScaler(),
            SMOTE(sampling_strategy='auto', random_state=42),
            ExtraTreesClassifier(
                n_estimators=200,
                class_weight='balanced',
                max_depth=10,
                min_samples_leaf=5,
                n_jobs=-1
            )
        )
        
        # 6. Time-series cross validation with fixed indexing
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced for stability
        for i, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            # Convert numpy arrays to pandas-compatible indices
            train_idx_pd = X_train.index[train_idx]
            val_idx_pd = X_train.index[val_idx]
            
            fold_metrics = {}
            for horizon in ['24h', '48h', '72h']:
                h_idx = ['aqi_current', 'aqi_24h', 'aqi_48h', 'aqi_72h'].index(f'aqi_{horizon}')
                
                # Fixed indexing using .loc[]
                X_fold_train = X_train.loc[train_idx_pd]
                y_fold_train = y_train.loc[train_idx_pd, y_train.columns[h_idx]]
                X_fold_val = X_train.loc[val_idx_pd]
                y_fold_val = y_train.loc[val_idx_pd, y_train.columns[h_idx]]
                
                model.fit(X_fold_train, y_fold_train)
                preds = model.predict(X_fold_val)
                
                metrics = evaluate_model(y_fold_val, preds)
                logging.info(
                    f"Fold {i+1} {horizon} - "
                    f"R²: {metrics['regression']['r2']:.2f} | "
                    f"MAE: {metrics['regression']['mae']:.1f}"
                )
        
        # 7. Final training and evaluation with detailed printing
        print("\nFinal Model Evaluation Results:")
        print("="*60)
        print(f"{'Horizon':<10}{'R²':<10}{'RMSE':<10}{'MAE':<10}{'Accuracy':<10}{'Balanced Acc':<15}")
        print("-"*60)
        
        for horizon in ['24h', '48h', '72h']:
            h_idx = ['aqi_current', 'aqi_24h', 'aqi_48h', 'aqi_72h'].index(f'aqi_{horizon}')
            
            model.fit(X_train, y_train.iloc[:, h_idx])
            test_preds = model.predict(X_test)
            metrics = evaluate_model(y_test.iloc[:, h_idx], test_preds)
            
            # Print metrics in a formatted way
            print(f"{horizon:<10}"
                  f"{metrics['regression']['r2']:.3f}{'':<2}"
                  f"{metrics['regression']['rmse']:.1f}{'':<4}"
                  f"{metrics['regression']['mae']:.1f}{'':<4}"
                  f"{metrics['classification']['accuracy']:.3f}{'':<4}"
                  f"{metrics['classification']['balanced_accuracy']:.3f}")
            
            # Save model and metrics
            joblib.dump(model, MODEL_DIR / f"aqi_{horizon}_model.pkl")
            with open(MODEL_DIR / f"aqi_{horizon}_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
        
        print("="*60)
        print("Note: RMSE and MAE are in AQI units (0-500 scale)")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise
