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
        cv_results = {h: [] for h in ['24h', '48h', '72h']}
        
        for i, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            fold_metrics = {}
            
            for horizon in ['24h', '48h', '72h']:
                h_idx = ['aqi_current', 'aqi_24h', 'aqi_48h', 'aqi_72h'].index(f'aqi_{horizon}')
                model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx, h_idx])
                preds = model.predict(X_train.iloc[val_idx])
                
                metrics = evaluate_model(y_train.iloc[val_idx, h_idx], preds)
                cv_results[horizon].append(metrics)
                
                logging.info(
                    f"Fold {i+1} {horizon} - "
                    f"R²: {metrics['regression']['r2']:.2f} | "
                    f"MAE: {metrics['regression']['mae']:.1f} | "
                    f"CatAcc: {metrics['classification']['category_accuracy']:.1%}"
                )
        
        # 7. Final training and evaluation
        for horizon in ['24h', '48h', '72h']:
            h_idx = ['aqi_current', 'aqi_24h', 'aqi_48h', 'aqi_72h'].index(f'aqi_{horizon}')
            
            # Train final model
            model.fit(X_train, y_train.iloc[:, h_idx])
            
            # Evaluate on test set
            test_preds = model.predict(X_test)
            test_metrics = evaluate_model(y_test.iloc[:, h_idx], test_preds, return_continuous=True)
            
            # Save model and metrics
            joblib.dump(model, MODEL_DIR / f"aqi_{horizon}_model.pkl")
            with open(MODEL_DIR / f"aqi_{horizon}_metrics.json", 'w') as f:
                json.dump(test_metrics, f, indent=2)
            
            # Log final performance
            logging.info(f"\n=== Final {horizon} Model Performance ===")
            logging.info(f"R²: {test_metrics['regression']['r2']:.3f}")
            logging.info(f"MAE: {test_metrics['regression']['mae']:.1f} AQI points")
            logging.info(f"Category Accuracy: {test_metrics['classification']['category_accuracy']:.1%}")
            logging.info(f"High AQI Recall: {test_metrics['classification']['high_aqi_recall']:.1%}")
            
            # Save predictions for analysis
            pd.DataFrame({
                'true_aqi': test_metrics['continuous_values']['true_aqi'],
                'pred_aqi': test_metrics['continuous_values']['pred_aqi']
            }).to_csv(MODEL_DIR / f"aqi_{horizon}_predictions.csv", index=False)
                
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    train_aqi_model()
