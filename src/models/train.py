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
            'mse': mean_squared_error(true_aqi, pred_aqi),  # Added MSE
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

def display_predictions_and_metrics(y_true, y_pred, horizon, fold=None):
    """Display predictions and key metrics in a formatted way"""
    # Convert to continuous AQI for display
    true_aqi = np.interp(y_true, [1, 6], [0, 500])
    pred_aqi = np.interp(y_pred, [1, 6], [0, 500])
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(true_aqi, pred_aqi))
    r2 = r2_score(true_aqi, pred_aqi)
    mse = mean_squared_error(true_aqi, pred_aqi)
    mae = mean_absolute_error(true_aqi, pred_aqi)
    
    # Display header
    fold_text = f"Fold {fold} - " if fold is not None else ""
    print(f"\n{'='*60}")
    print(f"{fold_text}{horizon} AQI Prediction Results")
    print(f"{'='*60}")
    
    # Display metrics
    print(f"Model Performance Metrics:")
    print(f"  R² Score:     {r2:.4f}")
    print(f"  RMSE:         {rmse:.2f}")
    print(f"  MSE:          {mse:.2f}")
    print(f"  MAE:          {mae:.2f}")
    
    # Display sample predictions (first 10)
    print(f"\nSample Predictions (First 10):")
    print(f"{'Index':<8} {'True AQI':<10} {'Pred AQI':<10} {'Error':<10}")
    print(f"{'-'*40}")
    
    for i in range(min(10, len(true_aqi))):
        error = abs(true_aqi[i] - pred_aqi[i])
        print(f"{i+1:<8} {true_aqi[i]:<10.1f} {pred_aqi[i]:<10.1f} {error:<10.1f}")
    
    # Statistical summary
    errors = np.abs(true_aqi - pred_aqi)
    print(f"\nPrediction Error Statistics:")
    print(f"  Mean Error:   {np.mean(errors):.2f}")
    print(f"  Std Error:    {np.std(errors):.2f}")
    print(f"  Max Error:    {np.max(errors):.2f}")
    print(f"  Min Error:    {np.min(errors):.2f}")
    
    return {
        'rmse': rmse,
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'predictions': pred_aqi.tolist(),
        'true_values': true_aqi.tolist()
    }

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
            # Ensure y_train and y_test are DataFrames, not Series
            if isinstance(y_train, pd.Series):
                y_train = targets.loc[y_train.index]
            if isinstance(y_test, pd.Series):
                y_test = targets.loc[y_test.index]
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
        
        # Store results for summary
        all_results = {}
        
        # 6. Time-series cross validation with enhanced display
        print("\n" + "="*80)
        print("CROSS-VALIDATION RESULTS")
        print("="*80)
        
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced for stability
        for i, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            # Convert numpy arrays to pandas-compatible indices
            train_idx_pd = X_train.index[train_idx]
            val_idx_pd = X_train.index[val_idx]
            
            fold_metrics = {}
            for horizon in ['24h', '48h', '72h']:
                # Get the correct column index for this horizon
                horizon_columns = ['aqi_current', 'aqi_24h', 'aqi_48h', 'aqi_72h']
                target_col = f'aqi_{horizon}'
                
                # Handle case where target_col might not exist or be named differently
                if target_col in y_train.columns:
                    target_series = y_train[target_col]
                elif horizon in y_train.columns:
                    target_series = y_train[horizon]
                else:
                    # Try to find by index
                    try:
                        h_idx = horizon_columns.index(target_col)
                        target_series = y_train.iloc[:, h_idx]
                    except (ValueError, IndexError):
                        logging.warning(f"Could not find target column for {horizon}, skipping...")
                        continue
                
                # Fixed indexing using .loc[]
                X_fold_train = X_train.loc[train_idx_pd]
                y_fold_train = target_series.loc[train_idx_pd]
                X_fold_val = X_train.loc[val_idx_pd]
                y_fold_val = target_series.loc[val_idx_pd]
                
                model.fit(X_fold_train, y_fold_train)
                preds = model.predict(X_fold_val)
                
                # Display predictions and metrics
                results = display_predictions_and_metrics(
                    y_fold_val, preds, horizon, fold=i+1
                )
                fold_metrics[horizon] = results
            
            all_results[f'fold_{i+1}'] = fold_metrics
        
        # 7. Final training and evaluation with enhanced display
        print("\n" + "="*80)
        print("FINAL TEST SET RESULTS")
        print("="*80)
        
        final_results = {}
        for horizon in ['24h', '48h', '72h']:
            # Get the correct target column for this horizon
            horizon_columns = ['aqi_current', 'aqi_24h', 'aqi_48h', 'aqi_72h']
            target_col = f'aqi_{horizon}'
            
            # Handle case where target_col might not exist or be named differently
            if target_col in y_train.columns:
                y_train_target = y_train[target_col]
                y_test_target = y_test[target_col]
            elif horizon in y_train.columns:
                y_train_target = y_train[horizon]
                y_test_target = y_test[horizon]
            else:
                # Try to find by index
                try:
                    h_idx = horizon_columns.index(target_col)
                    y_train_target = y_train.iloc[:, h_idx]
                    y_test_target = y_test.iloc[:, h_idx]
                except (ValueError, IndexError):
                    logging.error(f"Could not find target column for {horizon}, skipping...")
                    continue
            
            model.fit(X_train, y_train_target)
            test_preds = model.predict(X_test)
            
            # Display final test results
            results = display_predictions_and_metrics(
                y_test_target, test_preds, horizon
            )
            final_results[horizon] = results
            
            # Save model and enhanced metrics
            joblib.dump(model, MODEL_DIR / f"aqi_{horizon}_model.pkl")
            
            # Save detailed metrics including predictions
            detailed_metrics = evaluate_model(
                y_test_target, test_preds, return_continuous=True
            )
            detailed_metrics['display_results'] = results
            
            with open(MODEL_DIR / f"aqi_{horizon}_metrics.json", 'w') as f:
                json.dump(detailed_metrics, f, indent=2)
        
        # Summary table
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        print(f"{'Horizon':<10} {'R²':<8} {'RMSE':<8} {'MSE':<10} {'MAE':<8}")
        print("-" * 50)
        
        for horizon in ['24h', '48h', '72h']:
            r = final_results[horizon]
            print(f"{horizon:<10} {r['r2']:<8.3f} {r['rmse']:<8.1f} {r['mse']:<10.1f} {r['mae']:<8.1f}")
        
        return all_results, final_results
                
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

# Add this function to run the training
if __name__ == "__main__":
    cv_results, test_results = train_aqi_model()
    print(f"\nTraining completed successfully!")
    print(f"Results saved to {MODEL_DIR}")
