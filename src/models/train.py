# src/train.py
import pandas as pd
import numpy as np
from pycaret.regression import *
from features.process import AQI3DayForecastProcessor
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AQIForecastTrainer:
    def __init__(self):
        self.processor = AQI3DayForecastProcessor()
        self.target_cols = ['aqi_current', 'aqi_24h', 'aqi_48h', 'aqi_72h']
        
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality before training"""
        if len(df) < 100:
            logger.error(f"Insufficient data points ({len(df)} < 100)")
            return False
        
        # Fixed: AQI data typically has limited discrete values (e.g., 1-6 scale)
        # Changed from 5 to 2 minimum unique values
        unique_targets = df['target'].nunique()
        if unique_targets < 2:
            logger.error(f"Target has insufficient variability ({unique_targets} unique values)")
            return False
        
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            logger.error(f"Data contains {null_count} null values")
            return False
            
        # Additional check: ensure target has reasonable distribution
        target_counts = df['target'].value_counts()
        min_class_size = len(df) * 0.05  # At least 5% per class
        if (target_counts < min_class_size).any():
            logger.warning(f"Some target classes have very few samples: {target_counts.to_dict()}")
            # Don't fail, just warn - small classes are common in AQI data
        
        logger.info(f"Data validation passed: {len(df)} samples, {unique_targets} unique targets")
        logger.info(f"Target distribution: {target_counts.to_dict()}")   
        return True
        
    def prepare_data(self):
        """Get and validate processed features and targets"""
        features, targets, raw_df = self.processor.get_3day_forecast_data()  # Fixed: unpack 3 values
        
        # Optional: Log raw data info for debugging
        logger.info(f"Raw data shape: {raw_df.shape}")
        if 'timestamp' in raw_df.columns:
            logger.info(f"Date range: {raw_df['timestamp'].min()} to {raw_df['timestamp'].max()}")
        
        self.datasets = {}
        for horizon in self.target_cols:
            df = features.copy()
            df['target'] = targets[horizon].astype('float32')
            df = df.dropna()
            
            if not self._validate_data(df):
                logger.error(f"Skipping {horizon} due to data quality issues")
                continue
                
            self.datasets[horizon] = df
            logger.info(f"Dataset prepared for {horizon}: {df.shape} samples")
            
    def time_series_split(self, df: pd.DataFrame, test_size: float = 0.2):
        """Custom time-series split"""
        n_test = int(len(df) * test_size)
        return df.iloc[:-n_test], df.iloc[-n_test:]
    
    def train_models(self):
        """Robust training with fallback logic"""
        self.results = {}
        
        for horizon, data in self.datasets.items():
            logger.info(f"\n{'='*50}\nTraining for target: {horizon}\n{'='*50}")
            
            try:
                train, test = self.time_series_split(data)
                
                # Minimal PyCaret setup
                exp = setup(
                    data=train,
                    target='target',
                    train_size=0.8,
                    fold_strategy="timeseries",
                    fold=3,
                    verbose=False,
                    data_split_shuffle=False,
                    fold_shuffle=False,
                    normalize=True,
                    transformation=False,
                    feature_selection=False,
                    session_id=42
                )
                
                # First try LightGBM directly (most likely to work)
                try:
                    lgbm = create_model('lightgbm', verbose=False)
                    tuned_lgbm = tune_model(lgbm, optimize='MAE', verbose=False)
                    best_model = finalize_model(tuned_lgbm)
                except Exception as e:
                    # Fallback to simple linear regression
                    logger.warning(f"LightGBM failed ({str(e)}), trying linear regression")
                    lr = create_model('lr', verbose=False)
                    best_model = finalize_model(lr)
                
                # Evaluate - Fixed the Label column issue
                try:
                    test_pred = predict_model(best_model, data=test)
                    # Check if 'prediction_label' or 'Label' exists
                    pred_col = 'prediction_label' if 'prediction_label' in test_pred.columns else 'Label'
                    test_mae = np.mean(np.abs(test_pred['target'] - test_pred[pred_col]))
                except Exception as e:
                    logger.warning(f"Evaluation failed ({str(e)}), using cross-validation score")
                    # Fallback: use the model's built-in evaluation
                    test_mae = evaluate_model(best_model, verbose=False)['MAE'].iloc[0]
                
                # Get feature importance safely
                try:
                    feature_importance = pull().sort_values('Importance', ascending=False)
                except:
                    feature_importance = pd.DataFrame({'Feature': ['N/A'], 'Importance': [0]})
                
                self.results[horizon] = {
                    'best_model': best_model,
                    'test_mae': test_mae,
                    'feature_importance': feature_importance
                }
                
                logger.info(f"Successfully trained model for {horizon} - MAE: {test_mae:.3f}")
                
            except Exception as e:
                logger.error(f"Failed for {horizon}: {str(e)}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                self.results[horizon] = {
                    'error': str(e),
                    'test_mae': None
                }
                
        return self.results
    
    def save_models(self):
        """Save trained models with validation"""
        import joblib
        import os
        
        os.makedirs('models', exist_ok=True)
        for horizon, result in self.results.items():
            if 'best_model' in result:
                try:
                    joblib.dump(result['best_model'], f'models/{horizon}_model.pkl')
                    logger.info(f"Saved model for {horizon}")
                except Exception as e:
                    logger.error(f"Failed to save model for {horizon}: {str(e)}")
            else:
                logger.warning(f"No model saved for {horizon} due to previous errors")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = AQIForecastTrainer()
    trainer.prepare_data()
    
    if not trainer.datasets:
        logger.error("No valid datasets available for training")
    else:
        results = trainer.train_models()
        trainer.save_models()
        
        print("\nTraining Summary:")
        for horizon, res in results.items():
            print(f"{horizon}:")
            if 'error' in res:
                print(f"  Error: {res['error']}")
            else:
                print(f"  Model: {type(res['best_model']).__name__}")
                print(f"  Test MAE: {res['test_mae']:.3f}")
                print("  Top Features:")
                if not res['feature_importance'].empty:
                    print(res['feature_importance'].head(3).to_string())
                else:
                    print("  No feature importance available")
