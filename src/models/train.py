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
            logger.error("Insufficient data points (<100)")
            return False
        if df['target'].nunique() < 5:
            logger.error("Target has insufficient variability")
            return False
        if df.isnull().sum().sum() > 0:
            logger.error("Data contains null values")
            return False
        return True
        
    def prepare_data(self):
        """Get and validate processed features and targets"""
        features, targets = self.processor.get_3day_forecast_data()
        
        self.datasets = {}
        for horizon in self.target_cols:
            df = features.copy()
            df['target'] = targets[horizon].astype('float32')
            df = df.dropna()
            
            if not self._validate_data(df):
                logger.error(f"Skipping {horizon} due to data quality issues")
                continue
                
            self.datasets[horizon] = df
            
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
                except:
                    # Fallback to simple linear regression
                    logger.warning("LightGBM failed, trying linear regression")
                    lr = create_model('lr', verbose=False)
                    best_model = finalize_model(lr)
                
                # Evaluate
                test_pred = predict_model(best_model, data=test)
                test_mae = np.mean(np.abs(test_pred['target'] - test_pred['Label']))
                
                self.results[horizon] = {
                    'best_model': best_model,
                    'test_mae': test_mae,
                    'feature_importance': pull().sort_values('Importance', ascending=False)
                }
                
            except Exception as e:
                logger.error(f"Failed for {horizon}: {str(e)}")
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
                joblib.dump(result['best_model'], f'models/{horizon}_model.pkl')
                logger.info(f"Saved model for {horizon}")
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
                print(f"  Test MAE: {res['test_mae']:.2f}")
                print("  Top Features:")
                print(res['feature_importance'].head(3).to_string())
