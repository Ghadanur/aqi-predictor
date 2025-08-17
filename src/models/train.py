# src/train.py
import pandas as pd
import numpy as np
from pycaret.regression import *
from features.process import AQI3DayForecastProcessor
import logging
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AQIForecastTrainer:
    def __init__(self):
        self.processor = AQI3DayForecastProcessor()
        self.target_cols = ['aqi_current', 'aqi_24h', 'aqi_48h', 'aqi_72h']
        
    def prepare_data(self):
        """Get processed features and targets"""
        features, targets = self.processor.get_3day_forecast_data()
        
        # Combine features with each target horizon
        self.datasets = {}
        for horizon in self.target_cols:
            df = features.copy()
            df['target'] = targets[horizon]
            self.datasets[horizon] = df.dropna()
            
    def time_series_split(self, df: pd.DataFrame, test_size: float = 0.2):
        """Custom time-series split that maintains temporal order"""
        n_test = int(len(df) * test_size)
        train = df.iloc[:-n_test]
        test = df.iloc[-n_test:]
        return train, test
    
    def train_models(self):
        """Train and compare models for each forecast horizon with robust error handling"""
        self.results = {}
        
        for horizon, data in self.datasets.items():
            logger.info(f"\n{'='*50}\nTraining for target: {horizon}\n{'='*50}")
            
            try:
                # Time-series split
                train, test = self.time_series_split(data)
                
                # PyCaret setup
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
                    transformation=False,  # Disabled for time series
                    remove_multicollinearity=False,  # Disabled for time series
                    feature_selection=False,  # Disabled for time series
                    session_id=42,
                    use_gpu=False
                )
                
                # Compare models with error handling
                try:
                    best_models = compare_models(
                        sort='MAE',
                        include=['lightgbm', 'xgboost', 'catboost', 'rf', 'et'],  # Reduced set
                        n_select=1,  # Only get the best model
                        verbose=False,
                        error_score='raise'
                    )
                    
                    if not best_models or len(best_models) == 0:
                        raise ValueError("No models could be trained")
                        
                    best_model = best_models[0]
                    
                    # Evaluate
                    test_pred = predict_model(best_model, data=test)
                    test_mae = np.mean(np.abs(test_pred['target'] - test_pred['Label']))
                    
                    self.results[horizon] = {
                        'best_model': best_model,
                        'test_mae': test_mae,
                        'feature_importance': pull().sort_values('Importance', ascending=False)
                    }
                    
                    logger.info(f"Successfully trained {type(best_model).__name__} for {horizon}")
                    logger.info(f"Test MAE: {test_mae:.2f}")
                    
                except Exception as model_error:
                    logger.error(f"Model training failed for {horizon}: {str(model_error)}")
                    self.results[horizon] = {
                        'error': str(model_error),
                        'test_mae': None
                    }
                    
            except Exception as setup_error:
                logger.error(f"Setup failed for {horizon}: {str(setup_error)}")
                self.results[horizon] = {
                    'error': str(setup_error),
                    'test_mae': None
                }
                
        return self.results
    
    def save_models(self):
        """Save trained models"""
        import joblib
        import os
        
        os.makedirs('models', exist_ok=True)
        for horizon, result in self.results.items():
            model = result['best_model']
            joblib.dump(model, f'models/{horizon}_model.pkl')
            logger.info(f"Saved model for {horizon}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Train and evaluate models
    trainer = AQIForecastTrainer()
    trainer.prepare_data()
    results = trainer.train_models()
    trainer.save_models()
    
    # Print summary
    print("\nTraining Summary:")
    for horizon, res in results.items():
        print(f"{horizon}:")
        print(f"  Model: {type(res['best_model']).__name__}")
        print(f"  Test MAE: {res['test_mae']:.2f}")
        print("  Top 5 Features:")
        print(res['feature_importance'].head(5).to_string())

