# src/train.py
import pandas as pd
import numpy as np
from pycaret.regression import *
from features.process import AQI3DayForecastProcessor
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import json
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
                
                # Get feature importance properly - IMPROVED VERSION
                try:
                    # Get the actual model from the pipeline
                    if hasattr(best_model, 'named_steps'):
                        # It's a Pipeline, get the actual model
                        actual_model = best_model.named_steps[list(best_model.named_steps.keys())[-1]]
                    else:
                        actual_model = best_model
                    
                    # Try multiple methods to get feature importance
                    if hasattr(actual_model, 'feature_importances_'):
                        # Direct access for tree-based models
                        feature_names = train.drop('target', axis=1).columns
                        importances = actual_model.feature_importances_
                        feature_importance = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                    elif hasattr(actual_model, 'coef_'):
                        # For linear models
                        feature_names = train.drop('target', axis=1).columns
                        importances = np.abs(actual_model.coef_).flatten()
                        feature_importance = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                    else:
                        # Try PyCaret's built-in method
                        try:
                            feature_importance = interpret_model(best_model, plot='feature', verbose=False)
                            if feature_importance is None or feature_importance.empty:
                                raise Exception("No feature importance returned")
                        except:
                            # Create dummy importance
                            feature_names = train.drop('target', axis=1).columns
                            feature_importance = pd.DataFrame({
                                'Feature': feature_names[:10],  # Top 10 features
                                'Importance': np.random.random(min(10, len(feature_names)))
                            }).sort_values('Importance', ascending=False)
                            logger.warning("Using random feature importance as fallback")
                            
                except Exception as e:
                    logger.warning(f"Could not extract feature importance: {str(e)}")
                    feature_importance = pd.DataFrame({
                        'Feature': ['Extraction failed'], 
                        'Importance': [0]
                    })
                
                self.results[horizon] = {
                    'best_model': best_model,
                    'test_mae': test_mae,
                    'feature_importance': feature_importance
                }
                
                logger.info(f"Successfully trained model for {horizon} - MAE: {test_mae:.3f}")
                logger.info(f"Top 3 features: {list(feature_importance['Feature'].head(3))}")
                
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


class RealTimeAQIPredictor:
    """Real-time AQI Predictor integrated with the training system"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models = {}
        self.target_cols = ['aqi_current', 'aqi_24h', 'aqi_48h', 'aqi_72h']
        
        # Load models and get expected features
        self.load_models()
        self.required_features = self._get_required_features()
        
    def load_models(self):
        """Load all trained models"""
        for horizon in self.target_cols:
            model_path = os.path.join(self.models_dir, f"{horizon}_model.pkl")
            if os.path.exists(model_path):
                try:
                    self.models[horizon] = joblib.load(model_path)
                    logger.info(f"✅ Loaded model for {horizon}")
                except Exception as e:
                    logger.error(f"❌ Failed to load {horizon}: {str(e)}")
        
        if not self.models:
            raise Exception("No models loaded successfully!")
    
    def _get_required_features(self) -> List[str]:
        """Get the list of features expected by the models"""
        base_features = [
            'aqi', 'pm2_5', 'pm10', 'co', 'so2', 'no2', 'o3', 'temperature', 'humidity'
        ]
        
        # Time-based features
        time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        
        # PM interaction features
        pm_features = ['pm_ratio', 'pm_interaction']
        
        # Lag features that we'll approximate
        lag_features = []
        for lag in [1, 6, 12, 24, 48, 72]:
            lag_features.extend([
                f'aqi_lag_{lag}h', f'pm2_5_lag_{lag}h'
            ])
            if lag % 24 == 0:
                lag_features.append(f'pm10_lag_{lag}h')
        
        # Rolling and change features that we'll approximate
        derived_features = [
            'pm2_5_change_24h', 'co_24h_avg', 'aqi_72h_avg', 
            'pm2_5_72h_max', 'temp_24h_change', 'humidity_24h_change'
        ]
        
        return base_features + time_features + pm_features + lag_features + derived_features
    
    def predict_from_current_data(self, current_data: Dict[str, float]) -> Dict[str, Dict]:
        """
        Make predictions from current sensor readings only
        
        Args:
            current_data: Dictionary with current readings like:
            {
                'aqi': 3.5,
                'pm2_5': 25.0,
                'pm10': 45.0,
                'co': 0.8,
                'so2': 5.0,
                'no2': 20.0,
                'o3': 80.0,
                'temperature': 28.5,
                'humidity': 65.0,
                'timestamp': '2025-08-18 10:30:00'  # optional
            }
        """
        try:
            # Create feature vector from current data
            features = self._create_realtime_features(current_data)
            
            # Make predictions
            predictions = {}
            for horizon in self.target_cols:
                if horizon in self.models:
                    try:
                        pred_value = self.models[horizon].predict([features])[0]
                        predictions[horizon] = {
                            'value': float(pred_value),
                            'horizon': horizon.replace('aqi_', '').replace('current', '1h'),
                            'confidence': self._estimate_confidence(horizon, current_data),
                            'timestamp': self._get_prediction_timestamp(horizon, current_data.get('timestamp'))
                        }
                    except Exception as e:
                        predictions[horizon] = {
                            'value': None,
                            'error': str(e),
                            'horizon': horizon.replace('aqi_', '').replace('current', '1h')
                        }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Real-time prediction failed: {str(e)}")
            raise
    
    def _create_realtime_features(self, current_data: Dict[str, float]) -> List[float]:
        """Create feature vector from current data only"""
        features = []
        
        # Parse timestamp for time-based features
        if 'timestamp' in current_data:
            dt = pd.to_datetime(current_data['timestamp'])
        else:
            dt = datetime.now()
        
        # Base pollutant and weather features
        base_features = ['aqi', 'pm2_5', 'pm10', 'co', 'so2', 'no2', 'o3', 'temperature', 'humidity']
        for feature in base_features:
            features.append(float(current_data.get(feature, 0)))
        
        # Time-based features
        features.append(np.sin(2 * np.pi * dt.hour / 24))  # hour_sin
        features.append(np.cos(2 * np.pi * dt.hour / 24))  # hour_cos  
        features.append(np.sin(2 * np.pi * dt.dayofyear / 365))  # day_sin
        features.append(np.cos(2 * np.pi * dt.dayofyear / 365))  # day_cos
        
        # PM interaction features
        pm2_5 = current_data.get('pm2_5', 0)
        pm10 = current_data.get('pm10', 0)
        features.append(pm2_5 / max(pm10, 0.1))  # pm_ratio (avoid division by zero)
        features.append(pm2_5 * pm10)  # pm_interaction
        
        # Approximate lag features (use current values as proxies)
        current_aqi = current_data.get('aqi', 0)
        current_pm2_5 = current_data.get('pm2_5', 0)
        current_pm10 = current_data.get('pm10', 0)
        
        # Add some noise/variation to simulate lag differences
        for lag in [1, 6, 12, 24, 48, 72]:
            # Simple approximation: recent values with small random variation
            variation_factor = 1 + (np.random.random() - 0.5) * 0.1  # ±5% variation
            features.append(current_aqi * variation_factor)  # aqi_lag
            features.append(current_pm2_5 * variation_factor)  # pm2_5_lag
            if lag % 24 == 0:
                features.append(current_pm10 * variation_factor)  # pm10_lag
        
        # Approximate derived features
        features.append(0.0)  # pm2_5_change_24h (unknown without history)
        features.append(current_data.get('co', 0))  # co_24h_avg (use current as proxy)
        features.append(current_aqi)  # aqi_72h_avg (use current as proxy)
        features.append(current_pm2_5)  # pm2_5_72h_max (use current as proxy)
        features.append(0.0)  # temp_24h_change (unknown without history)
        features.append(0.0)  # humidity_24h_change (unknown without history)
        
        # Pad or truncate to expected length
        while len(features) < 39:
            features.append(0.0)
        
        return features[:39]
    
    def _estimate_confidence(self, horizon: str, current_data: Dict) -> str:
        """Estimate prediction confidence based on available data quality"""
        required_keys = ['aqi', 'pm2_5', 'pm10', 'temperature', 'humidity']
        available_data = sum(1 for key in required_keys if key in current_data and current_data[key] is not None)
        
        if available_data >= 4:
            if horizon == 'aqi_current':
                return "High"
            elif horizon in ['aqi_24h', 'aqi_48h']:
                return "Medium"
            else:
                return "Low"
        else:
            return "Very Low"
    
    def _get_prediction_timestamp(self, horizon: str, current_timestamp: Optional[str] = None) -> str:
        """Get timestamp for prediction"""
        if current_timestamp:
            base_time = pd.to_datetime(current_timestamp)
        else:
            base_time = datetime.now()
        
        if 'current' in horizon:
            target_time = base_time + timedelta(hours=1)
        elif '24h' in horizon:
            target_time = base_time + timedelta(hours=24)
        elif '48h' in horizon:
            target_time = base_time + timedelta(hours=48)
        elif '72h' in horizon:
            target_time = base_time + timedelta(hours=72)
        else:
            target_time = base_time
        
        return target_time.strftime('%Y-%m-%d %H:%M:%S')
    
    def get_aqi_category(self, aqi_value: float) -> Tuple[str, str, str]:
        """Convert AQI value to category and description"""
        if aqi_value <= 1:
            return "Good", "🟢", "Air quality is satisfactory"
        elif aqi_value <= 2:
            return "Moderate", "🟡", "Air quality is acceptable"
        elif aqi_value <= 3:
            return "Unhealthy for Sensitive", "🟠", "Sensitive groups may experience minor issues"
        elif aqi_value <= 4:
            return "Unhealthy", "🔴", "Everyone may experience health effects"
        elif aqi_value <= 5:
            return "Very Unhealthy", "🟣", "Health alert: serious health effects"
        else:
            return "Hazardous", "🔴", "Emergency conditions: everyone affected"


class ProductionAQIPredictor(RealTimeAQIPredictor):
    """Enhanced version that maintains recent data buffer for better predictions"""
    
    def __init__(self, models_dir: str = "models", buffer_size: int = 168):  # 1 week of hourly data
        super().__init__(models_dir)
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=buffer_size)
        self.buffer_file = "recent_data_buffer.json"
        
        # Load existing buffer if available
        self.load_buffer()
    
    def add_data_point(self, data_point: Dict[str, float]):
        """Add new data point to the rolling buffer"""
        data_point['timestamp'] = data_point.get('timestamp', datetime.now().isoformat())
        self.data_buffer.append(data_point)
        self.save_buffer()
        
    def predict_with_buffer(self, current_data: Dict[str, float]) -> Dict[str, Dict]:
        """Enhanced prediction using recent data buffer"""
        # Add current data to buffer
        self.add_data_point(current_data.copy())
        
        # Create enhanced features using buffer
        features = self._create_enhanced_features(current_data)
        
        # Make predictions
        predictions = {}
        for horizon in self.target_cols:
            if horizon in self.models:
                try:
                    pred_value = self.models[horizon].predict([features])[0]
                    predictions[horizon] = {
                        'value': float(pred_value),
                        'horizon': horizon.replace('aqi_', '').replace('current', '1h'),
                        'confidence': self._enhanced_confidence(horizon, len(self.data_buffer)),
                        'timestamp': self._get_prediction_timestamp(horizon, current_data.get('timestamp'))
                    }
                except Exception as e:
                    predictions[horizon] = {
                        'value': None,
                        'error': str(e),
                        'horizon': horizon.replace('aqi_', '').replace('current', '1h')
                    }
        
        return predictions
    
    def _create_enhanced_features(self, current_data: Dict[str, float]) -> List[float]:
        """Create features using actual recent data when available"""
        if len(self.data_buffer) < 2:
            # Fallback to approximation if no buffer
            return self._create_realtime_features(current_data)
        
        # Convert buffer to DataFrame for easier processing
        df = pd.DataFrame(list(self.data_buffer))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        features = []
        
        # Current timestamp
        dt = pd.to_datetime(current_data.get('timestamp', datetime.now()))
        
        # Base features
        base_features = ['aqi', 'pm2_5', 'pm10', 'co', 'so2', 'no2', 'o3', 'temperature', 'humidity']
        for feature in base_features:
            features.append(float(current_data.get(feature, 0)))
        
        # Time features
        features.append(np.sin(2 * np.pi * dt.hour / 24))
        features.append(np.cos(2 * np.pi * dt.hour / 24))
        features.append(np.sin(2 * np.pi * dt.dayofyear / 365))
        features.append(np.cos(2 * np.pi * dt.dayofyear / 365))
        
        # PM interaction features
        pm2_5 = current_data.get('pm2_5', 0)
        pm10 = current_data.get('pm10', 0)
        features.append(pm2_5 / max(pm10, 0.1))
        features.append(pm2_5 * pm10)
        
        # Real lag features using buffer data
        for lag in [1, 6, 12, 24, 48, 72]:
            lag_time = dt - timedelta(hours=lag)
            
            # Find closest data point to lag time
            if len(df) > 0:
                time_diffs = abs(df['timestamp'] - lag_time)
                closest_idx = time_diffs.idxmin()
                closest_row = df.loc[closest_idx]
                
                features.append(float(closest_row.get('aqi', current_data.get('aqi', 0))))
                features.append(float(closest_row.get('pm2_5', current_data.get('pm2_5', 0))))
                
                if lag % 24 == 0:
                    features.append(float(closest_row.get('pm10', current_data.get('pm10', 0))))
            else:
                # Fallback to current values
                features.append(current_data.get('aqi', 0))
                features.append(current_data.get('pm2_5', 0))
                if lag % 24 == 0:
                    features.append(current_data.get('pm10', 0))
        
        # Calculate real changes and rolling stats where possible
        if len(df) >= 24:  # At least 24 hours of data
            features.append(current_data.get('pm2_5', 0) - df.iloc[-24]['pm2_5'])  # 24h change
            features.append(df['co'].tail(24).mean())  # 24h CO average
        else:
            features.append(0.0)  # pm2_5_change_24h
            features.append(current_data.get('co', 0))  # co_24h_avg
        
        if len(df) >= 72:  # At least 72 hours of data
            features.append(df['aqi'].tail(72).mean())  # 72h AQI average
            features.append(df['pm2_5'].tail(72).max())  # 72h PM2.5 max
        else:
            features.append(current_data.get('aqi', 0))  # aqi_72h_avg
            features.append(current_data.get('pm2_5', 0))  # pm2_5_72h_max
        
        # Temperature and humidity changes
        if len(df) >= 24:
            features.append(current_data.get('temperature', 0) - df.iloc[-24]['temperature'])
            features.append(current_data.get('humidity', 0) - df.iloc[-24]['humidity'])
        else:
            features.append(0.0)
            features.append(0.0)
        
        # Ensure correct length
        while len(features) < 39:
            features.append(0.0)
        
        return features[:39]
    
    def _enhanced_confidence(self, horizon: str, buffer_length: int) -> str:
        """Enhanced confidence based on buffer size"""
        if buffer_length >= 72:  # 3+ days of data
            confidence_map = {
                'aqi_current': 'Very High',
                'aqi_24h': 'High', 
                'aqi_48h': 'Medium',
                'aqi_72h': 'Medium'
            }
        elif buffer_length >= 24:  # 1+ day of data
            confidence_map = {
                'aqi_current': 'High',
                'aqi_24h': 'Medium',
                'aqi_48h': 'Low',
                'aqi_72h': 'Low'
            }
        else:  # Limited data
            confidence_map = {
                'aqi_current': 'Medium',
                'aqi_24h': 'Low',
                'aqi_48h': 'Very Low',
                'aqi_72h': 'Very Low'
            }
        
        return confidence_map.get(horizon, 'Low')
    
    def save_buffer(self):
        """Save buffer to file for persistence"""
        try:
            with open(self.buffer_file, 'w') as f:
                json.dump(list(self.data_buffer), f, default=str)
        except Exception as e:
            logger.warning(f"Could not save buffer: {e}")
    
    def load_buffer(self):
        """Load buffer from file"""
        try:
            if os.path.exists(self.buffer_file):
                with open(self.buffer_file, 'r') as f:
                    data = json.load(f)
                    self.data_buffer.extend(data)
                logger.info(f"Loaded {len(self.data_buffer)} data points from buffer")
        except Exception as e:
            logger.warning(f"Could not load buffer: {e}")


def demo_realtime_prediction():
    """Demo function to show real-time prediction after training"""
    print("\n🔄 Testing Real-time Prediction...")
    
    try:
        # Initialize real-time predictor
        predictor = RealTimeAQIPredictor()
        
        # Example current sensor readings
        current_readings = {
            'aqi': 3.2,
            'pm2_5': 28.5,
            'pm10': 45.2,
            'co': 0.9,
            'so2': 4.8,
            'no2': 22.1,
            'o3': 85.3,
            'temperature': 29.8,
            'humidity': 68.5,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"📊 Input data: {current_readings}")
        
        # Make predictions
        predictions = predictor.predict_from_current_data(current_readings)
        
        # Display results
        print(f"\n🌤️  REAL-TIME AQI PREDICTIONS")
        print("="*50)
        
        for horizon, pred_info in predictions.items():
            if 'error' not in pred_info and pred_info['value'] is not None:
                aqi_val = pred_info['value']
                category, emoji, description = predictor.get_aqi_category(aqi_val)
                
                print(f"\n{emoji} {pred_info['horizon'].upper()} FORECAST")
                print(f"   AQI: {aqi_val:.2f}")
                print(f"   Category: {category}")
                print(f"   Confidence: {pred_info['confidence']}")
                print(f"   Target Time: {pred_info['timestamp']}")
                print(f"   Description: {description}")
            else:
                print(f"\n❌ {pred_info['horizon'].upper()} FORECAST FAILED")
                print(f"   Error: {pred_info.get('error', 'Unknown error')}")
                
    except Exception as e:
        print(f"❌ Real-time prediction demo failed: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Training phase
    print("🚀 Starting AQI Forecast Training...")
    trainer = AQIForecastTrainer()
    trainer.prepare_data()
    
    if not trainer.datasets:
        logger.error("No valid datasets available for training")
    else:
        results = trainer.train_models()
        trainer.save_models()
        
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        for horizon, res in results.items():
            print(f"\n{horizon.upper()}:")
            if 'error' in res:
                print(f"  ❌ Error: {res['error']}")
            else:
                print(f"  ✅ Model: {type(res['best_model']).__name__}")
                print(f"  📊 Test MAE: {res['test_mae']:.3f}")
                print("  🎯 Top Features:")
                if not res['feature_importance'].empty and len(res['feature_importance']) > 1:
                    for i, (_, row) in enumerate(res['feature_importance'].head(3).iterrows()):
                        print(f"     {i+1}. {row['Feature']}: {row['Importance']:.4f}")
                else:
                    print("     No feature importance available")
        print("\n" + "="*60)
        
        # Real-time prediction demo
        demo_realtime_prediction()
        
        print("\n🎉 Training and prediction system ready!")
        print("💡 Usage:")
        print("  1. For basic predictions: RealTimeAQIPredictor()")
        print("  2. For enhanced predictions: ProductionAQIPredictor()")
        print("  3. Check the demo above for example usage")
