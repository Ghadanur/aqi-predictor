# src/production_predictor.py
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime, timedelta
import json
import os

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
        
        # Make predictions (same as before but with better features)
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

# Usage example
def production_example():
    """Example of how to use in production"""
    predictor = ProductionAQIPredictor()
    
    # Simulate receiving data over time
    import time
    for i in range(5):
        # Simulate new sensor reading every hour
        current_reading = {
            'aqi': 3.0 + np.random.random(),
            'pm2_5': 25 + np.random.random() * 10,
            'pm10': 40 + np.random.random() * 15,
            'co': 0.8 + np.random.random() * 0.4,
            'so2': 5 + np.random.random() * 2,
            'no2': 20 + np.random.random() * 5,
            'o3': 80 + np.random.random() * 20,
            'temperature': 28 + np.random.random() * 4,
            'humidity': 65 + np.random.random() * 10,
        }
        
        predictions = predictor.predict_with_buffer(current_reading)
        print(f"\nHour {i+1} - Predictions made with {len(predictor.data_buffer)} data points in buffer")
        
        time.sleep(1)  # Simulate time passing
