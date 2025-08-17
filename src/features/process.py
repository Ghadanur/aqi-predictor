# src/process.py
import pandas as pd
import numpy as np
import hopsworks
import os
from datetime import datetime, timedelta
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class AQI3DayForecastProcessor:
    def __init__(self):
        self._connect_hopsworks()
        
    def _connect_hopsworks(self):
        """Initialize Hopsworks connection"""
        try:
            self.project = hopsworks.login(
                api_key_value=os.getenv("HOPSWORKS_API_KEY"),
                project="aqi_predictr"
            )
            self.fs = self.project.get_feature_store()
        except Exception as e:
            logger.error(f"Hopsworks connection failed: {str(e)}")
            raise

    def get_3day_forecast_data(self, lookback_days: int = 90) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retrieve and process data for 3-day forecasting
        Returns: 
            - features: DataFrame with engineered features
            - targets: DataFrame with 3-day horizon targets
        """
        # 1. Fetch 3 months of historical data
        raw_df = self._fetch_raw_data(lookback_days)
        
        # 2. Clean and validate
        clean_df = self._clean_data(raw_df)
        
        # 3. Create 3-day forecast features
        features = self._create_3day_features(clean_df)
        
        # 4. Prepare 3-day targets
        targets = self._create_3day_targets(clean_df)
        
        return features.iloc[:-72], targets.iloc[:-72],raw_df  # Exclude last 72h for valid split

    def _fetch_raw_data(self, lookback_days: int) -> pd.DataFrame:
        """Retrieve historical data from Hopsworks"""
        try:
            fg = self.fs.get_feature_group(
                name="aqi_weather_data",
                version=1
            )
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            return fg.select_all().filter(
                fg.timestamp >= start_date.strftime('%Y-%m-%d')
            ).read()
            
        except Exception as e:
            logger.error(f"Data retrieval failed: {str(e)}")
            raise

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data validation and cleaning"""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Validate ranges
        df = df[(df['aqi'] > 0) & (df['aqi'] <= 500)]
        df = df[(df['pm2_5'] >= 0) & (df['pm2_5'] <= 500)]
        
        # Forward-fill weather data
        df[['temperature', 'humidity']] = df[['temperature', 'humidity']].ffill()
        
        return df.dropna().sort_values('timestamp')

    def _create_3day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features optimized for 72-hour forecasting"""
    # Convert timestamp to numerical features first
        df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour/24)
        df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.dayofyear/365)
        df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.dayofyear/365)

        # 2. Enhanced PM features (high importance)
        df['pm_ratio'] = (df['pm2_5'] / df['pm10']).astype('float32')  # Composition ratio
        df['pm_interaction'] = (df['pm2_5'] * df['pm10']).astype('float32')  # Interaction term
        df['pm2_5_change_24h'] = df['pm2_5'].diff(24).astype('float32')  # Daily delta    
    # Lag features (3 days = 72 hours)
        for lag in [1, 6, 12, 24, 48, 72]:
            df[f'aqi_lag_{lag}h'] = df['aqi'].shift(lag).astype('float32')
            df[f'pm2_5_lag_{lag}h'] = df['pm2_5'].shift(lag).astype('float32')
            if lag % 24 == 0:  # Only daily lags for PM10
                df[f'pm10_lag_{lag}h'] = df['pm10'].shift(lag).astype('float32')   
    # Rolling statistics
        # 4. Simplified CO features (low importance)
        df['co_24h_avg'] = df['co'].rolling(24).mean().astype('float32')
        df['aqi_72h_avg'] = df['aqi'].rolling(72).mean().astype('float32')
        df['pm2_5_72h_max'] = df['pm2_5'].rolling(72).max().astype('float32')
    
    # Weather trends
        df['temp_24h_change'] = df['temperature'].diff(24).astype('float32')
        df['humidity_24h_change'] = df['humidity'].diff(24).astype('float32')
    
    # Drop original timestamp and ensure float dtypes
        df = df.drop('timestamp', axis=1)
        return df.astype('float32').dropna()

    def _create_3day_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create multi-horizon targets up to 72 hours"""
        targets = pd.DataFrame({
            'aqi_current': df['aqi'],
            'aqi_24h': df['aqi'].shift(-24),
            'aqi_48h': df['aqi'].shift(-48),
            'aqi_72h': df['aqi'].shift(-72)
        })
        return targets.dropna()
if __name__ == "__main__":
    processor = AQI3DayForecastProcessor()
    features, targets = processor.get_3day_forecast_data()
    
    print("\n=== Features Summary ===")
    print(f"Shape: {features.shape}")
    print(features.describe())
    
    print("\n=== Targets Summary ===")
    print(f"Shape: {targets.shape}")
    print(targets.describe())
    print(f"raw_df:{raw_df}")
    
    print("\n=== Target Value Counts ===")
    for col in targets.columns:
        print(f"\n{col}:")
        print(targets[col].value_counts().head())




