import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from process import AQI3DayForecastProcessor
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AQIModelTrainer:
    def __init__(self):
        self.processor = AQI3DayForecastProcessor()
        self.model = None
        self.scaler = StandardScaler()
        
    def prepare_data(self):
        """Get and preprocess data"""
        logger.info("Preparing data...")
        features, targets = self.processor.get_3day_forecast_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features.values,
            targets.values,
            test_size=0.2,
            shuffle=False  # Important for time series!
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Reshape for LSTM [samples, timesteps, features]
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """Create LSTM model architecture"""
        logger.info("Building model...")
        model = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(4)  # Output layer for 4 targets
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self):
        """Train the model"""
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10)
        
        logger.info("Training model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        logger.info("Evaluating model...")
        predictions = self.model.predict(X_test)
        
        # Calculate RMSE for each horizon
        rmse = {}
        for i, horizon in enumerate(['current', '24h', '48h', '72h']):
            rmse[f'rmse_{horizon}'] = np.sqrt(
                mean_squared_error(y_test[:, i], predictions[:, i])
            )
        
        return rmse
    
    def save_model(self, model_dir='models'):
        """Save model and scaler"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Keras model
        self.model.save(f'{model_dir}/aqi_forecast_model.h5')
        
        # Save scaler
        joblib.dump(self.scaler, f'{model_dir}/scaler.pkl')
        
        logger.info(f"Model saved to {model_dir}")

if __name__ == '__main__':
    trainer = AQIModelTrainer()
    history = trainer.train()
    
    # Evaluate and save
    X_train, X_test, y_train, y_test = trainer.prepare_data()
    metrics = trainer.evaluate(X_test, y_test)
    logger.info(f"Model metrics: {metrics}")
    
    trainer.save_model()
