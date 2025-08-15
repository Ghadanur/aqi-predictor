import logging
from pathlib import Path
import os
import sys
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Setup logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'aqi_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    # Now import other dependencies
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                               confusion_matrix, mean_absolute_error,
                               mean_squared_error, r2_score)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import make_pipeline as make_imb_pipeline
    from src.features.process import AQI3DayForecastProcessor
    
except ImportError as e:
    logger.critical(f"Critical import error: {str(e)}", exc_info=True)
    sys.exit(1)

# Constants with absolute paths
MODEL_DIR = Path(__file__).parent
os.makedirs(MODEL_DIR, exist_ok=True)
AQI_BINS = [0, 50, 100, 150, 200, 300, 500]
AQI_LABELS = [1, 2, 3, 4, 5, 6]

def train_aqi_model():
    """Main training function with enhanced logging"""
    try:
        logger.info("Starting AQI training pipeline")
        
        # 1. Data loading
        logger.info("Initializing data processor")
        processor = AQI3DayForecastProcessor()
        
        logger.info("Fetching 1 year of historical data")
        features, targets = processor.get_3day_forecast_data(lookback_days=365)
        logger.info(f"Loaded data: {features.shape[0]} samples, {features.shape[1]} features")
        
        # 2. Preprocessing
        logger.info("Preprocessing data")
        features, targets = preprocess_data(features, targets)
        
        # 3. Validation
        class_dist = targets.iloc[:, 0].value_counts(normalize=True)
        if len(class_dist) < 2:
            logger.error("Insufficient class diversity - only one AQI category present")
            return None
        
        # ... [rest of your existing code with added logger calls] ...

    except Exception as e:
        logger.critical(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("Script started")
    train_aqi_model()
    logger.info("Script completed")

