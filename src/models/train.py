import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_aqi_model():
    try:
        # 1. Load data (replace with your actual data loading)
        data = pd.read_csv('aqi_data.csv')
        
        # 2. Select only the key features we care about
        features = data[['pm2_5', 'pm10', 'co']]  # Add other confirmed important features
        targets = data[['aqi_current', 'aqi_24h', 'aqi_48h', 'aqi_72h']]
        
        # 3. Simple train-test split
        split_idx = int(0.8 * len(data))
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = targets.iloc[:split_idx], targets.iloc[split_idx:]
        
        # 4. Train separate models for each horizon
        models = {}
        horizons = ['24h', '48h', '72h']
        
        for horizon in horizons:
            # Simple Random Forest (adjust parameters as needed)
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train, y_train[f'aqi_{horizon}'])
            models[horizon] = model
            
            # Evaluate
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test[f'aqi_{horizon}'], preds)
            logging.info(f"{horizon} model accuracy: {accuracy:.2f}")
            
            # Save model
            joblib.dump(model, f'aqi_{horizon}_model.pkl')
        
        return models

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    train_aqi_model()
