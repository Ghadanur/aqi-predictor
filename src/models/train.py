# src/models/train.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from .explain import ForecastExplainer
from src.features.process import AQI3DayForecastProcessor

def train_3day_forecaster():
    # 1. Get processed data
    processor = AQI3DayForecastProcessor()
    features, targets = processor.get_3day_forecast_data(lookback_days=120)
    
    # 2. Time-based split
    split_idx = int(0.8 * len(features))
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = targets.iloc[:split_idx], targets.iloc[split_idx:]
    
    # 3. Train model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Save model
    joblib.dump(model, 'src/models/3day_forecaster.pkl')
    
    # 5. SHAP analysis
    explainer = ForecastExplainer('src/models/3day_forecaster.pkl')
    explainer.prepare_shap(X_train)
    
    # 6. Evaluate and visualize
    horizons = ['24h', '48h', '72h']
    preds = model.predict(X_test)
    
    for i, horizon in enumerate(horizons):
        rmse = mean_squared_error(y_test.iloc[:, i], preds[:, i], squared=False)
        print(f"RMSE for {horizon} forecast: {rmse:.2f}")
        
        explainer.visualize_horizon(
            X_test, 
            horizon,
            save_path='src/models/'
        )

if __name__ == "__main__":
    train_3day_forecaster()
