# src/models/train.py
from src.process import AQI3DayForecastProcessor
from sklearn.model_selection import TimeSeriesSplit

def train_3day_forecaster():
    # 1. Get processed data
    processor = AQI3DayForecastProcessor()
    features, targets = processor.get_3day_forecast_data(lookback_days=120)
    
    # 2. Time-based split (preserve temporal order)
    tscv = TimeSeriesSplit(n_splits=3)
    for train_idx, test_idx in tscv.split(features):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = targets.iloc[train_idx], targets.iloc[test_idx]
def evaluate_3day_forecast(model, X_test, y_test):
    horizons = ['24h', '48h', '72h']
    preds = model.predict(X_test)
    
    for i, horizon in enumerate(horizons):
        rmse = mean_squared_error(y_test.iloc[:, i+1], preds[:, i], squared=False)
        print(f"RMSE for {horizon} forecast: {rmse:.2f}")
        
        # 3. Train model on 3 targets simultaneously
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        
        # 4. Evaluate on 24h/48h/72h horizons
        evaluate_3day_forecast(model, X_test, y_test)
