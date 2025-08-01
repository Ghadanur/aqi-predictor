# src/models/train.py
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .explain import ForecastExplainer
from src.features.process import AQI3DayForecastProcessor

def plot_predictions(y_true, y_pred, horizon, save_path=None):
    """Visual comparison of predictions vs actuals"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--', alpha=0.8)
    plt.title(f'AQI Forecast Validation ({horizon} horizon)')
    plt.xlabel('Time Steps')
    plt.ylabel('AQI Value')
    plt.legend()
    if save_path:
        plt.savefig(f'{save_path}/validation_plot_{horizon}.png', bbox_inches='tight')
    plt.close()

def evaluate_forecast(y_true, y_pred, horizon):
    """Comprehensive forecast evaluation"""
    metrics = {
        'horizon': horizon,
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'mean_error': np.mean(y_pred - y_true),
        'std_error': np.std(y_pred - y_true),
        'max_error': np.max(np.abs(y_pred - y_true))
    }
    return metrics

def train_3day_forecaster():
    # 1. Get processed data
    processor = AQI3DayForecastProcessor()
    features, targets = processor.get_3day_forecast_data(lookback_days=120)
    
    # 2. Time-based split (preserve temporal order)
    split_idx = int(0.8 * len(features))
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = targets.iloc[:split_idx], targets.iloc[split_idx:]
    
    # 3. Train model with optimized parameters
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 4. Save model artifacts
    model_dir = 'src/models/'
    joblib.dump(model, f'{model_dir}3day_forecaster.pkl')
    
    # 5. Generate predictions
    preds = model.predict(X_test)
    
    # 6. Comprehensive validation
    validation_results = []
    horizons = ['24h', '48h', '72h']
    
    for i, horizon in enumerate(horizons):
        # Get valid indices (non-NaN)
        valid_idx = ~np.isnan(y_test.iloc[:, i+1])
        y_true = y_test.iloc[:, i+1][valid_idx]
        y_pred = preds[valid_idx, i] if preds.ndim > 1 else preds[valid_idx]
        
        # Evaluate
        metrics = evaluate_forecast(y_true, y_pred, horizon)
        validation_results.append(metrics)
        
        # Visual validation
        plot_predictions(y_true, y_pred, horizon, model_dir)
        
        # Generate SHAP explanations
        explainer = ForecastExplainer(f'{model_dir}3day_forecaster.pkl')
        explainer.prepare_shap(X_train)
        explainer.visualize_horizon(X_test[valid_idx], horizon, model_dir)
    
    # 7. Save validation report
    pd.DataFrame(validation_results).to_csv(f'{model_dir}validation_report.csv')
    
    # 8. Alert if critical errors found
    max_errors = [res['max_error'] for res in validation_results]
    if any(error > 50 for error in max_errors):
        print("ALERT: Some predictions >50 AQI points from actual values")
    
    return model

if __name__ == "__main__":
    trained_model = train_3day_forecaster()
    print("Model training and validation completed successfully")
