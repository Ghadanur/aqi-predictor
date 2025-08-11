import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

# Configuration
MODEL_DIR = Path(__file__).parent
os.makedirs(MODEL_DIR, exist_ok=True)

def plot_predictions(y_true, y_pred, horizon, save_path=None):
    """Quick visualization of predictions vs actuals"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--', alpha=0.8)
    plt.title(f'AQI Forecast ({horizon} horizon)')
    plt.xlabel('Time Steps')
    plt.ylabel('AQI Value')
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def evaluate_forecast(y_true, y_pred):
    """Core evaluation metrics"""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'samples': len(y_true)
    }

def train_3day_forecaster(processor):
    """Main training pipeline"""
    # 1. Get and split data
    features, targets = processor.get_3day_forecast_data(lookback_days=120)
    split_idx = int(0.8 * len(features))
    
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = targets.iloc[:split_idx], targets.iloc[split_idx:]

    # 2. Train model
    model = ExtraTreesRegressor(
        n_estimators=200,
        max_features='sqrt',
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 3. Evaluate predictions
    preds = model.predict(X_test)
    results = []
    
    for i, horizon in enumerate(['24h', '48h', '72h']):
        valid_mask = (~y_test.iloc[:, i].isna())
        y_true = y_test.loc[valid_mask].iloc[:, i]
        y_pred_h = preds[valid_mask, i]
        
        metrics = evaluate_forecast(y_true, y_pred_h)
        results.append({'horizon': horizon, **metrics})
        
        plot_predictions(
            y_true, y_pred_h, 
            horizon,
            os.path.join(MODEL_DIR, f'validation_{horizon}.png')
        )
    
    # 4. Save outputs
    joblib.dump(model, os.path.join(MODEL_DIR, 'aqi_forecaster.pkl'))
    pd.DataFrame(results).to_csv(os.path.join(MODEL_DIR, 'metrics.csv'), index=False)
    
    return model, results

if __name__ == "__main__":
    from features.process import AQI3DayForecastProcessor  # Or your import method
    
    processor = AQI3DayForecastProcessor()
    model, results = train_3day_forecaster(processor)
    
    print("\nTraining completed. Results:")
    for r in results:
        print(f"{r['horizon']}: R2={r['R2']:.2f}, RMSE={r['RMSE']:.2f}")
