# src/models/train.py
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import traceback
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    from src.models.explain import ForecastExplainer
    from src.features.process import AQI3DayForecastProcessor
except ImportError:
    # Fallback for direct execution
    from explain import ForecastExplainer
    from features.process import AQI3DayForecastProcessor

# Define the output directory relative to repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DATA_DIR = os.path.join(REPO_ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

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
        output_path = os.path.join(save_path, f'validation_plot_{horizon}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved validation plot to: {output_path}")
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
        'max_error': np.max(np.abs(y_pred - y_true)),
        'samples': len(y_true)
    }
    return metrics

def train_3day_forecaster():
    try:
        # 1. Get processed data
        processor = AQI3DayForecastProcessor()
        print("Fetching and processing data...")
        features, targets = processor.get_3day_forecast_data(lookback_days=120)
        
        # 2. Reset indices to ensure alignment
        features = features.reset_index(drop=True)
        targets = targets.reset_index(drop=True)
        
        # 3. Time-based split (preserve temporal order)
        split_idx = int(0.8 * len(features))
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = targets.iloc[:split_idx], targets.iloc[split_idx:]
        
        # 4. Verify alignment
        assert X_train.index.equals(y_train.index), "Train index mismatch"
        assert X_test.index.equals(y_test.index), "Test index mismatch"
        
        # 5. Train model with optimized parameters
        print("Training Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # 6. Save model artifacts to data folder
        model_path = os.path.join(DATA_DIR, '3day_forecaster.pkl')
        joblib.dump(model, model_path)
        print(f"Saved model to: {model_path}")
        
        # 7. Generate predictions
        print("Generating predictions...")
        preds = model.predict(X_test)
        
        # 8. Comprehensive validation
        validation_results = []
        horizons = ['24h', '48h', '72h']
        
        for i, horizon in enumerate(horizons):
            try:
                print(f"\nEvaluating {horizon} forecast...")
                # Get valid indices (non-NaN and aligned)
                valid_mask = (~np.isnan(y_test.iloc[:, i+1])) & (y_test.index.isin(X_test.index))
                
                # Get aligned data
                y_true = y_test.loc[valid_mask, y_test.columns[i+1]]
                y_pred = preds[valid_mask, i]
                
                # Verify we have data to evaluate
                if len(y_true) == 0:
                    print(f"Warning: No valid samples for {horizon} forecast")
                    continue
                
                # Evaluate
                metrics = evaluate_forecast(y_true, y_pred, horizon)
                validation_results.append(metrics)
                print(f"{horizon} metrics:", {k: round(v, 3) if isinstance(v, float) else v 
                                           for k, v in metrics.items()})
                
                # Visual validation
                plot_predictions(y_true, y_pred, horizon, DATA_DIR)
                
                # Generate SHAP explanations
                print("Generating SHAP explanation...")
                explainer = ForecastExplainer(model_path)
                explainer.prepare_shap(X_train)
                explainer.visualize_horizon(X_test.loc[valid_mask], horizon, DATA_DIR)
                print(f"Saved SHAP plot to: {os.path.join(DATA_DIR, f'shap_{horizon}.png')}")
                
            except Exception as e:
                print(f"Error evaluating {horizon} forecast: {str(e)}")
                continue
        
        # 9. Save validation report to data folder
        report_path = os.path.join(DATA_DIR, 'validation_report.csv')
        pd.DataFrame(validation_results).to_csv(report_path, index=False)
        print(f"\nSaved validation report to: {report_path}")
        
        # 10. Final checks
        max_errors = [res['max_error'] for res in validation_results if res is not None]
        if max_errors and any(error > 50 for error in max_errors):
            print("\nALERT: Some predictions >50 AQI points from actual values")
        
        return model

    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("Starting AQI 3-day forecast model training...")
        trained_model = train_3day_forecaster()
        
        # List all generated files
        print("\nGenerated files in data directory:")
        for f in sorted(os.listdir(DATA_DIR)):
            if f.startswith(('3day_forecaster', 'shap_', 'validation_')):
                print(f"- {f}")
        
        print("\nModel training and validation completed successfully")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {str(e)}")
        print("Traceback:", traceback.format_exc())
        sys.exit(1)

