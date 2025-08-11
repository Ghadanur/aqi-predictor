# src/models/train.py
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import traceback
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    from src.features.process import AQI3DayForecastProcessor
except ImportError:
    from features.process import AQI3DayForecastProcessor

# Define the output directory
MODEL_DIR = Path(__file__).parent
os.makedirs(MODEL_DIR, exist_ok=True)

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
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    percentage_errors = np.abs((y_true - y_pred) / y_true) * 100
    accuracy = 100 - np.mean(percentage_errors)
    
    return {
        'horizon': horizon,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Accuracy (%)': accuracy,
        'mean_error': np.mean(y_pred - y_true),
        'std_error': np.std(y_pred - y_true),
        'max_error': np.max(np.abs(y_pred - y_true)),
        'samples': len(y_true)
    }

def train_3day_forecaster():
    try:
        # 1. Get processed data
        processor = AQI3DayForecastProcessor()
        print("Fetching and processing data...")
        features, targets = processor.get_3day_forecast_data(lookback_days=120)
        
        # 2. Reset indices
        features = features.reset_index(drop=True)
        targets = targets.reset_index(drop=True)
        
        # 3. Time-based split
        split_idx = int(0.8 * len(features))
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = targets.iloc[:split_idx], targets.iloc[split_idx:]
        
        # 4. Verify alignment
        assert X_train.index.equals(y_train.index), "Train index mismatch"
        assert X_test.index.equals(y_test.index), "Test index mismatch"
        
        # 5. Train model
        print("Training ExtraTreesRegressor model...")
        model = ExtraTreesRegressor(
            n_estimators=200,
            max_features='sqrt',
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # 6. Save model
        model_path = os.path.join(MODEL_DIR, '3day_forecaster.pkl')
        joblib.dump(model, model_path)
        print(f"Saved model to: {model_path}")
        
        # 7. Generate predictions
        print("Generating predictions...")
        preds = model.predict(X_test)
        
        # 8. Validation
        validation_results = []
        horizon_map = {'24h': 0, '48h': 1, '72h': 2}

        for horizon_name, horizon_idx in horizon_map.items():
            try:
                print(f"\nEvaluating {horizon_name} forecast...")
                valid_mask = (~targets.iloc[:, horizon_idx].isna()) & (targets.index.isin(X_test.index))
                y_true = targets.loc[valid_mask, targets.columns[horizon_idx]]
                y_pred = preds[valid_mask, horizon_idx]
                
                assert len(y_true) == len(y_pred), "Alignment failed"
                print(f"Evaluating {len(y_true)} samples")
                
                metrics = evaluate_forecast(y_true, y_pred, horizon_name)
                validation_results.append(metrics)
                print(f"\n{horizon_name} Forecast Performance:")
                print(f"- Accuracy: {metrics['Accuracy (%)']:.2f}%")
                print(f"- RMSE: {metrics['RMSE']:.2f}")
                print(f"- MAE: {metrics['MAE']:.2f}")
                print(f"- R²: {metrics['R2']:.2f}")
                print(f"- Samples: {metrics['samples']}")
                
                plot_predictions(y_true, y_pred, horizon_name, MODEL_DIR)
                
            except Exception as e:
                print(f"Error evaluating {horizon_name} forecast: {str(e)}")
                continue
        
        # 9. Save report
        report_path = os.path.join(MODEL_DIR, 'validation_report.csv')
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
        
        print("\nGenerated files in data directory:")
        for f in sorted(os.listdir(MODEL_DIR)):
            if f.startswith(('3day_forecaster', 'validation_')):
                print(f"- {f}")
        
        print("\nModel training and validation completed successfully")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {str(e)}")
        print("Traceback:", traceback.format_exc())
        sys.exit(1)
