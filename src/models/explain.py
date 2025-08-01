# src/models/explain.py
import shap
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
import os
import joblib

class ForecastExplainer:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.explainer = None
        
    def prepare_shap(self, X_train: pd.DataFrame):
        """Initialize SHAP explainer (call this once during training)"""
        # For tree-based models
        if hasattr(self.model, 'estimators_'):
            self.explainer = shap.TreeExplainer(self.model.estimators_[0])
        # For neural networks/linear models
        else:
            self.explainer = shap.KernelExplainer(self.model.predict, X_train.iloc[:100])
        
        return self

    def analyze_horizon(self, X: pd.DataFrame, horizon: str) -> Dict:
        """
        Analyze feature importance for a specific forecast horizon
        Returns: {'feature': importance_score}
        """
        if not self.explainer:
            raise ValueError("Call prepare_shap() first")
            
        # Get SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # For multi-output models
        if isinstance(shap_values, list):
            horizon_idx = {'24h': 0, '48h': 1, '72h': 2}[horizon]
            shap_values = shap_values[horizon_idx]
        
        # Process importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=False)
        
        return importance.set_index('feature')['importance'].to_dict()

    def visualize_horizon(self, X: pd.DataFrame, horizon: str, save_path: str = None):
        """Generate SHAP visualization for a specific horizon"""
        horizon_idx = {'24h': 0, '48h': 1, '72h': 2}.get(horizon)
        
        plt.figure(figsize=(12, 8))
        
        if hasattr(self.model, 'estimators_'):  # Tree models
            shap_values = self.explainer.shap_values(X)
            if horizon_idx is not None:
                shap_values = shap_values[horizon_idx]
            shap.summary_plot(shap_values, X, show=False)
        else:  # Other models
            shap_values = self.explainer.shap_values(X.iloc[:100])
            shap.summary_plot(shap_values, X.iloc[:100], show=False)
        
        plt.title(f'Feature Importance for {horizon} Forecast')
        if save_path:
            plt.savefig(os.path.join(save_path, f'shap_{horizon}.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
