# src/models/explain.py
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import joblib

class ForecastExplainer:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.explainer = None
        
    def prepare_shap(self, X_train: pd.DataFrame):
        """Initialize SHAP explainer"""
        if hasattr(self.model, 'estimators_'):  # Tree models
            self.explainer = shap.TreeExplainer(self.model)
        else:  # Other models
            self.explainer = shap.KernelExplainer(self.model.predict, X_train.iloc[:100])
        return self

    def visualize_horizon(self, X: pd.DataFrame, horizon: str, save_path: str = None):
        """Generate SHAP visualization"""
        if not self.explainer:
            raise ValueError("Call prepare_shap() first")
            
        plt.figure(figsize=(12, 8))
        shap_values = self.explainer.shap_values(X)
        shap.summary_plot(shap_values, X, show=False)
        
        plt.title(f'Feature Importance for {horizon} Forecast')
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'shap_{horizon}.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
