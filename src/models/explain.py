import pandas as pd
def visualize_horizon(self, X: pd.DataFrame, horizon: str, save_path: str = None):
    """Generate SHAP visualization with dimension checks"""
    if not self.explainer:
        raise ValueError("Call prepare_shap() first")
    
    # Get the correct horizon index
    horizon_idx = {'24h': 0, '48h': 1, '72h': 2}.get(horizon)
    if horizon_idx is None:
        raise ValueError(f"Invalid horizon: {horizon}")

    # Get SHAP values for this horizon
    shap_values = self.explainer.shap_values(X)
    
    # Handle multi-output format
    if isinstance(shap_values, list):
        shap_values = shap_values[horizon_idx]
    
    # Verify dimensions match
    if shap_values.shape[1] != X.shape[1]:
        print(f"Warning: Dimension mismatch - Features: {X.shape[1]}, SHAP: {shap_values.shape[1]}")
        print("Using first {X.shape[1]} SHAP features")
        shap_values = shap_values[:, :X.shape[1]]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    
    plt.title(f'Feature Importance for {horizon} Forecast')
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'shap_{horizon}.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
