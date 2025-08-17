# src/train.py (final robust version)
class AQIForecastTrainer:
    # ... (keep existing methods until train_models)

    def train_models(self):
        """Train and compare models for each forecast horizon with robust error handling"""
        self.results = {}
        
        for horizon, data in self.datasets.items():
            logger.info(f"\n{'='*50}\nTraining for target: {horizon}\n{'='*50}")
            
            try:
                # Time-series split
                train, test = self.time_series_split(data)
                
                # PyCaret setup
                exp = setup(
                    data=train,
                    target='target',
                    train_size=0.8,
                    fold_strategy="timeseries",
                    fold=3,
                    verbose=False,
                    data_split_shuffle=False,
                    fold_shuffle=False,
                    normalize=True,
                    transformation=False,  # Disabled for time series
                    remove_multicollinearity=False,  # Disabled for time series
                    feature_selection=False,  # Disabled for time series
                    session_id=42,
                    use_gpu=False
                )
                
                # Compare models with error handling
                try:
                    best_models = compare_models(
                        sort='MAE',
                        include=['lightgbm', 'xgboost', 'catboost', 'rf', 'et'],  # Reduced set
                        n_select=1,  # Only get the best model
                        verbose=False,
                        error_score='raise'
                    )
                    
                    if not best_models or len(best_models) == 0:
                        raise ValueError("No models could be trained")
                        
                    best_model = best_models[0]
                    
                    # Evaluate
                    test_pred = predict_model(best_model, data=test)
                    test_mae = np.mean(np.abs(test_pred['target'] - test_pred['Label']))
                    
                    self.results[horizon] = {
                        'best_model': best_model,
                        'test_mae': test_mae,
                        'feature_importance': pull().sort_values('Importance', ascending=False)
                    }
                    
                    logger.info(f"Successfully trained {type(best_model).__name__} for {horizon}")
                    logger.info(f"Test MAE: {test_mae:.2f}")
                    
                except Exception as model_error:
                    logger.error(f"Model training failed for {horizon}: {str(model_error)}")
                    self.results[horizon] = {
                        'error': str(model_error),
                        'test_mae': None
                    }
                    
            except Exception as setup_error:
                logger.error(f"Setup failed for {horizon}: {str(setup_error)}")
                self.results[horizon] = {
                    'error': str(setup_error),
                    'test_mae': None
                }
                
        return self.results

    # ... (keep remaining methods)
