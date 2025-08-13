def train_3day_forecaster():
    try:
        # [Previous data loading code remains the same until model training]
        
        # Train final models with versioning
        logging.info("\nTraining final models...")
        models = {}
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for h_name, h_idx in horizon_map.items():
            # Calculate class weights
            classes, counts = np.unique(y_train.iloc[:, h_idx], return_counts=True)
            weights = compute_class_weight('balanced', classes=classes, 
                                         y=y_train.iloc[:, h_idx])
            class_weights = dict(zip(classes, weights))
            
            min_samples = min(counts)
            
            # Build appropriate pipeline based on sample size
            if min_samples >= 3:
                model = make_pipeline(
                    StandardScaler(),
                    CalibratedClassifierCV(
                        ExtraTreesClassifier(
                            n_estimators=300,
                            max_depth=10,
                            min_samples_leaf=3,
                            random_state=42,
                            n_jobs=-1,
                            class_weight=class_weights
                        ),
                        cv=min(3, min_samples),
                        method='isotonic'
                    )
                )
            else:
                logging.warning(f"Insufficient samples ({min_samples}) for calibration in {h_name}")
                model = make_pipeline(
                    StandardScaler(),
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                        class_weight=class_weights
                    )
                )
            
            try:
                model.fit(X_train, y_train.iloc[:, h_idx])
                models[h_name] = model
                
                # Save model
                model_path = os.path.join(MODEL_DIR, f'model_v{model_version}_{h_name}.pkl')
                joblib.dump(model, model_path)
                logging.info(f"Saved {h_name} model to: {model_path}")
                
            except Exception as e:
                logging.error(f"Failed to train {h_name} model: {str(e)}")
                continue
        
        # [Rest of the function remains the same]
