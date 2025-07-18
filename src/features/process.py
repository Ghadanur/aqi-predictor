import hopsworks

def get_features():
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    # Read latest data
    fg = fs.get_feature_group("aqi_realtime", version=1)
    df = fg.read()
    
    # Feature engineering (same as before)
    df["hour"] = df["timestamp"].dt.hour
    return df