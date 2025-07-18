# src/models/train.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load processed data
data = pd.read_csv("data/processed/aqi_processed.csv")

# Train/Test split
train = data.iloc[:-100]  # Most recent 100 rows for testing
test = data.iloc[-100:]

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(train[["hour", "day_of_week", "pm25_24h_avg"]], train["pm2_5"])

# Save model
joblib.dump(model, "src/models/artifacts/model.joblib")