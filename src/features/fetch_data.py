# src/features/fetch_data.py
import hopsworks
import pandas as pd
from datetime import datetime

def fetch_and_store():
    # 1. Fetch data from API (example)
    api_data = {"timestamp": [datetime.now()], "pm2_5": [12.4], "pm10": [23.1]}
    df = pd.DataFrame(api_data)
    
    # 2. Connect to Hopsworks
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project="your_project_name"
    )
    fs = project.get_feature_store()
    
    # 3. Create/Get feature group
    fg = fs.get_or_create_feature_group(
        name="aqi_realtime",
        version=1,
        primary_key=["timestamp"],
        description="Real-time AQI data",
        time_travel_format="HUDI"
    )
    
    # 4. Insert data
    fg.insert(df)
    
if __name__ == "__main__":
    fetch_and_store()