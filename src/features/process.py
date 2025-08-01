import hopsworks
import pandas as pd

# 1. Connect to Feature Store
project = hopsworks.login(
    os.getenv("HOPSWORKS_API_KEY")
    project="aqi_predictr"
)
fs = project.get_feature_store()

# 2. Access Feature Group
fg = fs.get_feature_group(
    name="aqi_weather_data",
    version=1
)

# 3. Retrieve Data (with optional filters)
query = fg.select_all()  

# Add time filter (last 30 days)
from datetime import datetime, timedelta
start_date = datetime.now() - timedelta(days=30)
query = query.filter(fg.timestamp >= start_date)

# 4. Execute Query
df = query.read()
print(f"Retrieved {len(df)} records")
