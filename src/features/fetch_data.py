import os
import requests
import pandas as pd
import hopsworks

def fetch_openweather():
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    LAT, LON = "24.9056", "67.0822"
    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    
    response = requests.get(url).json()
    pollution = response["list"][0]
    
    return {
        "timestamp": pd.Timestamp.now(),
        "aqi": pollution["main"]["aqi"],
        "pm2_5": pollution["components"]["pm2_5"],
        "pm10": pollution["components"]["pm10"],
        "co": pollution["components"]["co"],
        "no2": pollution["components"]["no2"],
        "o3": pollution["components"]["o3"],
        "so2": pollution["components"]["so2"]
    }

def fetch_accuweather():
    API_KEY = os.getenv("ACCUWEATHER_API_KEY")
    LOCATION_KEY = "261158"  # Karachi
    
    url = f"https://dataservice.accuweather.com/currentconditions/v1/{LOCATION_KEY}?apikey={API_KEY}&details=true"
    response = requests.get(url).json()[0]
    
    return {
        "temperature": response["Temperature"]["Metric"]["Value"],
        "humidity": response["RelativeHumidity"],
        "wind_speed": response["Wind"]["Speed"]["Metric"]["Value"],
        "pressure": response["Pressure"]["Metric"]["Value"],
        "uv_index": response["UVIndex"]
    }

def save_to_hopsworks(data):
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    
    # Create/update feature group
    fg = fs.get_or_create_feature_group(
        name="aqi_weather_data",
        version=1,
        primary_key=["timestamp"],
        description="Combined AQI and weather data"
    )
    fg.insert(pd.DataFrame([data]))

if __name__ == "__main__":
    # Fetch and merge data
    aqi_data = fetch_openweather()
    weather_data = fetch_accuweather()
    combined_data = {**aqi_data, **weather_data}
    
    # Save to Hopsworks
    save_to_hopsworks(combined_data)
