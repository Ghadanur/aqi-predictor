import os
import requests
import pandas as pd
import hopsworks
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_openweather():
    """Fetch air pollution data from OpenWeather API"""
    try:
        API_KEY = os.getenv("OPENWEATHER_API_KEY")
        if not API_KEY:
            raise ValueError("OPENWEATHER_API_KEY environment variable not set")
            
        LAT, LON = "24.9056", "67.0822"
        url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raises exception for 4XX/5XX responses
        
        data = response.json()
        
        if not data.get("list"):
            raise ValueError("No pollution data in OpenWeather response")
            
        pollution = data["list"][0]
        
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
        
    except Exception as e:
        logger.error(f"OpenWeather API Error: {str(e)}")
        return None

def fetch_accuweather():
    """Fetch weather data from AccuWeather API"""
    try:
        API_KEY = os.getenv("ACCUWEATHER_API_KEY")
        if not API_KEY:
            raise ValueError("ACCUWEATHER_API_KEY environment variable not set")
            
        LOCATION_KEY = "261158"  # Karachi
        url = f"https://dataservice.accuweather.com/currentconditions/v1/{LOCATION_KEY}?apikey={API_KEY}&details=true"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data or len(data) == 0:
            raise ValueError("Empty response from AccuWeather API")
            
        current = data[0]
        
        return {
            "temperature": current.get("Temperature", {}).get("Metric", {}).get("Value"),
            "humidity": current.get("RelativeHumidity"),
            "wind_speed": current.get("Wind", {}).get("Speed", {}).get("Metric", {}).get("Value"),
            "pressure": current.get("Pressure", {}).get("Metric", {}).get("Value"),
            "uv_index": current.get("UVIndex")
        }
        
    except Exception as e:
        logger.error(f"AccuWeather API Error: {str(e)}")
        return None

def save_to_hopsworks(data):
    try:
        project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            project="your_project_name"  # Add your exact project name
        )
        fs = project.get_feature_store()
        
        fg = fs.get_or_create_feature_group(
            name="aqi_weather_data",
            version=1,
            primary_key=["timestamp"],
            description="Combined AQI and weather data"
        )
        
        # Convert timestamp to string for Hopsworks compatibility
        data["timestamp"] = data["timestamp"].isoformat()
        df = pd.DataFrame([data])
        
        fg.insert(df)
        logger.info("Data saved successfully")
        
    except Exception as e:
        logger.error(f"Hopsworks Error: {str(e)}")
        raise
        
if __name__ == "__main__":
    logger.info("Starting data fetch...")
    
    # Fetch data from both APIs
    aqi_data = fetch_openweather()
    weather_data = fetch_accuweather()
    
    if not aqi_data or not weather_data:
        logger.error("Failed to fetch data from one or more APIs")
        exit(1)
        
    # Merge data
    combined_data = {**aqi_data, **weather_data}
    logger.info(f"Combined data: {combined_data}")
    
    # Save to Hopsworks
    try:
        save_to_hopsworks(combined_data)
        logger.info("Data pipeline completed successfully")
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}")
        exit(1)
