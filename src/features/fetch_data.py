import os
import requests
import pandas as pd
import hopsworks
from datetime import datetime
import logging
from numbers import Number

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_float(value):
    """Convert value to float if it's a number, otherwise return None"""
    if isinstance(value, Number):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(f"Could not convert {value} to float")
        return None

def fetch_openweather():
    """Fetch air pollution data from OpenWeather API"""
    try:
        API_KEY = os.getenv("OPENWEATHER_API_KEY")
        if not API_KEY:
            raise ValueError("OPENWEATHER_API_KEY environment variable not set")
            
        LAT, LON = "24.9056", "67.0822"
        url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("list"):
            raise ValueError("No pollution data in OpenWeather response")
            
        pollution = data["list"][0]
        
        return {
            "timestamp": pd.Timestamp.now(),
            "aqi": ensure_float(pollution["main"]["aqi"]),
            "pm2_5": ensure_float(pollution["components"]["pm2_5"]),
            "pm10": ensure_float(pollution["components"]["pm10"]),
            "co": ensure_float(pollution["components"]["co"]),
            "no2": ensure_float(pollution["components"]["no2"]),
            "o3": ensure_float(pollution["components"]["o3"]),
            "so2": ensure_float(pollution["components"]["so2"])
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
            "temperature": ensure_float(current.get("Temperature", {}).get("Metric", {}).get("Value")),
            "humidity": ensure_float(current.get("RelativeHumidity")),
            "wind_speed": ensure_float(current.get("Wind", {}).get("Speed", {}).get("Metric", {}).get("Value")),
            "pressure": ensure_float(current.get("Pressure", {}).get("Metric", {}).get("Value")),
            "uv_index": ensure_float(current.get("UVIndex"))
        }
        
    except Exception as e:
        logger.error(f"AccuWeather API Error: {str(e)}")
        return None

def save_to_hopsworks(data):
    try:
        # Validate data before processing
        if not all(isinstance(v, (float, int)) for k, v in data.items() if k != 'timestamp'):
            logger.error("Invalid data types detected")
            raise ValueError("Data contains non-numeric values")
            
        project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            project="aqi_predictr" 
        )
        fs = project.get_feature_store()
        
        # Convert and enforce float types
        data["timestamp"] = data["timestamp"].isoformat()
        df = pd.DataFrame([data]).astype({
            'aqi': 'float64',
            'pm2_5': 'float64',
            'pm10': 'float64',
            'co': 'float64',
            'no2': 'float64',
            'o3': 'float64',
            'so2': 'float64',
            'temperature': 'float64',
            'humidity': 'float64',
            'wind_speed': 'float64',
            'pressure': 'float64',
            'uv_index': 'float64'
        })
        
        # Get or create feature group
        fg = fs.get_or_create_feature_group(
            name="aqi_weather_data",
            version=1,
            primary_key=["timestamp"],
            description="AQI and weather data with enforced float types"
        )
        fg.insert(df)
        
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
    
    # Validate all fields are numeric
    if not all(isinstance(v, (float, int)) for k, v in combined_data.items() if k != 'timestamp'):
        logger.error("Non-numeric values detected in combined data")
        exit(1)
    
    # Save to Hopsworks
    try:
        save_to_hopsworks(combined_data)
        logger.info("Data pipeline completed successfully")
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}")
        exit(1)
