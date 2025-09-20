from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import requests
from dotenv import load_dotenv

# Load .env
load_dotenv()

MAPMYINDIA_KEY = os.getenv("MAPMYINDIA_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")

# --- Load model ---
with open("model_ET.pkl", "rb") as f:
    model = pickle.load(f)

# --- mappings (kept from your original code) ---
state_mapping = {0: 'West Bengal'}

district_mapping = {
    0: '24 parganas north', 6: '24 parganas south', 12: 'alipurduar',
    18: 'bankura', 24: 'birbhum', 30: 'coochbehar', 36: 'darjeeling',
    42: 'dinajpur dakshin', 48: 'dinajpur uttar', 54: 'hooghly',
    60: 'howrah', 66: 'jalpaiguri', 72: 'jhargram', 75: 'kalimpong',
    78: 'maldah', 84: 'medinipur east', 90: 'medinipur west',
    96: 'murshidabad', 102: 'nadia', 108: 'paschim bardhaman',
    111: 'purba bardhaman', 117: 'purulia'
}

crop_mapping = {
    0: 'Arhar/Tur', 1: 'Bajra', 2: 'Barley', 3: 'Castor seed',
    4: 'Coconut', 5: 'Cotton(lint)', 6: 'Gram', 7: 'Groudnut',
    8: 'Horse-gram', 9: 'Jowar', 10: 'Jute', 11: 'Khesari',
    12: 'Linseed', 13: 'Maize', 14: 'Masoor', 15: 'Mesta',
    16: 'Moong(Green Gram)', 17: 'Moth', 18: 'Niger seed',
    19: 'Other Kharif pulses', 20: 'Other Rabi pulses',
    21: 'Peas & Beans (Pulses)', 22: 'Potato', 23: 'Ragi',
    24: 'Rapeseed &Mustard', 25: 'Rice', 26: 'Safflower',
    27: 'Sannhamp', 28: 'Sesamum', 29: 'Small millets',
    30: 'Soyabean', 31: 'Sugarcane', 32: 'Sunflower',
    33: 'Tobacco', 34: 'Urad', 35: 'Wheat',
}

season_mapping = {
    0: 'Autumn', 1: 'Kharif', 2: 'Rabi', 3: 'Summer', 4: 'Whole Year'
}

app = Flask(__name__)

@app.route("/")
def home():
    return render_template(
        "index.html",
        states=state_mapping,
        districts=district_mapping,
        crops=crop_mapping,
        seasons=season_mapping,
    )

@app.route("/reverse_geocode", methods=["POST"])
def reverse_geocode():
    """
    Accepts JSON: { "lat": <float>, "lon": <float> }
    Returns JSON with:
      - matched_district_name (string)
      - district_code (int or null)
      - weather fields: high_temp, low_temp, avg_temp, rainfall_mm, high_humidity, low_humidity
    """
    try:
        payload = request.get_json() or {}
        lat = payload.get("lat")
        lon = payload.get("lon")

        if lat is None or lon is None:
            return jsonify({"error": "Missing lat/lon"}), 400

        # 1) MapmyIndia Reverse Geocode
        district_name = None
        if MAPMYINDIA_KEY:
            # Example endpoint (MapmyIndia docs show: /advancedmaps/v1/<key>/rev_geocode?lat=...&lng=...)
            mm_url = f"https://apis.mapmyindia.com/advancedmaps/v1/{MAPMYINDIA_KEY}/rev_geocode"
            params = {"lat": lat, "lng": lon}
            try:
                mm_res = requests.get(mm_url, params=params, timeout=6)
                mm_json = mm_res.json()
                # Attempt to extract district from returned JSON robustly.
                # MapmyIndia responses vary across versions; search for 'district' key anywhere.
                def find_key(obj, keyname):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if k.lower() == keyname.lower():
                                return v
                            found = find_key(v, keyname)
                            if found:
                                return found
                    elif isinstance(obj, list):
                        for item in obj:
                            found = find_key(item, keyname)
                            if found:
                                return found
                    return None

                district_name = find_key(mm_json, "district")
                if not district_name:
                    # fallback: try adminInfo or subdistrict/locality fields
                    district_name = find_key(mm_json, "adminInfo") or find_key(mm_json, "subdistrict") or find_key(mm_json, "city") or find_key(mm_json, "state")
            except Exception as ex:
                # non-fatal: continue without district
                district_name = None

        # Normalize district name for matching
        matched_code = None
        matched_name = None
        if district_name:
            d_norm = str(district_name).strip().lower()
            # try to match with district_mapping by substring match
            for code, name in district_mapping.items():
                if name and name.lower() in d_norm:
                    matched_code = code
                    matched_name = name
                    break
            # If not matched exactly, try reverse: see if any value contains the district fragment
            if matched_code is None:
                for code, name in district_mapping.items():
                    if d_norm in name.lower() or name.lower() in d_norm:
                        matched_code = code
                        matched_name = name
                        break

        # 2) OpenWeather current weather
        weather_data = {
            "avg_temp": None,
            "high_temp": None,
            "low_temp": None,
            "rainfall_mm": 0.0,
            "high_humidity": None,
            "low_humidity": None,
        }
        if OPENWEATHER_KEY:
            ow_url = "https://api.openweathermap.org/data/2.5/weather"
            params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_KEY, "units": "metric"}
            try:
                ow_res = requests.get(ow_url, params=params, timeout=6)
                ow_json = ow_res.json()
                main = ow_json.get("main", {})
                rain = ow_json.get("rain", {})  # may include "1h" or "3h"
                # temperatures
                temp = main.get("temp")
                temp_min = main.get("temp_min")
                temp_max = main.get("temp_max")
                humidity = main.get("humidity")

                # Fill weather_data with sensible fallbacks
                weather_data["avg_temp"] = temp if temp is not None else None
                weather_data["high_temp"] = temp_max if temp_max is not None else temp
                weather_data["low_temp"] = temp_min if temp_min is not None else temp
                # rainfall: prefer 1h then 3h then 0
                rainfall = 0.0
                if isinstance(rain, dict):
                    rainfall = float(rain.get("1h") or rain.get("3h") or 0.0)
                weather_data["rainfall_mm"] = rainfall
                weather_data["high_humidity"] = humidity
                weather_data["low_humidity"] = humidity
            except Exception as ex:
                # non-fatal: leave weather_data as None/0
                pass

        response = {
            "matched_district_name": matched_name or district_name,
            "district_code": matched_code,
            "weather": weather_data,
            "raw_lat": lat,
            "raw_lon": lon
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form inputs
        data = {
            'State_Code': int(request.form['State_Code']),
            'District_Code': int(request.form['District_Code']),
            'Crop_Code': int(request.form['Crop_Code']),
            'Season_Code': int(request.form['Season_Code']),
            'Major Soil Type': request.form['Major_Soil_Type'],
            'Second Major Soil Type': request.form['Second_Major_Soil_Type'],
            'Irrigation Used': request.form['Irrigation_Used'],
            'Area_Hectares': float(request.form['Area_Hectares']),
            'Production': float(request.form['Production']),
            'Year_Numeric': int(request.form['Year_Numeric']),
            # The following are auto-filled hidden fields populated by client JS
            'Highest Temperature(in degree celsius)': float(request.form.get('High_Temp', 0.0)),
            'Lowest Temperature(in degree celsius)': float(request.form.get('Low_Temp', 0.0)),
            'Average Temperature(in degree celsius)': float(request.form.get('Avg_Temp', 0.0)),
            'Average Rainfall(past 5 years)': float(request.form.get('Rainfall', 0.0)),
            'Highest Humidity(past 5 years)': float(request.form.get('High_Humidity', 0.0)),
            'Lowest Humidity(past 5 years)': float(request.form.get('Low_Humidity', 0.0))
        }

        # Feature engineering (same as your original)
        data['Log_Area'] = np.log1p(data['Area_Hectares'])
        data['Log_Production'] = np.log1p(data['Production'])
        data['Temp_Range'] = data['Highest Temperature(in degree celsius)'] - data['Lowest Temperature(in degree celsius)']
        data['Temp_Anomaly'] = 0
        data['Humidity_Range'] = data['Highest Humidity(past 5 years)'] - data['Lowest Humidity(past 5 years)']
        data['Relative_Area'] = np.nan
        data['Crop_Diversity'] = np.nan
        data['District_Yield_Avg_3yr'] = np.nan

        # Column order
        categorical = [
            'State_Code','District_Code','Crop_Code','Season_Code',
            'Major Soil Type','Second Major Soil Type','Irrigation Used'
        ]
        numeric = [
            'Area_Hectares','Year_Numeric','Log_Area','Log_Production',
            'Relative_Area','Crop_Diversity','District_Yield_Avg_3yr',
            'Highest Temperature(in degree celsius)',
            'Lowest Temperature(in degree celsius)',
            'Average Temperature(in degree celsius)',
            'Temp_Range','Temp_Anomaly',
            'Average Rainfall(past 5 years)',
            'Highest Humidity(past 5 years)','Lowest Humidity(past 5 years)',
            'Humidity_Range'
        ]

        input_df = pd.DataFrame([data], columns=categorical + numeric)

        # Prediction
        prediction = model.predict(input_df)[0]

        return render_template("result.html", prediction=prediction)

    except Exception as e:
        return render_template("result.html", error=str(e))


if __name__ == "__main__":
    # In production, use a proper WSGI server
    app.run(debug=True, host="0.0.0.0", port=5000)
