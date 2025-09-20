from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
import requests
import os

# Load model
with open("model_ET.pkl", "rb") as f:
    model = pickle.load(f)

# Mappings
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

# API keys from environment variables
MAPMYINDIA_KEY = os.getenv("MAPMYINDIA_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")

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
    try:
        data = request.get_json()
        lat = data.get("lat")
        lon = data.get("lon")

        if not lat or not lon:
            return jsonify({"error": "Latitude/Longitude missing"}), 400

        # ---- MapmyIndia API to get district ----
        mapmi_url = f"https://apis.mapmyindia.com/advancedmaps/v1/{MAPMYINDIA_KEY}/rev_geocode?lat={lat}&lng={lon}"
        resp = requests.get(mapmi_url, timeout=5)
        resp.raise_for_status()
        district_name = resp.json().get("results", [{}])[0].get("admin_area4", "")

        # Match district_name to district_mapping
        matched_code = None
        for code, name in district_mapping.items():
            if name.lower() in district_name.lower():
                matched_code = code
                break

        # ---- OpenWeather API to get weather ----
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
        wresp = requests.get(weather_url, timeout=5)
        wresp.raise_for_status()
        wdata = wresp.json()

        weather = {
            "high_temp": wdata["main"]["temp_max"],
            "low_temp": wdata["main"]["temp_min"],
            "avg_temp": wdata["main"]["temp"],
            "rainfall_mm": wdata.get("rain", {}).get("1h", 0),
            "high_humidity": wdata["main"]["humidity"],
            "low_humidity": wdata["main"]["humidity"]
        }

        return jsonify({
            "district_code": matched_code,
            "matched_district_name": district_name,
            "weather": weather
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
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
            'Highest Temperature(in degree celsius)': float(request.form['High_Temp']),
            'Lowest Temperature(in degree celsius)': float(request.form['Low_Temp']),
            'Average Temperature(in degree celsius)': float(request.form['Avg_Temp']),
            'Average Rainfall(past 5 years)': float(request.form['Rainfall']),
            'Highest Humidity(past 5 years)': float(request.form['High_Humidity']),
            'Lowest Humidity(past 5 years)': float(request.form['Low_Humidity'])
        }

        # Feature engineering
        data['Log_Area'] = np.log1p(data['Area_Hectares'])
        data['Log_Production'] = np.log1p(data['Production'])
        data['Temp_Range'] = data['Highest Temperature(in degree celsius)'] - data['Lowest Temperature(in degree celsius)']
        data['Temp_Anomaly'] = 0
        data['Humidity_Range'] = data['Highest Humidity(past 5 years)'] - data['Lowest Humidity(past 5 years)']
        data['Relative_Area'] = np.nan
        data['Crop_Diversity'] = np.nan
        data['District_Yield_Avg_3yr'] = np.nan

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
        prediction = model.predict(input_df)[0]

        return render_template("result.html", prediction=prediction)

    except Exception as e:
        return render_template("result.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
