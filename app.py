from flask import Flask, jsonify, request
import joblib
import pandas as pd
import numpy as np
import requests
import os

app = Flask(__name__)

model        = joblib.load("safety_model.pkl")
le           = joblib.load("label_encoder.pkl")
grid         = pd.read_csv("grid_lookup.csv")
FEATURE_COLS = open("feature_cols.txt").read().strip().split("\n")

FAST2SMS_KEY   = "YOUR_FAST2SMS_API_KEY"
GUARDIAN_PHONE = "91XXXXXXXXXX"

def send_sms(lat, lon, risk_score):
    message = (
        f"SAFETY ALERT! Unsafe location detected. "
        f"Location: https://maps.google.com/?q={lat},{lon} "
        f"Risk Score: {risk_score:.2f}. Please check immediately!"
    )
    url     = "https://www.fast2sms.com/dev/bulkV2"
    payload = {
        "route"    : "v3",
        "sender_id": "TXTIND",
        "message"  : message,
        "language" : "english",
        "flash"    : 0,
        "numbers"  : GUARDIAN_PHONE
    }
    headers = {"authorization": FAST2SMS_KEY, "Content-Type": "application/json"}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

@app.route("/predict", methods=["GET"])
def predict():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "Missing or invalid lat/lon"}), 400

    distances   = np.sqrt((grid["latitude"] - lat)**2 + (grid["longitude"] - lon)**2)
    nearest_row = grid.iloc[distances.idxmin()]
    features    = pd.DataFrame([nearest_row[FEATURE_COLS].values], columns=FEATURE_COLS)

    prediction  = int(model.predict(features)[0])
    risk_score  = round(float(model.predict_proba(features)[0][1]), 4)

    sms_sent = False
    if prediction == 1:
        send_sms(lat, lon, risk_score)
        sms_sent = True

    return jsonify({
        "latitude"  : lat,
        "longitude" : lon,
        "label"     : "UNSAFE" if prediction == 1 else "SAFE",
        "risk_score": risk_score,
        "sms_sent"  : sms_sent,
        "area_id"   : nearest_row["area_id"]
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Women Safety API is running", "usage": "/predict?lat=13.05&lon=80.25"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)