from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

# Load model
model_data = joblib.load("quantized_linreg_model.pkl")

scaler = model_data["scaler"]
w_q = model_data["w_q"]
s_w = model_data["s_w"]
b = model_data["b"]
s_x = model_data["s_x"]
all_features = scaler.feature_names_in_    # full list of 5 original features
kept_features = model_data["features"]     # the 2 features kept after pruning

app = FastAPI()

def predict_quantized(trailwood, wilmington):
    # Build input with ALL original 5 features
    row = {feature: 0.0 for feature in all_features}

    # Insert the two kept features with real input values
    row[kept_features[0]] = trailwood
    row[kept_features[1]] = wilmington

    # Convert to DataFrame
    X = pd.DataFrame([row])

    # SCALE using the original 5-feature scaler
    X_scaled = scaler.transform(X)

    # QUANTIZE input
    X_q = np.clip(np.round(X_scaled / s_x), -127, 127).astype(np.int8)

    # INTEGER MATMUL
    y_int = X_q.astype(np.int32) @ w_q.astype(np.int32)

    # DEQUANTIZE
    y = (s_x * s_w) * y_int + b

    return float(y[0])

@app.get("/")
def home():
    return {"status": "Walnut Creek Water Level Predictor is Running"}

@app.get("/predict")
def predict(trailwood: float, wilmington: float):
    pred = predict_quantized(trailwood, wilmington)
    return {
        "trailwood_input": trailwood,
        "wilmington_input": wilmington,
        "predicted_south_state_gage_height": pred
    }
