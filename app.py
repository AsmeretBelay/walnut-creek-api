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

# The scaler was trained on 5 original features:
all_features = [
    "Rain_Gage_at_Lake_Johnson_354546078422045_lag2",
    "Walnut_Creek_at_Buck_Jones_Road_02087337_lag1",
    "Walnut_Creek_at_South_Wilmington_St_0208734795_lag1",
    "Walnut_Creek_at_Trailwood_Drive_0208734210_lag1",
    "drt"
]

# The 2 pruned (kept) features:
kept_features = [
    "Walnut_Creek_at_South_Wilmington_St_0208734795_lag1",
    "Walnut_Creek_at_Trailwood_Drive_0208734210_lag1"
]

app = FastAPI()

def predict_quantized(trailwood, wilmington):
    # Build full 5-feature input
    row = {
        "Rain_Gage_at_Lake_Johnson_354546078422045_lag2": 0.0,
        "Walnut_Creek_at_Buck_Jones_Road_02087337_lag1": 0.0,
        "Walnut_Creek_at_South_Wilmington_St_0208734795_lag1": wilmington,
        "Walnut_Creek_at_Trailwood_Drive_0208734210_lag1": trailwood,
        "drt": 0.0
    }

    X = pd.DataFrame([row])

    # SCALE
    X_scaled = scaler.transform(X)

    # QUANTIZE
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
