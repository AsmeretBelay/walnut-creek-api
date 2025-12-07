from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# ======================================================
# LOAD MODEL
# ======================================================
model_data = joblib.load("quantized_linreg_model.pkl")

scaler = model_data["scaler"]
w_q_pruned = model_data["w_q"]          # length 2
s_w = model_data["s_w"]
b = model_data["b"]
s_x = model_data["s_x"]

# ======================================================
# FULL 5 FEATURE NAMES (in training order)
# ======================================================
all_features = [
    "Rain_Gage_at_Lake_Johnson_354546078422045_lag2",
    "Walnut_Creek_at_Buck_Jones_Road_02087337_lag1",
    "Walnut_Creek_at_South_Wilmington_St_0208734795_lag1",
    "Walnut_Creek_at_Trailwood_Drive_0208734210_lag1",
    "drt"
]

# The 2 pruned features that remain
kept_features = [
    "Walnut_Creek_at_South_Wilmington_St_0208734795_lag1",
    "Walnut_Creek_at_Trailwood_Drive_0208734210_lag1"
]

# ======================================================
# EXPAND QUANTIZED WEIGHT VECTOR FROM 2 → 5
# ======================================================
w_q_full = np.zeros(len(all_features), dtype=np.int8)

for i, fname in enumerate(kept_features):
    full_index = all_features.index(fname)
    w_q_full[full_index] = w_q_pruned[i]

# replace pruned weights with padded full-size weight vector
w_q = w_q_full


# ======================================================
# QUANTIZED PREDICTION FUNCTION
# ======================================================
def predict_quantized(trailwood, wilmington):

    # Build a row with 5 features
    row = {
        "Rain_Gage_at_Lake_Johnson_354546078422045_lag2": 0.0,
        "Walnut_Creek_at_Buck_Jones_Road_02087337_lag1": 0.0,
        "Walnut_Creek_at_South_Wilmington_St_0208734795_lag1": wilmington,
        "Walnut_Creek_at_Trailwood_Drive_0208734210_lag1": trailwood,
        "drt": 0.0
    }

    X = pd.DataFrame([row])

    # Scale using original scaler (trained on 5 features)
    X_scaled = scaler.transform(X)

    # Quantize input
    X_q = np.clip(np.round(X_scaled / s_x), -127, 127).astype(np.int8)

    # Integer matrix multiply
    y_int = X_q.astype(np.int32) @ w_q.astype(np.int32)

    # Dequantize
    y = (s_x * s_w) * y_int + b

    return float(y[0])


# ======================================================
# API ROUTES
# ======================================================
@app.get("/")
def home():
    return {"status": "Walnut Creek Water Level Predictor API is running."}


@app.get("/predict")
def predict(trailwood: float, wilmington: float):
    pred = predict_quantized(trailwood, wilmington)
    return {
        "input_trailwood": trailwood,
        "input_wilmington": wilmington,
        "predicted_south_state_gage_height": pred
    }
