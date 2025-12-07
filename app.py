from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

# Load quantized model (this file must be in the same folder)
model_data = joblib.load("quantized_linreg_model.pkl")

scaler = model_data["scaler"]
w_q = model_data["w_q"]
s_w = model_data["s_w"]
b = model_data["b"]
s_x = model_data["s_x"]
features = model_data["features"]  # ['Trailwood_lag1', 'Wilmington_lag1']

app = FastAPI()

def predict_quantized(X_new):
    # 1. Scale inputs
    X_scaled = scaler.transform(X_new)

    # 2. Quantize inputs
    X_q = np.clip(np.round(X_scaled / s_x), -127, 127).astype(np.int8)

    # 3. Integer matmul
    y_int = X_q.astype(np.int32) @ w_q.astype(np.int32)

    # 4. Dequantize back to float
    y = (s_x * s_w) * y_int + b

    # Return scalar
    return float(y[0])

@app.get("/")
def home():
    # Just a health check: confirms API is running
    return {"status": "Walnut Creek Water Level Predictor is Running"}

@app.get("/predict")
def predict(trailwood: float, wilmington: float):
    """
    Inputs:
      trailwood  - upstream gage height at Trailwood (lag1)
      wilmington - upstream gage height at South Wilmington (lag1)

    Output:
      Predicted South State Street gage height
    """
    X = pd.DataFrame([[trailwood, wilmington]], columns=features)
    pred = predict_quantized(X)
    return {
        "trailwood_input": trailwood,
        "wilmington_input": wilmington,
        "predicted_south_state_gage_height": pred
    }
