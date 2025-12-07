# Walnut Creek Water Level Prediction API  
Pruned + Quantized Linear Regression Model (INT8)  
Deployed Using FastAPI + Render (Free Cloud Hosting)

## Project Overview

This API predicts the water level at **South State Street (USGS 02087358)** in Raleigh, NC using two upstream gauges:

- Trailwood Drive (lag1)
- South Wilmington Street (lag1)

The model was compressed using pruning (LASSO) and quantization (INT8) to create a lightweight, real-time prediction system.

## Deployment Instructions (Render)

1. Upload these files to a GitHub repository:
   - `app.py`
   - `quantized_linreg_model.pkl`
   - `requirements.txt`
   - `README.md`

2. Go to https://render.com  
3. Click **New → Web Service**  
4. Connect your GitHub repo  
5. Configure:

- **Runtime:** Python 3  
- **Start Command:**
  ```bash
  uvicorn app:app --host 0.0.0.0 --port 10000
