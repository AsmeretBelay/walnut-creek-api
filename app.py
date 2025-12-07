# Load model
model_data = joblib.load("quantized_linreg_model.pkl")

scaler = model_data["scaler"]
w_q_pruned = model_data["w_q"]   # shape (2,)
s_w = model_data["s_w"]
b = model_data["b"]
s_x = model_data["s_x"]

# Full feature list used by scaler
all_features = [
    "Rain_Gage_at_Lake_Johnson_354546078422045_lag2",
    "Walnut_Creek_at_Buck_Jones_Road_02087337_lag1",
    "Walnut_Creek_at_South_Wilmington_St_0208734795_lag1",
    "Walnut_Creek_at_Trailwood_Drive_0208734210_lag1",
    "drt"
]

# Pruned/kept features
kept_features = [
    "Walnut_Creek_at_South_Wilmington_St_0208734795_lag1",
    "Walnut_Creek_at_Trailwood_Drive_0208734210_lag1"
]

# Expand pruned weights to a 5-feature vector
w_q_full = np.zeros(len(all_features), dtype=np.int8)
for i, fname in enumerate(kept_features):
    full_index = all_features.index(fname)
    w_q_full[full_index] = w_q_pruned[i]

# Use the full-length quantized weight vector
w_q = w_q_full
