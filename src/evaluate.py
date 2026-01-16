# src/evaluate.py

import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# -----------------------------
# 1. Paths
# -----------------------------

DATA_PATH = Path("data/processed/train.csv")
MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("metrics.json")

# -----------------------------
# 2. Load data and model
# -----------------------------

print("Loading processed data...")
df = pd.read_csv(DATA_PATH)

print("Loading trained model...")
model = joblib.load(MODEL_PATH)

# -----------------------------
# 3. Split features and target
# -----------------------------

TARGET = "SalePrice"

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Evaluate model
# -----------------------------

print("Evaluating model...")
y_pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))

# -----------------------------
# 5. Save metrics
# -----------------------------

metrics = {
    "rmse": rmse
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"RMSE: {rmse}")
print(f"Metrics saved to {METRICS_PATH}")
