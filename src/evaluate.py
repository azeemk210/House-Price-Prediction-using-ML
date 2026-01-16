# src/evaluate.py

import json
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
from preprocess import TARGET, num_cols, cat_cols

# -----------------------------
# 0. Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Evaluate or predict using the trained model.")
parser.add_argument("--test-file", type=str, help="Path to the test CSV file.")
parser.add_argument("--predict", action="store_true", help="Generate predictions instead of evaluation.")
args = parser.parse_args()

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

if args.predict:
    print("Generating predictions...")
    test_data = pd.read_csv(args.test_file)

    # Preprocess test data
    if "Id" in test_data.columns:
        test_data = test_data.drop(columns=["Id"])

    test_data[num_cols] = test_data[num_cols].fillna(test_data[num_cols].median())
    test_data[cat_cols] = test_data[cat_cols].fillna("Missing")
    test_data = pd.get_dummies(test_data, columns=cat_cols, drop_first=True)

    # Align columns with training data
    missing_cols = set(X.columns) - set(test_data.columns)
    for col in missing_cols:
        test_data[col] = 0
    test_data = test_data[X.columns]

    predictions = model.predict(test_data)

    output_data = {"Predictions": predictions}

    # Check for ground truth and calculate error
    if TARGET in test_data.columns:
        ground_truth = test_data[TARGET]
        errors = predictions - ground_truth
        output_data["Ground Truth"] = ground_truth
        output_data["Errors"] = errors

    output_path = Path("predictions.csv")
    pd.DataFrame(output_data).to_csv(output_path, index=False)
    print(f"Predictions and errors saved to {output_path}")
else:
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
