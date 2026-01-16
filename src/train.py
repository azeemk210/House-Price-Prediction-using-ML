# src/train.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# -----------------------------
# 1. Paths
# -----------------------------

INPUT_PATH = Path("data/processed/train.csv")
MODEL_PATH = Path("models/model.pkl")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 2. Load processed data
# -----------------------------

print("Loading processed data...")
df = pd.read_csv(INPUT_PATH)

# -----------------------------
# 3. Split features and target
# -----------------------------

TARGET = "SalePrice"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Simple train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Train model
# -----------------------------

print("Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5. Save model
# -----------------------------

joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
