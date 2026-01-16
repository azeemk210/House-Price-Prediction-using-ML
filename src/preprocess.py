# src/preprocess.py

import pandas as pd
from pathlib import Path

# -----------------------------
# 1. Define input and output paths
# -----------------------------

# Raw data path (tracked by DVC)
INPUT_PATH = Path("data/raw/train.csv")

# Processed data output (will be tracked by DVC as output)
OUTPUT_PATH = Path("data/processed/train.csv")

# Ensure output directory exists
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 2. Load raw data
# -----------------------------

print("Loading raw training data...")
df = pd.read_csv(INPUT_PATH)

# -----------------------------
# 3. Basic preprocessing
# -----------------------------

# Drop ID column if present
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# Separate target before encoding
TARGET = "SalePrice"
y = df[TARGET]
X = df.drop(columns=[TARGET])

# Numerical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())

# Categorical columns
cat_cols = X.select_dtypes(include=["object"]).columns
X[cat_cols] = X[cat_cols].fillna("Missing")

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Reattach target
df_processed = pd.concat([X, y], axis=1)

# -----------------------------
# 4. Save processed data
# -----------------------------

print("Saving processed training data...")
df_processed.to_csv(OUTPUT_PATH, index=False)

print("Preprocessing completed successfully.")
