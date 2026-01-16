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

# Example: remove ID column if present
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

# Handle missing values (VERY BASIC for now)
# Numerical columns -> fill with median
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical columns -> fill with "Missing"
cat_cols = df.select_dtypes(include=["object"]).columns
df[cat_cols] = df[cat_cols].fillna("Missing")

# -----------------------------
# 4. Save processed data
# -----------------------------

print("Saving processed training data...")
df.to_csv(OUTPUT_PATH, index=False)

print("Preprocessing completed successfully.")
