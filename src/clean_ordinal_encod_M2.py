import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# ==========================================================
# Load Flight Delay Prediction Dataset
# ==========================================================

DATASET_PATH = r"C:/Users/hp/PycharmProjects/FlightFore/datasets/flights_5000_each_month_with_class_labels (1).csv"

df = pd.read_csv(DATASET_PATH)

# Create a copy
data = df.copy()

# ==========================================================
# Remove Leading and Trailing Spaces
# ==========================================================

for col in data.select_dtypes(include=["object", "string"]).columns:
    data[col] = data[col].astype(str).str.strip()

# ==========================================================
# Missing Values
# ==========================================================

print("=" * 60)
print("Missing Values Before Cleaning")
print("=" * 60)
print(data.isnull().sum())

# ==========================================================
# Remove Duplicate Records
# ==========================================================

before = len(data)

data = data.drop_duplicates()

after = len(data)

print("\nDuplicate Rows Removed :", before - after)

# ==========================================================
# Target Column
# ==========================================================

if "delay" in data.columns:
    target_col = "delay"
else:
    raise ValueError("Target column 'delay' not found.")

print("\nTarget Column :", target_col)

# ==========================================================
# Numerical & Categorical Columns
# ==========================================================

num_cols = data.select_dtypes(include=np.number).columns.tolist()

cat_cols = data.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

# Remove target from numerical columns

if target_col in num_cols:
    feature_num_cols = [c for c in num_cols if c != target_col]
else:
    feature_num_cols = num_cols

print("\nNumerical Features")
print(feature_num_cols)

print("\nCategorical Features")
print(cat_cols)

# ==========================================================
# Numerical Missing Values
# ==========================================================

if feature_num_cols:

    num_imputer = SimpleImputer(strategy="mean")

    data[feature_num_cols] = num_imputer.fit_transform(
        data[feature_num_cols]
    )

# ==========================================================
# Categorical Missing Values
# ==========================================================

if cat_cols:

    cat_imputer = SimpleImputer(strategy="most_frequent")

    data[cat_cols] = cat_imputer.fit_transform(
        data[cat_cols]
    )

# ==========================================================
# Ordinal Encoding
# ==========================================================

if cat_cols:

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    encoded = encoder.fit_transform(data[cat_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=[
            "Ordinal_" + col
            for col in cat_cols
        ]
    )

    final_output = pd.concat(
        [
            data[feature_num_cols].reset_index(drop=True),
            encoded_df.reset_index(drop=True),
            data[[target_col]].reset_index(drop=True)
        ],
        axis=1
    )

else:

    final_output = data.copy()

# ==========================================================
# Missing Values After Cleaning
# ==========================================================

print("\n")
print("=" * 60)
print("Missing Values After Cleaning")
print("=" * 60)
print(final_output.isnull().sum())

# ==========================================================
# Dataset Information
# ==========================================================

print("\n")
print("=" * 60)
print("FINAL DATASET")
print("=" * 60)

print("Shape :", final_output.shape)

print("\nColumns")
print(final_output.columns.tolist())

print("\nFirst Five Rows")
print(final_output.head())

# ==========================================================
# Save Dataset
# ==========================================================

OUTPUT_FILE = r"C:/Users/hp/PycharmProjects/FlightFore/datasets/flight_ordinal_encoded.csv"

final_output.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n")
print("=" * 60)
print("Ordinal Encoding Completed Successfully")
print("Output File:")
print(OUTPUT_FILE)
print("=" * 60)