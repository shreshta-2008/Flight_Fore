import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# ==========================================================
# Load Flight Delay Prediction Dataset
# Original dataset will NOT be modified
# ==========================================================

df = pd.read_csv(
    "C:/Users/hp/PycharmProjects/FlightFore/datasets/flights_5000_each_month_with_class_labels (1).csv"
)


# Create a copy for processing
data = df.copy()


# ==========================================================
# 1. Remove Leading and Trailing Spaces
# ==========================================================

for col in data.select_dtypes(include="str").columns:
    data[col] = data[col].str.strip()


# ==========================================================
# 2. Identify Missing Values
# ==========================================================

print("Missing Values Before Cleaning:")
print(data.isnull().sum())


# ==========================================================
# 3. Remove Duplicate Records
# ==========================================================

before = data.shape[0]

data = data.drop_duplicates()

after = data.shape[0]

print(
    "\nDuplicate Flight Records Removed:",
    before - after
)


# ==========================================================
# Separate Numerical and Categorical Columns
# ==========================================================

num_cols = data.select_dtypes(
    include=np.number
).columns.tolist()

cat_cols = data.select_dtypes(
    exclude=np.number
).columns.tolist()


# ==========================================================
# Separate Target Variable
# ==========================================================

target_col = "Delay"

# Do not encode the target as an input feature
if target_col in num_cols:
    num_feature_cols = [
        col for col in num_cols
        if col != target_col
    ]
else:
    num_feature_cols = num_cols

print("\nNumerical Feature Columns:")
print(num_feature_cols)

print("\nCategorical Columns:")
print(cat_cols)

print("\nTarget Column:")
print(target_col)


# ==========================================================
# 4. Fill Missing Numerical Values with Mean
# ==========================================================

if len(num_feature_cols) > 0:

    num_imputer = SimpleImputer(
        strategy="mean"
    )

    data[num_feature_cols] = num_imputer.fit_transform(
        data[num_feature_cols]
    )


# ==========================================================
# 5. Fill Missing Categorical Values with Mode
# ==========================================================

if len(cat_cols) > 0:

    cat_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    data[cat_cols] = cat_imputer.fit_transform(
        data[cat_cols]
    )


# ==========================================================
# 6. Label Encoding
# ==========================================================

label_encoders = {}


for col in cat_cols:

    encoder = LabelEncoder()

    data[col] = encoder.fit_transform(
        data[col]
    )

    label_encoders[col] = encoder

    print(
        f"\nLabel Encoding completed for: {col}"
    )

    print(
        "Number of categories:",
        len(encoder.classes_)
    )


# ==========================================================
# 7. Check Missing Values After Cleaning
# ==========================================================

print("\nMissing Values After Cleaning:")
print(data.isnull().sum())


# ==========================================================
# 8. Display Final Dataset Information
# ==========================================================

print("\n======================================")
print("FINAL FLIGHT DATASET")
print("======================================")

print("\nDataset Shape:")
print(data.shape)

print("\nDataset Columns:")
print(data.columns.tolist())

print("\nFirst 5 Records:")
print(data.head())


# ==========================================================
# Save Result
# ==========================================================

output_file = "C:/Users/hp/PycharmProjects/FlightFore/datasets/flight_label_encoded.csv"

data.to_csv(
    output_file,
    index=False
)


print("\n======================================")
print("Original dataset is NOT modified.")
print("Flight Label Encoding completed successfully.")
print("Output file:")
print(output_file)
print("======================================")