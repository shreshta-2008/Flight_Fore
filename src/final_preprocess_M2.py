# --------------------------------------------
# Flight Delay Prediction Dataset Preprocessing
# --------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import (LabelEncoder, StandardScaler)


# --------------------------------------------
# Read Original Flight Dataset
# --------------------------------------------

input_file = (
    "C:/Users/srinu/PycharmProjects/FlightFore/datasets/"
    "flights_5000_each_month_with_class_labels (1).csv"
)

output_file = (
    "C:/Users/srinu/PycharmProjects/FlightFore/datasets/"
    "final_flight_preprocess_M2.csv"
)


df = pd.read_csv(input_file)


# Create a copy so original dataset remains unchanged

processed_df = df.copy()


print("=" * 70)
print("FLIGHT DELAY PREDICTION PREPROCESSING")
print("=" * 70)

print("\nOriginal Dataset Shape:")
print(processed_df.shape)


# --------------------------------------------
# Clean Column Names
# --------------------------------------------

processed_df.columns = (
    processed_df.columns
    .str.strip()
)


# --------------------------------------------
# Target Column
# --------------------------------------------

target_column = "delay"


if target_column not in processed_df.columns:

    raise ValueError(
        "Target column 'delay' was not found in the dataset."
    )


print("\nTarget Column:")
print(target_column)


# --------------------------------------------
# Remove Duplicate Records
# --------------------------------------------

duplicate_count = processed_df.duplicated().sum()

print("\nDuplicate Records Found:")
print(duplicate_count)


processed_df = (
    processed_df
    .drop_duplicates()
    .copy()
)


print(
    "Dataset Shape After Duplicate Removal:",
    processed_df.shape
)


# --------------------------------------------
# Clean Text Data
# --------------------------------------------

categorical_cols = (
    processed_df
    .select_dtypes(include=["object"])
    .columns
    .tolist()
)


for col in categorical_cols:

    # Remove leading and trailing spaces

    processed_df[col] = (
        processed_df[col]
        .str.strip()
    )

    # Convert text to lowercase

    processed_df[col] = (
        processed_df[col]
        .str.lower()
    )


# --------------------------------------------
# Handle Missing Values
# --------------------------------------------

print("\nMissing Values Before Imputation:")

missing_before = (
    processed_df
    .isnull()
    .sum()
)

print(
    missing_before[
        missing_before > 0
    ]
)


# --------------------------------------------
# Numeric Columns
# --------------------------------------------

numeric_cols = (
    processed_df
    .select_dtypes(include=np.number)
    .columns
    .tolist()
)


for col in numeric_cols:

    # Do not modify target unnecessarily

    if col != target_column:

        median_value = (
            processed_df[col]
            .median()
        )

        processed_df[col] = (
            processed_df[col]
            .fillna(median_value)
        )


# --------------------------------------------
# Categorical Columns
# --------------------------------------------

for col in categorical_cols:

    mode_values = (
        processed_df[col]
        .mode()
    )

    if not mode_values.empty:

        processed_df[col] = (
            processed_df[col]
            .fillna(mode_values.iloc[0])
        )


# --------------------------------------------
# Remove Data Leakage
# --------------------------------------------

# These columns contain information that becomes available
# after or during the flight and therefore should NOT be
# used to predict whether a flight will be delayed.

leakage_columns = [
    "DEPARTURE_TIME",
    "DEPARTURE_DELAY",
    "TAXI_OUT",
    "WHEELS_OFF",
    "ELAPSED_TIME",
    "AIR_TIME",
    "WHEELS_ON",
    "TAXI_IN",
    "ARRIVAL_TIME",
    "ARRIVAL_DELAY",
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY",
    "CANCELLATION_REASON"
]


# Keep only columns that exist

leakage_columns = [
    col
    for col in leakage_columns
    if col in processed_df.columns
]


processed_df = processed_df.drop(
    columns=leakage_columns
)


print("\nLeakage Columns Removed:")

print(leakage_columns)


# --------------------------------------------
# Remove High-Cardinality Identifier
# --------------------------------------------

# TAIL_NUMBER identifies a particular aircraft.
# It is removed to reduce unnecessary overfitting.

identifier_columns = [
    "TAIL_NUMBER"
]


identifier_columns = [
    col
    for col in identifier_columns
    if col in processed_df.columns
]


processed_df = processed_df.drop(
    columns=identifier_columns
)


print("\nIdentifier Columns Removed:")

print(identifier_columns)


# --------------------------------------------
# Re-identify Columns After Removal
# --------------------------------------------

numeric_cols = (
    processed_df
    .select_dtypes(include=np.number)
    .columns
    .tolist()
)


categorical_cols = (
    processed_df
    .select_dtypes(include=["object"])
    .columns
    .tolist()
)


# --------------------------------------------
# Label Encoding
# --------------------------------------------

print("\nCategorical Columns to Encode:")

print(categorical_cols)


for col in categorical_cols:

    encoder = LabelEncoder()

    processed_df[col] = (
        encoder.fit_transform(
            processed_df[col].astype(str)
        )
    )


# --------------------------------------------
# Feature Scaling
# --------------------------------------------

# Re-identify numeric columns after encoding

numeric_cols = (
    processed_df
    .select_dtypes(include=np.number)
    .columns
    .tolist()
)


# Do NOT scale the target column

feature_numeric_cols = [
    col
    for col in numeric_cols
    if col != target_column
]


scaler = StandardScaler()


if len(feature_numeric_cols) > 0:

    processed_df[
        feature_numeric_cols
    ] = scaler.fit_transform(
        processed_df[
            feature_numeric_cols
        ]
    )


# --------------------------------------------
# Final Missing Value Check
# --------------------------------------------

print("\nMissing Values After Processing:")

remaining_missing = (
    processed_df
    .isnull()
    .sum()
)

print(
    remaining_missing[
        remaining_missing > 0
    ]
)


# --------------------------------------------
# Final Dataset Information
# --------------------------------------------

print("\n" + "=" * 70)
print("FINAL PREPROCESSED DATASET")
print("=" * 70)


print("\nOriginal Dataset Shape:")
print(df.shape)


print("\nProcessed Dataset Shape:")
print(processed_df.shape)


print("\nFinal Columns:")

print(
    processed_df.columns.tolist()
)


print("\nFirst 5 Records:")

print(
    processed_df.head()
)


# --------------------------------------------
# Target Distribution
# --------------------------------------------

print("\nTarget Distribution:")

print(
    processed_df[target_column]
    .value_counts()
)


# --------------------------------------------
# Save Preprocessed Dataset
# --------------------------------------------

processed_df.to_csv(
    output_file,
    index=False
)


# --------------------------------------------
# Completion Message
# --------------------------------------------

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nOriginal dataset was NOT modified.")

print("\nInput File:")
print(input_file)

print("\nOutput File:")
print(output_file)

print("\nFinal Dataset Shape:")
print(processed_df.shape)

print("=" * 70)