# ---------------------------------------------------------
# Numeric Column Pre-processing Techniques
# Mean, Median, Mode
# Feature Scaling, Standardization, and Normalization
# Save all results in ONE CSV file
# ---------------------------------------------------------
# Check whether scikit-learn is installed.
# If not, install using:
# pip install scikit-learn
# ---------------------------------------------------------

import pandas as pd
import numpy as np

# ---------------------------------------------------------
# IMPORTANT:
# Use a non-GUI Matplotlib backend.
# This prevents the Tkinter "init.tcl" error.
# ---------------------------------------------------------

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer


# ------------------------------------------------------------
# Load Flight Delay Dataset
# ------------------------------------------------------------

file_path = (
    "C:/Users/hp/PycharmProjects/FlightFore/"
    "datasets/flights_5000_each_month_with_class_labels (1).csv"
)

df = pd.read_csv(file_path)


print("Original Flight Dataset")
print("------------------------")
print(df.head())


print("\nDataset Shape:", df.shape)


print("\nData Types:")
print("------------------------")
print(df.dtypes)


print("\nMissing Values:")
print("------------------------")
print(df.isnull().sum())


print("\nDuplicate Records:", df.duplicated().sum())


# ---------------------------------------------------
# Step 2: Remove Duplicate Records
# ---------------------------------------------------

df = df.drop_duplicates()


# ---------------------------------------------------
# Step 3: Handle Missing Values
# ---------------------------------------------------

# ---------------------------------------------------
# Numerical Columns
# ---------------------------------------------------

numerical_columns = df.select_dtypes(
    include=["number"]
).columns


# Fill missing numerical values with mean

for column in numerical_columns:

    df[column] = df[column].fillna(
        df[column].mean()
    )


# ---------------------------------------------------
# Categorical Columns
# ---------------------------------------------------

categorical_columns = df.select_dtypes(
    include=["object", "string"]
).columns


# Fill missing categorical values with mode

for column in categorical_columns:

    if not df[column].mode().empty:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


# ---------------------------------------------------
# Step 4: Remove Leading and Trailing Spaces
# ---------------------------------------------------

for column in categorical_columns:

    df[column] = df[column].astype("string").str.strip()


# ------------------------------------------------------------
# Select Numeric Columns
# ------------------------------------------------------------

numeric_columns = df.select_dtypes(
    include=["number"]
).columns


# ------------------------------------------------------------
# IMPORTANT:
# Do not scale the target variable "Delay".
# It should remain unchanged for machine-learning
# classification.
# ------------------------------------------------------------

target_column = "Delay"


if target_column in numeric_columns:

    feature_numeric_columns = [
        col for col in numeric_columns
        if col != target_column
    ]

else:

    feature_numeric_columns = list(
        numeric_columns
    )


print("\nNumeric Feature Columns:")
print(list(feature_numeric_columns))


print("\nTarget Column:")
print(target_column)


# ------------------------------------------------------------
# Standardization (Z-score)
# Mean = 0
# Standard Deviation = 1
# ------------------------------------------------------------

standard_scaler = StandardScaler()


standardized = standard_scaler.fit_transform(
    df[feature_numeric_columns]
)


standardized_df = pd.DataFrame(
    standardized,
    columns=[
        col + "_Standardized"
        for col in feature_numeric_columns
    ],
    index=df.index
)


# ------------------------------------------------------------
# Feature Scaling (Min-Max Scaling)
# Values between 0 and 1
# ------------------------------------------------------------

minmax_scaler = MinMaxScaler()


scaled = minmax_scaler.fit_transform(
    df[feature_numeric_columns]
)


scaled_df = pd.DataFrame(
    scaled,
    columns=[
        col + "_Scaled"
        for col in feature_numeric_columns
    ],
    index=df.index
)


# ------------------------------------------------------------
# Normalization (L2 Normalization)
# Each row becomes a unit vector
# ------------------------------------------------------------

normalizer = Normalizer(
    norm="l2"
)


normalized = normalizer.fit_transform(
    df[feature_numeric_columns]
)


normalized_df = pd.DataFrame(
    normalized,
    columns=[
        col + "_Normalized"
        for col in feature_numeric_columns
    ],
    index=df.index
)


# ------------------------------------------------------------
# Combine all preprocessing results
# ------------------------------------------------------------
# Using pd.concat() avoids the Pandas
# "DataFrame is highly fragmented" warning.
# ------------------------------------------------------------

df = pd.concat(
    [
        df,
        standardized_df,
        scaled_df,
        normalized_df
    ],
    axis=1
)


# ------------------------------------------------------------
# Display Results after Pre-processing
# ------------------------------------------------------------

print(
    "\nDisplay Results after Preprocessed "
    "Flight Dataset"
)

print(df.head())


print("\nDataset Shape:", df.shape)


print("\nDataset Information")

df.info()


print("\nColumns in Dataset:")
print(df.columns)


print("\nMissing Values After Preprocessing")
print(df.isnull().sum())


print("\nDuplicate Records After Preprocessing")
print(df.duplicated().sum())


# ---------------------------------------------------
# Save Preprocessed Dataset
# ---------------------------------------------------

output_file = (
    "C:/Users/hp/PycharmProjects/FlightFore/"
    "datasets/flight_minmax_standardized_normalized.csv"
)


df.to_csv(
    output_file,
    index=False
)


print(
    "\nPreprocessed flight dataset "
    "saved successfully."
)

print("Output File:")
print(output_file)


# ---------------------------------------------------
# Display Histogram of Preprocessed Data
# ---------------------------------------------------

print("\nCreating histogram...")


pf = pd.read_csv(
    output_file
)


# ---------------------------------------------------
# Select only numerical columns for histograms
# ---------------------------------------------------

histogram_columns = pf.select_dtypes(
    include=["number"]
).columns


# ---------------------------------------------------
# Limit the number of histograms
# ---------------------------------------------------
# Your dataset contains many numerical columns after
# adding Standardized, Scaled and Normalized columns.
# Plotting every column can create a huge figure.
# ---------------------------------------------------

histogram_columns = histogram_columns[:20]


try:

    pf[histogram_columns].hist(
        figsize=(16, 12),
        bins=10,
        edgecolor="black"
    )


    plt.suptitle(
        "Histogram of Preprocessed Flight Dataset",
        fontsize=16,
        fontweight="bold"
    )


    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )


    # ------------------------------------------------
    # Save histogram instead of plt.show()
    # This works without Tkinter.
    # ------------------------------------------------

    histogram_file = (
        "C:/Users/hp/PycharmProjects/FlightFore/"
        "datasets/flight_preprocessed_histogram.png"
    )


    plt.savefig(
        histogram_file,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close("all")


    print("\nHistogram saved successfully.")
    print("Histogram File:")
    print(histogram_file)


except Exception as e:

    print("\nHistogram could not be generated.")
    print("Reason:", e)


# ---------------------------------------------------
# Final Message
# ---------------------------------------------------

print("\n========================================")
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("========================================")

print("\nFinal Dataset Shape:", df.shape)

print("\nCSV File:")
print(output_file)

print("\nHistogram File:")
print(
    "C:/Users/hp/PycharmProjects/FlightFore/"
    "datasets/flight_preprocessed_histogram.png"
)