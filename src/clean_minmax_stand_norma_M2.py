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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Load Flight Delay Dataset
# ------------------------------------------------------------

file_path = "C:/Users/srinu/PycharmProjects/FlightFore/datasets/flights_5000_each_month_with_class_labels (1).csv"

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

# Numerical Columns

numerical_columns = df.select_dtypes(
    include=['int64', 'float64']
).columns


# Fill missing numerical values with mean

for column in numerical_columns:

    df[column] = df[column].fillna(
        df[column].mean()
    )


# Categorical Columns

categorical_columns = df.select_dtypes(
    include=['object']
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

    df[column] = df[column].str.strip()


# ------------------------------------------------------------
# Select Numeric Columns
# ------------------------------------------------------------

numeric_columns = df.select_dtypes(
    include=['int64', 'float64']
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
# Mean = 0, Standard Deviation = 1
# ------------------------------------------------------------

standard_scaler = StandardScaler()


standardized = standard_scaler.fit_transform(
    df[feature_numeric_columns]
)


for i, col in enumerate(feature_numeric_columns):

    df[col + "_Standardized"] = standardized[:, i]


# ------------------------------------------------------------
# Feature Scaling (Min-Max Scaling)
# Values between 0 and 1
# ------------------------------------------------------------

minmax_scaler = MinMaxScaler()


scaled = minmax_scaler.fit_transform(
    df[feature_numeric_columns]
)


for i, col in enumerate(feature_numeric_columns):

    df[col + "_Scaled"] = scaled[:, i]


# ------------------------------------------------------------
# Normalization (L2 Normalization)
# Each row becomes a unit vector
# ------------------------------------------------------------

normalizer = Normalizer(
    norm='l2'
)


normalized = normalizer.fit_transform(
    df[feature_numeric_columns]
)


for i, col in enumerate(feature_numeric_columns):

    df[col + "_Normalized"] = normalized[:, i]


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

output_file = "C:/Users/srinu/PycharmProjects/FlightFore/datasets/flight_minmax_standardized_normalized.csv"


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

pf = pd.read_csv(
    output_file
)


# Select only numerical columns for histograms

histogram_columns = pf.select_dtypes(
    include=['int64', 'float64']
).columns


pf[histogram_columns].hist(
    figsize=(16, 12),
    bins=10,
    edgecolor='black'
)


plt.suptitle(
    "Histogram of Preprocessed Flight Dataset",
    fontsize=16,
    fontweight="bold"
)


plt.tight_layout()


plt.show()