# ============================================================
# FLIGHT DELAY PREDICTION - IMPROVED DATA PREPROCESSING
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# 1. FILE PATHS
# ============================================================

input_file = (
    "C:/Users/srinu/PycharmProjects/FlightFore/datasets/"
    "flights_5000_each_month_with_class_labels (1).csv"
)

output_file = (
    "C:/Users/srinu/PycharmProjects/FlightFore/datasets/"
    "flight_delay_preprocessed.csv"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(input_file)

# Create a copy so original dataset is never modified
data = df.copy()

print("\n============================================================")
print("FLIGHT DELAY DATASET PREPROCESSING")
print("============================================================")

print("\nOriginal Dataset Shape:")
print(data.shape)


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

data.columns = data.columns.str.strip()

print("\nDataset Columns:")
print(data.columns.tolist())


# ============================================================
# 4. REMOVE LEADING/TRAILING SPACES
# ============================================================

for col in data.select_dtypes(include="object").columns:

    data[col] = data[col].str.strip()


# ============================================================
# 5. TARGET COLUMN
# ============================================================

target_column = "delay"

if target_column not in data.columns:

    raise ValueError(
        f"Target column '{target_column}' was not found."
    )

print("\nTarget Column:")
print(target_column)


# ============================================================
# 6. CHECK TARGET VALUES
# ============================================================

print("\nTarget Distribution:")

print(
    data[target_column].value_counts()
)


print("\nTarget Percentage:")

print(
    data[target_column]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 7. MISSING VALUE ANALYSIS
# ============================================================

print("\n============================================================")
print("MISSING VALUES BEFORE CLEANING")
print("============================================================")

missing_before = data.isnull().sum()

print(
    missing_before[
        missing_before > 0
    ]
)


# ============================================================
# 8. REMOVE DUPLICATE RECORDS
# ============================================================

duplicate_count = data.duplicated().sum()

print("\nDuplicate Records Found:")
print(duplicate_count)

if duplicate_count > 0:

    data = data.drop_duplicates()

print("\nDataset Shape After Duplicate Removal:")
print(data.shape)


# ============================================================
# 9. HANDLE MISSING VALUES IN CATEGORICAL COLUMNS
# ============================================================

categorical_columns = data.select_dtypes(
    include="object"
).columns.tolist()


for col in categorical_columns:

    mode_values = data[col].mode()

    if len(mode_values) > 0:

        data[col] = data[col].fillna(
            mode_values.iloc[0]
        )


# ============================================================
# 10. HANDLE MISSING VALUES IN NUMERICAL COLUMNS
# ============================================================

numerical_columns = data.select_dtypes(
    include=np.number
).columns.tolist()


for col in numerical_columns:

    if col != target_column:

        median_value = data[col].median()

        data[col] = data[col].fillna(
            median_value
        )


# ============================================================
# 11. CONVERT TIME COLUMNS
# ============================================================

# Flight times are stored as HHMM.
# Example:
# 1032 -> 10:32
# 2055 -> 20:55

time_columns = [
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL"
]


for col in time_columns:

    if col in data.columns:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

        hour = (
            data[col] // 100
        )

        minute = (
            data[col] % 100
        )

        # Convert to minutes from midnight
        data[col + "_MINUTES"] = (
            hour * 60 + minute
        )

        # Extract hour
        data[col + "_HOUR"] = hour


# ============================================================
# 12. CREATE ROUTE FEATURE
# ============================================================

if (
    "ORIGIN_AIRPORT" in data.columns
    and
    "DESTINATION_AIRPORT" in data.columns
):

    data["ROUTE"] = (
        data["ORIGIN_AIRPORT"].astype(str)
        + "_"
        + data["DESTINATION_AIRPORT"].astype(str)
    )


# ============================================================
# 13. CREATE WEEKEND FEATURE
# ============================================================

if "DAY_OF_WEEK" in data.columns:

    data["IS_WEEKEND"] = (
        data["DAY_OF_WEEK"] >= 6
    ).astype(int)


# ============================================================
# 14. CREATE SCHEDULED TIME FEATURES
# ============================================================

if (
    "SCHEDULED_DEPARTURE_MINUTES" in data.columns
    and
    "SCHEDULED_ARRIVAL_MINUTES" in data.columns
):

    data["SCHEDULED_FLIGHT_DURATION"] = (
        data["SCHEDULED_ARRIVAL_MINUTES"]
        - data["SCHEDULED_DEPARTURE_MINUTES"]
    )

    # Correct overnight flights
    data.loc[
        data["SCHEDULED_FLIGHT_DURATION"] < 0,
        "SCHEDULED_FLIGHT_DURATION"
    ] += 1440


# ============================================================
# 15. REMOVE DATA LEAKAGE
# ============================================================

# These columns contain information that becomes available
# only after the flight has started or after the delay occurs.
#
# Using them for prediction would make the model unrealistically
# accurate.

leakage_columns = [

    "DEPARTURE_TIME",
    "DEPARTURE_DELAY",

    "WHEELS_OFF",
    "TAXI_OUT",

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


# Remove only columns that actually exist
leakage_columns = [
    col for col in leakage_columns
    if col in data.columns
]


data = data.drop(
    columns=leakage_columns
)


print("\nLeakage Columns Removed:")

print(leakage_columns)


# ============================================================
# 16. REMOVE HIGH-CARDINALITY IDENTIFIERS
# ============================================================

# TAIL_NUMBER identifies an individual aircraft.
# Keeping it directly can cause overfitting.
#
# FLIGHT_NUMBER is also treated as an identifier rather than
# a meaningful continuous numerical variable.

identifier_columns = [

    "TAIL_NUMBER",
    "FLIGHT_NUMBER"
]


identifier_columns = [
    col for col in identifier_columns
    if col in data.columns
]


data = data.drop(
    columns=identifier_columns
)


print("\nIdentifier Columns Removed:")

print(identifier_columns)


# ============================================================
# 17. TARGET ENCODING
# ============================================================

# Target encoding is applied only to categorical variables.
#
# Smoothing is used so that categories with very few records
# do not receive extreme target values.

categorical_columns = data.select_dtypes(
    include="object"
).columns.tolist()


global_mean = data[target_column].mean()

target_encoded_columns = []


for col in categorical_columns:

    # Calculate category statistics
    statistics = (
        data.groupby(col)[target_column]
        .agg(["mean", "count"])
    )

    # Smoothing parameter
    smoothing = 10

    statistics["smoothed_mean"] = (
        (
            statistics["count"]
            * statistics["mean"]
        )
        +
        (
            smoothing
            * global_mean
        )
    ) / (
        statistics["count"]
        + smoothing
    )

    encoded_column = (
        "Target_" + col
    )

    data[encoded_column] = (
        data[col].map(
            statistics["smoothed_mean"]
        )
    )

    target_encoded_columns.append(
        encoded_column
    )


# ============================================================
# 18. REMOVE ORIGINAL CATEGORICAL COLUMNS
# ============================================================

data = data.drop(
    columns=categorical_columns
)


# ============================================================
# 19. FINAL MISSING VALUE CHECK
# ============================================================

# Any numerical values created during preprocessing
# are filled using their median.

for col in data.columns:

    if col != target_column:

        if data[col].isnull().sum() > 0:

            data[col] = data[col].fillna(
                data[col].median()
            )


print("\n============================================================")
print("MISSING VALUES AFTER PREPROCESSING")
print("============================================================")

missing_after = data.isnull().sum()

print(
    missing_after[
        missing_after > 0
    ]
)


# ============================================================
# 20. CHECK INFINITE VALUES
# ============================================================

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)


for col in data.columns:

    if data[col].isnull().sum() > 0:

        data[col] = data[col].fillna(
            data[col].median()
        )


# ============================================================
# 21. ENSURE TARGET IS INTEGER
# ============================================================

data[target_column] = (
    data[target_column]
    .astype(int)
)


# ============================================================
# 22. DISPLAY FINAL INFORMATION
# ============================================================

print("\n============================================================")
print("FINAL PREPROCESSED DATASET")
print("============================================================")


print("\nFinal Dataset Shape:")

print(data.shape)


print("\nFinal Dataset Columns:")

for i, col in enumerate(
    data.columns,
    start=1
):

    print(
        i,
        "->",
        col
    )


print("\nFirst 5 Records:")

print(
    data.head()
)


print("\nFinal Target Distribution:")

print(
    data[target_column]
    .value_counts()
)


# ============================================================
# 23. FINAL DATA QUALITY CHECK
# ============================================================

print("\n============================================================")
print("FINAL DATA QUALITY CHECK")
print("============================================================")


print(
    "Total Rows:",
    len(data)
)


print(
    "Total Columns:",
    len(data.columns)
)


print(
    "Duplicate Rows:",
    data.duplicated().sum()
)


print(
    "Missing Values:",
    data.isnull().sum().sum()
)


print(
    "Infinite Values:",
    np.isinf(
        data.select_dtypes(
            include=np.number
        )
    ).sum().sum()
)


# ============================================================
# 24. SAVE PREPROCESSED DATASET
# ============================================================

data.to_csv(
    output_file,
    index=False
)


# ============================================================
# 25. COMPLETION MESSAGE
# ============================================================

print("\n============================================================")
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("============================================================")

print("\nOriginal Dataset:")
print("NOT MODIFIED")

print("\nProcessed Dataset Saved At:")

print(output_file)

print("\n============================================================")