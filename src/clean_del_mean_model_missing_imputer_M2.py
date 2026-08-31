import pandas as pd
import numpy as np


# ============================================================
# FLIGHT DELAY PREDICTION DATASET
# MISSING VALUE HANDLING USING PANDAS
#
# Techniques:
# 1. Deletion
# 2. Mean Imputation
# 3. Median Imputation
# 4. Model-Based Imputation
# 5. Missing Indicator Features
#
# Original dataset is NOT modified.
# All results are stored in ONE CSV file.
# ============================================================


# ------------------------------------------------------------
# 1. READ DATASET
# ------------------------------------------------------------

input_file = (
    "C:/Users/srinu/PycharmProjects/FlightFore/datasets/"
    "flights_5000_each_month_with_class_labels (1).csv"
)

output_file = (
    "C:/Users/srinu/PycharmProjects/FlightFore/datasets/"
    "flight_missing_values_processed.csv"
)


df = pd.read_csv(input_file)


print("=" * 70)
print("ORIGINAL FLIGHT DELAY PREDICTION DATASET")
print("=" * 70)

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ------------------------------------------------------------
# 2. CLEAN COLUMN NAMES
# ------------------------------------------------------------

df.columns = df.columns.str.strip()


print("\nDataset Columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 3. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES IN ORIGINAL DATASET")
print("=" * 70)

print(df.isnull().sum())


# ------------------------------------------------------------
# 4. CREATE ORIGINAL COPY
# ------------------------------------------------------------

# The original dataset is never modified.

df_original = df.copy()


# ============================================================
# METHOD 1: DELETION
# ============================================================

# Delete rows containing any missing value.

df_deletion = df_original.dropna().copy()


print("\n" + "=" * 70)
print("1. DELETION")
print("=" * 70)

print("Original rows:", len(df_original))
print("Rows after deletion:", len(df_deletion))
print(
    "Rows deleted:",
    len(df_original) - len(df_deletion)
)


# ------------------------------------------------------------
# Create deletion indicator
# ------------------------------------------------------------

# 1 = row contains missing value
# 0 = row contains no missing value

deletion_indicator = (
    df_original.isnull()
    .any(axis=1)
).astype(int)


# ============================================================
# IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ============================================================

numeric_columns = (
    df_original
    .select_dtypes(include="number")
    .columns
    .tolist()
)

categorical_columns = (
    df_original
    .select_dtypes(exclude="number")
    .columns
    .tolist()
)


print("\n" + "=" * 70)
print("COLUMN INFORMATION")
print("=" * 70)

print("\nNumerical Columns:")
print(numeric_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# ============================================================
# METHOD 2: MEAN IMPUTATION
# ============================================================

df_mean = df_original.copy()


for column in numeric_columns:

    if df_mean[column].isnull().any():

        mean_value = df_mean[column].mean()

        df_mean[column] = (
            df_mean[column]
            .fillna(mean_value)
        )

        print(
            "\nMean used for",
            column,
            "=",
            mean_value
        )


# ------------------------------------------------------------
# Handle categorical missing values using mode
# ------------------------------------------------------------

for column in categorical_columns:

    if df_mean[column].isnull().any():

        mode_values = df_mean[column].mode()

        if len(mode_values) > 0:

            df_mean[column] = (
                df_mean[column]
                .fillna(mode_values.iloc[0])
            )


print("\n" + "=" * 70)
print("2. MEAN IMPUTATION")
print("=" * 70)

print(df_mean.head())


# ============================================================
# METHOD 3: MEDIAN IMPUTATION
# ============================================================

df_median = df_original.copy()


for column in numeric_columns:

    if df_median[column].isnull().any():

        median_value = (
            df_median[column].median()
        )

        df_median[column] = (
            df_median[column]
            .fillna(median_value)
        )

        print(
            "\nMedian used for",
            column,
            "=",
            median_value
        )


# ------------------------------------------------------------
# Handle categorical missing values using mode
# ------------------------------------------------------------

for column in categorical_columns:

    if df_median[column].isnull().any():

        mode_values = df_median[column].mode()

        if len(mode_values) > 0:

            df_median[column] = (
                df_median[column]
                .fillna(mode_values.iloc[0])
            )


print("\n" + "=" * 70)
print("3. MEDIAN IMPUTATION")
print("=" * 70)

print(df_median.head())


# ============================================================
# METHOD 4: MODEL-BASED IMPUTATION
# ============================================================

# A simple linear regression model is calculated using
# Pandas operations.
#
# For every numerical column containing missing values,
# another numerical column is automatically selected
# as the predictor.


df_model = df_original.copy()


print("\n" + "=" * 70)
print("4. MODEL-BASED IMPUTATION")
print("=" * 70)


if len(numeric_columns) >= 2:

    for target_column in numeric_columns:

        # Check whether target contains missing values

        if not df_model[target_column].isnull().any():

            continue


        # ----------------------------------------------------
        # Select predictor column
        # ----------------------------------------------------

        predictor_column = None


        for column in numeric_columns:

            if column != target_column:

                if (
                    df_model[column]
                    .notnull()
                    .sum()
                    >= 2
                ):

                    predictor_column = column

                    break


        # If predictor is unavailable

        if predictor_column is None:

            continue


        # ----------------------------------------------------
        # Training data
        # ----------------------------------------------------

        training_data = df_model[
            df_model[target_column].notnull()
            &
            df_model[predictor_column].notnull()
        ].copy()


        if len(training_data) < 2:

            continue


        x = training_data[
            predictor_column
        ]

        y = training_data[
            target_column
        ]


        # ----------------------------------------------------
        # Calculate linear regression
        #
        # y = slope*x + intercept
        # ----------------------------------------------------

        x_mean = x.mean()
        y_mean = y.mean()


        numerator = (
            (x - x_mean)
            *
            (y - y_mean)
        ).sum()


        denominator = (
            (x - x_mean) ** 2
        ).sum()


        if denominator == 0:

            continue


        slope = (
            numerator
            /
            denominator
        )


        intercept = (
            y_mean
            -
            slope * x_mean
        )


        # ----------------------------------------------------
        # Find missing target rows
        # ----------------------------------------------------

        missing_rows = df_model[
            df_model[target_column].isnull()
            &
            df_model[predictor_column].notnull()
        ].copy()


        # ----------------------------------------------------
        # Predict missing values
        # ----------------------------------------------------

        if len(missing_rows) > 0:

            predicted_values = (
                slope
                *
                missing_rows[predictor_column]
                +
                intercept
            )


            df_model.loc[
                missing_rows.index,
                target_column
            ] = predicted_values


            print(
                "\nTarget column:",
                target_column
            )

            print(
                "Predictor column:",
                predictor_column
            )

            print(
                "Slope:",
                slope
            )

            print(
                "Intercept:",
                intercept
            )

            print(
                "Number of values predicted:",
                len(missing_rows)
            )


# ------------------------------------------------------------
# Median fallback
# ------------------------------------------------------------

# If model-based prediction could not fill some values,
# median is used as a fallback.

for column in numeric_columns:

    if df_model[column].isnull().any():

        median_value = (
            df_model[column].median()
        )

        df_model[column] = (
            df_model[column]
            .fillna(median_value)
        )


# ------------------------------------------------------------
# Handle categorical columns
# ------------------------------------------------------------

for column in categorical_columns:

    if df_model[column].isnull().any():

        mode_values = (
            df_model[column].mode()
        )

        if len(mode_values) > 0:

            df_model[column] = (
                df_model[column]
                .fillna(mode_values.iloc[0])
            )


print("\nModel-Based Imputation Result:")
print(df_model.head())


# ============================================================
# METHOD 5: MISSING INDICATOR FEATURES
# ============================================================

print("\n" + "=" * 70)
print("5. MISSING INDICATOR FEATURES")
print("=" * 70)


df_indicator = df_original.copy()


# Create an indicator for every original column

for column in df_original.columns:

    indicator_column = (
        column + "_Missing"
    )

    df_indicator[indicator_column] = (
        df_original[column]
        .isnull()
        .astype(int)
    )


print("\nMissing Indicator Result:")
print(df_indicator.head())


# ============================================================
# CREATE ONE FINAL OUTPUT DATAFRAME
# ============================================================

final_result = df_original.copy()


# ------------------------------------------------------------
# A. Deletion indicator
# ------------------------------------------------------------

final_result[
    "Deletion_Row_Removed"
] = deletion_indicator


# ------------------------------------------------------------
# B. Mean-imputed values
# ------------------------------------------------------------

for column in numeric_columns:

    final_result[
        column + "_Mean_Imputed"
    ] = df_mean[column]


# ------------------------------------------------------------
# C. Median-imputed values
# ------------------------------------------------------------

for column in numeric_columns:

    final_result[
        column + "_Median_Imputed"
    ] = df_median[column]


# ------------------------------------------------------------
# D. Model-based imputed values
# ------------------------------------------------------------

for column in numeric_columns:

    final_result[
        column + "_Model_Imputed"
    ] = df_model[column]


# ------------------------------------------------------------
# E. Missing indicator features
# ------------------------------------------------------------

for column in df_original.columns:

    final_result[
        column + "_Missing"
    ] = (
        df_original[column]
        .isnull()
        .astype(int)
    )


# ============================================================
# SAVE ALL RESULTS INTO ONE CSV FILE
# ============================================================

final_result.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL OUTPUT")
print("=" * 70)

print(final_result.head())

print("\nFinal Dataset Shape:")
print(final_result.shape)


# ============================================================
# CHECK REMAINING MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("REMAINING MISSING VALUES")
print("=" * 70)


print("\nMean-Imputed Dataset:")
print(
    df_mean.isnull().sum()
)


print("\nMedian-Imputed Dataset:")
print(
    df_median.isnull().sum()
)


print("\nModel-Imputed Dataset:")
print(
    df_model.isnull().sum()
)


# ============================================================
# VERIFY ORIGINAL DATASET WAS NOT MODIFIED
# ============================================================

df_check = pd.read_csv(input_file)


if df_original.equals(df_check):

    print(
        "\nOriginal flight dataset was NOT modified."
    )

else:

    print(
        "\nWARNING: Original flight dataset was modified!"
    )


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FLIGHT DATA MISSING-VALUE PROCESSING COMPLETED")
print("=" * 70)

print("\nInput file:")
print(input_file)

print("\nOutput file:")
print(output_file)

print(
    "\nAll five missing-value handling techniques "
    "are stored in ONE CSV file."
)

print("=" * 70)