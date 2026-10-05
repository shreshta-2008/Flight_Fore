# ============================================================
# LINEAR REGRESSION - FLIGHT DELAY PREDICTION DATASET
# ============================================================

import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. FILE PATHS
# ============================================================

DATASET_PATH = "C:/Users/hp/PycharmProjects/FlightFore/datasets/final_flight_preprocess_M2.csv"

OUTPUT_FOLDER = (
    "C:/Users/hp/PycharmProjects/FlightFore/"
    "outputs/Linear_Regression_with_Metrics_M2"
)

IMAGE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "images"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )


df = pd.read_csv(DATASET_PATH)


print("=" * 60)
print("LINEAR REGRESSION - FLIGHT DELAY PREDICTION")
print("=" * 60)


print("\nDataset Shape:")
print(df.shape)


print("\nFirst 5 Records:")
print(df.head())


# ============================================================
# 3. DISPLAY COLUMN NAMES
# ============================================================

print("\nDataset Columns:")

for column in df.columns:
    print(column)


# ============================================================
# 4. SELECT FEATURES AND TARGET
# ============================================================

# Multiple Linear Regression:
#
# y = b0 + b1x1 + b2x2 + ... + bnxn
#
# FlightFore features used for prediction BEFORE the flight:
#
# MONTH
# DAY
# DAY_OF_WEEK
# FLIGHT_NUMBER
# SCHEDULED_DEPARTURE
# SCHEDULED_TIME
# DISTANCE
# SCHEDULED_ARRIVAL
#
# Target:
# delay
#
# We are NOT using:
#
# AIRLINE
# ORIGIN_AIRPORT
# DESTINATION_AIRPORT
#
# because these are categorical columns in this dataset.
#
# We also do not use:
#
# DIVERTED
# CANCELLED
#
# because these can represent events occurring during/after
# the flight and are not suitable as clean pre-departure
# prediction features.


feature_columns = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "FLIGHT_NUMBER",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_TIME",
    "DISTANCE",
    "SCHEDULED_ARRIVAL"
]


# y = Flight Delay
target_column = "delay"


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    feature_columns +
    [target_column]
)


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    print("\nERROR!")

    print(
        "The following columns were not found:"
    )

    print(
        missing_columns
    )

    print(
        "\nAvailable columns are:"
    )

    print(
        list(df.columns)
    )

    raise ValueError(
        "Please check feature_columns and "
        "target_column according to your dataset."
    )


print("\nSelected Features:")

for column in feature_columns:
    print("-", column)


print("\nTarget:")
print("-", target_column)


# ============================================================
# 6. CREATE MODEL DATA
# ============================================================

model_df = df[
    required_columns
].copy()


# ============================================================
# 7. CONVERT DATA TO NUMERIC
# ============================================================

# Make sure all selected features are numeric.

for column in feature_columns:

    model_df[column] = pd.to_numeric(
        model_df[column],
        errors="coerce"
    )


model_df[target_column] = pd.to_numeric(
    model_df[target_column],
    errors="coerce"
)


# ============================================================
# 8. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values before cleaning:")

print(
    model_df.isnull().sum()
)


# Replace missing feature values with
# the median of that feature.

for column in feature_columns:

    median_value = model_df[column].median()

    model_df[column] = model_df[column].fillna(
        median_value
    )


# Remove rows where target is missing.

model_df = model_df.dropna(
    subset=[target_column]
)


print("\nMissing values after cleaning:")

print(
    model_df.isnull().sum()
)


# ============================================================
# 9. DEFINE X AND Y
# ============================================================

X = model_df[
    feature_columns
]

y = model_df[
    target_column
]


print("\nFinal X shape:")
print(X.shape)


print("\nFinal y shape:")
print(y.shape)


# ============================================================
# 10. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)


print(
    "\nTraining samples:",
    len(X_train)
)


print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# 11. CREATE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 12. TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


print(
    "\nModel training completed."
)


# ============================================================
# 13. MODEL COEFFICIENTS
# ============================================================

# Linear Regression equation:
#
# y = b0 +
#     b1*x1 +
#     b2*x2 +
#     ...
#     bn*xn


print("\n" + "=" * 60)

print(
    "MODEL COEFFICIENTS"
)

print("=" * 60)


print(
    "\nIntercept is b0:"
)

print(
    model.intercept_
)


print(
    "\nCoefficients values:"
)


coefficient_df = pd.DataFrame({

    "Feature": feature_columns,

    "Coefficient": model.coef_

})


print(
    coefficient_df
)


# ============================================================
# 14. LINEAR REGRESSION EQUATION
# ============================================================

equation = (
    f"{target_column} = "
    f"{model.intercept_:.4f}"
)


for feature, coefficient in zip(
    feature_columns,
    model.coef_
):

    equation += (
        f" + ({coefficient:.4f} × {feature})"
    )


print(
    "\nLinear Regression Equation:"
)

print(
    equation
)


# ============================================================
# 15. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 16. EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)


mse = mean_squared_error(
    y_test,
    y_pred
)


rmse = np.sqrt(
    mse
)


r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)

print(
    "MODEL EVALUATION"
)

print("=" * 60)


print(
    f"MAE  : {mae:.4f}"
)


print(
    f"MSE  : {mse:.4f}"
)


print(
    f"RMSE : {rmse:.4f}"
)


print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# 17. CREATE PREDICTION RESULTS
# ============================================================

results = X_test.copy()


results[
    "Actual"
] = y_test.values


results[
    "Predicted"
] = y_pred


results[
    "Residual"
] = (
    results["Actual"]
    -
    results["Predicted"]
)


results[
    "Absolute_Error"
] = abs(
    results["Residual"]
)


# ============================================================
# 18. SAVE PREDICTION RESULTS
# ============================================================

prediction_file = os.path.join(

    OUTPUT_FOLDER,

    "linear_regression_predictions.csv"

)


results.to_csv(

    prediction_file,

    index=False

)


print(
    "\nPrediction results saved to:"
)

print(
    prediction_file
)


# ============================================================
# 19. SAVE MODEL COEFFICIENTS
# ============================================================

coefficient_file = os.path.join(

    OUTPUT_FOLDER,

    "linear_regression_coefficients.csv"

)


coefficient_df.to_csv(

    coefficient_file,

    index=False

)


print(
    "\nCoefficient results saved to:"
)

print(
    coefficient_file
)


# ============================================================
# 20. SAVE MODEL METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "Metric": [

        "MAE",

        "MSE",

        "RMSE",

        "R2"

    ],

    "Value": [

        mae,

        mse,

        rmse,

        r2

    ]

})


metrics_file = os.path.join(

    OUTPUT_FOLDER,

    "linear_regression_metrics.csv"

)


metrics_df.to_csv(

    metrics_file,

    index=False

)


print(
    "\nMetrics saved to:"
)

print(
    metrics_file
)


# ============================================================
# 21. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(

    y_test,

    y_pred,

    alpha=0.6

)


# Perfect prediction line

minimum = min(

    y_test.min(),

    y_pred.min()

)


maximum = max(

    y_test.max(),

    y_pred.max()

)


plt.plot(

    [minimum, maximum],

    [minimum, maximum],

    linestyle="--"

)


plt.xlabel(
    "Actual Delay"
)


plt.ylabel(
    "Predicted Delay"
)


plt.title(
    "Linear Regression: Actual vs Predicted Flight Delay"
)


plt.grid(
    True
)


actual_predicted_image = os.path.join(

    IMAGE_FOLDER,

    "actual_vs_predicted.png"

)


plt.savefig(

    actual_predicted_image,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 22. RESIDUAL GRAPH
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(

    y_pred,

    results["Residual"],

    alpha=0.6

)


plt.axhline(

    y=0,

    linestyle="--"

)


plt.xlabel(
    "Predicted Delay"
)


plt.ylabel(
    "Residual"
)


plt.title(
    "Residual Plot - Linear Regression"
)


plt.grid(
    True
)


residual_image = os.path.join(

    IMAGE_FOLDER,

    "residual_plot.png"

)


plt.savefig(

    residual_image,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 23. COEFFICIENT GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.bar(

    coefficient_df["Feature"],

    coefficient_df["Coefficient"]

)


plt.xlabel(
    "Features"
)


plt.ylabel(
    "Coefficient"
)


plt.title(
    "Linear Regression Feature Coefficients"
)


plt.xticks(

    rotation=30,

    ha="right"

)


plt.grid(
    axis="y"
)


coefficient_image = os.path.join(

    IMAGE_FOLDER,

    "feature_coefficients.png"

)


plt.savefig(

    coefficient_image,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 24. SAVE EQUATION
# ============================================================

equation_file = os.path.join(

    OUTPUT_FOLDER,

    "linear_regression_equation.txt"

)


with open(

    equation_file,

    "w",

    encoding="utf-8"

) as file:

    file.write(
        "Linear Regression Equation\n"
    )

    file.write(
        "=" * 40 + "\n"
    )

    file.write(
        equation
    )


# ============================================================
# 25. SAVE IMAGE INFORMATION
# ============================================================

image_information = pd.DataFrame({

    "Image": [

        "actual_vs_predicted.png",

        "residual_plot.png",

        "feature_coefficients.png"

    ],

    "Description": [

        "Comparison between actual and predicted flight delay",

        "Residual plot of the Linear Regression model",

        "Linear Regression coefficients for each feature"

    ]

})


image_information_file = os.path.join(

    OUTPUT_FOLDER,

    "image_information.csv"

)


image_information.to_csv(

    image_information_file,

    index=False

)


# ============================================================
# 26. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)

print(
    "PROCESS COMPLETED SUCCESSFULLY"
)

print("=" * 60)


print(
    "\nOutput Folder:"
)

print(
    OUTPUT_FOLDER
)


print(
    "\nGenerated Files:"
)


print(
    "- linear_regression_predictions.csv"
)


print(
    "- linear_regression_coefficients.csv"
)


print(
    "- linear_regression_metrics.csv"
)


print(
    "- linear_regression_equation.txt"
)


print(
    "- image_information.csv"
)


print(
    "\nGenerated Images:"
)


print(
    "- actual_vs_predicted.png"
)


print(
    "- residual_plot.png"
)


print(
    "- feature_coefficients.png"
)


print(
    "\nAll files have been saved successfully."
)