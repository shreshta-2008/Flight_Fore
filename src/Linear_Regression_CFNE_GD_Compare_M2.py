# ============================================================
# LINEAR REGRESSION
# Closed-Form Normal Equation vs Gradient Descent
#
# FlightFore - Flight Delay Dataset
#
# Images are stored in ONE separate folder
# Original dataset is NOT modified
# ============================================================


import os
import numpy as np
import pandas as pd
import matplotlib


# ============================================================
# USE NON-GUI BACKEND
# Prevents Tkinter / init.tcl errors in PyCharm
# ============================================================

matplotlib.use("Agg")

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD DATASET
# ============================================================


DATASET = (
    "C:/Users/hp/PycharmProjects/FlightFore/"
    "datasets/final_flight_preprocess_M2.csv"
)


data = pd.read_csv(
    DATASET
)


print("\n========== DATASET ==========")


print(
    data.head()
)


print(
    "\nColumns:"
)


print(
    data.columns.tolist()
)


print(
    "\nDataset shape:"
)


print(
    data.shape
)


# ============================================================
# 2. CREATE IMAGE OUTPUT FOLDER
# ============================================================


IMAGE_FOLDER = (
    "C:/Users/hp/PycharmProjects/FlightFore/"
    "outputs/Linear_Regression_CFNE_GD_Compare_M2"
)


os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)


print(
    "\nImage output folder:"
)


print(
    IMAGE_FOLDER
)


# ============================================================
# 3. TARGET
# ============================================================
#
# Target:
#
# delay = 0 -> Flight is not delayed
# delay = 1 -> Flight is delayed
#
# Linear Regression treats this binary target as a numerical
# value and predicts a continuous value.
#
# Example:
#
# 0.20 -> likely not delayed
# 0.80 -> likely delayed
#
# ============================================================


TARGET = "delay"


if TARGET not in data.columns:

    raise ValueError(
        f"Target column '{TARGET}' "
        "was not found in the dataset."
    )


# Remove rows where target is missing

data = data.dropna(
    subset=[TARGET]
).copy()


# Make target numeric

data[TARGET] = pd.to_numeric(
    data[TARGET],
    errors="coerce"
)


data = data.dropna(
    subset=[TARGET]
).copy()


# ============================================================
# 4. SELECT FEATURES
# ============================================================
#
# We use only features that are known BEFORE departure.
#
# We do NOT use:
#
# DEPARTURE_TIME
# DEPARTURE_DELAY
# TAXI_OUT
# WHEELS_OFF
# ELAPSED_TIME
# AIR_TIME
# WHEELS_ON
# TAXI_IN
# ARRIVAL_TIME
# ARRIVAL_DELAY
# DIVERTED
# CANCELLED
# CANCELLATION_REASON
# AIR_SYSTEM_DELAY
# SECURITY_DELAY
# AIRLINE_DELAY
# LATE_AIRCRAFT_DELAY
# WEATHER_DELAY
#
# These can cause data leakage.
#
# YEAR is also excluded because it is constant (2015).
# ============================================================


FEATURES = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "FLIGHT_NUMBER",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_TIME",
    "DISTANCE",
    "SCHEDULED_ARRIVAL"
]


print(
    "\n========== SELECTED FEATURES =========="
)


for feature in FEATURES:

    print(
        "-",
        feature
    )


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================


missing_features = [
    feature
    for feature in FEATURES
    if feature not in data.columns
]


if len(missing_features) > 0:

    raise ValueError(
        "The following required columns are missing:\n"
        + str(missing_features)
    )


# ============================================================
# 6. EXTRACT X AND y
# ============================================================


X = data[
    FEATURES
].copy()


y = data[
    TARGET
].copy()


# ============================================================
# 7. CONVERT FEATURES TO NUMERIC
# ============================================================


for feature in FEATURES:

    X[feature] = pd.to_numeric(
        X[feature],
        errors="coerce"
    )


# ============================================================
# 8. HANDLE MISSING VALUES
# ============================================================


print(
    "\nMissing values before cleaning:"
)


print(
    X.isnull().sum()
)


for feature in FEATURES:

    median_value = X[
        feature
    ].median()


    X[feature] = X[
        feature
    ].fillna(
        median_value
    )


print(
    "\nMissing values after cleaning:"
)


print(
    X.isnull().sum().sum()
)


# ============================================================
# 9. CONVERT TO NUMPY ARRAYS
# ============================================================


X = X.values.astype(
    float
)


y = y.values.astype(
    float
)


print(
    "\nX shape:",
    X.shape
)


print(
    "y shape:",
    y.shape
)


# ============================================================
# 10. REMOVE CONSTANT FEATURES
# ============================================================
#
# A constant feature has exactly the same value in every row.
#
# Such a feature provides no information to the model.
#
# It can also contribute to a singular X^T X matrix.
# ============================================================


variances = np.var(
    X,
    axis=0
)


non_constant_indices = np.where(
    variances > 1e-12
)[0]


removed_features = [
    FEATURES[i]
    for i in range(
        len(FEATURES)
    )
    if i not in non_constant_indices
]


if len(removed_features) > 0:

    print(
        "\nConstant features removed:"
    )

    for feature in removed_features:

        print(
            "-",
            feature
        )


X = X[
    :,
    non_constant_indices
]


ACTIVE_FEATURES = [
    FEATURES[i]
    for i in non_constant_indices
]


print(
    "\nActive features:"
)


for feature in ACTIVE_FEATURES:

    print(
        "-",
        feature
    )


# ============================================================
# 11. TRAIN-TEST SPLIT
# ============================================================


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print(
    "\n========== TRAIN / TEST SPLIT =========="
)


print(
    "Training samples:",
    X_train.shape[0]
)


print(
    "Testing samples:",
    X_test.shape[0]
)


# ============================================================
# 12. FEATURE SCALING
# ============================================================
#
# StandardScaler:
#
# x_scaled = (x - mean) / standard deviation
#
# Scaling is especially important for Gradient Descent.
#
# We use the SAME scaled features for both models so that
# the comparison is fair.
# ============================================================


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 13. CLOSED FORM NORMAL EQUATION
# ============================================================
#
# Linear Regression equation:
#
# y = X theta
#
# Normal Equation:
#
# theta = (X^T X)^(-1) X^T y
#
# However, X^T X can be singular.
#
# Therefore we use:
#
# theta = pinv(X^T X) X^T y
#
# np.linalg.pinv() calculates the Moore-Penrose
# pseudo-inverse and avoids the Singular Matrix error.
# ============================================================


print(
    "\n========== CLOSED FORM =========="
)


# Add bias column

X_train_bias = np.c_[
    np.ones(
        (
            X_train_scaled.shape[0],
            1
        )
    ),
    X_train_scaled
]


X_test_bias = np.c_[
    np.ones(
        (
            X_test_scaled.shape[0],
            1
        )
    ),
    X_test_scaled
]


# ------------------------------------------------------------
# Calculate X^T X
# ------------------------------------------------------------


XTX = (
    X_train_bias.T
    @
    X_train_bias
)


print(
    "X^T X shape:",
    XTX.shape
)


# ------------------------------------------------------------
# Check matrix rank
# ------------------------------------------------------------


matrix_rank = np.linalg.matrix_rank(
    XTX
)


matrix_size = XTX.shape[0]


print(
    "Matrix rank:",
    matrix_rank
)


print(
    "Matrix size:",
    matrix_size
)


if matrix_rank < matrix_size:

    print(
        "Warning: X^T X is singular."
    )

    print(
        "Using pseudo-inverse."
    )

else:

    print(
        "X^T X is full rank."
    )


# ------------------------------------------------------------
# Normal Equation using pseudo-inverse
# ------------------------------------------------------------


theta = (
    np.linalg.pinv(
        XTX
    )
    @
    X_train_bias.T
    @
    y_train
)


# ============================================================
# 14. CLOSED FORM PREDICTION
# ============================================================


pred_normal = (
    X_test_bias
    @
    theta
)


# ============================================================
# 15. CLOSED FORM METRICS
# ============================================================


mse_normal = mean_squared_error(
    y_test,
    pred_normal
)


rmse_normal = np.sqrt(
    mse_normal
)


r2_normal = r2_score(
    y_test,
    pred_normal
)


print(
    "\n------ Closed Form Normal Equation ------"
)


print(
    "\nCoefficients:"
)


print(
    theta
)


print(
    "\nMSE:",
    mse_normal
)


print(
    "RMSE:",
    rmse_normal
)


print(
    "R2 Score:",
    r2_normal
)


# ============================================================
# 16. GRADIENT DESCENT
# ============================================================


print(
    "\n========== GRADIENT DESCENT =========="
)


# X already contains scaled features.

X_train_gd = np.c_[
    np.ones(
        (
            X_train_scaled.shape[0],
            1
        )
    ),
    X_train_scaled
]


X_test_gd = np.c_[
    np.ones(
        (
            X_test_scaled.shape[0],
            1
        )
    ),
    X_test_scaled
]


# Number of training samples

m = len(
    y_train
)


# Initialize theta

theta_gd = np.zeros(
    X_train_gd.shape[1]
)


# Learning rate

learning_rate = 0.01


# Number of iterations

epochs = 1000


print(
    "Learning rate:",
    learning_rate
)


print(
    "Epochs:",
    epochs
)


# ============================================================
# 17. STORE LOSS FOR EACH EPOCH
# ============================================================


loss_history = []


# ============================================================
# 18. GRADIENT DESCENT ITERATIONS
# ============================================================


for epoch in range(
    epochs
):


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = (
        X_train_gd
        @
        theta_gd
    )


    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    errors = (
        predictions
        -
        y_train
    )


    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    gradients = (
        2 / m
    ) * (
        X_train_gd.T
        @
        errors
    )


    # --------------------------------------------------------
    # Update parameters
    # --------------------------------------------------------

    theta_gd -= (
        learning_rate
        *
        gradients
    )


    # --------------------------------------------------------
    # Calculate MSE
    # --------------------------------------------------------

    current_predictions = (
        X_train_gd
        @
        theta_gd
    )


    loss = np.mean(
        (
            current_predictions
            -
            y_train
        ) ** 2
    )


    loss_history.append(
        loss
    )


    # --------------------------------------------------------
    # Display progress
    # --------------------------------------------------------

    if (
        epoch + 1
    ) % 100 == 0:

        print(
            f"Epoch {epoch + 1}/{epochs}"
            f" - MSE: {loss:.8f}"
        )


# ============================================================
# 19. GRADIENT DESCENT PREDICTION
# ============================================================


pred_gd = (
    X_test_gd
    @
    theta_gd
)


# ============================================================
# 20. GRADIENT DESCENT METRICS
# ============================================================


mse_gd = mean_squared_error(
    y_test,
    pred_gd
)


rmse_gd = np.sqrt(
    mse_gd
)


r2_gd = r2_score(
    y_test,
    pred_gd
)


print(
    "\n------ Gradient Descent ------"
)


print(
    "\nCoefficients:"
)


print(
    theta_gd
)


print(
    "\nMSE:",
    mse_gd
)


print(
    "RMSE:",
    rmse_gd
)


print(
    "R2 Score:",
    r2_gd
)


# ============================================================
# 21. COMPARISON
# ============================================================


print(
    "\n=========== Comparison ==========="
)


print(
    "\nNormal Equation"
)


print(
    "MSE =",
    mse_normal
)


print(
    "RMSE =",
    rmse_normal
)


print(
    "R2 =",
    r2_normal
)


print(
    "\nGradient Descent"
)


print(
    "MSE =",
    mse_gd
)


print(
    "RMSE =",
    rmse_gd
)


print(
    "R2 =",
    r2_gd
)


# ============================================================
# 22. CREATE COMPARISON TABLE
# ============================================================


comparison = pd.DataFrame({

    "Method": [
        "Closed Form Normal Equation",
        "Gradient Descent"
    ],

    "MSE": [
        mse_normal,
        mse_gd
    ],

    "RMSE": [
        rmse_normal,
        rmse_gd
    ],

    "R2_Score": [
        r2_normal,
        r2_gd
    ]

})


print(
    "\n========== MODEL COMPARISON TABLE =========="
)


print(
    comparison
)


comparison.to_csv(
    os.path.join(
        IMAGE_FOLDER,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# IMAGE 1
# ACTUAL VS PREDICTED VALUES
# ============================================================


plt.figure(
    figsize=(8, 6)
)


plt.scatter(
    y_test,
    pred_normal,
    alpha=0.5,
    label="Normal Equation"
)


plt.scatter(
    y_test,
    pred_gd,
    alpha=0.5,
    label="Gradient Descent"
)


# ------------------------------------------------------------
# Perfect prediction line
# ------------------------------------------------------------


minimum = min(
    y_test.min(),
    pred_normal.min(),
    pred_gd.min()
)


maximum = max(
    y_test.max(),
    pred_normal.max(),
    pred_gd.max()
)


plt.plot(
    [
        minimum,
        maximum
    ],
    [
        minimum,
        maximum
    ],
    linestyle="--",
    label="Perfect Prediction"
)


plt.xlabel(
    "Actual Delay"
)


plt.ylabel(
    "Predicted Delay"
)


plt.title(
    "Actual vs Predicted Values"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


image1 = os.path.join(
    IMAGE_FOLDER,
    "actual_vs_predicted.png"
)


plt.savefig(
    image1,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nImage saved:"
)


print(
    image1
)


# ============================================================
# IMAGE 2
# RESIDUAL COMPARISON
# ============================================================
#
# Residual:
#
# Residual = Actual - Predicted
#
# A good regression model should have residuals
# distributed around zero.
# ============================================================


normal_residuals = (
    y_test
    -
    pred_normal
)


gd_residuals = (
    y_test
    -
    pred_gd
)


plt.figure(
    figsize=(9, 6)
)


plt.scatter(
    pred_normal,
    normal_residuals,
    alpha=0.5,
    label="Normal Equation"
)


plt.scatter(
    pred_gd,
    gd_residuals,
    alpha=0.5,
    label="Gradient Descent"
)


plt.axhline(
    y=0,
    linestyle="--"
)


plt.xlabel(
    "Predicted Values"
)


plt.ylabel(
    "Residuals"
)


plt.title(
    "Residual Comparison"
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


image2 = os.path.join(
    IMAGE_FOLDER,
    "residual_comparison.png"
)


plt.savefig(
    image2,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Image saved:"
)


print(
    image2
)


# ============================================================
# IMAGE 3
# GRADIENT DESCENT LOSS CURVE
# ============================================================


plt.figure(
    figsize=(9, 6)
)


plt.plot(
    range(
        1,
        epochs + 1
    ),
    loss_history
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Mean Squared Error"
)


plt.title(
    "Gradient Descent Convergence"
)


plt.grid(
    True
)


plt.tight_layout()


image3 = os.path.join(
    IMAGE_FOLDER,
    "gradient_descent_loss.png"
)


plt.savefig(
    image3,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Image saved:"
)


print(
    image3
)


# ============================================================
# 23. SAVE PREDICTION RESULTS
# ============================================================


prediction_results = pd.DataFrame({

    "Actual": y_test,

    "Normal_Equation_Predicted":
        pred_normal,

    "Gradient_Descent_Predicted":
        pred_gd,

    "Normal_Equation_Residual":
        normal_residuals,

    "Gradient_Descent_Residual":
        gd_residuals

})


prediction_results.to_csv(
    os.path.join(
        IMAGE_FOLDER,
        "prediction_results.csv"
    ),
    index=False
)


# ============================================================
# 24. SAVE MODEL PARAMETERS
# ============================================================


parameter_results = pd.DataFrame({

    "Parameter": [
        "Bias"
    ] + ACTIVE_FEATURES,

    "Closed_Form": theta,

    "Gradient_Descent": theta_gd

})


parameter_results.to_csv(
    os.path.join(
        IMAGE_FOLDER,
        "model_parameters.csv"
    ),
    index=False
)


# ============================================================
# 25. SAVE IMAGE INFORMATION
# ============================================================


image_info = pd.DataFrame({

    "Image": [

        "actual_vs_predicted.png",

        "residual_comparison.png",

        "gradient_descent_loss.png"

    ],

    "Description": [

        "Actual values versus predictions from both methods",

        "Residual comparison between Normal Equation "
        "and Gradient Descent",

        "MSE loss across Gradient Descent epochs"

    ]

})


image_info.to_csv(
    os.path.join(
        IMAGE_FOLDER,
        "image_information.csv"
    ),
    index=False
)


# ============================================================
# 26. SAVE FEATURES USED
# ============================================================


features_info = pd.DataFrame({

    "Feature": ACTIVE_FEATURES

})


features_info.to_csv(
    os.path.join(
        IMAGE_FOLDER,
        "features_used.csv"
    ),
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================


print(
    "\n=========================================="
)


print(
    "PROCESS COMPLETED SUCCESSFULLY"
)


print(
    "=========================================="
)


print(
    "\nAll images are stored in ONE folder:"
)


print(
    IMAGE_FOLDER
)


print(
    "\nGenerated images:"
)


print(
    "1. actual_vs_predicted.png"
)


print(
    "2. residual_comparison.png"
)


print(
    "3. gradient_descent_loss.png"
)


print(
    "\nGenerated CSV files:"
)


print(
    "4. model_comparison.csv"
)


print(
    "5. prediction_results.csv"
)


print(
    "6. model_parameters.csv"
)


print(
    "7. image_information.csv"
)


print(
    "8. features_used.csv"
)


print(
    "\nOriginal dataset was NOT modified."
)


print(
    "\n=========================================="
)