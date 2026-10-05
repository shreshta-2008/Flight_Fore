import os
import numpy as np
import pandas as pd
import matplotlib

# Use non-GUI backend so PyCharm does not give Tkinter errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET = "C:/Users/hp/PycharmProjects/FlightFore/datasets/final_flight_preprocess_M2.csv"

OUTPUT_FOLDER = (
    "C:/Users/hp/PycharmProjects/FlightFore/"
    "outputs/Logistic_Regression_Binary_Classify_M2"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD FLIGHT DATASET
# ============================================================

df = pd.read_csv(DATASET)

print("\n========== DATASET ==========")

print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset shape:")
print(df.shape)


# ============================================================
# 3. CHECK TARGET
# ============================================================

TARGET = "delay"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found in the dataset."
    )


# Remove rows where target is missing
df = df.dropna(subset=[TARGET]).copy()


# Make sure target is integer
df[TARGET] = df[TARGET].astype(int)


print("\nTarget:", TARGET)

print("\nTarget distribution:")
print(df[TARGET].value_counts().sort_index())


# ============================================================
# 4. SELECT PRE-DEPARTURE FEATURES
# ============================================================
#
# We are predicting whether a flight will be delayed.
#
# Therefore, we should only use information available
# BEFORE the flight departs.
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
# These are post-departure or leakage-related variables.
# ============================================================


FEATURES = [
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "FLIGHT_NUMBER",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_TIME",
    "DISTANCE",
    "SCHEDULED_ARRIVAL"
]


# Check that all required columns exist
missing_features = [
    feature for feature in FEATURES
    if feature not in df.columns
]

if len(missing_features) > 0:

    raise ValueError(
        "The following required feature columns are missing:\n"
        + str(missing_features)
    )


print("\nFeatures:")
for feature in FEATURES:
    print("-", feature)

print("\nNumber of features:", len(FEATURES))


# ============================================================
# 5. SELECT X AND y
# ============================================================

X = df[FEATURES].copy()

y = df[TARGET].values


# Convert all selected features to numeric
for feature in FEATURES:

    X[feature] = pd.to_numeric(
        X[feature],
        errors="coerce"
    )


# Replace missing numeric values with median
for feature in FEATURES:

    median_value = X[feature].median()

    X[feature] = X[feature].fillna(
        median_value
    )


X = X.values


# Make sure target is integer
y = y.astype(int)


print("\nX shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)


# ============================================================
# 6. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


print("\n========== DATA SPLIT ==========")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 7. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


print("\nStandardization completed.")


# ============================================================
# 8. SIGMOID FUNCTION
# ============================================================

def sigmoid(z):

    """
    Sigmoid function:

                  1
    sigmoid(z) = -------
                1 + e^(-z)

    Converts the linear score into a probability.
    """

    # Prevent overflow in exponential calculation
    z = np.clip(
        z,
        -500,
        500
    )

    return 1 / (
        1 + np.exp(-z)
    )


# ============================================================
# 9. SIGMOID GRAPH
# ============================================================

z_values = np.linspace(
    -10,
    10,
    500
)

sigmoid_values = sigmoid(
    z_values
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    z_values,
    sigmoid_values,
    linewidth=3
)


plt.axhline(
    0.5,
    linestyle="--",
    label="Threshold = 0.5"
)


plt.axvline(
    0,
    linestyle="--"
)


plt.xlabel(
    "z"
)

plt.ylabel(
    "Sigmoid(z)"
)


plt.title(
    "Sigmoid Function"
)


plt.legend()

plt.grid(
    alpha=0.3
)


plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_sigmoid.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 10. CROSS-ENTROPY LOSS
# ============================================================

def cross_entropy_loss(
    y_true,
    y_probability
):

    """
    Binary Cross-Entropy Loss:

             -1/m Σ [
                 y log(p)
                 +
                 (1-y) log(1-p)
             ]

    Lower loss means better prediction.
    """

    epsilon = 1e-15


    y_probability = np.clip(
        y_probability,
        epsilon,
        1 - epsilon
    )


    loss = -np.mean(
        y_true * np.log(
            y_probability
        )
        +
        (1 - y_true)
        *
        np.log(
            1 - y_probability
        )
    )


    return loss


# ============================================================
# 11. INITIALIZE LOGISTIC REGRESSION
# ============================================================

number_of_features = X_train_scaled.shape[1]


weights = np.zeros(
    number_of_features
)


bias = 0.0


learning_rate = 0.05


epochs = 3000


loss_history = []


print("\n========== LOGISTIC REGRESSION ==========")

print(
    "Number of features:",
    number_of_features
)

print(
    "Learning rate:",
    learning_rate
)

print(
    "Epochs:",
    epochs
)


# ============================================================
# 12. TRAIN USING GRADIENT DESCENT
# ============================================================

m = len(y_train)


print("\nTraining started...")


for epoch in range(epochs):

    # --------------------------------------------------------
    # Linear model
    #
    # z = w1*x1 + w2*x2 + ... + wn*xn + b
    # --------------------------------------------------------

    z = np.dot(
        X_train_scaled,
        weights
    ) + bias


    # --------------------------------------------------------
    # Sigmoid
    # --------------------------------------------------------

    probability = sigmoid(
        z
    )


    # --------------------------------------------------------
    # Cross-entropy loss
    # --------------------------------------------------------

    loss = cross_entropy_loss(
        y_train,
        probability
    )


    loss_history.append(
        loss
    )


    # --------------------------------------------------------
    # Calculate error
    # --------------------------------------------------------

    error = (
        probability - y_train
    )


    # --------------------------------------------------------
    # Gradient for weights
    # --------------------------------------------------------

    dw = (
        1 / m
    ) * np.dot(
        X_train_scaled.T,
        error
    )


    # --------------------------------------------------------
    # Gradient for bias
    # --------------------------------------------------------

    db = (
        1 / m
    ) * np.sum(
        error
    )


    # --------------------------------------------------------
    # Update weights
    # --------------------------------------------------------

    weights -= (
        learning_rate * dw
    )


    # --------------------------------------------------------
    # Update bias
    # --------------------------------------------------------

    bias -= (
        learning_rate * db
    )


    # Print progress
    if (epoch + 1) % 500 == 0:

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {loss:.6f}"
        )


print("\nTraining completed.")


# ============================================================
# 13. DISPLAY MODEL PARAMETERS
# ============================================================

print(
    "\n========== MODEL PARAMETERS =========="
)


for feature, weight in zip(
    FEATURES,
    weights
):

    print(
        f"{feature}: {weight:.6f}"
    )


print(
    f"Bias: {bias:.6f}"
)


# ============================================================
# 14. CROSS-ENTROPY LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.plot(
    range(1, epochs + 1),
    loss_history,
    linewidth=2
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Cross-Entropy Loss"
)


plt.title(
    "Logistic Regression - Cross-Entropy Loss"
)


plt.grid(
    alpha=0.3
)


plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "02_cross_entropy_loss.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 15. PREDICTION FUNCTIONS
# ============================================================

def predict_probability(X):

    """
    Calculate probability that the flight
    belongs to delay = 1.
    """

    z = np.dot(
        X,
        weights
    ) + bias


    return sigmoid(
        z
    )


def predict(
    X,
    threshold=0.5
):

    """
    Convert probability into binary class.

    probability >= 0.5 -> Delayed
    probability < 0.5  -> Not Delayed
    """

    probability = predict_probability(
        X
    )


    return (
        probability >= threshold
    ).astype(int)


# ============================================================
# 16. TEST SET PREDICTION
# ============================================================

test_probability = predict_probability(
    X_test_scaled
)


y_pred = predict(
    X_test_scaled
)


# ============================================================
# 17. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "\n========== MODEL PERFORMANCE =========="
)


print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


print(
    "\nConfusion Matrix:"
)


print(cm)


plt.figure(
    figsize=(7, 6)
)


plt.imshow(
    cm,
    cmap="Blues"
)


plt.colorbar()


plt.xticks(
    [0, 1],
    [
        "Not Delayed",
        "Delayed"
    ]
)


plt.yticks(
    [0, 1],
    [
        "Not Delayed",
        "Delayed"
    ]
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.title(
    "Flight Delay - Confusion Matrix"
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=16
        )


plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "03_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 19. CALCULATE ADDITIONAL METRICS
# ============================================================

TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]


if (TP + FP) != 0:

    precision = TP / (
        TP + FP
    )

else:

    precision = 0


if (TP + FN) != 0:

    recall = TP / (
        TP + FN
    )

else:

    recall = 0


if (precision + recall) != 0:

    f1_score = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )

else:

    f1_score = 0


print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall: {recall * 100:.2f}%"
)

print(
    f"F1 Score: {f1_score * 100:.2f}%"
)


# ============================================================
# 20. DECISION BOUNDARY AS HYPERPLANE
# ============================================================
#
# Logistic regression equation:
#
# z = w1*x1 + w2*x2 + ... + wn*xn + b
#
# Decision boundary occurs when:
#
# probability = 0.5
#
# sigmoid(z) = 0.5
#
# Therefore:
#
# z = 0
#
# Hence:
#
# w1*x1 + w2*x2 + ... + wn*xn + b = 0
#
# This is a HYPERPLANE.
#
# Our model has multiple features, so we visualize
# a 2D cross-section.
#
# We use:
#
# MONTH
# DISTANCE
#
# and keep the remaining features fixed at their
# average values.
# ============================================================


BOUNDARY_FEATURE_1 = "MONTH"

BOUNDARY_FEATURE_2 = "DISTANCE"


feature1_index = FEATURES.index(
    BOUNDARY_FEATURE_1
)

feature2_index = FEATURES.index(
    BOUNDARY_FEATURE_2
)


# ============================================================
# 21. CREATE GRID FOR DECISION BOUNDARY
# ============================================================

feature1_min = X[:, feature1_index].min()

feature1_max = X[:, feature1_index].max()


feature2_min = X[:, feature2_index].min()

feature2_max = X[:, feature2_index].max()


feature1_values = np.linspace(
    feature1_min,
    feature1_max,
    300
)


feature2_values = np.linspace(
    feature2_min,
    feature2_max,
    300
)


FEATURE1_GRID, FEATURE2_GRID = np.meshgrid(
    feature1_values,
    feature2_values
)


# ============================================================
# 22. CREATE GRID USING MEAN VALUES
# ============================================================

grid = np.zeros(
    (
        FEATURE1_GRID.size,
        len(FEATURES)
    )
)


# Set all features to their mean
for i in range(
    len(FEATURES)
):

    grid[:, i] = np.mean(
        X[:, i]
    )


# Replace selected two features
grid[:, feature1_index] = (
    FEATURE1_GRID.ravel()
)


grid[:, feature2_index] = (
    FEATURE2_GRID.ravel()
)


# ============================================================
# 23. STANDARDIZE GRID
# ============================================================

grid_scaled = scaler.transform(
    grid
)


# ============================================================
# 24. CALCULATE GRID PROBABILITY
# ============================================================

grid_probability = predict_probability(
    grid_scaled
)


grid_probability = grid_probability.reshape(
    FEATURE1_GRID.shape
)


# ============================================================
# 25. PLOT DECISION BOUNDARY
# ============================================================

plt.figure(
    figsize=(10, 7)
)


# Probability regions
contour = plt.contourf(
    FEATURE1_GRID,
    FEATURE2_GRID,
    grid_probability,
    levels=50,
    cmap="RdYlGn",
    alpha=0.35
)


plt.colorbar(
    contour,
    label="Delay Probability"
)


# Hyperplane cross-section
plt.contour(
    FEATURE1_GRID,
    FEATURE2_GRID,
    grid_probability,
    levels=[0.5],
    colors="black",
    linewidths=3
)


# ============================================================
# Plot original data
# ============================================================

not_delayed = (
    df[TARGET] == 0
)


delayed = (
    df[TARGET] == 1
)


plt.scatter(
    df.loc[
        not_delayed,
        BOUNDARY_FEATURE_1
    ],
    df.loc[
        not_delayed,
        BOUNDARY_FEATURE_2
    ],
    s=20,
    alpha=0.5,
    label="Not Delayed"
)


plt.scatter(
    df.loc[
        delayed,
        BOUNDARY_FEATURE_1
    ],
    df.loc[
        delayed,
        BOUNDARY_FEATURE_2
    ],
    s=20,
    alpha=0.5,
    label="Delayed"
)


plt.xlabel(
    BOUNDARY_FEATURE_1
)


plt.ylabel(
    BOUNDARY_FEATURE_2
)


plt.title(
    "Logistic Regression Decision Boundary\n"
    "Flight Delay Prediction"
)


plt.legend()


plt.grid(
    alpha=0.2
)


plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "04_decision_boundary_hyperplane.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# 26. PREDICT A NEW FLIGHT
# ============================================================
#
# Example flight:
#
# YEAR                 = 2015
# MONTH                = 6
# DAY                  = 15
# DAY_OF_WEEK          = 1
# FLIGHT_NUMBER        = 100
# SCHEDULED_DEPARTURE  = 1200
# SCHEDULED_TIME       = 180
# DISTANCE             = 1000
# SCHEDULED_ARRIVAL    = 1500
#
# You can change these values.
# ============================================================


new_flight = pd.DataFrame(
    [[
        2015,   # YEAR
        6,      # MONTH
        15,     # DAY
        1,      # DAY_OF_WEEK
        100,    # FLIGHT_NUMBER
        1200,   # SCHEDULED_DEPARTURE
        180,    # SCHEDULED_TIME
        1000,   # DISTANCE
        1500    # SCHEDULED_ARRIVAL
    ]],
    columns=FEATURES
)


# Convert to numpy
new_flight_array = (
    new_flight.values
)


# Standardize new flight
new_flight_scaled = scaler.transform(
    new_flight_array
)


# Calculate probability
new_probability = predict_probability(
    new_flight_scaled
)[0]


# Binary prediction
new_prediction = int(
    new_probability >= 0.5
)


# ============================================================
# 27. DISPLAY NEW FLIGHT PREDICTION
# ============================================================

print(
    "\n========== NEW FLIGHT PREDICTION =========="
)


print(
    "\nInput Flight:"
)


for feature in FEATURES:

    print(
        f"{feature}: "
        f"{new_flight.iloc[0][feature]}"
    )


print(
    "\nDelay Probability:",
    f"{new_probability * 100:.2f}%"
)


if new_prediction == 1:

    print(
        "Prediction: FLIGHT IS LIKELY TO BE DELAYED"
    )

else:

    print(
        "Prediction: FLIGHT IS LIKELY TO BE ON TIME"
    )


# ============================================================
# 28. SAVE PREDICTION RESULTS
# ============================================================

results = pd.DataFrame({

    "Actual": y_test,

    "Predicted": y_pred,

    "Delay_Probability": test_probability

})


results.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "prediction_results.csv"
    ),
    index=False
)


# ============================================================
# 29. SAVE MODEL PARAMETERS
# ============================================================

parameter_results = pd.DataFrame({

    "Feature": FEATURES,

    "Weight": weights

})


parameter_results.loc[
    len(parameter_results)
] = [
    "Bias",
    bias
]


parameter_results.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "model_parameters.csv"
    ),
    index=False
)


# ============================================================
# 30. SAVE METRICS
# ============================================================

metrics_results = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Value": [
        accuracy,
        precision,
        recall,
        f1_score
    ]

})


metrics_results.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "model_metrics.csv"
    ),
    index=False
)


# ============================================================
# 31. SAVE CONFUSION MATRIX
# ============================================================

confusion_results = pd.DataFrame(
    cm,
    index=[
        "Actual_Not_Delayed",
        "Actual_Delayed"
    ],
    columns=[
        "Predicted_Not_Delayed",
        "Predicted_Delayed"
    ]
)


confusion_results.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "confusion_matrix.csv"
    )
)


# ============================================================
# 32. SAVE LOSS HISTORY
# ============================================================

loss_results = pd.DataFrame({

    "Epoch": range(
        1,
        epochs + 1
    ),

    "Cross_Entropy_Loss": loss_history

})


loss_results.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "loss_history.csv"
    ),
    index=False
)


# ============================================================
# 33. SAVE FEATURE LIST
# ============================================================

feature_results = pd.DataFrame({

    "Feature": FEATURES

})


feature_results.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "features_used.csv"
    ),
    index=False
)


# ============================================================
# 34. FINISH
# ============================================================

print(
    "\n============================================"
)

print(
    "PROGRAM COMPLETED SUCCESSFULLY"
)

print(
    "============================================"
)


print(
    "\nOutput folder:"
)


print(
    os.path.abspath(
        OUTPUT_FOLDER
    )
)


print(
    "\nGenerated files:"
)


print(
    "1. 01_sigmoid.png"
)

print(
    "2. 02_cross_entropy_loss.png"
)

print(
    "3. 03_confusion_matrix.png"
)

print(
    "4. 04_decision_boundary_hyperplane.png"
)

print(
    "5. prediction_results.csv"
)

print(
    "6. model_parameters.csv"
)

print(
    "7. model_metrics.csv"
)

print(
    "8. confusion_matrix.csv"
)

print(
    "9. loss_history.csv"
)

print(
    "10. features_used.csv"
)

print(
    "\n============================================"
)