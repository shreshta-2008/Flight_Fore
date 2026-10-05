# =============================================================================
# LOGISTIC REGRESSION
# BINARY FLIGHT DELAY PREDICTION
#
# Dataset:
# flights_5000_each_month_with_class_labels (1).csv
#
# Target:
# delay
#   0 = Not Delayed
#   1 = Delayed
#
# IMPORTANT:
# 1. The original CSV is never modified.
# 2. The dataset contains numeric and categorical columns.
# 3. Numeric missing values are median-imputed.
# 4. Categorical missing values are filled with the most frequent value.
# 5. Categorical features are one-hot encoded.
# 6. Matplotlib uses the Agg backend to avoid Tkinter errors.
# =============================================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

# Use a non-GUI backend before importing pyplot.
# This avoids: _tkinter.TclError: can't delete Tcl command
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

warnings.filterwarnings("ignore")


# =============================================================================
# 1. FILE SETTINGS
# =============================================================================

# Change only this path if your project is stored somewhere else.
DATASET_FILE = (
    "C:/Users/hp/PycharmProjects/FlightFore/datasets/flights_5000_each_month_with_class_labels (1).csv"
)

OUTPUT_FOLDER = (
    "C:/Users/hp/PycharmProjects/FlightFore/outputs/Logistic_Regression_Flights_Delay_Outputs"
)

TARGET_COLUMN = "delay"


# =============================================================================
# 2. OUTPUT SUBFOLDERS
# =============================================================================

METRICS_FOLDER = os.path.join(OUTPUT_FOLDER, "metrics")
PREDICTIONS_FOLDER = os.path.join(OUTPUT_FOLDER, "predictions")
CONFUSION_FOLDER = os.path.join(OUTPUT_FOLDER, "confusion_matrix")
CHARTS_FOLDER = os.path.join(OUTPUT_FOLDER, "charts")
FEATURE_FOLDER = os.path.join(OUTPUT_FOLDER, "feature_coefficients")
MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "model")

folders = [
    OUTPUT_FOLDER,
    METRICS_FOLDER,
    PREDICTIONS_FOLDER,
    CONFUSION_FOLDER,
    CHARTS_FOLDER,
    FEATURE_FOLDER,
    MODEL_FOLDER
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)


# =============================================================================
# 3. PROGRAM HEADER
# =============================================================================

print("\n")
print("=" * 80)
print("                 LOGISTIC REGRESSION")
print("              FLIGHT DELAY PREDICTION")
print("=" * 80)


# =============================================================================
# 4. CHECK DATASET
# =============================================================================

if not os.path.exists(DATASET_FILE):
    print("\nERROR: Dataset not found.")
    print("\nExpected path:")
    print(DATASET_FILE)
    print(
        "\nPlace the CSV file inside the Dataset folder "
        "with the exact filename."
    )
    raise SystemExit


print("\nDataset found successfully.")
print("Dataset path:")
print(DATASET_FILE)


# =============================================================================
# 5. READ DATASET
# =============================================================================

# pd.read_csv() only reads the dataset.
# The original CSV is not changed.

df_original = pd.read_csv(DATASET_FILE)

print("\nDataset loaded successfully.")
print("Rows    :", df_original.shape[0])
print("Columns :", df_original.shape[1])


# =============================================================================
# 6. CREATE COPY
# =============================================================================

# All further operations are performed on this copy.

data = df_original.copy(deep=True)


# =============================================================================
# 7. DISPLAY DATASET INFORMATION
# =============================================================================

print("\n" + "=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

print("\nFirst 5 rows:")
print(data.head())

print("\nColumn names:")
print(data.columns.tolist())

print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())


# =============================================================================
# 8. CHECK TARGET COLUMN
# =============================================================================

if TARGET_COLUMN not in data.columns:
    raise ValueError(
        "\nTarget column '" + TARGET_COLUMN +
        "' does not exist in the dataset.\n"
        "Available columns:\n" +
        str(data.columns.tolist())
    )

print("\nTarget column:", TARGET_COLUMN)


# =============================================================================
# 9. TARGET CLASS DISTRIBUTION
# =============================================================================

print("\n" + "=" * 80)
print("TARGET CLASS DISTRIBUTION")
print("=" * 80)

class_counts = data[TARGET_COLUMN].value_counts(dropna=False).sort_index()

print(class_counts)

number_of_classes = data[TARGET_COLUMN].nunique(dropna=True)

print("\nNumber of classes:", number_of_classes)

if number_of_classes != 2:
    raise ValueError(
        "This Logistic Regression program expects exactly 2 classes "
        "in the 'delay' column."
    )


# =============================================================================
# 10. SEPARATE FEATURES AND TARGET
# =============================================================================

X = data.drop(columns=[TARGET_COLUMN]).copy()
y = data[TARGET_COLUMN].copy()

# Remove rows where the target itself is missing.
# This does not modify the original CSV.
valid_target_rows = y.notna()

X = X.loc[valid_target_rows].copy()
y = y.loc[valid_target_rows].copy()


# =============================================================================
# 11. ENSURE TARGET IS BINARY NUMERIC
# =============================================================================

print("\n" + "=" * 80)
print("TARGET VALUES")
print("=" * 80)

print("Unique target values:", sorted(y.unique().tolist()))

# The supplied dataset already uses:
# 0 = not delayed
# 1 = delayed

try:
    y = pd.to_numeric(y)
except Exception as exc:
    raise ValueError(
        "The target column 'delay' must contain numeric binary values 0 and 1."
    ) from exc

unique_target_values = sorted(y.unique().tolist())

if set(unique_target_values) != {0, 1}:
    raise ValueError(
        "The 'delay' column must contain exactly 0 and 1.\n"
        "Found: " + str(unique_target_values)
    )

y = y.astype(int)


# =============================================================================
# 12. FEATURE INFORMATION
# =============================================================================

print("\n" + "=" * 80)
print("FEATURE INFORMATION")
print("=" * 80)

print("\nNumber of original features:", X.shape[1])

numeric_columns = X.select_dtypes(
    include=[np.number]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()

print("\nNumeric features:", len(numeric_columns))
for column in numeric_columns:
    print("  -", column)

print("\nCategorical features:", len(categorical_columns))
for column in categorical_columns:
    print("  -", column)


# =============================================================================
# 13. SAVE DATASET INFORMATION
# =============================================================================

information_file = os.path.join(
    OUTPUT_FOLDER,
    "dataset_information.txt"
)

with open(information_file, "w", encoding="utf-8") as file:
    file.write("LOGISTIC REGRESSION - FLIGHT DELAY DATASET INFORMATION\n")
    file.write("=" * 80 + "\n\n")

    file.write(f"Dataset file: {DATASET_FILE}\n")
    file.write(f"Rows: {data.shape[0]}\n")
    file.write(f"Columns: {data.shape[1]}\n")
    file.write(f"Target column: {TARGET_COLUMN}\n")
    file.write(f"Number of classes: {number_of_classes}\n\n")

    file.write("Target class distribution:\n")
    file.write(str(class_counts))
    file.write("\n\n")

    file.write("Numeric features:\n")
    for column in numeric_columns:
        file.write(f"- {column}\n")

    file.write("\nCategorical features:\n")
    for column in categorical_columns:
        file.write(f"- {column}\n")

    file.write("\nData types:\n")
    file.write(str(data.dtypes))

    file.write("\n\nMissing values:\n")
    file.write(str(data.isnull().sum()))


# =============================================================================
# 14. PREPROCESSING
# =============================================================================
#
# Numeric columns:
#   Missing values -> median
#
# Categorical columns:
#   Missing values -> most frequent
#   Then one-hot encoding
#
# The original dataset is not changed.
# =============================================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ],
    remainder="drop"
)


# =============================================================================
# 15. TRAIN-TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT")
print("=" * 80)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

print(
    "\nTraining percentage:",
    round(len(X_train) / len(X) * 100, 2),
    "%"
)

print(
    "Testing percentage :",
    round(len(X_test) / len(X) * 100, 2),
    "%"
)


# =============================================================================
# 16. LOGISTIC REGRESSION MODEL
# =============================================================================

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION")
print("=" * 80)

# Because the target has two classes (0 and 1), this is binary
# Logistic Regression rather than Multinomial Logistic Regression.

classifier = LogisticRegression(
    solver="lbfgs",
    max_iter=2000,
    random_state=42
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)


# =============================================================================
# 17. TRAIN MODEL
# =============================================================================

print("\nTraining Logistic Regression model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed successfully.")


# =============================================================================
# 18. PREDICT TEST DATA
# =============================================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# =============================================================================
# 19. CALCULATE METRICS
# =============================================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="binary",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="binary",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="binary",
    zero_division=0
)

cross_entropy = log_loss(
    y_test,
    model.predict_proba(X_test),
    labels=[0, 1]
)


# =============================================================================
# 20. PRINT MODEL RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("MODEL RESULTS")
print("=" * 80)

print(f"\nAccuracy             : {accuracy:.4f}")
print(f"Accuracy (%)         : {accuracy * 100:.2f}%")
print(f"Precision            : {precision:.4f}")
print(f"Recall               : {recall:.4f}")
print(f"F1 Score             : {f1:.4f}")
print(f"Cross-Entropy Loss   : {cross_entropy:.4f}")


# =============================================================================
# 21. CLASSIFICATION REPORT
# =============================================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Not Delayed (0)",
        "Delayed (1)"
    ],
    zero_division=0
)

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)
print(report)


# =============================================================================
# 22. SAVE MODEL METRICS
# =============================================================================

metrics_file = os.path.join(
    METRICS_FOLDER,
    "model_metrics.txt"
)

with open(metrics_file, "w", encoding="utf-8") as file:
    file.write("LOGISTIC REGRESSION - FLIGHT DELAY PREDICTION\n")
    file.write("=" * 80 + "\n\n")

    file.write(f"Dataset: {DATASET_FILE}\n")
    file.write(f"Target: {TARGET_COLUMN}\n")
    file.write("Class 0: Not Delayed\n")
    file.write("Class 1: Delayed\n\n")

    file.write("MODEL PERFORMANCE\n")
    file.write("-" * 40 + "\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Accuracy (%): {accuracy * 100:.2f}%\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1 Score: {f1:.4f}\n")
    file.write(f"Cross-Entropy Loss: {cross_entropy:.4f}\n\n")

    file.write("CLASSIFICATION REPORT\n")
    file.write("-" * 40 + "\n")
    file.write(report)


# =============================================================================
# 23. CONFUSION MATRIX
# =============================================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)
print(cm)

cm_df = pd.DataFrame(
    cm,
    index=["Actual_0_Not_Delayed", "Actual_1_Delayed"],
    columns=["Predicted_0_Not_Delayed", "Predicted_1_Delayed"]
)

cm_csv_path = os.path.join(
    CONFUSION_FOLDER,
    "confusion_matrix.csv"
)

cm_df.to_csv(cm_csv_path)


# =============================================================================
# 24. CONFUSION MATRIX GRAPH
# =============================================================================

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    [0, 1],
    ["Not Delayed (0)", "Delayed (1)"],
    rotation=20
)

plt.yticks(
    [0, 1],
    ["Not Delayed (0)", "Delayed (1)"]
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()

confusion_image_path = os.path.join(
    CONFUSION_FOLDER,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_image_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# 25. PERFORMANCE GRAPH
# =============================================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

metric_values = [
    accuracy * 100,
    precision * 100,
    recall * 100,
    f1 * 100
]

plt.figure(figsize=(10, 6))

bars = plt.bar(
    metric_names,
    metric_values
)

plt.title(
    "Logistic Regression Performance"
)

plt.xlabel("Metrics")
plt.ylabel("Score (%)")
plt.ylim(0, 100)

for bar, value in zip(bars, metric_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        min(value + 1, 99),
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

performance_path = os.path.join(
    CHARTS_FOLDER,
    "performance_graph.png"
)

plt.savefig(
    performance_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# 26. ACTUAL VS PREDICTED CHART
# =============================================================================

actual_counts = y_test.value_counts().reindex(
    [0, 1],
    fill_value=0
)

predicted_counts = pd.Series(y_pred).value_counts().reindex(
    [0, 1],
    fill_value=0
)

comparison_df = pd.DataFrame({
    "Actual": actual_counts.values,
    "Predicted": predicted_counts.values
}, index=[0, 1])

plt.figure(figsize=(9, 6))

x = np.arange(2)
width = 0.35

plt.bar(
    x - width / 2,
    comparison_df["Actual"],
    width,
    label="Actual"
)

plt.bar(
    x + width / 2,
    comparison_df["Predicted"],
    width,
    label="Predicted"
)

plt.xlabel("Flight Delay Class")
plt.ylabel("Number of Flights")
plt.title("Actual vs Predicted Flight Delay")

plt.xticks(
    x,
    ["Not Delayed (0)", "Delayed (1)"]
)

plt.legend()
plt.tight_layout()

actual_predicted_path = os.path.join(
    CHARTS_FOLDER,
    "actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# 27. CLASS DISTRIBUTION CHART
# =============================================================================

class_distribution = (
    y.value_counts()
    .sort_index()
)

class_distribution_df = pd.DataFrame({
    "Class": class_distribution.index,
    "Count": class_distribution.values
})

class_distribution_df.to_csv(
    os.path.join(
        CHARTS_FOLDER,
        "class_distribution.csv"
    ),
    index=False
)

plt.figure(figsize=(8, 6))

plt.bar(
    class_distribution.index.astype(str),
    class_distribution.values
)

plt.xlabel("Flight Delay Class")
plt.ylabel("Number of Flights")
plt.title("Flight Delay Class Distribution")

plt.xticks(
    [0, 1],
    ["Not Delayed (0)", "Delayed (1)"]
)

plt.tight_layout()

class_distribution_path = os.path.join(
    CHARTS_FOLDER,
    "class_distribution.png"
)

plt.savefig(
    class_distribution_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# 28. ROC CURVE
# =============================================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC curve (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.tight_layout()

roc_path = os.path.join(
    CHARTS_FOLDER,
    "roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# 29. GET PROCESSED FEATURE NAMES
# =============================================================================

feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


# =============================================================================
# 30. SAVE MODEL COEFFICIENTS
# =============================================================================

trained_classifier = (
    model
    .named_steps["classifier"]
)

coefficients = trained_classifier.coef_[0]

coefficient_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute_Coefficient": np.abs(coefficients)
})

coefficient_df = (
    coefficient_df
    .sort_values(
        by="Absolute_Coefficient",
        ascending=False
    )
)

coefficient_path = os.path.join(
    FEATURE_FOLDER,
    "model_coefficients.csv"
)

coefficient_df.to_csv(
    coefficient_path,
    index=False
)


# =============================================================================
# 31. TOP FEATURE COEFFICIENTS GRAPH
# =============================================================================

top_coefficients = (
    coefficient_df
    .head(15)
    .sort_values(
        by="Coefficient"
    )
)

plt.figure(figsize=(11, 8))

plt.barh(
    top_coefficients["Feature"],
    top_coefficients["Coefficient"]
)

plt.xlabel("Logistic Regression Coefficient")
plt.ylabel("Feature")
plt.title("Top 15 Logistic Regression Feature Coefficients")

plt.tight_layout()

coefficient_chart_path = os.path.join(
    FEATURE_FOLDER,
    "top_feature_coefficients.png"
)

plt.savefig(
    coefficient_chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# 32. CREATE PREDICTION OUTPUT
# =============================================================================

prediction_output = X_test.copy()

prediction_output["Actual_Class"] = y_test.values

prediction_output["Predicted_Class"] = y_pred

prediction_output["Prediction_Probability_Delayed"] = y_probability

prediction_output["Correct"] = (
    y_test.values == y_pred
)

prediction_output["Actual_Label"] = (
    prediction_output["Actual_Class"]
    .map({
        0: "Not Delayed",
        1: "Delayed"
    })
)

prediction_output["Predicted_Label"] = (
    prediction_output["Predicted_Class"]
    .map({
        0: "Not Delayed",
        1: "Delayed"
    })
)

prediction_path = os.path.join(
    PREDICTIONS_FOLDER,
    "predictions.csv"
)

prediction_output.to_csv(
    prediction_path,
    index=False
)


# =============================================================================
# 33. SAVE ACTUAL VS PREDICTED
# =============================================================================

actual_predicted = pd.DataFrame({
    "Actual_Class": y_test.values,
    "Predicted_Class": y_pred,
    "Actual_Label": [
        "Not Delayed" if value == 0 else "Delayed"
        for value in y_test.values
    ],
    "Predicted_Label": [
        "Not Delayed" if value == 0 else "Delayed"
        for value in y_pred
    ],
    "Prediction_Probability_Delayed": y_probability,
    "Correct": y_test.values == y_pred
})

actual_predicted.to_csv(
    os.path.join(
        PREDICTIONS_FOLDER,
        "actual_vs_predicted.csv"
    ),
    index=False
)


# =============================================================================
# 34. SAVE CLASS DISTRIBUTION
# =============================================================================

class_distribution_df.to_csv(
    os.path.join(
        PREDICTIONS_FOLDER,
        "class_distribution.csv"
    ),
    index=False
)


# =============================================================================
# 35. SAVE ROC DATA
# =============================================================================

roc_df = pd.DataFrame({
    "False_Positive_Rate": fpr,
    "True_Positive_Rate": tpr,
    "Threshold": thresholds
})

roc_df.to_csv(
    os.path.join(
        CHARTS_FOLDER,
        "roc_curve_data.csv"
    ),
    index=False
)


# =============================================================================
# 36. SAVE MODEL
# =============================================================================

model_file = os.path.join(
    MODEL_FOLDER,
    "logistic_regression_flight_delay.pkl"
)

joblib.dump(
    model,
    model_file
)


# =============================================================================
# 37. SAVE MODEL PARAMETERS
# =============================================================================

parameters_file = os.path.join(
    MODEL_FOLDER,
    "model_parameters.txt"
)

with open(parameters_file, "w", encoding="utf-8") as file:
    file.write("LOGISTIC REGRESSION MODEL PARAMETERS\n")
    file.write("=" * 80 + "\n\n")

    file.write(f"Solver: {trained_classifier.solver}\n")
    file.write(f"Max Iterations: {trained_classifier.max_iter}\n")
    file.write(f"Random State: {trained_classifier.random_state}\n")
    file.write(f"Number of input features before encoding: {X.shape[1]}\n")
    file.write(f"Number of features after encoding: {len(feature_names)}\n")
    file.write(f"Training samples: {len(X_train)}\n")
    file.write(f"Testing samples: {len(X_test)}\n")
    file.write(f"ROC AUC: {roc_auc:.4f}\n")


# =============================================================================
# 38. FINAL MESSAGE
# =============================================================================

print("\n" + "=" * 80)
print("COMPLETED SUCCESSFULLY")
print("=" * 80)

print("\nOriginal dataset was NOT modified.")

print("\nTarget column:", TARGET_COLUMN)
print("Class 0 = Not Delayed")
print("Class 1 = Delayed")

print("\nFinal model performance:")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")

print("\nAll generated files are stored in:")
print(os.path.abspath(OUTPUT_FOLDER))

print("\nGenerated folders:")
for folder in sorted(folders):
    print("  ", folder)

print("\n" + "=" * 80)
