# ================================================================
# RANDOM FOREST CLASSIFIER
# FLIGHT DELAY PREDICTION USING FLIGHTS DATASET
# ================================================================
#
# Dataset:
# flights_5000_each_month_with_class_labels (1).csv
#
# Target column:
# delay
#
# IMPORTANT:
# 1. Original CSV dataset is NEVER modified.
# 2. Preprocessing is performed inside a pipeline.
# 3. Matplotlib uses Agg backend to avoid Tkinter errors.
# 4. Categorical columns are automatically one-hot encoded.
# 5. Only PRE-DEPARTURE features are used for prediction.
# 6. Post-departure information is removed to avoid data leakage.
# 7. Class imbalance is handled using class_weight="balanced".
#
# OUTPUTS:
#   1. Accuracy
#   2. Precision
#   3. Recall
#   4. F1 Score
#   5. Confusion Matrix
#   6. Performance Graph
#   7. Actual vs Predicted Chart
#   8. Class Distribution Chart
#   9. Feature Importance Chart
#  10. Random Forest Tree Visualization
#  11. Classification Report
#  12. Prediction CSV
#  13. Random Forest Model
#  14. Model Parameters
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import os
import pickle

import numpy as np
import pandas as pd

# IMPORTANT:
# Use non-GUI backend before importing pyplot.
# This prevents Tkinter-related errors in PyCharm.

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import plot_tree

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ================================================================
# 2. DATASET PATH
# ================================================================

DATASET_PATH = (
    "C:/Users/hp/PycharmProjects/FlightFore/"
    "datasets/flights_5000_each_month_with_class_labels (1).csv"
)


# ================================================================
# 3. OUTPUT MAIN FOLDER
# ================================================================

OUTPUT_FOLDER = (
    "C:/Users/hp/PycharmProjects/FlightFore/"
    "outputs/Random_Forest_Flights_Delay_Outputs"
)


# ================================================================
# 4. TARGET COLUMN
# ================================================================

TARGET_COLUMN = "delay"


# ================================================================
# 5. OUTPUT SUBFOLDERS
# ================================================================

METRICS_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "metrics"
)

PREDICTIONS_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "predictions"
)

CONFUSION_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "confusion_matrix"
)

CHARTS_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "charts"
)

TREE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "random_forest_tree"
)

FEATURE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "feature_importance"
)

MODEL_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "model"
)


# ================================================================
# 6. CREATE OUTPUT FOLDERS
# ================================================================

folders = [
    METRICS_FOLDER,
    PREDICTIONS_FOLDER,
    CONFUSION_FOLDER,
    CHARTS_FOLDER,
    TREE_FOLDER,
    FEATURE_FOLDER,
    MODEL_FOLDER
]

for folder in folders:

    os.makedirs(
        folder,
        exist_ok=True
    )


# ================================================================
# 7. PROGRAM HEADER
# ================================================================

print("\n")

print("=" * 80)

print(
    "             RANDOM FOREST FLIGHT DELAY PREDICTION"
)

print("=" * 80)


# ================================================================
# 8. CHECK DATASET
# ================================================================

if not os.path.exists(DATASET_PATH):

    print("\nERROR: Dataset not found.")

    print("\nCheck dataset path:")
    print(DATASET_PATH)

    raise SystemExit


print("\nDataset found successfully.")

print("Dataset path:")

print(DATASET_PATH)


# ================================================================
# 9. LOAD RAW DATASET
# ================================================================

df = pd.read_csv(
    DATASET_PATH
)

print("\nDataset loaded successfully.")

print(
    "Number of rows    :",
    df.shape[0]
)

print(
    "Number of columns :",
    df.shape[1]
)


# ================================================================
# 10. CREATE COPY
# ================================================================

# The original dataframe remains untouched.

data = df.copy(
    deep=True
)


# ================================================================
# 11. DISPLAY DATASET INFORMATION
# ================================================================

print("\nDataset columns:")

print(
    list(data.columns)
)

print("\nFirst five records:")

print(
    data.head()
)


# ================================================================
# 12. IDENTIFY TARGET COLUMN
# ================================================================

if TARGET_COLUMN not in data.columns:

    print(
        "\nERROR: Target column was not found."
    )

    print("\nExpected target column:")

    print(
        TARGET_COLUMN
    )

    print("\nAvailable columns:")

    print(
        list(data.columns)
    )

    raise SystemExit


print(
    "\nTarget column:",
    TARGET_COLUMN
)


# ================================================================
# 13. TARGET DISTRIBUTION
# ================================================================

print("\nTarget class distribution:")

print(
    data[TARGET_COLUMN].value_counts(
        dropna=False
    )
)


# ================================================================
# 14. REMOVE MISSING TARGET ROWS
# ================================================================

data_model = data.dropna(
    subset=[TARGET_COLUMN]
).copy()


print(
    "\nRecords used for modeling:",
    len(data_model)
)


# ================================================================
# 15. SEPARATE X AND y
# ================================================================

X = data_model.drop(
    columns=[TARGET_COLUMN]
).copy()

y = data_model[TARGET_COLUMN].copy()


# ================================================================
# 16. CLEAN / CONVERT TARGET
# ================================================================

print("\nTarget values before conversion:")

print(
    y.unique()
)


if y.dtype == "object" or str(y.dtype) == "string":

    y = (
        y.astype(str)
        .str.strip()
        .str.lower()
    )

    target_mapping = {

        "yes": 1,

        "no": 0,

        "placed": 1,

        "not placed": 0,

        "true": 1,

        "false": 0,

        "delayed": 1,

        "not delayed": 0
    }

    y = y.map(
        target_mapping
    )


elif y.dtype == bool:

    y = y.astype(int)


else:

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    unique_values = sorted(
        y.dropna().unique()
    )

    if set(unique_values) != {0, 1}:

        if len(unique_values) == 2:

            mapping = {
                unique_values[0]: 0,
                unique_values[1]: 1
            }

            y = y.map(
                mapping
            )

        else:

            raise ValueError(
                "The target column must contain exactly two classes."
            )


if y.isnull().any():

    raise ValueError(
        "Target values could not be converted to 0 and 1."
    )


y = y.astype(int)


print(
    "\nTarget distribution after conversion:"
)

print(
    y.value_counts()
)


# ================================================================
# 17. REMOVE COMPLETELY EMPTY COLUMNS
# ================================================================

empty_columns = X.columns[
    X.isnull().all()
].tolist()


if len(empty_columns) > 0:

    print(
        "\nCompletely empty columns found:"
    )

    print(
        empty_columns
    )

    X = X.drop(
        columns=empty_columns
    )


# ================================================================
# 18. PRE-DEPARTURE FEATURE SELECTION
# ================================================================
#
# IMPORTANT:
#
# The purpose of FlightFore is to predict flight delay.
#
# Therefore, we should not use information that becomes available
# only after the flight has already started or completed.
#
# Examples of leakage:
#
# DEPARTURE_DELAY
# ARRIVAL_DELAY
# TAXI_OUT
# WHEELS_OFF
# AIR_TIME
# WHEELS_ON
# TAXI_IN
# ELAPSED_TIME
# ARRIVAL_TIME
# AIR_SYSTEM_DELAY
# SECURITY_DELAY
# AIRLINE_DELAY
# LATE_AIRCRAFT_DELAY
# WEATHER_DELAY
#
# These are removed.
#
# ================================================================


PRE_DEPARTURE_FEATURES = [

    "YEAR",

    "MONTH",

    "DAY",

    "DAY_OF_WEEK",

    "AIRLINE",

    "FLIGHT_NUMBER",

    "TAIL_NUMBER",

    "ORIGIN_AIRPORT",

    "DESTINATION_AIRPORT",

    "SCHEDULED_DEPARTURE",

    "SCHEDULED_TIME",

    "DISTANCE"
]


# ================================================================
# 19. CHECK AVAILABLE FEATURES
# ================================================================

available_features = [
    column
    for column in PRE_DEPARTURE_FEATURES
    if column in X.columns
]


missing_features = [
    column
    for column in PRE_DEPARTURE_FEATURES
    if column not in X.columns
]


print("\nPre-departure features used:")

print(
    available_features
)


if len(missing_features) > 0:

    print(
        "\nFeatures not found in dataset:"
    )

    print(
        missing_features
    )


# Keep only available pre-departure features.

X = X[
    available_features
].copy()


# ================================================================
# 20. DISPLAY FINAL FEATURES
# ================================================================

print("\nFinal features used for Random Forest:")

print(
    list(X.columns)
)


# ================================================================
# 21. IDENTIFY NUMERIC FEATURES
# ================================================================

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()


# ================================================================
# 22. IDENTIFY CATEGORICAL FEATURES
# ================================================================
#
# IMPORTANT:
# "str" is explicitly included to avoid the Pandas 4 warning:
#
# Pandas4Warning:
# Explicitly pass 'str' to include...
#
# ================================================================

categorical_features = X.select_dtypes(
    include=[
        "str",
        "category",
        "bool"
    ]
).columns.tolist()


print("\nNumeric features:")

print(
    numeric_features
)


print("\nCategorical features:")

print(
    categorical_features
)


# ================================================================
# 23. NUMERIC PREPROCESSING
# ================================================================

numeric_transformer = Pipeline(
    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


# ================================================================
# 24. CATEGORICAL PREPROCESSING
# ================================================================

categorical_transformer = Pipeline(
    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",

            OneHotEncoder(
                handle_unknown="ignore",

                sparse_output=True
            )
        )
    ]
)


# ================================================================
# 25. COLUMN TRANSFORMER
# ================================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",

            numeric_transformer,

            numeric_features
        ),

        (
            "categorical",

            categorical_transformer,

            categorical_features
        )
    ],

    remainder="drop",

    sparse_threshold=0.3
)


# ================================================================
# 26. TRAIN TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:")

print(
    len(X_train)
)


print("\nTesting samples:")

print(
    len(X_test)
)


# ================================================================
# 27. DISPLAY TRAINING CLASS DISTRIBUTION
# ================================================================

print("\nTraining class distribution:")

print(
    y_train.value_counts()
)


print("\nTesting class distribution:")

print(
    y_test.value_counts()
)


# ================================================================
# 28. CREATE RANDOM FOREST CLASSIFIER
# ================================================================

random_forest = RandomForestClassifier(

    n_estimators=100,

    criterion="gini",

    max_depth=10,

    min_samples_split=10,

    min_samples_leaf=5,

    max_features="sqrt",

    bootstrap=True,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)


# ================================================================
# 29. CREATE COMPLETE PIPELINE
# ================================================================

model = Pipeline(

    steps=[

        (
            "preprocessor",

            preprocessor
        ),

        (
            "classifier",

            random_forest
        )
    ]
)


# ================================================================
# 30. TRAIN RANDOM FOREST
# ================================================================

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)

print(
    "Random Forest training completed."
)


# ================================================================
# 31. GENERATE PREDICTIONS
# ================================================================

print(
    "\nGenerating predictions..."
)

y_pred = model.predict(
    X_test
)

print(
    "Prediction completed."
)


# ================================================================
# 32. CALCULATE PERFORMANCE METRICS
# ================================================================
#
# average="binary" means:
#
# Positive class = 1
# 1 = delayed
#
# This gives a meaningful delay-detection F1 score.
#
# ================================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)


recall = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)


f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)


# ================================================================
# 33. DISPLAY PERFORMANCE
# ================================================================

print("\n")

print("=" * 80)

print(
    "                RANDOM FOREST PERFORMANCE"
)

print("=" * 80)


print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


print("\nPercentage:")

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


# ================================================================
# 34. SAVE METRICS
# ================================================================

metrics_df = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score"
    ],

    "Score": [

        accuracy,

        precision,

        recall,

        f1
    ],

    "Percentage": [

        accuracy * 100,

        precision * 100,

        recall * 100,

        f1 * 100
    ]
})


metrics_path = os.path.join(

    METRICS_FOLDER,

    "random_forest_metrics.csv"
)


metrics_df.to_csv(

    metrics_path,

    index=False
)


# ================================================================
# 35. CLASSIFICATION REPORT
# ================================================================

classification_report_result = classification_report(

    y_test,

    y_pred,

    output_dict=True,

    zero_division=0
)


classification_report_df = pd.DataFrame(

    classification_report_result
).transpose()


classification_report_path = os.path.join(

    METRICS_FOLDER,

    "classification_report.csv"
)


classification_report_df.to_csv(

    classification_report_path
)


# ================================================================
# 36. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(

    y_test,

    y_pred
)


random_forest_classifier = (

    model.named_steps["classifier"]
)


class_labels = (

    random_forest_classifier.classes_
)


print("\n")

print("=" * 80)

print(
    "                    CONFUSION MATRIX"
)

print("=" * 80)


print(cm)


# ================================================================
# 37. SAVE CONFUSION MATRIX CSV
# ================================================================

cm_df = pd.DataFrame(

    cm,

    index=[

        "Actual_" + str(label)

        for label in class_labels
    ],

    columns=[

        "Predicted_" + str(label)

        for label in class_labels
    ]
)


cm_csv_path = os.path.join(

    CONFUSION_FOLDER,

    "confusion_matrix.csv"
)


cm_df.to_csv(

    cm_csv_path
)


# ================================================================
# 38. CONFUSION MATRIX GRAPH
# ================================================================

plt.figure(
    figsize=(8, 6)
)


plt.imshow(
    cm
)


plt.title(
    "Random Forest - Flight Delay Confusion Matrix"
)


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "Actual Label"
)


plt.xticks(

    range(len(class_labels)),

    class_labels
)


plt.yticks(

    range(len(class_labels)),

    class_labels
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


# ================================================================
# 39. PERFORMANCE GRAPH
# ================================================================

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


plt.figure(

    figsize=(10, 6)
)


bars = plt.bar(

    metric_names,

    metric_values
)


plt.title(

    "Random Forest Performance"
)


plt.xlabel(

    "Metrics"
)


plt.ylabel(

    "Score (%)"
)


plt.ylim(

    0,

    100
)


for bar, value in zip(

    bars,

    metric_values
):

    plt.text(

        bar.get_x() + bar.get_width() / 2,

        value + 1,

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


# ================================================================
# 40. ACTUAL VS PREDICTED CHART
# ================================================================

actual_counts = y_test.value_counts()

predicted_counts = pd.Series(
    y_pred
).value_counts()


comparison_df = pd.DataFrame({

    "Actual": actual_counts,

    "Predicted": predicted_counts

}).fillna(0)


comparison_df = comparison_df.reindex(

    class_labels
)


plt.figure(

    figsize=(9, 6)
)


x = np.arange(

    len(class_labels)
)


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


plt.xlabel(

    "Flight Delay Class"
)


plt.ylabel(

    "Number of Flights"
)


plt.title(

    "Actual vs Predicted Flight Delay"
)


plt.xticks(

    x,

    class_labels
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


# ================================================================
# 41. CLASS DISTRIBUTION CHART
# ================================================================

class_counts = (

    y.value_counts()
    .sort_index()
)


plt.figure(

    figsize=(8, 6)
)


plt.bar(

    class_counts.index.astype(str),

    class_counts.values
)


plt.xlabel(

    "Flight Delay Class"
)


plt.ylabel(

    "Number of Flights"
)


plt.title(

    "Flight Delay Class Distribution"
)


plt.xticks(

    rotation=45
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


# ================================================================
# 42. GET FEATURE NAMES
# ================================================================

feature_names = (

    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


# ================================================================
# 43. FEATURE IMPORTANCE
# ================================================================

feature_importances = (

    random_forest_classifier
    .feature_importances_
)


feature_importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": feature_importances
})


feature_importance_df = (

    feature_importance_df
    .sort_values(

        by="Importance",

        ascending=False
    )
)


# ================================================================
# 44. SAVE FEATURE IMPORTANCE CSV
# ================================================================

feature_importance_path = os.path.join(

    FEATURE_FOLDER,

    "feature_importance.csv"
)


feature_importance_df.to_csv(

    feature_importance_path,

    index=False
)


# ================================================================
# 45. FEATURE IMPORTANCE CHART
# ================================================================

top_features = (

    feature_importance_df
    .head(15)
    .sort_values(

        by="Importance"
    )
)


plt.figure(

    figsize=(10, 7)
)


plt.barh(

    top_features["Feature"],

    top_features["Importance"]
)


plt.xlabel(

    "Importance"
)


plt.ylabel(

    "Feature"
)


plt.title(

    "Top 15 Random Forest Feature Importances"
)


plt.tight_layout()


feature_chart_path = os.path.join(

    FEATURE_FOLDER,

    "feature_importance.png"
)


plt.savefig(

    feature_chart_path,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ================================================================
# 46. RANDOM FOREST TREE VISUALIZATION
# ================================================================

print("\n")

print("=" * 80)

print(
    "             RANDOM FOREST TREE VISUALIZATION"
)

print("=" * 80)


print(
    "\nGenerating Random Forest Tree 1..."
)


# Select first trained tree.

first_tree = (

    random_forest_classifier
    .estimators_[0]
)


plt.figure(

    figsize=(30, 18)
)


plot_tree(

    first_tree,

    feature_names=feature_names,

    class_names=[

        str(label)

        for label in class_labels
    ],

    filled=True,

    rounded=True,

    proportion=False,

    precision=2,

    fontsize=7
)


plt.title(

    "Random Forest - Tree 1",

    fontsize=20
)


plt.tight_layout()


# ================================================================
# 47. SAVE RANDOM FOREST TREE
# ================================================================

tree_image_path = os.path.join(

    TREE_FOLDER,

    "random_forest_tree_1.png"
)


plt.savefig(

    tree_image_path,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print(
    "\nRandom Forest Tree 1 generated successfully."
)


print(
    "Tree image saved at:"
)


print(
    tree_image_path
)


# ================================================================
# 48. SAVE TEST PREDICTIONS
# ================================================================

test_predictions = X_test.copy()


test_predictions["Actual"] = (

    y_test.values
)


test_predictions["Predicted"] = (

    y_pred
)


prediction_path = os.path.join(

    PREDICTIONS_FOLDER,

    "test_predictions.csv"
)


test_predictions.to_csv(

    prediction_path,

    index=False
)


# ================================================================
# 49. SAVE TRAINED MODEL
# ================================================================

model_path = os.path.join(

    MODEL_FOLDER,

    "random_forest_flight_delay_model.pkl"
)


with open(

    model_path,

    "wb"

) as file:

    pickle.dump(

        model,

        file
    )


# ================================================================
# 50. SAVE RANDOM FOREST PARAMETERS
# ================================================================

parameters_df = pd.DataFrame({

    "Parameter": [

        "Algorithm",

        "Number of Trees",

        "Criterion",

        "Maximum Depth",

        "Minimum Samples Split",

        "Minimum Samples Leaf",

        "Maximum Features",

        "Bootstrap",

        "Class Weight",

        "Random State"
    ],

    "Value": [

        "Random Forest Classifier",

        random_forest.n_estimators,

        random_forest.criterion,

        random_forest.max_depth,

        random_forest.min_samples_split,

        random_forest.min_samples_leaf,

        random_forest.max_features,

        random_forest.bootstrap,

        random_forest.class_weight,

        random_forest.random_state
    ]
})


parameters_path = os.path.join(

    METRICS_FOLDER,

    "random_forest_parameters.csv"
)


parameters_df.to_csv(

    parameters_path,

    index=False
)


# ================================================================
# 51. SAVE FEATURE LIST
# ================================================================

feature_list_df = pd.DataFrame({

    "Feature": available_features

})


feature_list_path = os.path.join(

    METRICS_FOLDER,

    "features_used.csv"
)


feature_list_df.to_csv(

    feature_list_path,

    index=False
)


# ================================================================
# 52. FINAL RESULT
# ================================================================

print("\n")

print("=" * 80)

print(
    "          RANDOM FOREST FLIGHT DELAY COMPLETED"
)

print("=" * 80)


print(
    "\nOriginal RAW dataset was NOT modified."
)


print(
    "\nDataset:"
)


print(
    DATASET_PATH
)


print(
    "\nAll outputs stored in:"
)


print(
    OUTPUT_FOLDER
)


print("\n")


print(
    "FINAL PERFORMANCE"
)


print("-" * 50)


print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)


print(
    f"Precision : {precision * 100:.2f}%"
)


print(
    f"Recall    : {recall * 100:.2f}%"
)


print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


print("\n")


print(
    "OUTPUT FOLDERS"
)


print("-" * 50)


print(
    "1. metrics"
)


print(
    "2. predictions"
)


print(
    "3. confusion_matrix"
)


print(
    "4. charts"
)


print(
    "5. random_forest_tree"
)


print(
    "6. feature_importance"
)


print(
    "7. model"
)


print("\n")


print("=" * 80)

print(
    "                PROGRAM FINISHED"
)

print("=" * 80)