# ============================================================
# FLIGHTFORE - FLIGHT DELAY DATASET
# K-MEANS AND K-MEANS++ CLUSTERING
#
# Features:
#   MONTH
#   DAY
#   DAY_OF_WEEK
#   FLIGHT_NUMBER
#   SCHEDULED_DEPARTURE
#   SCHEDULED_TIME
#   DISTANCE
#   SCHEDULED_ARRIVAL
#
# Methods:
#   1. K-Means
#   2. K-Means++
#   3. Elbow Method
#   4. Silhouette Score
#   5. Calinski-Harabasz Score
#   6. Davies-Bouldin Score
#   7. PCA visualization
#   8. 3D clustering visualization
#
# IMPORTANT:
# - Uses the PREPROCESSED FlightFore dataset
# - Original dataset is NOT modified
# - "delay" is NOT used for clustering
# - delay is used only after clustering for comparison
# - Categorical columns are not used directly
# - Outputs are stored in separate folders
# ============================================================


import os
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    accuracy_score
)


# ============================================================
# 1. WARNING SETTINGS
# ============================================================

warnings.filterwarnings("ignore")


# ============================================================
# 2. INPUT FILE
# ============================================================

INPUT_FILE = (
    r"C:/Users/hp/PycharmProjects/FlightFore/"
    r"datasets/final_flight_preprocess_M2.csv"
)


# ============================================================
# 3. OUTPUT FOLDER
# ============================================================

OUTPUT_FOLDER = (
    r"C:/Users/hp/PycharmProjects/FlightFore/"
    r"outputs/K_Means_K++Means_Elbow_Silhoute_M4_Outputs"
)


# ============================================================
# 4. CREATE SEPARATE OUTPUT FOLDERS
# ============================================================

KMEANS_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "KMeans"
)

KMEANS_PP_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "KMeansPlusPlus"
)

ELBOW_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "Elbow"
)

SILHOUETTE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "Silhouette"
)

ACCURACY_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "Accuracy"
)

PCA_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "PCA"
)


for folder in [
    KMEANS_FOLDER,
    KMEANS_PP_FOLDER,
    ELBOW_FOLDER,
    SILHOUETTE_FOLDER,
    ACCURACY_FOLDER,
    PCA_FOLDER
]:

    os.makedirs(
        folder,
        exist_ok=True
    )


# ============================================================
# 5. LOAD PREPROCESSED DATASET
# ============================================================

print("=" * 75)
print("K-MEANS AND K-MEANS++")
print("FLIGHTFORE - FLIGHT DELAY CLUSTERING")
print("=" * 75)


if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        "\nPreprocessed dataset was not found.\n\n"
        "Check this path:\n"
        + INPUT_FILE
    )


data = pd.read_csv(
    INPUT_FILE
)


print(
    "\nPreprocessed dataset loaded successfully."
)

print(
    "Rows    :",
    data.shape[0]
)

print(
    "Columns :",
    data.shape[1]
)


# ============================================================
# 6. DISPLAY COLUMNS
# ============================================================

print(
    "\nAvailable columns:"
)

print(
    "-" * 75
)


for i, column in enumerate(
    data.columns,
    1
):

    print(
        i,
        ".",
        column
    )


# ============================================================
# 7. COLUMN-NAME NORMALIZATION
# ============================================================

def normalize_column_name(column):

    return (
        str(column)
        .strip()
        .lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
    )


normalized_columns = {
    normalize_column_name(column): column
    for column in data.columns
}


# ============================================================
# 8. FEATURES USED FOR CLUSTERING
# ============================================================

# These are numeric flight-related features.
#
# delay is intentionally NOT included here because K-Means
# is an unsupervised algorithm.
#
# We want the clusters to be formed from flight characteristics
# rather than from the known delay label.


required_features = [

    "MONTH",

    "DAY",

    "DAY_OF_WEEK",

    "FLIGHT_NUMBER",

    "SCHEDULED_DEPARTURE",

    "SCHEDULED_TIME",

    "DISTANCE",

    "SCHEDULED_ARRIVAL"

]


# ============================================================
# 9. FIND FEATURES
# ============================================================

feature_columns = []


for feature in required_features:

    normalized_feature = normalize_column_name(
        feature
    )

    if normalized_feature in normalized_columns:

        feature_columns.append(
            normalized_columns[
                normalized_feature
            ]
        )

    else:

        raise ValueError(
            f"\nRequired feature '{feature}' "
            f"was not found in the dataset."
            "\n\nAvailable columns:\n"
            + "\n".join(
                data.columns.astype(str)
            )
        )


# ============================================================
# 10. DISPLAY SELECTED FEATURES
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "FEATURES USED FOR CLUSTERING"
)

print(
    "=" * 75
)


for feature in feature_columns:

    print(
        "✓",
        feature
    )


print(
    "\nTotal clustering features:",
    len(feature_columns)
)


# ============================================================
# 11. SELECT CLUSTERING DATA
# ============================================================

X = data[
    feature_columns
].copy()


# ============================================================
# 12. CONVERT FEATURES TO NUMERIC
# ============================================================

for column in feature_columns:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# ============================================================
# 13. HANDLE INFINITE VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 14. CHECK MISSING VALUES
# ============================================================

print(
    "\nMissing values before cleaning:"
)

print(
    X.isnull().sum()
)


# ============================================================
# 15. HANDLE MISSING VALUES
# ============================================================

for column in feature_columns:

    median_value = X[column].median()

    X[column] = X[column].fillna(
        median_value
    )


print(
    "\nMissing values after preprocessing:"
)

print(
    X.isnull().sum()
)


# ============================================================
# 16. CHECK FOR CONSTANT FEATURES
# ============================================================

constant_features = []

for column in feature_columns:

    if X[column].nunique() <= 1:

        constant_features.append(
            column
        )


if constant_features:

    print(
        "\nConstant features found:"
    )

    print(
        constant_features
    )

    X = X.drop(
        columns=constant_features
    )

    feature_columns = [
        column
        for column in feature_columns
        if column not in constant_features
    ]


print(
    "\nFinal clustering features:"
)

for feature in feature_columns:

    print(
        "-",
        feature
    )


# ============================================================
# 17. STANDARDISATION
# ============================================================
#
# K-Means is distance-based.
#
# Therefore, all features are standardized so that a feature
# with a large numerical range does not dominate the clustering.
# ============================================================

scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X
)


print(
    "\nFeature standardisation completed."
)


# ============================================================
# 18. K RANGE
# ============================================================

K_VALUES = range(
    2,
    11
)


# ============================================================
# 19. K-MEANS - ELBOW AND SILHOUETTE
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "K-MEANS ANALYSIS"
)

print(
    "=" * 75
)


kmeans_inertia = []

kmeans_silhouette = []


for k in K_VALUES:

    model = KMeans(

        n_clusters=k,

        init="random",

        n_init=10,

        random_state=42

    )


    labels = model.fit_predict(
        X_scaled
    )


    inertia = model.inertia_


    silhouette = silhouette_score(

        X_scaled,

        labels

    )


    kmeans_inertia.append(
        inertia
    )


    kmeans_silhouette.append(
        silhouette
    )


    print(

        f"K = {k:2d} | "

        f"Inertia = {inertia:.4f} | "

        f"Silhouette = {silhouette:.4f}"

    )


# ============================================================
# 20. K-MEANS++ - ELBOW AND SILHOUETTE
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "K-MEANS++ ANALYSIS"
)

print(
    "=" * 75
)


kmeans_pp_inertia = []

kmeans_pp_silhouette = []


for k in K_VALUES:

    model = KMeans(

        n_clusters=k,

        init="k-means++",

        n_init=10,

        random_state=42

    )


    labels = model.fit_predict(
        X_scaled
    )


    inertia = model.inertia_


    silhouette = silhouette_score(

        X_scaled,

        labels

    )


    kmeans_pp_inertia.append(
        inertia
    )


    kmeans_pp_silhouette.append(
        silhouette
    )


    print(

        f"K = {k:2d} | "

        f"Inertia = {inertia:.4f} | "

        f"Silhouette = {silhouette:.4f}"

    )


# ============================================================
# 21. SELECT BEST K
# ============================================================
#
# The maximum Silhouette Score is used to automatically select
# the best K.
#
# The Elbow graphs can also be inspected visually.
# ============================================================

K_LIST = list(
    K_VALUES
)


best_k_kmeans = K_LIST[
    np.argmax(
        kmeans_silhouette
    )
]


best_k_kmeans_pp = K_LIST[
    np.argmax(
        kmeans_pp_silhouette
    )
]


print(
    "\n" + "=" * 75
)

print(
    "SELECTED NUMBER OF CLUSTERS"
)

print(
    "=" * 75
)


print(
    "K-Means best K       :",
    best_k_kmeans
)


print(
    "K-Means++ best K     :",
    best_k_kmeans_pp
)


# ============================================================
# 22. SAVE ELBOW DATA
# ============================================================

elbow_data = pd.DataFrame({

    "K": K_LIST,

    "KMeans_Inertia": kmeans_inertia,

    "KMeansPlusPlus_Inertia":
        kmeans_pp_inertia

})


elbow_data.to_csv(

    os.path.join(
        ELBOW_FOLDER,
        "elbow_values.csv"
    ),

    index=False

)


# ============================================================
# 23. SAVE SILHOUETTE DATA
# ============================================================

silhouette_data = pd.DataFrame({

    "K": K_LIST,

    "KMeans_Silhouette":
        kmeans_silhouette,

    "KMeansPlusPlus_Silhouette":
        kmeans_pp_silhouette

})


silhouette_data.to_csv(

    os.path.join(
        SILHOUETTE_FOLDER,
        "silhouette_values.csv"
    ),

    index=False

)


# ============================================================
# 24. ELBOW GRAPH - K-MEANS
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    K_LIST,

    kmeans_inertia,

    marker="o"

)


plt.xlabel(
    "Number of Clusters (K)"
)


plt.ylabel(
    "Inertia"
)


plt.title(
    "Elbow Method - K-Means"
)


plt.xticks(
    K_LIST
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        ELBOW_FOLDER,

        "elbow_kmeans.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 25. ELBOW GRAPH - K-MEANS++
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    K_LIST,

    kmeans_pp_inertia,

    marker="o"

)


plt.xlabel(
    "Number of Clusters (K)"
)


plt.ylabel(
    "Inertia"
)


plt.title(
    "Elbow Method - K-Means++"
)


plt.xticks(
    K_LIST
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        ELBOW_FOLDER,

        "elbow_kmeans_plus_plus.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 26. COMBINED ELBOW GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    K_LIST,

    kmeans_inertia,

    marker="o",

    label="K-Means"

)


plt.plot(

    K_LIST,

    kmeans_pp_inertia,

    marker="s",

    label="K-Means++"

)


plt.xlabel(
    "Number of Clusters (K)"
)


plt.ylabel(
    "Inertia"
)


plt.title(
    "Elbow Method Comparison"
)


plt.xticks(
    K_LIST
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        ELBOW_FOLDER,

        "elbow_comparison.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 27. SILHOUETTE GRAPH - K-MEANS
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    K_LIST,

    kmeans_silhouette,

    marker="o"

)


plt.xlabel(
    "Number of Clusters (K)"
)


plt.ylabel(
    "Silhouette Score"
)


plt.title(
    "Silhouette Score - K-Means"
)


plt.xticks(
    K_LIST
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        SILHOUETTE_FOLDER,

        "silhouette_kmeans.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 28. SILHOUETTE GRAPH - K-MEANS++
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    K_LIST,

    kmeans_pp_silhouette,

    marker="o"

)


plt.xlabel(
    "Number of Clusters (K)"
)


plt.ylabel(
    "Silhouette Score"
)


plt.title(
    "Silhouette Score - K-Means++"
)


plt.xticks(
    K_LIST
)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        SILHOUETTE_FOLDER,

        "silhouette_kmeans_plus_plus.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 29. COMBINED SILHOUETTE GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    K_LIST,

    kmeans_silhouette,

    marker="o",

    label="K-Means"

)


plt.plot(

    K_LIST,

    kmeans_pp_silhouette,

    marker="s",

    label="K-Means++"

)


plt.xlabel(
    "Number of Clusters (K)"
)


plt.ylabel(
    "Silhouette Score"
)


plt.title(
    "Silhouette Score Comparison"
)


plt.xticks(
    K_LIST
)


plt.legend()


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        SILHOUETTE_FOLDER,

        "silhouette_comparison.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 30. FINAL K-MEANS
# ============================================================

kmeans_model = KMeans(

    n_clusters=best_k_kmeans,

    init="random",

    n_init=10,

    random_state=42

)


kmeans_labels = kmeans_model.fit_predict(
    X_scaled
)


# ============================================================
# 31. FINAL K-MEANS++
# ============================================================

kmeans_pp_model = KMeans(

    n_clusters=best_k_kmeans_pp,

    init="k-means++",

    n_init=10,

    random_state=42

)


kmeans_pp_labels = (
    kmeans_pp_model.fit_predict(
        X_scaled
    )
)


# ============================================================
# 32. SAVE K-MEANS CLUSTERED DATA
# ============================================================

kmeans_result = X.copy()


# +1 makes cluster numbers easier to read:
# 1, 2, 3 instead of 0, 1, 2

kmeans_result[
    "KMeans_Cluster"
] = kmeans_labels + 1


kmeans_result.to_csv(

    os.path.join(

        KMEANS_FOLDER,

        "kmeans_clustered_data.csv"

    ),

    index=False

)


# ============================================================
# 33. SAVE K-MEANS++ CLUSTERED DATA
# ============================================================

kmeans_pp_result = X.copy()


kmeans_pp_result[
    "KMeansPlusPlus_Cluster"
] = (
    kmeans_pp_labels + 1
)


kmeans_pp_result.to_csv(

    os.path.join(

        KMEANS_PP_FOLDER,

        "kmeans_plus_plus_clustered_data.csv"

    ),

    index=False

)


# ============================================================
# 34. FINAL METRICS - K-MEANS
# ============================================================

km_silhouette = silhouette_score(

    X_scaled,

    kmeans_labels

)


km_calinski = calinski_harabasz_score(

    X_scaled,

    kmeans_labels

)


km_davies = davies_bouldin_score(

    X_scaled,

    kmeans_labels

)


# ============================================================
# 35. FINAL METRICS - K-MEANS++
# ============================================================

pp_silhouette = silhouette_score(

    X_scaled,

    kmeans_pp_labels

)


pp_calinski = calinski_harabasz_score(

    X_scaled,

    kmeans_pp_labels

)


pp_davies = davies_bouldin_score(

    X_scaled,

    kmeans_pp_labels

)


# ============================================================
# 36. SAVE K-MEANS METRICS
# ============================================================

kmeans_metrics = pd.DataFrame({

    "Method": [
        "K-Means"
    ],

    "Number_of_Clusters": [
        best_k_kmeans
    ],

    "Inertia": [
        kmeans_model.inertia_
    ],

    "Silhouette_Score": [
        km_silhouette
    ],

    "Calinski_Harabasz_Score": [
        km_calinski
    ],

    "Davies_Bouldin_Score": [
        km_davies
    ]

})


kmeans_metrics.to_csv(

    os.path.join(

        KMEANS_FOLDER,

        "kmeans_metrics.csv"

    ),

    index=False

)


# ============================================================
# 37. SAVE K-MEANS++ METRICS
# ============================================================

kmeans_pp_metrics = pd.DataFrame({

    "Method": [
        "K-Means++"
    ],

    "Number_of_Clusters": [
        best_k_kmeans_pp
    ],

    "Inertia": [
        kmeans_pp_model.inertia_
    ],

    "Silhouette_Score": [
        pp_silhouette
    ],

    "Calinski_Harabasz_Score": [
        pp_calinski
    ],

    "Davies_Bouldin_Score": [
        pp_davies
    ]

})


kmeans_pp_metrics.to_csv(

    os.path.join(

        KMEANS_PP_FOLDER,

        "kmeans_plus_plus_metrics.csv"

    ),

    index=False

)


# ============================================================
# 38. COMPARISON METRICS
# ============================================================

comparison_metrics = pd.DataFrame({

    "Method": [
        "K-Means",
        "K-Means++"
    ],

    "Best_K": [
        best_k_kmeans,
        best_k_kmeans_pp
    ],

    "Inertia": [
        kmeans_model.inertia_,
        kmeans_pp_model.inertia_
    ],

    "Silhouette_Score": [
        km_silhouette,
        pp_silhouette
    ],

    "Calinski_Harabasz_Score": [
        km_calinski,
        pp_calinski
    ],

    "Davies_Bouldin_Score": [
        km_davies,
        pp_davies
    ]

})


comparison_metrics.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "kmeans_kmeansplusplus_comparison.csv"

    ),

    index=False

)


# ============================================================
# 39. 3-D K-MEANS CLUSTERING GRAPH
# ============================================================
#
# Four features are used for clustering.
#
# Only the first three features are displayed in this 3-D
# graph because a normal 3-D graph can only show three axes.
# ============================================================

fig = plt.figure(
    figsize=(11, 8)
)


ax = fig.add_subplot(

    111,

    projection="3d"

)


scatter = ax.scatter(

    X[feature_columns[0]],

    X[feature_columns[1]],

    X[feature_columns[2]],

    c=kmeans_labels,

    s=20,

    alpha=0.7

)


ax.set_xlabel(
    feature_columns[0]
)


ax.set_ylabel(
    feature_columns[1]
)


ax.set_zlabel(
    feature_columns[2]
)


ax.set_title(
    "K-Means Clustering"
)


fig.colorbar(

    scatter,

    ax=ax,

    label="Cluster"

)


plt.tight_layout()


plt.savefig(

    os.path.join(

        KMEANS_FOLDER,

        "kmeans_3D_clustering.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 40. 3-D K-MEANS++ CLUSTERING GRAPH
# ============================================================

fig = plt.figure(
    figsize=(11, 8)
)


ax = fig.add_subplot(

    111,

    projection="3d"

)


scatter = ax.scatter(

    X[feature_columns[0]],

    X[feature_columns[1]],

    X[feature_columns[2]],

    c=kmeans_pp_labels,

    s=20,

    alpha=0.7

)


ax.set_xlabel(
    feature_columns[0]
)


ax.set_ylabel(
    feature_columns[1]
)


ax.set_zlabel(
    feature_columns[2]
)


ax.set_title(
    "K-Means++ Clustering"
)


fig.colorbar(

    scatter,

    ax=ax,

    label="Cluster"

)


plt.tight_layout()


plt.savefig(

    os.path.join(

        KMEANS_PP_FOLDER,

        "kmeans_plus_plus_3D_clustering.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 41. PCA 2-D VISUALISATION
# ============================================================
#
# PCA is used ONLY to visualize the clustering results.
#
# Clustering itself was performed using all selected features.
# ============================================================

pca = PCA(
    n_components=2
)


X_pca = pca.fit_transform(
    X_scaled
)


explained_variance = (
    pca.explained_variance_ratio_
)


print(
    "\nPCA Explained Variance:"
)


print(
    "PC1:",
    f"{explained_variance[0] * 100:.2f}%"
)


print(
    "PC2:",
    f"{explained_variance[1] * 100:.2f}%"
)


print(
    "Total:",
    f"{explained_variance.sum() * 100:.2f}%"
)


# ============================================================
# 42. SAVE PCA DATA
# ============================================================

pca_result = pd.DataFrame({

    "PC1": X_pca[:, 0],

    "PC2": X_pca[:, 1],

    "KMeans_Cluster":
        kmeans_labels + 1,

    "KMeansPlusPlus_Cluster":
        kmeans_pp_labels + 1

})


pca_result.to_csv(

    os.path.join(

        PCA_FOLDER,

        "pca_cluster_data.csv"

    ),

    index=False

)


# ============================================================
# 43. K-MEANS PCA GRAPH
# ============================================================

plt.figure(
    figsize=(10, 7)
)


scatter = plt.scatter(

    X_pca[:, 0],

    X_pca[:, 1],

    c=kmeans_labels,

    s=20,

    alpha=0.7

)


plt.xlabel(
    "Principal Component 1"
)


plt.ylabel(
    "Principal Component 2"
)


plt.title(
    "K-Means Clusters - PCA Visualization"
)


plt.colorbar(

    scatter,

    label="Cluster"

)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        KMEANS_FOLDER,

        "kmeans_PCA_clustering.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 44. K-MEANS++ PCA GRAPH
# ============================================================

plt.figure(
    figsize=(10, 7)
)


scatter = plt.scatter(

    X_pca[:, 0],

    X_pca[:, 1],

    c=kmeans_pp_labels,

    s=20,

    alpha=0.7

)


plt.xlabel(
    "Principal Component 1"
)


plt.ylabel(
    "Principal Component 2"
)


plt.title(
    "K-Means++ Clusters - PCA Visualization"
)


plt.colorbar(

    scatter,

    label="Cluster"

)


plt.grid(
    True
)


plt.tight_layout()


plt.savefig(

    os.path.join(

        KMEANS_PP_FOLDER,

        "kmeans_plus_plus_PCA_clustering.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 45. CLUSTER COUNTS - K-MEANS
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "K-MEANS CLUSTER COUNTS"
)

print(
    "=" * 75
)


kmeans_counts = (

    kmeans_result[
        "KMeans_Cluster"
    ]

    .value_counts()

    .sort_index()

)


print(
    kmeans_counts
)


kmeans_counts_df = pd.DataFrame({

    "Cluster": kmeans_counts.index,

    "Number_of_Flights":
        kmeans_counts.values

})


kmeans_counts_df.to_csv(

    os.path.join(

        KMEANS_FOLDER,

        "kmeans_cluster_counts.csv"

    ),

    index=False

)


# ============================================================
# 46. CLUSTER COUNTS - K-MEANS++
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "K-MEANS++ CLUSTER COUNTS"
)

print(
    "=" * 75
)


kmeans_pp_counts = (

    kmeans_pp_result[
        "KMeansPlusPlus_Cluster"
    ]

    .value_counts()

    .sort_index()

)


print(
    kmeans_pp_counts
)


kmeans_pp_counts_df = pd.DataFrame({

    "Cluster": kmeans_pp_counts.index,

    "Number_of_Flights":
        kmeans_pp_counts.values

})


kmeans_pp_counts_df.to_csv(

    os.path.join(

        KMEANS_PP_FOLDER,

        "kmeans_plus_plus_cluster_counts.csv"

    ),

    index=False

)


# ============================================================
# 47. CHECK DELAY TARGET
# ============================================================
#
# The delay column is NOT used to create the clusters.
#
# After clustering, we compare each cluster with delay.
#
# This helps us understand whether some clusters contain a
# larger proportion of delayed flights.
# ============================================================

if "delay" in data.columns:

    print(
        "\n" + "=" * 75
    )

    print(
        "CLUSTER VS FLIGHT DELAY ANALYSIS"
    )

    print(
        "=" * 75
    )


    delay = pd.to_numeric(

        data["delay"],

        errors="coerce"

    )


    valid_delay = delay.notna()


    delay_valid = delay[
        valid_delay
    ].reset_index(
        drop=True
    )


    km_labels_valid = pd.Series(

        kmeans_labels

    )[valid_delay].reset_index(
        drop=True
    )


    pp_labels_valid = pd.Series(

        kmeans_pp_labels

    )[valid_delay].reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # K-MEANS DELAY ANALYSIS
    # --------------------------------------------------------

    km_delay_table = pd.DataFrame({

        "Cluster":
            km_labels_valid + 1,

        "Delay":
            delay_valid

    })


    km_delay_summary = (

        km_delay_table

        .groupby("Cluster")["Delay"]

        .agg([
            "count",
            "mean",
            "sum"
        ])

        .reset_index()

    )


    km_delay_summary.rename(

        columns={

            "count":
                "Number_of_Flights",

            "mean":
                "Delay_Rate",

            "sum":
                "Number_of_Delayed_Flights"

        },

        inplace=True

    )


    km_delay_summary[
        "Delay_Rate_Percentage"
    ] = (

        km_delay_summary[
            "Delay_Rate"
        ]

        * 100

    )


    print(
        "\nK-Means cluster delay summary:"
    )


    print(
        km_delay_summary
    )


    km_delay_summary.to_csv(

        os.path.join(

            KMEANS_FOLDER,

            "kmeans_delay_summary.csv"

        ),

        index=False

    )


    # --------------------------------------------------------
    # K-MEANS++ DELAY ANALYSIS
    # --------------------------------------------------------

    pp_delay_table = pd.DataFrame({

        "Cluster":
            pp_labels_valid + 1,

        "Delay":
            delay_valid

    })


    pp_delay_summary = (

        pp_delay_table

        .groupby("Cluster")["Delay"]

        .agg([
            "count",
            "mean",
            "sum"
        ])

        .reset_index()

    )


    pp_delay_summary.rename(

        columns={

            "count":
                "Number_of_Flights",

            "mean":
                "Delay_Rate",

            "sum":
                "Number_of_Delayed_Flights"

        },

        inplace=True

    )


    pp_delay_summary[
        "Delay_Rate_Percentage"
    ] = (

        pp_delay_summary[
            "Delay_Rate"
        ]

        * 100

    )


    print(
        "\nK-Means++ cluster delay summary:"
    )


    print(
        pp_delay_summary
    )


    pp_delay_summary.to_csv(

        os.path.join(

            KMEANS_PP_FOLDER,

            "kmeans_plus_plus_delay_summary.csv"

        ),

        index=False

    )


    # ========================================================
    # 48. CLUSTER DELAY RATE GRAPH - K-MEANS
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )


    plt.bar(

        km_delay_summary[
            "Cluster"
        ].astype(str),

        km_delay_summary[
            "Delay_Rate_Percentage"
        ]

    )


    plt.xlabel(
        "Cluster"
    )


    plt.ylabel(
        "Delayed Flights (%)"
    )


    plt.title(
        "K-Means Cluster-wise Flight Delay Rate"
    )


    plt.grid(
        axis="y"
    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            KMEANS_FOLDER,

            "kmeans_cluster_delay_rate.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    # ========================================================
    # 49. CLUSTER DELAY RATE GRAPH - K-MEANS++
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )


    plt.bar(

        pp_delay_summary[
            "Cluster"
        ].astype(str),

        pp_delay_summary[
            "Delay_Rate_Percentage"
        ]

    )


    plt.xlabel(
        "Cluster"
    )


    plt.ylabel(
        "Delayed Flights (%)"
    )


    plt.title(
        "K-Means++ Cluster-wise Flight Delay Rate"
    )


    plt.grid(
        axis="y"
    )


    plt.tight_layout()


    plt.savefig(

        os.path.join(

            KMEANS_PP_FOLDER,

            "kmeans_plus_plus_cluster_delay_rate.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


else:

    print(
        "\n'delay' column was not found."
    )

    print(
        "Delay analysis was skipped."
    )


# ============================================================
# 50. SAVE STANDARDIZATION INFORMATION
# ============================================================

scaling_information = pd.DataFrame({

    "Feature": feature_columns,

    "Mean": scaler.mean_,

    "Standard_Deviation": scaler.scale_

})


scaling_information.to_csv(

    os.path.join(

        OUTPUT_FOLDER,

        "standardization_information.csv"

    ),

    index=False

)


# ============================================================
# 51. SAVE PCA INFORMATION
# ============================================================

pca_information = pd.DataFrame({

    "Component": [
        "PC1",
        "PC2"
    ],

    "Explained_Variance_Ratio": [

        explained_variance[0],

        explained_variance[1]

    ],

    "Explained_Variance_Percentage": [

        explained_variance[0] * 100,

        explained_variance[1] * 100

    ]

})


pca_information.to_csv(

    os.path.join(

        PCA_FOLDER,

        "pca_information.csv"

    ),

    index=False

)


# ============================================================
# 52. FINAL RESULTS
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "FINAL CLUSTERING RESULTS"
)

print(
    "=" * 75
)


print(
    "\nK-MEANS"
)

print(
    "-" * 40
)


print(
    "Best K:",
    best_k_kmeans
)


print(
    "Inertia:",
    round(
        kmeans_model.inertia_,
        4
    )
)


print(
    "Silhouette:",
    round(
        km_silhouette,
        4
    )
)


print(
    "Calinski-Harabasz:",
    round(
        km_calinski,
        4
    )
)


print(
    "Davies-Bouldin:",
    round(
        km_davies,
        4
    )
)


print(
    "\nK-MEANS++"
)

print(
    "-" * 40
)


print(
    "Best K:",
    best_k_kmeans_pp
)


print(
    "Inertia:",
    round(
        kmeans_pp_model.inertia_,
        4
    )
)


print(
    "Silhouette:",
    round(
        pp_silhouette,
        4
    )
)


print(
    "Calinski-Harabasz:",
    round(
        pp_calinski,
        4
    )
)


print(
    "Davies-Bouldin:",
    round(
        pp_davies,
        4
    )
)


# ============================================================
# 53. OUTPUT LOCATION
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "OUTPUTS SAVED"
)

print(
    "=" * 75
)


print(
    "\nMain folder:"
)

print(
    OUTPUT_FOLDER
)


print(
    "\nK-Means:"
)

print(
    KMEANS_FOLDER
)


print(
    "\nK-Means++:"
)

print(
    KMEANS_PP_FOLDER
)


print(
    "\nElbow:"
)

print(
    ELBOW_FOLDER
)


print(
    "\nSilhouette:"
)

print(
    SILHOUETTE_FOLDER
)


print(
    "\nPCA:"
)

print(
    PCA_FOLDER
)


print(
    "\nAccuracy:"
)

print(
    ACCURACY_FOLDER
)


print(
    "\nOriginal/preprocessed input dataset was NOT modified."
)


print(
    "\nPROGRAM COMPLETED SUCCESSFULLY."
)


print(
    "=" * 75
)