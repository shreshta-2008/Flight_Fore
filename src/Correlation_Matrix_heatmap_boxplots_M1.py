import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================
# 1. Create Output Directory
# =============================================================

output_dir = "C:/Users/srinu/PycharmProjects/FlightFore/outputs/Boxplots_correlation"
os.makedirs(output_dir, exist_ok=True)


# =============================================================
# 2. Load Flight Dataset
# =============================================================

dataset_path = "C:/Users/srinu/PycharmProjects/FlightFore/datasets/flights_5000_each_month_with_class_labels (1).csv"

df = pd.read_csv(dataset_path)

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

print("\n--- Dataset Columns ---")
print(df.columns.tolist())


# =============================================================
# 3. Select Numerical Features
# =============================================================

numerical_cols = df.select_dtypes(
    include=[np.number]
).columns.tolist()


# Remove target column from feature correlation
# so that the feature-to-feature correlation is clearly shown

feature_numerical_cols = [
    col for col in numerical_cols
    if col != "Delay"
]


print("\n--- Numerical Features ---")
print(feature_numerical_cols)


# =============================================================
# 4. Compute Correlation Matrix
# =============================================================

corr_matrix = df[feature_numerical_cols].corr()


print("\n--- Correlation Matrix ---")
print(corr_matrix)


# =============================================================
# 5. Generate Correlation Heatmap
# =============================================================

plt.figure(
    figsize=(16, 12)
)

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    square=False,
    linewidths=0.5
)

plt.title(
    "Correlation Heatmap of Flight Numerical Features",
    fontsize=16,
    fontweight="bold"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()


# Save heatmap

heatmap_path = os.path.join(
    output_dir,
    "flight_correlation_heatmap.png"
)

plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nExported correlation heatmap to: {heatmap_path}"
)


# =============================================================
# 6. Correlation of Features with Delay
# =============================================================

target_col = "Delay"


if target_col in df.columns:

    target_correlation = (
        df[numerical_cols]
        .corr()[target_col]
        .sort_values(
            ascending=False
        )
    )

    print("\n--- Correlation with Delay ---")
    print(target_correlation)


    # Remove Delay itself from the plot

    target_correlation = target_correlation.drop(
        target_col
    )


    # ---------------------------------------------------------
    # Plot Feature Correlation with Delay
    # ---------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    target_correlation.plot(
        kind="bar"
    )

    plt.title(
        "Correlation of Numerical Features with Flight Delay",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel(
        "Flight Features"
    )

    plt.ylabel(
        "Correlation with Delay"
    )

    plt.xticks(
        rotation=75
    )

    plt.axhline(
        y=0,
        linewidth=1
    )

    plt.tight_layout()


    delay_corr_path = os.path.join(
        output_dir,
        "features_vs_delay_correlation.png"
    )

    plt.savefig(
        delay_corr_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Exported feature-delay correlation plot to: "
        f"{delay_corr_path}"
    )


else:

    print(
        f"\nTarget column '{target_col}' "
        "not found in dataset."
    )


# =============================================================
# 7. Produce Boxplots:
#    Numerical Features vs Delay
# =============================================================

if target_col in df.columns:

    print("\n--- Generating Boxplots ---")


    for col in feature_numerical_cols:

        # Skip columns that have no useful variation

        if df[col].nunique() <= 1:
            continue


        plt.figure(
            figsize=(7, 5)
        )


        sns.boxplot(
            x=target_col,
            y=col,
            data=df,
            hue=target_col,
            legend=False
        )


        plt.title(
            f"{col} vs Flight Delay",
            fontsize=12,
            fontweight="bold"
        )


        plt.xlabel(
            "Flight Delay Class"
        )


        plt.ylabel(
            col
        )


        plt.tight_layout()


        # Save individual boxplot

        boxplot_filename = (
            f"boxplot_{col}_vs_Delay.png"
        )

        boxplot_path = os.path.join(
            output_dir,
            boxplot_filename
        )


        plt.savefig(
            boxplot_path,
            dpi=300,
            bbox_inches="tight"
        )


        plt.close()


        print(
            f"Exported boxplot to: {boxplot_path}"
        )


else:

    print(
        f"\nTarget column '{target_col}' "
        "not found. Skipping boxplots."
    )


# =============================================================
# 8. Delay Class Distribution
# =============================================================

if target_col in df.columns:

    print("\n--- Flight Delay Class Distribution ---")

    print(
        df[target_col].value_counts()
    )


    plt.figure(
        figsize=(6, 5)
    )


    sns.countplot(
        x=target_col,
        data=df
    )


    plt.title(
        "Flight Delay Class Distribution",
        fontsize=14,
        fontweight="bold"
    )


    plt.xlabel(
        "Delay Class"
    )


    plt.ylabel(
        "Number of Flights"
    )


    plt.tight_layout()


    delay_distribution_path = os.path.join(
        output_dir,
        "flight_delay_distribution.png"
    )


    plt.savefig(
        delay_distribution_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Exported delay distribution plot to: "
        f"{delay_distribution_path}"
    )


# =============================================================
# 9. Airline-wise Delay Analysis
# =============================================================

if "AIRLINE" in df.columns and target_col in df.columns:

    airline_delay = (
        df.groupby("AIRLINE")[target_col]
        .mean()
        .sort_values(ascending=False)
    )


    print("\n--- Airline-wise Delay Rate ---")
    print(airline_delay)


    plt.figure(
        figsize=(10, 6)
    )


    airline_delay.plot(
        kind="bar"
    )


    plt.title(
        "Average Delay Rate by Airline",
        fontsize=14,
        fontweight="bold"
    )


    plt.xlabel(
        "Airline"
    )


    plt.ylabel(
        "Delay Rate"
    )


    plt.xticks(
        rotation=45
    )


    plt.tight_layout()


    airline_path = os.path.join(
        output_dir,
        "airline_delay_rate.png"
    )


    plt.savefig(
        airline_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Exported airline delay plot to: {airline_path}"
    )


# =============================================================
# 10. Monthly Delay Analysis
# =============================================================

if "MONTH" in df.columns and target_col in df.columns:

    monthly_delay = (
        df.groupby("MONTH")[target_col]
        .mean()
    )


    plt.figure(
        figsize=(9, 5)
    )


    monthly_delay.plot(
        marker="o"
    )


    plt.title(
        "Monthly Flight Delay Rate",
        fontsize=14,
        fontweight="bold"
    )


    plt.xlabel(
        "Month"
    )


    plt.ylabel(
        "Delay Rate"
    )


    plt.xticks(
        range(1, 13)
    )


    plt.grid(
        True,
        alpha=0.3
    )


    plt.tight_layout()


    monthly_path = os.path.join(
        output_dir,
        "monthly_delay_rate.png"
    )


    plt.savefig(
        monthly_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Exported monthly delay plot to: {monthly_path}"
    )


# =============================================================
# 11. Final Message
# =============================================================

print("\n==============================================")
print("Flight EDA Tasks Completed Successfully!")
print("==============================================")

print(
    f"\nAll generated files are available in:\n"
    f"{output_dir}"
)