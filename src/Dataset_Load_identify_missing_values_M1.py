import pandas as pd

# ============================================================
# IMPORTANT:
# Use the non-interactive Agg backend.
# This prevents the Tkinter / FigureCanvasAgg errors.
# ============================================================

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD THE FLIGHT DATASET
# ============================================================

file_path = (
    "C:/Users/hp/PycharmProjects/FlightFore/"
    "datasets/flights_5000_each_month_with_class_labels (1).csv"
)

df = pd.read_csv(file_path)


# ============================================================
# 2. RETRIEVE DATA IN DIFFERENT WAYS
# ============================================================

# ------------------------------------------------------------
# View the first 5 rows
# ------------------------------------------------------------

print("--- First 5 Rows ---")
print(df.head())

print("-" * 50)


# ------------------------------------------------------------
# Print first 6 columns
# ------------------------------------------------------------

print("---- First 6 Columns ----")

subset = df.iloc[:, 0:6]

print(subset)

print("-" * 50)


# ============================================================
# 3. IDENTIFY MISSING VALUES PER COLUMN
# ============================================================

missing_counts = df.isnull().sum()

print("----- Missing Values Per Column -----")
print(missing_counts)

print("-" * 50)


# ============================================================
# 4. DETECT DUPLICATE ROWS
# ============================================================

# Keeps the first occurrence and marks subsequent
# duplicate rows as True

duplicate_rows = df[df.duplicated()]

print(
    f"Total duplicate rows detected: "
    f"{len(duplicate_rows)}"
)

print(duplicate_rows)

print("-" * 50)


# ============================================================
# 5. MISSINGNESS HEATMAP
# ============================================================

print("Creating missing values heatmap...")


plt.figure(figsize=(14, 7))


sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False,
    cmap="viridis"
)


plt.title(
    "Missing Values Heatmap - Flight Dataset",
    fontsize=16,
    fontweight="bold"
)


plt.xlabel(
    "Flight Dataset Attributes"
)

plt.ylabel(
    "Records"
)


plt.tight_layout()


# ------------------------------------------------------------
# Save heatmap instead of plt.show()
# ------------------------------------------------------------

heatmap_file = (
    "C:/Users/hp/PycharmProjects/FlightFore/"
    "datasets/missing_values_heatmap_M1.png"
)


plt.savefig(
    heatmap_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print("\nMissing values heatmap saved successfully.")
print("Heatmap File:")
print(heatmap_file)


# ============================================================
# 6. SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("FLIGHT DATASET SUMMARY")
print("=" * 50)


print(
    "Number of Rows    :",
    df.shape[0]
)


print(
    "Number of Columns :",
    df.shape[1]
)


print(
    "Total Missing Values :",
    df.isnull().sum().sum()
)


print(
    "Total Duplicate Rows :",
    len(duplicate_rows)
)


print("=" * 50)


# ============================================================
# 7. FINAL STATUS
# ============================================================

print("\n==========================================")
print("M1 DATASET ANALYSIS COMPLETED SUCCESSFULLY")
print("==========================================")

print("\nDataset File:")
print(file_path)

print("\nHeatmap File:")
print(heatmap_file)