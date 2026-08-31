import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD THE FLIGHT DATASET
# ============================================================

df = pd.read_csv(
    "C:/Users/srinu/PycharmProjects/FlightFore/datasets/flights_5000_each_month_with_class_labels (1).csv"
)


# ============================================================
# 2. RETRIEVE DATA IN DIFFERENT WAYS
# ============================================================

# View the first 5 rows
print("--- First 5 Rows ---")
print(df.head())

print("-" * 50)


# Print first 6 columns
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

# Keeps the first occurrence and marks subsequent duplicates as True
duplicate_rows = df[df.duplicated()]

print(f"Total duplicate rows detected: {len(duplicate_rows)}")

print(duplicate_rows)

print("-" * 50)


# ============================================================
# 5. MISSINGNESS HEATMAP
# ============================================================

plt.figure(figsize=(12, 6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False,
    cmap="viridis"
)

plt.title("Missing Values Heatmap - Flight Dataset")

plt.xlabel("Flight Dataset Attributes")
plt.ylabel("Records")

plt.tight_layout()

plt.show()


# ============================================================
# 6. SUMMARY
# ============================================================

print("\n" + "=" * 50)
print("FLIGHT DATASET SUMMARY")
print("=" * 50)

print("Number of Rows    :", df.shape[0])
print("Number of Columns :", df.shape[1])

print("Total Missing Values :", df.isnull().sum().sum())

print("Total Duplicate Rows :", len(duplicate_rows))

print("=" * 50)