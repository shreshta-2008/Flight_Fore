import pandas as pd

# ============================================================
# IMPORTANT:
# Use the non-interactive Agg backend.
# This prevents Tkinter / FigureCanvasAgg errors.
# ============================================================

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("1. Load the dataset")

file_path = (
    "C:/Users/hp/PycharmProjects/FlightFore/"
    "datasets/flights_5000_each_month_with_class_labels (1).csv"
)

try:

    df = pd.read_csv(file_path)

    print("-----------------------------------")
    print("1. Dataset Contents:")
    print("-----------------------------------")
    print(df)

    # ========================================================
    # Dataset shape
    # ========================================================

    print("-----------------------------------")
    print("\n2. Number of Rows and Columns:", df.shape)
    print("-----------------------------------")

    # ========================================================
    # Column names
    # ========================================================

    print("\n3. Column Names:")
    print("-----------------------------------")
    print(df.columns.tolist())

    # ========================================================
    # Configure pandas display
    # ========================================================

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # ========================================================
    # First 10 rows
    # ========================================================

    print("-----------------------------------")
    print("\n4. First 10 Records")
    print("-----------------------------------")
    print(df.head(10))

    # ========================================================
    # Last 10 rows
    # ========================================================

    print("-----------------------------------")
    print("Last 10 Records")
    print("-----------------------------------")
    print(df.tail(10))


    # ============================================================
    # 2. UNDERSTAND THE DATASET
    # ============================================================

    print("\n" + "=" * 60)
    print("2. UNDERSTAND THE DATASET")
    print("=" * 60)

    # ========================================================
    # Data types
    # ========================================================

    print("-----------------------------------")
    print("5. Data Types of Columns:")
    print("-----------------------------------")
    print(df.dtypes)

    # ========================================================
    # Column names with data types
    # ========================================================

    print("-----------------------------------")
    print("6. Column Names with Data Types")
    print("-----------------------------------")

    print("\nColumn Name\t\tData Type")
    print("-" * 50)

    for column in df.columns:
        print(
            f"{column:<25} {df[column].dtype}"
        )

    # ========================================================
    # Dataset information
    # ========================================================

    print("-----------------------------------")
    print("7. Dataset Summary and Information")
    print("-----------------------------------")

    df.info()

    print("\n" + "=" * 50 + "\n")


    # ============================================================
    # 3. NUMERICAL COLUMNS
    # ============================================================

    print("8. Display Numerical Columns")
    print("-----------------------------------")

    # Use "number" instead of only int64/float64
    # This is more robust with different Pandas versions.

    numeric_df = df.select_dtypes(
        include=["number"]
    )

    print("Numerical Columns:")
    print(numeric_df.columns.tolist())

    print("\nMissing Values in Numeric Attributes:")
    print(numeric_df.isnull().sum())

    print(
        "\n9. Total Missing Numeric Values:",
        numeric_df.isnull().sum().sum()
    )


    # ============================================================
    # 4. FLOAT COLUMNS
    # ============================================================

    print("-----------------------------------")
    print("10. Float Attribute Names:")
    print("-----------------------------------")

    float_columns = df.select_dtypes(
        include=["floating"]
    ).columns

    for column in float_columns:
        print(column)

    print("\n11. Missing Values in Float Attributes")
    print("=" * 50)

    print(
        df[float_columns].isnull().sum()
    )

    print(
        "\n12. Total Missing Float Values:",
        df[float_columns].isnull().sum().sum()
    )


    # ============================================================
    # 5. CATEGORICAL COLUMNS
    # ============================================================

    # IMPORTANT:
    # Include both object and string dtypes.
    # This avoids the Pandas 4 warning.

    categorical_df = df.select_dtypes(
        include=["object", "string"]
    )

    print("-----------------------------------")
    print("13. Display Categorical Attributes:")
    print("-----------------------------------")

    print("Categorical Columns:")
    print(categorical_df.columns.tolist())

    print("\nMissing Values in Categorical Attributes")
    print("=" * 50)

    print(
        categorical_df.isnull().sum()
    )

    print(
        "\n14. Total Missing Categorical Values:",
        categorical_df.isnull().sum().sum()
    )


    # ============================================================
    # 6. MISSING VALUES
    # ============================================================

    print("-----------------------------------")
    print("15. Missing Values in Each Column")
    print("-" * 40)

    print(
        df.isnull().sum()
    )

    total_missing = (
        df.isnull().sum().sum()
    )

    print("-----------------------------------")
    print(
        "16. Total Missing Values:",
        total_missing
    )


    # ============================================================
    # 7. DUPLICATE RECORDS
    # ============================================================

    print("-----------------------------------")

    duplicate_count = (
        df.duplicated().sum()
    )

    print(
        "17. Number of Duplicate Records:",
        duplicate_count
    )


    # ============================================================
    # 8. STATISTICAL SUMMARY
    # ============================================================

    print("-----------------------------------")
    print("18. Statistical Overview")
    print("-----------------------------------")

    print(
        df.describe()
    )


    # ============================================================
    # 9. FLIGHT DELAY ANALYSIS
    # ============================================================

    print("\n" + "=" * 60)
    print("19. FLIGHT DELAY ANALYSIS")
    print("=" * 60)

    print("\nDeparture Delay Statistics:")

    print(
        df["DEPARTURE_DELAY"].describe()
    )

    print("\nArrival Delay Statistics:")

    print(
        df["ARRIVAL_DELAY"].describe()
    )


    # ============================================================
    # 10. HISTOGRAM - DEPARTURE DELAY
    # ============================================================

    print("-----------------------------------")
    print("20. Histogram of Departure Delay")
    print("-----------------------------------")

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        df["DEPARTURE_DELAY"].dropna(),
        bins=30,
        edgecolor="black"
    )

    plt.title(
        "Histogram of Departure Delay"
    )

    plt.xlabel(
        "Departure Delay (minutes)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.grid(True)

    plt.tight_layout()

    # Save instead of plt.show()

    departure_histogram = (
        "C:/Users/hp/PycharmProjects/FlightFore/"
        "datasets/departure_delay_histogram.png"
    )

    plt.savefig(
        departure_histogram,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Departure delay histogram saved:"
    )

    print(
        departure_histogram
    )


    # ============================================================
    # 11. HISTOGRAM - ARRIVAL DELAY
    # ============================================================

    print("-----------------------------------")
    print("21. Histogram of Arrival Delay")
    print("-----------------------------------")

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        df["ARRIVAL_DELAY"].dropna(),
        bins=30,
        edgecolor="black"
    )

    plt.title(
        "Histogram of Arrival Delay"
    )

    plt.xlabel(
        "Arrival Delay (minutes)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.grid(True)

    plt.tight_layout()

    # Save instead of plt.show()

    arrival_histogram = (
        "C:/Users/hp/PycharmProjects/FlightFore/"
        "datasets/arrival_delay_histogram.png"
    )

    plt.savefig(
        arrival_histogram,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Arrival delay histogram saved:"
    )

    print(
        arrival_histogram
    )


    # ============================================================
    # 12. AIRLINE DISTRIBUTION
    # ============================================================

    print("-----------------------------------")
    print("22. Number of Flights by Airline")
    print("-----------------------------------")

    airline_counts = (
        df["AIRLINE"].value_counts()
    )

    print(
        airline_counts
    )

    plt.figure(
        figsize=(10, 5)
    )

    airline_counts.plot(
        kind="bar"
    )

    plt.title(
        "Number of Flights by Airline"
    )

    plt.xlabel(
        "Airline"
    )

    plt.ylabel(
        "Number of Flights"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    # Save instead of plt.show()

    airline_chart = (
        "C:/Users/hp/PycharmProjects/FlightFore/"
        "datasets/flights_by_airline.png"
    )

    plt.savefig(
        airline_chart,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Airline distribution chart saved:"
    )

    print(
        airline_chart
    )


    # ============================================================
    # 13. CANCELLATION ANALYSIS
    # ============================================================

    print("-----------------------------------")
    print("23. Flight Cancellation Analysis")
    print("-----------------------------------")

    print("Cancelled Flights:")

    print(
        df["CANCELLED"].value_counts()
    )

    print("\nCancellation Reasons:")

    print(
        df["CANCELLATION_REASON"].value_counts(
            dropna=False
        )
    )


    # ============================================================
    # 14. DIVERTED FLIGHTS
    # ============================================================

    print("-----------------------------------")
    print("24. Diverted Flight Analysis")
    print("-----------------------------------")

    print(
        df["DIVERTED"].value_counts()
    )


    # ============================================================
    # 15. CORRELATION MATRIX
    # ============================================================

    print("-----------------------------------")
    print("25. Correlation Analysis")
    print("-----------------------------------")

    correlation = (
        numeric_df.corr()
    )

    print(
        correlation
    )

    plt.figure(
        figsize=(14, 10)
    )

    sns.heatmap(
        correlation,
        annot=False,
        cmap="coolwarm"
    )

    plt.title(
        "Correlation Matrix of Numerical Flight Attributes"
    )

    plt.tight_layout()

    # Save instead of plt.show()

    correlation_heatmap = (
        "C:/Users/hp/PycharmProjects/FlightFore/"
        "datasets/correlation_matrix.png"
    )

    plt.savefig(
        correlation_heatmap,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Correlation heatmap saved:"
    )

    print(
        correlation_heatmap
    )


    # ============================================================
    # COMPLETION
    # ============================================================

    print("\n" + "=" * 60)
    print("DATASET ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        "Dataset Shape:",
        df.shape
    )

    print(
        "Total Missing Values:",
        total_missing
    )

    print(
        "Duplicate Records:",
        duplicate_count
    )

    print("\nGenerated Graph Files:")
    print(
        "1.",
        departure_histogram
    )
    print(
        "2.",
        arrival_histogram
    )
    print(
        "3.",
        airline_chart
    )
    print(
        "4.",
        correlation_heatmap
    )


# ============================================================
# ERROR HANDLING
# ============================================================

except FileNotFoundError:

    print(
        f"Error: The file at '{file_path}' was not found."
    )

except Exception as e:

    print(
        f"An error occurred: {e}"
    )