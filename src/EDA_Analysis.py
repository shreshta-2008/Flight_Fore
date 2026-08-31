# EDA_Analysis.py
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

DATASET_PATH = "C:/Users/srinu/PycharmProjects/FlightFore/datasets/flights_5000_each_month_with_class_labels (1).csv"
OUTPUT_FOLDER = "C:/Users/srinu/PycharmProjects/FlightFore/outputs/EDA_Analysis_outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"]=(8,5)

df = pd.read_csv(DATASET_PATH)

print("="*60)
print(df.head())
print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
df.info()
print(df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=["object","string","category","bool"]).columns.tolist()
target = "delay" if "delay" in df.columns else None

print("Creating summary...")
df.describe(include="all").to_csv(os.path.join(OUTPUT_FOLDER,"Statistical_Summary.csv"))

print("Histograms / Boxplots...")
for col in numeric_cols:
    s=df[col].dropna()
    plt.figure(); sns.histplot(s,bins=25,kde=True,color="skyblue")
    plt.title(f"Histogram - {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER,f"Histogram_{col}.png")); plt.close()

    plt.figure(); sns.boxplot(y=s,color="orange")
    plt.title(f"Boxplot - {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER,f"Boxplot_{col}.png")); plt.close()

print("Correlation...")
corr=df[numeric_cols].corr(numeric_only=True)
corr.to_csv(os.path.join(OUTPUT_FOLDER,"Correlation_Matrix.csv"))
plt.figure(figsize=(15,10))
sns.heatmap(corr,annot=True,cmap="coolwarm",fmt=".2f")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER,"Correlation_Heatmap.png")); plt.close()

print("Countplots...")
for col in categorical_cols:
    plt.figure(figsize=(10,5))
    order=df[col].astype(str).value_counts().index
    sns.countplot(data=df,x=col,order=order)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER,f"Countplot_{col}.png")); plt.close()

print("Pairplot...")
if len(numeric_cols)>1:
    sample=df[numeric_cols].sample(min(1000,len(df)),random_state=42)
    g=sns.pairplot(sample)
    g.savefig(os.path.join(OUTPUT_FOLDER,"Pairplot.png"))
    plt.close('all')

print("Missing values heatmap...")
plt.figure(figsize=(12,6))
sns.heatmap(df.sample(min(1000,len(df)),random_state=42).isnull(),cbar=False,cmap="viridis")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER,"Missing_Values_Heatmap.png")); plt.close()

if {"DEPARTURE_DELAY","ARRIVAL_DELAY"}.issubset(df.columns):
    print("Scatter...")
    plt.figure(figsize=(8,6))
    d=df.sample(min(3000,len(df)),random_state=42)
    if target:
        sns.scatterplot(data=d,x="DEPARTURE_DELAY",y="ARRIVAL_DELAY",hue=target,alpha=0.6)
    else:
        sns.scatterplot(data=d,x="DEPARTURE_DELAY",y="ARRIVAL_DELAY",alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER,"Departure_vs_Arrival.png")); plt.close()

if target:
    plt.figure()
    sns.countplot(data=df,x=target)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER,"Target_Distribution.png")); plt.close()

for c,name in [("AIRLINE","Flights_By_Airline"),("MONTH","Flights_By_Month"),
("DAY_OF_WEEK","Flights_By_DayOfWeek"),("ORIGIN_AIRPORT","Origin_Airport_Distribution"),
("DESTINATION_AIRPORT","Destination_Airport_Distribution")]:
    if c in df.columns:
        plt.figure(figsize=(12,5))
        sns.countplot(data=df,x=c,order=df[c].astype(str).value_counts().index)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER,name+".png")); plt.close()

if {"AIRLINE","DEPARTURE_DELAY"}.issubset(df.columns):
    plt.figure(figsize=(12,6))
    avg=df.groupby("AIRLINE")["DEPARTURE_DELAY"].mean().reset_index()
    sns.barplot(data=avg,x="AIRLINE",y="DEPARTURE_DELAY")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER,"Average_Departure_Delay_By_Airline.png")); plt.close()

if "DISTANCE" in df.columns:
    plt.figure()
    sns.histplot(df["DISTANCE"],bins=30,kde=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER,"Distance_Distribution.png")); plt.close()

print("="*60)
print("EDA Completed Successfully.")
print("Outputs saved to:",OUTPUT_FOLDER)