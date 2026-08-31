import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

INPUT_FILE = r"C:/Users/srinu/PycharmProjects/FlightFore/datasets/flights_5000_each_month_with_class_labels (1).csv"
OUTPUT_FILE = r"C:/Users/srinu/PycharmProjects/FlightFore/datasets/flight_one_hot_encoded.csv"

data = pd.read_csv(INPUT_FILE).copy()

print("="*60)
print("Original Shape:", data.shape)
print("Duplicates:", data.duplicated().sum())

for col in data.select_dtypes(include=["object","string"]).columns:
    data[col] = data[col].astype(str).str.strip()

print("\nMissing Values Before:")
print(data.isnull().sum())

data = data.drop_duplicates()

if "delay" not in data.columns:
    raise ValueError("Target column 'delay' not found.")

target_col="delay"

num_cols=data.select_dtypes(include=np.number).columns.tolist()
cat_cols=data.select_dtypes(include=["object","string","category"]).columns.tolist()

if target_col in num_cols:
    num_cols.remove(target_col)
if target_col in cat_cols:
    cat_cols.remove(target_col)

for c in ["TAIL_NUMBER","FLIGHT_NUMBER"]:
    if c in cat_cols:
        cat_cols.remove(c)

num_imp=SimpleImputer(strategy="mean")
if num_cols:
    data[num_cols]=num_imp.fit_transform(data[num_cols])

cat_imp=SimpleImputer(strategy="most_frequent")
if cat_cols:
    data[cat_cols]=cat_imp.fit_transform(data[cat_cols])

if cat_cols:
    enc=OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int8)
    arr=enc.fit_transform(data[cat_cols])
    enc_df=pd.DataFrame(arr, columns=enc.get_feature_names_out(cat_cols))
else:
    enc_df=pd.DataFrame()

numeric_df=data[num_cols].reset_index(drop=True)
target_df=data[[target_col]].reset_index(drop=True)

final_output=pd.concat([numeric_df, enc_df.reset_index(drop=True), target_df], axis=1)

print("\nMissing Values After:")
print(final_output.isnull().sum().sum())
print("Final Shape:", final_output.shape)

final_output.to_csv(OUTPUT_FILE,index=False,encoding="utf-8")
print("Saved to:", OUTPUT_FILE)