import pandas as pd
import numpy as np

INPUT_FILE=r"C:/Users/hp/PycharmProjects/FlightFore/datasets/flights_5000_each_month_with_class_labels (1).csv"
OUTPUT_FILE=r"C:/Users/hp/PycharmProjects/FlightFore/datasets/flight_embedded_encoded.csv"

data=pd.read_csv(INPUT_FILE).copy()

for col in data.select_dtypes(include=["object","string"]).columns:
    data[col]=data[col].astype(str).str.strip()

print("Missing Values Before:")
print(data.isnull().sum())
print("Duplicates:",data.duplicated().sum())
data=data.drop_duplicates()

target_col="delay"
if target_col not in data.columns:
    raise ValueError("Target column 'delay' not found.")

num_cols=data.select_dtypes(include=np.number).columns.tolist()
cat_cols=data.select_dtypes(include=["object","string","category"]).columns.tolist()

if target_col in num_cols: num_cols.remove(target_col)
if target_col in cat_cols: cat_cols.remove(target_col)

for c in ["TAIL_NUMBER","FLIGHT_NUMBER"]:
    if c in cat_cols:
        cat_cols.remove(c)

for c in num_cols:
    data[c]=data[c].fillna(data[c].mean())

for c in cat_cols:
    m=data[c].mode()
    if not m.empty:
        data[c]=data[c].fillna(m.iloc[0])

embedding_size=3
np.random.seed(42)
embedding_output=pd.DataFrame(index=data.index)

for col in cat_cols:
    cats=data[col].unique()
    emb={cat:np.random.uniform(-1,1,embedding_size) for cat in cats}
    arr=np.vstack(data[col].map(emb).to_numpy())
    emb_df=pd.DataFrame(arr,columns=[f"Embedding_{col}_{i+1}" for i in range(embedding_size)])
    embedding_output=pd.concat([embedding_output.reset_index(drop=True),emb_df.reset_index(drop=True)],axis=1)

final_output=pd.concat([
    data[num_cols].reset_index(drop=True),
    embedding_output.reset_index(drop=True),
    data[[target_col]].reset_index(drop=True)
],axis=1)

print("Missing After:",final_output.isnull().sum().sum())
print("Final Shape:",final_output.shape)
final_output.to_csv(OUTPUT_FILE,index=False)
print("Saved:",OUTPUT_FILE)