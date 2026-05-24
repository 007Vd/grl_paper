import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT/"data"/"processed"

gspc_path=PROCESSED_DATA_DIR/"GSPC_merged.csv"
df=pd.read_csv(gspc_path)

df["date"] = pd.to_datetime(df["date"])
df["future_5day_mean"]=df["close"].shift(-5).rolling(5).mean()

df["target"]=(df["future_5day_mean"]>df["close"]).astype(int)
df = df.dropna()
print(df[[ "close","future_5day_mean","target"]].tail(10))

print("\nTARGET DISTRIBUTION:\n")
print(
    df["target"].value_counts(normalize=True)
)

save_path=PROCESSED_DATA_DIR/"GSPC_targeted.csv"
df.to_csv(save_path,index=False)
print(f"saved:{save_path}")