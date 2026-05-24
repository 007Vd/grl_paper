import pandas as pd
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
RAW_DATA_DIR=PROJECT_ROOT/"data"/"raw"
files=list(RAW_DATA_DIR.glob("*.csv"))
print(f"total files present :{len(files)}")

for file in files:
    print("\n")
    print(f"FILE:{file.name}\n")

    df = pd.read_csv(file, parse_dates=["Date"])  
    print(df.head(5))

    print(f"Columns:\n{df.columns}")
    print(df["Date"].min())
    print(df["Date"].max())


    print(f"\nSHAPE:\n{df.shape}")
    print(f"\nNULLS:\n{df.isnull().sum()}")
    print(f"\nTYPES:\n{df.dtypes}")
    print("\nDATE COUNT:")
    print(df["Date"].nunique())



summary=[]

print("SUMMARY")

for file in files:
    df=pd.read_csv(file)
    summary.append({
        "file":file.name,
        "rows":len(df),
        "nulls":df.isnull().sum().sum()
    })

summary_df = pd.DataFrame(summary)

print(summary_df)