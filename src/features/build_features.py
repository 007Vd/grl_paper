import pandas as pd
import numpy as np
from pathlib import Path
from stockstats import StockDataFrame

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
csv_files = []

for file in RAW_DATA_DIR.glob("*.csv"):

    df = pd.read_csv(file)

    columns = [col.lower() for col in df.columns]

    required_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    if all(col in columns for col in required_cols):

        csv_files.append(file)

def build_technical_features(df):
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    indicator_df = df[[
        "open",
        "close",
        "high",
        "low",
        "volume"
    ]].copy()

    stock_df = StockDataFrame.retype(indicator_df)
    df["sma_5"] = stock_df["close_5_sma"]
    df["sma_10"] = stock_df["close_10_sma"]
    df["ema_5"] = stock_df["close_5_ema"]
    df["ema_10"] = stock_df["close_10_ema"]
    df["rsi_14"] = stock_df["rsi_14"]
    df["macd"] = stock_df["macd"]
    df["cci"] = stock_df["cci"]
    df["adx"] = stock_df["adx"]
    df["kdjk"] = stock_df["kdjk"]

    return df
for file in csv_files:
    print(f"n processing {file.name}")
    try:
        df=pd.read_csv(file)
        feature_df=build_technical_features(df)
        save_name=file.stem+"_features.csv"
        save_path=PROCESSED_DATA_DIR/save_name
        feature_df.to_csv(save_path,index=False)
        print(f"saved:{save_path}")

    except Exception as e:
        print(f"Error PRocessing{file.name}:{e}")