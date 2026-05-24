import pandas as pd
from fredapi import Fred
from pathlib import Path
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
load_dotenv(PROJECT_ROOT/".env")
FRED_API_KEY=os.getenv("FRED_API_KEY")

fred=Fred(api_key=FRED_API_KEY)

FRED_FEATURES = [
    "RECPROUSM156N",
    "CORESTICKM159SFRBATL",
    "PCETRIM12M159SFRBDAL",
    "CPALTT01USM657N",
    "PSAVERT",
    "AISRSA",
    "ANFCI",
    "UNEMPLOY"
]
macro_data={}
for feature in FRED_FEATURES:
    print(f"downlaoding {feature}")
    try:
        series=fred.get_series(feature)
        df=pd.DataFrame(series,columns=[feature])
        df.index.name="Date"
        macro_data[feature]=df
        print(f"{feature}: {df.shape}")
    except Exception as e:
        print(f"Error downloading {feature}: {e}")


for feature, df in macro_data.items():
    save_path = RAW_DATA_DIR / f"{feature}.csv"
    df.to_csv(save_path)
    print(f"saved {save_path}")