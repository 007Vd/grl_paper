import yfinance as yf
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR=PROJECT_ROOT/"data"/"raw"
print(PROJECT_ROOT)
print(RAW_DATA_DIR)
RAW_DATA_DIR.mkdir(parents=True,exist_ok=True)

START_DATE = "2010-01-01"
END_DATE = "2024-01-01"

# define all required assets
TOP10_STOCK=[
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOG",
    "GOOGL",
    "BRK-B",
    "AMZN",
    "TSLA",
    "JNJ",
    "UNH"]

MARKET_NODES=[
    "^GSPC", #S&P500
    "^N225", #Nikkei 225
    "^FTSE", #FTSE100
    "^TNX", #10Y Treasury Yield
    "^FVX", #5Y Treasury Yield
    "000001.SS",# Shanghai Composite
    "XLK",       # Technology ETF
    "XLF",       # Financial ETF
    "XLY",       # Consumer Discretionary ETF
    "XLV"        # Healthcare ETF
     ]

ALL_ASSETS=TOP10_STOCK+MARKET_NODES

# asset downloading
all_data={}

for ticker in ALL_ASSETS:
    print(f"downloading {ticker}...")
    try:
        df=yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False
        )
        if df.empty:
            print(f"WARNING: NO DATA FOUND FOR {ticker}")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
             # name index properly
            df.index.name = "Date"
        df["Ticker"]=ticker
        all_data[ticker]=df
        print(f"{ticker} downloaded :{df.shape}")

    except Exception as e:
        print(f"ERROR downlaoding {ticker}: {e}")


# raw csv files path 

for ticker,df in all_data.items():
    safe_name=ticker.replace("^","")
    save_path=RAW_DATA_DIR/f"{safe_name}.csv"

    df.to_csv(save_path)
    print(f"saved {save_path}")

# draw downlaods sumary

print("\n DOWNOAD SUMMARY\n")
for ticker,df in all_data.items():
    print(f"{ticker}:{df.shape}")

