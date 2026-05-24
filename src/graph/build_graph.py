import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DATA_DIR=PROJECT_ROOT/"data"/"graph"

GRAPH_DATA_DIR.mkdir(parents=True,exist_ok=True)

TOP_FEATURES=[
    "trimmed_pce",
    "adj_close",
    "sma_10",
    "rsi_14",
    "personal_savings_rate",
    "macd",
    "open",
    "real_auto_sales",
    "ema_10",
    "cpi",
    "volume",
    "financial_conditions",
    "core_sticky_inflation",
    "recession_probability",
    "ema_5",
    "sma_5"
]

NODES=[
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOG",
    "GOOGL",
    "AMZN",
    "TSLA",
    "BRK-B",
    "JNJ",
    "UNH",
    "XLK",
    "XLF",
    "XLY",
    "XLV",
    "GSPC",
    "N225",
    "FTSE",
    "TNX",
    "FVX"
]

node_features=[]
for node in NODES:
    file_path=PROCESSED_DATA_DIR/f"{node}_merged.csv"
    df=pd.read_csv(file_path)
    latest_row=df.iloc[-1]
    feature_vector=latest_row[TOP_FEATURES].values
    node_features.append(feature_vector)

x=torch.tensor(node_features,dtype=torch.float)

print("\nNode Feature Matrix\n")
print(x)
print("\nSHAPE\n")
print(x.shape)


node_to_idx={
    node:idx
    for idx, node in enumerate(NODES)
    }
EDGES = [

    # Tech stocks ↔ XLK
    ("AAPL", "XLK"),
    ("MSFT", "XLK"),
    ("NVDA", "XLK"),
    ("GOOG", "XLK"),
    ("GOOGL", "XLK"),
    ("AMZN", "XLK"),

    # Healthcare
    ("JNJ", "XLV"),
    ("UNH", "XLV"),

    # Consumer
    ("AMZN", "XLY"),
    ("TSLA", "XLY"),

    # All major stocks ↔ GSPC
    ("AAPL", "GSPC"),
    ("MSFT", "GSPC"),
    ("NVDA", "GSPC"),
    ("GOOG", "GSPC"),
    ("GOOGL", "GSPC"),
    ("AMZN", "GSPC"),
    ("TSLA", "GSPC"),
    ("BRK-B", "GSPC"),
    ("JNJ", "GSPC"),
    ("UNH", "GSPC"),

    # Treasury influence
    ("TNX", "GSPC"),
    ("FVX", "GSPC"),

    # International indices
    ("N225", "GSPC"),
    ("FTSE", "GSPC")
]
edges=[]
for src,dst in EDGES:
    edges.append((
        node_to_idx[src],node_to_idx[dst]
    ))
    edges.append((node_to_idx[dst],node_to_idx[src]
    ))

edge_index=torch.tensor(edges,dtype=torch.long).t().contiguous()
print("\nEDGE INDEX:\n")
print(edge_index)
print("\nEDGE SHAPE:\n")
print(edge_index.shape)

graph_data=Data(x=x,edge_index=edge_index)
print("\nGRAPH OBJECT:\n")
print(graph_data)

save_path=GRAPH_DATA_DIR/"financial_graph.pt"
torch.save(graph_data,save_path)

print(f"graph saved to{save_path}")