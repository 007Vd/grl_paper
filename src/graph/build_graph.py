import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch_geometric.data import Data

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_DATA_DIR    = PROJECT_ROOT / "data" / "graph"
GRAPH_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
WINDOW = 30          # rolling days used as temporal context
CORR_THRESHOLD = 0.5 # minimum |correlation| to add an edge

TOP_FEATURES = [
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
    "sma_5",
]

NODES = [
    "AAPL", "MSFT", "NVDA", "GOOG", "GOOGL",
    "AMZN", "TSLA", "BRK-B", "JNJ", "UNH",
    "XLK",  "XLF",  "XLY",  "XLV",
    "GSPC", "N225", "FTSE", "TNX", "FVX",
]

# ─────────────────────────────────────────────
#  1. Load data for every node
# ─────────────────────────────────────────────
dataframes: dict[str, pd.DataFrame] = {}

for node in NODES:
    path = PROCESSED_DATA_DIR / f"{node}_merged.csv"
    df   = pd.read_csv(path)

    # Ensure features exist; fill missing macro cols with forward-fill
    for feat in TOP_FEATURES:
        if feat not in df.columns:
            df[feat] = np.nan
    df[TOP_FEATURES] = df[TOP_FEATURES].ffill().bfill()

    dataframes[node] = df
    print(f"[load] {node:8s}  rows={len(df)}")

# ─────────────────────────────────────────────
#  2. Build node feature matrix  (shape: N × (WINDOW × F))
#     Each node → last WINDOW rows flattened, then mean+std summary
#     appended for richer representation.
# ─────────────────────────────────────────────
node_features = []

for node in NODES:
    df     = dataframes[node]
    window = df[TOP_FEATURES].iloc[-WINDOW:].values          # (WINDOW, F)

    if len(window) < WINDOW:
        # Pad with zeros if history is shorter than WINDOW
        pad    = np.zeros((WINDOW - len(window), len(TOP_FEATURES)))
        window = np.vstack([pad, window])

    # Normalize within the window (z-score per feature)
    mean = window.mean(axis=0, keepdims=True)
    std  = window.std(axis=0, keepdims=True) + 1e-8
    window_norm = (window - mean) / std

    # Summary stats (mean + std across window) → shape (2*F,)
    summary = np.concatenate([window_norm.mean(0), window_norm.std(0)])

    node_features.append(summary)

x = torch.tensor(np.array(node_features), dtype=torch.float)

print(f"\n[features] Node feature matrix shape : {x.shape}")
print(f"           Features per node          : {x.shape[1]}  "
      f"(mean+std of {WINDOW} days × {len(TOP_FEATURES)} features)")

# ─────────────────────────────────────────────
#  3. Correlation-based edges  (data-driven)
# ─────────────────────────────────────────────
# Use pct_change of adj_close over the full history
returns: dict[str, pd.Series] = {}
for node in NODES:
    df = dataframes[node]
    returns[node] = (
        df["adj_close"]
        .pct_change()
        .dropna()
        .reset_index(drop=True)
    )

# Align all series to the shortest common length
min_len = min(len(s) for s in returns.values())
returns_df = pd.DataFrame(
    {node: s.iloc[-min_len:].values for node, s in returns.items()},
    columns=NODES,
)
corr_matrix = returns_df.corr()

edge_src, edge_dst, edge_weights = [], [], []

for i, n1 in enumerate(NODES):
    for j, n2 in enumerate(NODES):
        if i >= j:
            continue
        corr = corr_matrix.loc[n1, n2]
        if pd.isna(corr):
            continue
        if abs(corr) >= CORR_THRESHOLD:
            # Bidirectional
            edge_src     += [i, j]
            edge_dst     += [j, i]
            edge_weights += [float(corr), float(corr)]

# ─────────────────────────────────────────────
#  4. Sector / domain hard edges  (structural prior)
#     These supplement the correlation edges, especially
#     for pairs that may be below the threshold but are
#     known to be structurally related.
# ─────────────────────────────────────────────
node_to_idx = {node: idx for idx, node in enumerate(NODES)}

HARD_EDGES = [
    # Tech ETF
    ("AAPL",  "XLK"), ("MSFT",  "XLK"), ("NVDA",  "XLK"),
    ("GOOG",  "XLK"), ("GOOGL", "XLK"), ("AMZN",  "XLK"),
    # Finance ETF
    ("BRK-B", "XLF"),
    # Consumer ETF
    ("AMZN",  "XLY"), ("TSLA",  "XLY"),
    # Healthcare ETF
    ("JNJ",   "XLV"), ("UNH",   "XLV"),
    # GOOG / GOOGL are the same company
    ("GOOG",  "GOOGL"),
    # Macro / index connections
    ("TNX",   "GSPC"), ("FVX",   "GSPC"),
    ("N225",  "GSPC"), ("FTSE",  "GSPC"),
]

existing_edges = set(zip(edge_src, edge_dst))

for src_name, dst_name in HARD_EDGES:
    i = node_to_idx[src_name]
    j = node_to_idx[dst_name]
    for a, b in [(i, j), (j, i)]:
        if (a, b) not in existing_edges:
            edge_src.append(a)
            edge_dst.append(b)
            edge_weights.append(1.0)   # neutral weight for structural edges
            existing_edges.add((a, b))

# ─────────────────────────────────────────────
#  5. Assemble PyG Data object
# ─────────────────────────────────────────────
edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).contiguous()
edge_attr  = torch.tensor(edge_weights, dtype=torch.float)

graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

print(f"\n[graph]  Nodes        : {graph_data.num_nodes}")
print(f"         Edges (dir.) : {graph_data.num_edges}")
print(f"         Node feat dim: {graph_data.num_node_features}")
print(f"         Edge attr dim: {edge_attr.shape}")
print(f"\n{graph_data}")

# ─────────────────────────────────────────────
#  6. Save
# ─────────────────────────────────────────────
save_path = GRAPH_DATA_DIR / "financial_graph.pt"
torch.save(graph_data, save_path)
print(f"\n[saved]  {save_path}")

# ─────────────────────────────────────────────
#  7. Also save a snapshot list for temporal GNN usage
#     Each snapshot = one trading day's graph state
# ─────────────────────────────────────────────
SNAPSHOT_WINDOW = 60          # how many days of snapshots to generate
snapshots = []

# Find the common date range across all nodes
min_rows = min(len(dataframes[n]) for n in NODES)
SNAPSHOT_START = max(WINDOW, min_rows - SNAPSHOT_WINDOW)

for t in range(SNAPSHOT_START, min_rows):
    snap_features = []
    for node in NODES:
        df = dataframes[node]
        # Slice the window ending at timestep t
        start = max(0, t - WINDOW)
        w = df[TOP_FEATURES].iloc[start:t].values

        if len(w) < WINDOW:
            pad = np.zeros((WINDOW - len(w), len(TOP_FEATURES)))
            w   = np.vstack([pad, w])

        mean = w.mean(axis=0, keepdims=True)
        std  = w.std(axis=0, keepdims=True) + 1e-8
        w_norm = (w - mean) / std
        summary = np.concatenate([w_norm.mean(0), w_norm.std(0)])
        snap_features.append(summary)

    snap_x    = torch.tensor(np.array(snap_features), dtype=torch.float)
    snap_data = Data(x=snap_x, edge_index=edge_index, edge_attr=edge_attr)
    snapshots.append(snap_data)

snap_save_path = GRAPH_DATA_DIR / "financial_graph_snapshots.pt"
torch.save(snapshots, snap_save_path)
print(f"[saved]  {len(snapshots)} temporal snapshots → {snap_save_path}")
