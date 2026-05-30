import sys
import torch
import pandas as pd
import numpy as np

from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

sys.path.append(
    str(PROJECT_ROOT)
)

from src.models.graphsage_model import GraphSAGE


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
    "sma_5"
]

ALL_NODES = [
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

TRADABLE_ASSETS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOG",
    "GOOGL",
    "AMZN",
    "TSLA",
    "BRK-B",
    "JNJ",
    "UNH"
]


PROCESSED_DATA_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed"
)

GRAPH_MODEL_PATH = (
    PROJECT_ROOT /
    "data" /
    "graph" /
    "graphsage_model.pth"
)

EMBEDDINGS_DIR = (
    PROJECT_ROOT /
    "data" /
    "embeddings"
)

EMBEDDINGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


price_data = {}

for node in ALL_NODES:

    file_path = (
        PROCESSED_DATA_DIR /
        f"{node}_merged.csv"
    )

    df = pd.read_csv(file_path)

    price_data[node] = df


graphsage = GraphSAGE(
    input_dim=32,
    hidden_dim=32,
    output_dim=16
)

graphsage.load_state_dict(
    torch.load(
        GRAPH_MODEL_PATH,
        map_location=torch.device("cpu")
    )
)

graphsage.eval()


node_to_idx = {
    node: idx
    for idx, node in enumerate(ALL_NODES)
}


EDGES = [

    ("AAPL", "XLK"),
    ("MSFT", "XLK"),
    ("NVDA", "XLK"),
    ("GOOG", "XLK"),
    ("GOOGL", "XLK"),
    ("AMZN", "XLK"),

    ("JNJ", "XLV"),
    ("UNH", "XLV"),

    ("AMZN", "XLY"),
    ("TSLA", "XLY"),

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

    ("TNX", "GSPC"),
    ("FVX", "GSPC"),

    ("N225", "GSPC"),
    ("FTSE", "GSPC")
]

edges = []

for src, dst in EDGES:

    edges.append((
        node_to_idx[src],
        node_to_idx[dst]
    ))

    edges.append((
        node_to_idx[dst],
        node_to_idx[src]
    ))

edge_index = torch.tensor(
    edges,
    dtype=torch.long
).t().contiguous()


num_timesteps = min(
    len(df)
    for df in price_data.values()
)

all_rows = []


for timestep in range(num_timesteps - 5):

    if timestep % 100 == 0:

        print(
            f"processing timestep "
            f"{timestep}/{num_timesteps}"
        )

    node_features = []

    for node in ALL_NODES:

        df = price_data[node]

        window_size = 30

        start_idx = max(
            0,
            timestep - window_size
        )

        window_df = df.iloc[
            start_idx:timestep + 1
        ]

        feature_means = (
            window_df[TOP_FEATURES]
            .mean()
            .values
        )

        feature_stds = (
            window_df[TOP_FEATURES]
            .std()
            .fillna(0)
            .values
        )

        feature_vector = np.concatenate([
            feature_means,
            feature_stds
        ]).astype(np.float32)

        node_features.append(
            feature_vector
        )

    node_features = np.array(
        node_features,
        dtype=np.float32
    )

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    with torch.no_grad():

        embeddings = graphsage(
            x,
            edge_index
        )

    embeddings = embeddings.numpy()

    for asset in TRADABLE_ASSETS:

        asset_idx = node_to_idx[asset]

        asset_embedding = embeddings[
            asset_idx
        ]

        asset_df = price_data[asset]

        current_close = (
            asset_df.iloc[timestep]["close"]
        )

        future_close = (
            asset_df.iloc[
                timestep + 5
            ]["close"]
        )

        target = int(
            future_close >
            current_close
        )

        row_dict = {

            "date":
            asset_df.iloc[timestep]["date"],

            "ticker":
            asset,

            "target":
            target
        }

        for emb_idx in range(16):

            row_dict[
                f"emb_{emb_idx}"
            ] = asset_embedding[
                emb_idx
            ]

        all_rows.append(
            row_dict
        )


embedding_df = pd.DataFrame(
    all_rows
)

save_path = (
    EMBEDDINGS_DIR /
    "graph_embeddings.csv"
)

embedding_df.to_csv(
    save_path,
    index=False
)

print("\nEMBEDDINGS GENERATED\n")

print(
    f"SAVED TO:\n{save_path}"
)

print("\nDATASET SHAPE:\n")

print(
    embedding_df.shape
)

print("\nHEAD:\n")

print(
    embedding_df.head()
)