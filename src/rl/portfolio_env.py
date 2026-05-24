import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from gymnasium import spaces
from pathlib import Path
from torch_geometric.data import Data
import sys
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


class PortfolioEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.assets = [
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

        self.num_assets = len(self.assets)

        self.embedding_dim = 16

        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_assets,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.num_assets,
                self.embedding_dim
            ),
            dtype=np.float32
        )

        PROJECT_ROOT = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

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

        self.price_data = {}

        for node in ALL_NODES:

            file_path = (
                PROCESSED_DATA_DIR /
                f"{node}_merged.csv"
            )

            df = pd.read_csv(file_path)

            self.price_data[node] = df

        returns = []

        min_length = float("inf")

        for asset in self.assets:

            df = self.price_data[asset]

            asset_returns = (
                df["close"]
                .pct_change()
                .fillna(0)
                .values
            )

            returns.append(asset_returns)

            min_length = min(
                min_length,
                len(asset_returns)
            )

        aligned_returns = []

        for asset_returns in returns:

            aligned_returns.append(
                asset_returns[-min_length:]
            )

        self.returns_matrix = np.array(
            aligned_returns
        ).T

        print(
            "\nRETURNS MATRIX SHAPE:\n"
        )

        print(
            self.returns_matrix.shape
        )

        self.graphsage = GraphSAGE(
            input_dim=16,
            hidden_dim=32,
            output_dim=16
        )

        self.graphsage.load_state_dict(
            torch.load(
                GRAPH_MODEL_PATH,
                map_location=torch.device("cpu")
            )
        )

        self.graphsage.eval()

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

        self.edge_index = edge_index

        self.current_step = 0

        self.portfolio_value = 1.0

        self.max_steps = min_length - 1

    def _get_state(self):

        node_features = []

        for node in ALL_NODES:

            df = self.price_data[node]

            row = df.iloc[self.current_step]

            feature_vector = (
                row[TOP_FEATURES]
                .values
                .astype(np.float32)
            )

            node_features.append(
                feature_vector
            )

        x = torch.tensor(
            node_features,
            dtype=torch.float
        )

        with torch.no_grad():

            embeddings = self.graphsage(
                x,
                self.edge_index
            )

        tradable_embeddings = embeddings[
            :self.num_assets
        ]

        return tradable_embeddings.numpy()

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        self.current_step = 0

        self.portfolio_value = 1.0

        state = self._get_state()

        info = {}

        return state, info

    def step(self, action):

        self.current_step += 1

        action = np.clip(
            action,
            0,
            1
        )

        action = action / (
            action.sum() + 1e-8
        )

        asset_returns = (
            self.returns_matrix[
                self.current_step
            ]
        )

        reward = np.dot(
            action,
            asset_returns
        )

        self.portfolio_value *= (
            1 + reward
        )

        next_state = self._get_state()

        terminated = (
            self.current_step >=
            self.max_steps
        )

        truncated = False

        info = {
            "portfolio_value":
            self.portfolio_value
        }

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info
        )