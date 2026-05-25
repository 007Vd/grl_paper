import gymnasium as gym
import numpy as np
import pandas as pd

from gymnasium import spaces
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

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

    def __init__(
        self,
        start_idx=0,
        end_idx=None
    ):

        super().__init__()

        self.start_idx = start_idx
        self.end_idx = end_idx

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

        self.feature_dim = len(TOP_FEATURES)

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
                self.feature_dim
            ),
            dtype=np.float32
        )

        PROCESSED_DATA_DIR = (
            PROJECT_ROOT /
            "data" /
            "processed"
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

        if self.end_idx is None:

            self.end_idx = len(
                self.returns_matrix
            )

        self.returns_matrix = (
            self.returns_matrix[
                self.start_idx:self.end_idx
            ]
        )

        print(
            "\nRETURNS MATRIX SHAPE:\n"
        )

        print(
            self.returns_matrix.shape
        )

        self.current_step = 0

        self.portfolio_value = 1.0

        self.transaction_cost_rate = 0.001

        self.risk_penalty_rate = 0.001

        self.previous_weights = np.ones(
            self.num_assets
        ) / self.num_assets

        self.portfolio_returns_history = []

        self.max_steps = (
            len(self.returns_matrix) - 1
        )

    def _get_state(self):

        node_features = []

        for node in ALL_NODES:

            df = self.price_data[node]

            actual_idx = (
                self.start_idx +
                self.current_step
            )

            row = df.iloc[actual_idx]

            feature_vector = (
                row[TOP_FEATURES]
                .values
                .astype(np.float32)
            )

            node_features.append(
                feature_vector
            )

        node_features = np.array(
            node_features,
            dtype=np.float32
        )

        tradable_nodes = node_features[
            :self.num_assets
        ]

        return tradable_nodes

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        self.current_step = 0

        self.portfolio_value = 1.0

        self.previous_weights = np.ones(
            self.num_assets
        ) / self.num_assets

        self.portfolio_returns_history = []

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

        portfolio_return = np.dot(
            action,
            asset_returns
        )

        self.portfolio_returns_history.append(
            portfolio_return
        )

        if len(
            self.portfolio_returns_history
        ) >= 20:

            volatility = np.std(
                self.portfolio_returns_history[-20:]
            )

        else:

            volatility = 0

        turnover = np.sum(
            np.abs(
                action -
                self.previous_weights
            )
        )

        transaction_cost = (
            self.transaction_cost_rate *
            turnover
        )

        risk_penalty = (
            self.risk_penalty_rate *
            volatility
        )

        reward = (
            portfolio_return
            - transaction_cost
            - risk_penalty
        )

        self.previous_weights = action.copy()

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
            self.portfolio_value,

            "portfolio_return":
            portfolio_return,

            "transaction_cost":
            transaction_cost,

            "turnover":
            turnover,

            "volatility":
            volatility,

            "risk_penalty":
            risk_penalty
        }

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info
        )