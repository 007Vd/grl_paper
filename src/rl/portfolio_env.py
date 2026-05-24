import gymnasium as gym
import numpy as np
import pandas as pd

from gymnasium import spaces
from pathlib import Path


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

        self.price_data = {}

        for asset in self.assets:

            file_path = (
                PROCESSED_DATA_DIR /
                f"{asset}_merged.csv"
            )

            df = pd.read_csv(file_path)

            self.price_data[asset] = df

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

                min_length = min(min_length,len(asset_returns))


        aligned_returns = []
        for asset_returns in returns:
              aligned_returns.append(asset_returns[-min_length:])
        self.returns_matrix = np.array(aligned_returns).T

        print(
            "\nRETURNS MATRIX SHAPE:\n"
        )

        print(
            self.returns_matrix.shape
        )

        self.current_step = 0

        self.portfolio_value = 1.0

    def _get_state(self):

        return np.random.randn(
            self.num_assets,
            self.embedding_dim
        ).astype(np.float32)

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

        action = np.clip(action, 0, 1)

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
            len(self.returns_matrix) - 1
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