import sys
import numpy as np
import pandas as pd
import gymnasium as gym

from gymnasium import spaces

from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

sys.path.append(
    str(PROJECT_ROOT)
)


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

        self.num_assets = len(
            self.assets
        )

        self.sequence_length = 20

        self.current_step = (
            self.sequence_length
        )

        self.initial_balance = 1.0

        self.portfolio_value = (
            self.initial_balance
        )

        self.data_dir = (
            PROJECT_ROOT /
            "data" /
            "processed"
        )

        self.embedding_path = (
            PROJECT_ROOT /
            "data" /
            "embeddings" /
            "graph_embeddings.csv"
        )

        self.embedding_df = pd.read_csv(
            self.embedding_path
        )

        self.price_data = {}

        for asset in self.assets:

            asset_df = pd.read_csv(
                self.data_dir /
                f"{asset}_merged.csv"
            )

            self.price_data[
                asset
            ] = asset_df

        self.embedding_cols = [

            col
            for col in self.embedding_df.columns
            if "emb_" in col
        ]

        grouped = self.embedding_df.groupby(
            "date"
        )

        self.daily_embeddings = []

        for date, group in grouped:

            if len(group) != 10:

                continue

            group = group.sort_values(
                "ticker"
            )

            embedding_vector = (
                group[
                    self.embedding_cols
                ]
                .values
                .flatten()
            )

            self.daily_embeddings.append(
                embedding_vector
            )

        self.daily_embeddings = np.array(
            self.daily_embeddings,
            dtype=np.float32
        )

        self.num_steps = len(
            self.daily_embeddings
        )

        self.action_space = spaces.Box(

            low=0,

            high=1,

            shape=(
                self.num_assets,
            ),

            dtype=np.float32
        )

        self.observation_space = spaces.Box(

            low=-np.inf,

            high=np.inf,

            shape=(
                self.sequence_length,
                160
            ),

            dtype=np.float32
        )

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        self.current_step = (
            self.sequence_length
        )

        self.portfolio_value = (
            self.initial_balance
        )

        state = self._get_state()

        return state, {}

    def _get_state(self):

        state = self.daily_embeddings[

            self.current_step -
            self.sequence_length:

            self.current_step
        ]

        return state.astype(
            np.float32
        )

    def step(
        self,
        action
    ):

        weights = (
            action /
            (
                np.sum(action)
                + 1e-8
            )
        )

        portfolio_return = 0

        for idx, asset in enumerate(
            self.assets
        ):

            asset_df = self.price_data[
                asset
            ]

            current_close = (
                asset_df.iloc[
                    self.current_step
                ]["close"]
            )

            future_close = (
                asset_df.iloc[
                    self.current_step + 5
                ]["close"]
            )

            asset_return = (
                (
                    future_close -
                    current_close
                )
                /
                current_close
            )

            portfolio_return += (

                weights[idx] *
                asset_return
            )

        self.portfolio_value *= (
            1 + portfolio_return
        )

        reward = (
            portfolio_return
        )

        self.current_step += 1

        terminated = (
            self.current_step >=
            self.num_steps - 5
        )

        truncated = False

        next_state = (
            self._get_state()
        )

        info = {

            "portfolio_value":
            self.portfolio_value,

            "portfolio_return":
            portfolio_return
        }

        return (

            next_state,

            reward,

            terminated,

            truncated,

            info
        )