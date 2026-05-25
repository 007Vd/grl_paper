import sys

from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

sys.path.append(
    str(PROJECT_ROOT)
)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from portfolio_env import PortfolioEnv

from src.models.graphsage_extractor import (
    GraphSAGEExtractor
)


env = PortfolioEnv(
    start_idx=0,
    end_idx=2200
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./checkpoints/",
    name_prefix="ppo_grl"
)

model = PPO(

    policy="MlpPolicy",

    env=env,

    learning_rate=1e-4,

    n_steps=32,

    batch_size=8,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.5,

    ent_coef=0.01,

    verbose=1,

    tensorboard_log="./ppo_logs/",

    policy_kwargs=dict(

        features_extractor_class=
        GraphSAGEExtractor,

        features_extractor_kwargs=dict(
            features_dim=128
        )
    )
)

model.learn(
    total_timesteps=500000,
    callback=checkpoint_callback
)

model.save(
    "ppo_portfolio_agent"
)

print("\nTRAINING COMPLETE\n")

print(
    "MODEL SAVED AS: "
    "ppo_portfolio_agent.zip"
)