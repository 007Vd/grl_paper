import sys
import torch

from pathlib import Path

from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import (
    DummyVecEnv
)

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

sys.path.append(
    str(PROJECT_ROOT)
)

from portfolio_env import PortfolioEnv


env = DummyVecEnv([

    lambda: PortfolioEnv()
])


policy_kwargs = dict(

    activation_fn=torch.nn.ReLU,

    net_arch=dict(

        pi=[256, 128],

        vf=[256, 128]
    )
)

model = PPO(

    "MlpPolicy",

    env,

    learning_rate=0.0003,

    n_steps=2048,

    batch_size=256,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.005,

    verbose=1,

    policy_kwargs=policy_kwargs
)

model.learn(

    total_timesteps=3000000
)

MODEL_PATH = (
    PROJECT_ROOT /
    "data" /
    "rl_models" /
    "ppo_portfolio"
)

MODEL_PATH.parent.mkdir(

    parents=True,

    exist_ok=True
)

model.save(
    MODEL_PATH
)

print(
    f"\nMODEL SAVED TO:\n"
    f"{MODEL_PATH}"
)