import sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from stable_baselines3 import PPO

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

sys.path.append(
    str(PROJECT_ROOT)
)

from portfolio_env import PortfolioEnv


MODEL_PATH = (
    PROJECT_ROOT /
    "data" /
    "rl_models" /
    "ppo_portfolio.zip"
)

env = PortfolioEnv()

model = PPO.load(
    MODEL_PATH
)

obs, _ = env.reset()

portfolio_values = []

rewards = []

done = False

while not done:

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    done = (
        terminated or
        truncated
    )

    portfolio_values.append(

        info["portfolio_value"]
    )

    rewards.append(
        reward
    )

portfolio_values = np.array(
    portfolio_values
)

rewards = np.array(
    rewards
)

total_return = (
    portfolio_values[-1] - 1
)

volatility = np.std(
    rewards
)

sharpe_ratio = (
    np.mean(rewards)
    /
    (
        volatility + 1e-8
    )
)

running_max = np.maximum.accumulate(
    portfolio_values
)

drawdowns = (
    portfolio_values -
    running_max
) / running_max

max_drawdown = drawdowns.min()

print("\nFINAL PORTFOLIO VALUE:\n")

print(
    portfolio_values[-1]
)

print("\nTOTAL RETURN:\n")

print(
    total_return
)

print("\nVOLATILITY:\n")

print(
    volatility
)

print("\nSHARPE RATIO:\n")

print(
    sharpe_ratio
)

print("\nMAX DRAWDOWN:\n")

print(
    max_drawdown
)

plt.figure(figsize=(12,6))

plt.plot(portfolio_values)

plt.title(
    "PPO Portfolio Value Over Time"
)

plt.xlabel("Timestep")

plt.ylabel("Portfolio Value")

plt.grid(True)

plt.show()

plt.figure(figsize=(12,6))

plt.plot(rewards)

plt.title(
    "PPO Reward Dynamics"
)

plt.xlabel("Timestep")

plt.ylabel("Reward")

plt.grid(True)

plt.show()