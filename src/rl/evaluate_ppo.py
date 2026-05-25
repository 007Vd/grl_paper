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
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from portfolio_env import PortfolioEnv

from src.models.graphsage_extractor import (
    GraphSAGEExtractor
)


env = PortfolioEnv(
    start_idx=2800,
    end_idx=3400
)

model = PPO.load(
    "ppo_portfolio_agent",
    env=env
)

state, info = env.reset()

portfolio_values = []

rewards = []

done = False

while not done:

    action, _ = model.predict(
        state,
        deterministic=True
    )

    next_state, reward, terminated, truncated, info = env.step(action)

    portfolio_values.append(
        info["portfolio_value"]
    )

    rewards.append(reward)

    state = next_state

    done = terminated or truncated

print("\nFINAL PORTFOLIO VALUE:\n")

print(portfolio_values[-1])

total_return = (
    portfolio_values[-1] - 1
)

volatility = np.std(rewards)

sharpe_ratio = (
    np.mean(rewards)
    /
    (np.std(rewards) + 1e-8)
)

portfolio_array = np.array(
    portfolio_values
)

running_max = np.maximum.accumulate(
    portfolio_array
)

drawdowns = (
    portfolio_array -
    running_max
) / running_max

max_drawdown = drawdowns.min()

print("\nTOTAL RETURN:\n")
print(total_return)

print("\nVOLATILITY:\n")
print(volatility)

print("\nSHARPE RATIO:\n")
print(sharpe_ratio)

print("\nMAX DRAWDOWN:\n")
print(max_drawdown)

plt.figure(figsize=(12,6))

plt.plot(portfolio_values)

plt.title(
    "PPO Portfolio Value Over Time"
)

plt.xlabel("Timestep")

plt.ylabel("Portfolio Value")

plt.grid(True)

plt.savefig(
    "portfolio_value.png"
)

plt.close()

plt.figure(figsize=(12,6))

plt.plot(rewards)

plt.title(
    "PPO Reward Dynamics"
)

plt.xlabel("Timestep")

plt.ylabel("Reward")

plt.grid(True)

plt.savefig(
    "reward_dynamics.png"
)

plt.close()