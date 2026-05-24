import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from portfolio_env import PortfolioEnv

env = PortfolioEnv()
model = PPO.load("ppo_portfolio_agent")

state, info = env.reset()

portfolio_values = []
rewards = []
done = False

while not done:
    action, _ = model.predict(state, deterministic=True)
    next_state, reward, terminated, truncated, info = env.step(action)
    portfolio_values.append(info["portfolio_value"])
    rewards.append(reward)
    state=next_state
    done = terminated or truncated

print("\nFINAL PORTFOLIO VALUE:\n")

print(portfolio_values[-1])

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