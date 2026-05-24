from stable_baselines3 import PPO
from portfolio_env import PortfolioEnv

env = PortfolioEnv()
model = PPO(policy="MlpPolicy",env=env,verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_portfolio_agent")
