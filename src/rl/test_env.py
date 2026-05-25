from portfolio_env import PortfolioEnv

env = PortfolioEnv()

state, info = env.reset()

print("\nINITIAL STATE SHAPE:\n")
print(state.shape)

action = env.action_space.sample()

print("\nSAMPLED ACTION:\n")
print(action)

next_state, reward, terminated, truncated, info = env.step(action)

print("\nNEXT STATE SHAPE:\n")
print(next_state.shape)

print("\nREWARD:\n")
print(reward)

print("\nPORTFOLIO VALUE:\n")
print(info["portfolio_value"])

print("\nPORTFOLIO RETURN:\n")
print(info["portfolio_return"])

print("\nTRANSACTION COST:\n")
print(info["transaction_cost"])

print("\nTURNOVER:\n")
print(info["turnover"])

print("\nVOLATILITY:\n")
print(info["volatility"])

print("\nRISK PENALTY:\n")
print(info["risk_penalty"])

print("\nTERMINATED:\n")
print(terminated)

print("\nTRUNCATED:\n")
print(truncated)