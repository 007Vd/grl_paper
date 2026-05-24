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

print("\nTERMINATED:\n")
print(terminated)

print("\nTRUNCATED:\n")
print(truncated)