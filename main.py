from kaggle_environments import make

try:
    env = make("orbit_wars", configuration={"seed": 42}, debug=True)
except Exception as e:
    print(f"Failed to create environment: {e}")
    print("Try: pip install --upgrade kaggle-environments")
    exit(1)

# Inspect the environment spec before running
print("=== ENV SPEC ===")
print("Action Space:", env.specification.action)
print("Observation Space:", env.specification.observation)
print("Configuration:", env.configuration)

try:
    env.run(["my_agent.py", "random"])
except Exception as e:
    print(f"Failed to run environment: {e}")
    exit(1)

# Results
final = env.steps[-1]
print("\n--- Final Results ---")
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

rewards = [s.reward for s in final]
winner = rewards.index(max(rewards))
print(f"\nWinner: Player {winner}")

env.render(mode="ipython", width=800, height=600)