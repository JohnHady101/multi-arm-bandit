import numpy as np

# 1. Initialize the default random number generator
rng = np.random.default_rng()
AVG_REWARDS = (
    [0],
    [0]
)


Q = (
    [0, 0, 0],
    [3, 3, 3]
)

EPSILON = (
    0.1,
    0.1
)

def choose(action, q, avg_rewards):
    reward = rng.normal(loc=2*action, scale= 1, size= None)
    q[action] += q[action] + (1 / len(avg_rewards)) * (reward - q[action])
    avg_rewards.append(avg_rewards[-1] + (1 / len(avg_rewards)) * (reward - avg_rewards[-1]))
    


class ploitation:
    def __init__(self, method: str, epsilon: float):
        self.method = method
        self.epsilon = epsilon

    def the_action_chosen(self, q):
        if self.method == "epsilon-greedy":
            return self._epsilon_greedy(self.epsilon, q)


    def _epsilon_greedy(self, epsilon, q):
        random_action = np.random.choice(range(len(q)), size = 1)[0] # let's see if range works or have to make a list
        best_action_at_t = np.argmax(q)

        chosen = np.random.choice([random_action, best_action_at_t], size = None, p=[epsilon, 1 - epsilon])

        return chosen # it returns an index


p = (
    ploitation("epsilon-greedy", epsilon= EPSILON[0]),
    ploitation("epsilon-greedy", epsilon= EPSILON[1])

)

for i in range(500):
    action= [0, 0]
    action[0] = p[0].the_action_chosen(Q[0])
    action[1] = p[1].the_action_chosen(Q[1])
    choose(action = action[0], q = Q[0], avg_rewards= AVG_REWARDS[0])
    choose(action = action[1], q = Q[1], avg_rewards= AVG_REWARDS[1])



import matplotlib.pyplot as plt


# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(range(len(AVG_REWARDS[0])), AVG_REWARDS[0], marker='o', linestyle='-', color='blue', linewidth=2, markersize=0)
plt.plot(range(len(AVG_REWARDS[1])), AVG_REWARDS[1], marker='o', linestyle='-', color='red', linewidth=2, markersize=0)


# Add labels and title
plt.xlabel('Index', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('List Values by Index', fontsize=14)

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()
