import numpy as np

# 1. Initialize the default random number generator
rng = np.random.default_rng()

TOTAL_REWARD = 0
LIST_TOTAL_REWARDS = [0]
Q = [0, 0, 0]

def choose(i, alpha: float):
    global TOTAL_REWARD
    reward = rng.normal(loc=2*i, scale= 1, size= None)
    Q[i] += Q[i] + alpha * (reward - Q[i])
    TOTAL_REWARD += reward
    LIST_TOTAL_REWARDS.append(TOTAL_REWARD)


class ploitation:
    def __init__(self, method: str, epsilon: float):
        self.method = method
        self.epsilon = epsilon

    def the_action_chosen(self):
        if self.method == "epsilon-greedy":
            return self._epsilon_greedy(self.epsilon)


    def _epsilon_greedy(self, epsilon):
        random_action = np.random.choice(range(len(Q)), size = 1)[0] # let's see if range works or have to make a list
        best_action_at_t = np.argmax(Q)

        chosen = np.random.choice([random_action, best_action_at_t], size = None, p=[epsilon, 1 - epsilon])

        return chosen # it returns an index


p = ploitation("epsilon-greedy", 0.1)

for i in range(500):
    action = p.the_action_chosen()
    choose(action, alpha= 0.3)

N = 10    
LAST_N = []


for i in range(N, 500):
    _cache = 0
    for j in range(i-N, i):
        _cache += LIST_TOTAL_REWARDS[j]
    LAST_N.append(_cache) 



import matplotlib.pyplot as plt


# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(range(len(LAST_N)), LAST_N, marker='o', linestyle='-', color='blue', linewidth=2, markersize=0)

# Add labels and title
plt.xlabel('Index', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('List Values by Index', fontsize=14)

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Show the plot
plt.tight_layout()
plt.show()
