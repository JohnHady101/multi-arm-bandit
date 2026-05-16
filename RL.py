import torch
import gym
import random
import numpy as np

env = gym.make("CartPole-v1")

states = env.observation_space.shape[0]

actions = env.action_space.n

episodes = 10

# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False 
#     score = 0 
    
#     while not done:
#         env.render()
#         action = random.choice([0,1])
#         observation, reward, terminated, truncated, info = env.step(action)
#         score+=reward
#     print('Episode:{} Score:{}'.format(episode, score))

# make a torch nn that takes state as input
model = torch.nn.Sequential(
    torch.nn.Linear(states, 24),
    torch.nn.ReLU(),
    torch.nn.Linear(24, 24),
    torch.nn.ReLU(),
    torch.nn.Linear(24, actions)
)

# make neural network class
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.network = model

    def forward(self, x):
        return self.network(x)
    
# make a function that takes state as input and returns action
def get_action(state):
    state = torch.FloatTensor(state)
    action = model(state)
    action = torch.argmax(action).item()
    return action

x = torch.tensor([1.0, 0.0, 0.0, 0.0])

y = NeuralNetwork()(x) 

print(y)