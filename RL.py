from time import time

import torch
import gym
import random
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")  

states = env.observation_space.shape[0]
actions = env.action_space.n

episodes = 1
# score = 0

print("env.action_space.shape:", env.action_space)  

for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # random action
        observation, reward, terminated, truncated, info = env.step(action)
        # score += reward


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
    
