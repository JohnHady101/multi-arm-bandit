import torch
import torch.nn as nn
import torch.optim as optim
import gym
from torchrl.envs import GymWrapper
from torchrl.modules import QValueActor, MLP
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.collectors import SyncDataCollector


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# 1. Environment
# Using openai gym directly and wrapping it for torchrl compatibility
base_env = gym.make("CartPole-v1")
env = GymWrapper(base_env, device=device)


# 2. Q-network
# make torch nn module
class net(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=20, output_size=1):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.hiddenlayer = torch.nn.Linear(hidden_size, hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        
        
        self._init_weights_gaussian()
    
    def _init_weights_xavier(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def _init_weights_gaussian(self):
        torch.nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.normal_(self.fc2.weight, mean=0, std=0.1)
        torch.nn.init.zeros_(self.fc2.bias)

    def _init_weights_kaiming(self):
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x
    
qnet = net(input_size=4, hidden_size=128, output_size=2).to(device)
actor = QValueActor(qnet, spec=env.action_spec).to(device)

# 3. Replay Buffer
replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(10000),
    batch_size=64,
)


# 4. Data Collector
collector = SyncDataCollector(
    env,
    actor,
    frames_per_batch=1000,
    total_frames=50000,
    device=device,
)


# 5. Loss + Target network
loss_module = DQNLoss(actor, action_space=env.action_spec)
target_updater = SoftUpdate(loss_module, eps=0.995)

optimizer = optim.Adam(loss_module.parameters(), lr=1e-3)

# 6. Training Loop
for i, data in enumerate(collector):

    # store transitions
    replay_buffer.extend(data)

    if len(replay_buffer) < 1000:
        continue

    # sample batch
    batch = replay_buffer.sample()

    # compute loss
    loss = loss_module(batch)
    loss_value = loss["loss"]

    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()

    target_updater.step()

    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss_value.item():.4f}")
# 7. Run and visualize the trained policy
print("\nTesting the trained policy with rendering...")
from tensordict import TensorDict

try:
    # Use standard gym for rendering
    # For gym 0.26.2+, render_mode="human" is the preferred way
    eval_env = gym.make("CartPole-v1", render_mode="human")
    
    # Gym 0.26.2+ API: reset returns (obs, info)
    obs, info = eval_env.reset()
    done = False
    total_reward = 0
    
    with torch.no_grad():
        while not done:
            # Prepare observation for the actor
            obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            td = TensorDict({"observation": obs_tensor}, batch_size=[1])
            
            # Use actor to get action
            actor(td)
            action = td["action"].item()
            
            # Gym 0.26.2+ API: step returns (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # render_mode="human" handles the window display automatically in most cases,
            # but we can call it if needed.
            eval_env.render()
            
    print(f"Rollout complete. Total reward: {total_reward}")
    eval_env.close()

except Exception as e:
    print(f"Could not run with rendering: {e}")
    # Fallback to non-rendered rollout if something goes wrong
    print("Falling back to torchrl rollout...")
    with torch.no_grad():
        rollout = env.rollout(max_steps=1000, policy=actor)
        print(f"Rollout complete. Total reward: {rollout['next', 'reward'].sum().item()}")

# Optional: Display the first few transitions
print("First 5 observations:\n", rollout["observation"][:5])
print("First 5 actions:\n", rollout["action"][:5])


