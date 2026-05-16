import torch
import torch.nn as nn
import torch.optim as optim

from torchrl.envs import GymEnv
from torchrl.modules import QValueActor, MLP
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.collectors import SyncDataCollector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Environment
env = GymEnv("CartPole-v1", device=device)

# 2. Q-network
net = MLP(
    in_features=env.observation_spec.shape[-1],
    out_features=env.action_spec.shape[-1],
    depth=2,
    num_cells=128,
).to(device)

actor = QValueActor(net, spec=env.action_spec).to(device)

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