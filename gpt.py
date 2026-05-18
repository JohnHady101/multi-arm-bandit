import torch
import torch.optim as optim
import gym
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs.transforms import (
    StepCounter,
    InitTracker,
    RewardSum,
    DoubleToFloat,
    Compose,
)
from torchrl.modules import QValueActor
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.collectors import SyncDataCollector
from torchrl.modules import EGreedyModule


device = torch.device("cpu")

# ── 1. Environment ────────────────────────────────────────────────────────────
# Wrap with TransformedEnv so:
#   • InitTracker   → adds "is_init" flag, enabling proper resets mid-batch
#   • DoubleToFloat → CartPole returns float64; torchrl needs float32
#   • StepCounter   → tracks steps and enforces the 500-step truncation limit
#   • RewardSum     → accumulates episode reward (handy for logging)
base_env = gym.make("CartPole-v1")
env = TransformedEnv(
    GymWrapper(base_env, device=device),
    transform=Compose(
        InitTracker(),
        DoubleToFloat(),          # float64 → float32
        StepCounter(max_steps=500),
        RewardSum(),
    ),
)
env.set_seed(42)

# ── 2. Q-network ──────────────────────────────────────────────────────────────
class QNet(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


qnet = QNet().to(device)
actor = QValueActor(qnet, spec=env.action_spec).to(device)

# ── 3. Replay Buffer ──────────────────────────────────────────────────────────
replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(10_000),
    batch_size=64,
)

# ── 4. Data Collector ─────────────────────────────────────────────────────────
# frames_per_batch=200 gives several complete episodes per iteration so the
# buffer fills quickly; total_frames=50_000 for a decent training run.
collector = SyncDataCollector(
    env,
    actor,
    frames_per_batch=200,
    total_frames=20_000,
    device=device,
    reset_at_each_iter=False,   # let episode boundaries reset the env naturally
)

# ── 5. Loss + Target network ──────────────────────────────────────────────────
loss_module = DQNLoss(actor, action_space=env.action_spec, delay_value=True)
target_updater = SoftUpdate(loss_module, eps=0.99)
optimizer = optim.Adam(loss_module.parameters(), lr=1e-3)

# ── 6. Training Loop ──────────────────────────────────────────────────────────
for i, data in enumerate(collector):
    replay_buffer.extend(data)

    if len(replay_buffer) < 1000:
        continue

    batch = replay_buffer.sample()
    loss_dict = loss_module(batch)
    loss_value = loss_dict["loss"]

    optimizer.zero_grad()
    loss_value.backward()
    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=1.0)
    optimizer.step()
    target_updater.step()

    if i % 10 == 0:
        # "episode_reward" is written by RewardSum at the end of each episode
        ep_rewards = data.get(("next", "episode_reward"), None)
        if ep_rewards is not None:
            # Only report timesteps where the episode just finished
            done_mask = data.get(("next", "done"), data.get(("next", "terminated")))
            if done_mask is not None and done_mask.any():
                mean_ep_r = ep_rewards[done_mask.squeeze(-1)].mean().item()
                print(f"Iter {i:4d} | loss {loss_value.item():.4f} | mean ep reward {mean_ep_r:.1f}")
                continue
        print(f"Iter {i:4d} | loss {loss_value.item():.4f}")

# ── 7. Evaluate the trained policy ───────────────────────────────────────────
print("\nTesting trained policy…")
import time
from tensordict import TensorDict

try:
    eval_env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0

    with torch.no_grad():
        while not done:
            obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            td = TensorDict({"observation": obs_tensor}, batch_size=[1])
            actor(td)
            action = td["action"].item()
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            eval_env.render()
            time.sleep(0.02)  # ~50 fps, keeps the window alive between frames

    print(f"Eval episode total reward: {total_reward}")
    time.sleep(3)  # pause so you can see the final frame before the window closes
    eval_env.close()

except Exception as e:
    print(f"Rendering failed ({e}), falling back to torchrl rollout…")
    with torch.no_grad():
        rollout = env.rollout(max_steps=500, policy=actor)
        print(f"Rollout total reward: {rollout['next', 'reward'].sum().item():.1f}")
    print("First 5 observations:\n", rollout["observation"][:5])
    print("First 5 actions:\n", rollout["action"][:5])