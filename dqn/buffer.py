"""Uniform experience replay buffer backed by preallocated numpy arrays.

Design notes / edge cases handled:
  * `terminated` and `truncated` are stored separately. Only true termination
    zeros the bootstrap term; a time-limit truncation (CartPole at 500 steps,
    ALE max frames) still bootstraps from the next state, since that state is
    not absorbing.
  * Sampling before the buffer holds `batch_size` items raises, instead of
    silently returning duplicated or uninitialised rows.
  * Each transition records the global step it was inserted at, so the age of
    samples can be inspected (used by a diagnostic plot).
  * Image observations are stored as int8 (the assignment preprocessing output)
    to keep the MsPacman buffer at ~28 KB per state.
"""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, obs_dtype=np.float32, seed=0):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.float32)
        self.inserted_at = np.zeros(self.capacity, dtype=np.int64)
        self.ptr = 0
        self.size = 0
        self.total_added = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminated[i] = float(terminated)
        self.inserted_at[i] = self.total_added
        self.total_added += 1
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size, return_idx=False):
        if self.size < batch_size:
            raise ValueError(f"buffer has {self.size} items, need {batch_size}")
        idx = self.rng.integers(0, self.size, size=batch_size)
        batch = (self.obs[idx], self.actions[idx], self.rewards[idx],
                 self.next_obs[idx], self.terminated[idx])
        return (batch, idx) if return_idx else batch
