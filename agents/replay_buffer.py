"""Fixed-size ring buffer of transitions (local to one agent; never leaves the agent)."""
from __future__ import annotations

from typing import Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_actions: int, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.ones((capacity, n_actions), dtype=bool)
        self.size = 0
        self._pos = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool,
            next_mask: np.ndarray) -> None:
        i = self._pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i], self.next_masks[i] = next_obs, float(done), next_mask
        self._pos = (self._pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = self._rng.integers(0, self.size, size=batch_size)  # with replacement
        return (self.obs[idx], self.actions[idx], self.rewards[idx], self.next_obs[idx], self.dones[idx],
                self.next_masks[idx])
