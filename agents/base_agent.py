"""Abstract interface of a learning traffic-signal agent."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    agent_id: str

    @abstractmethod
    def act(self, obs: np.ndarray, mask: np.ndarray, epsilon: float = 0.0) -> int:
        """Choose a valid action (index into ``mask``); ``epsilon`` is the exploration rate."""

    @abstractmethod
    def store(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray,
              done: bool, next_mask: np.ndarray) -> None:
        """Record one transition of this agent's own experience."""

    @abstractmethod
    def learn(self) -> float | None:
        """Perform one gradient update if enough data exists; return the loss or None."""
