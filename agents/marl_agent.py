"""DQN agent for one signalised intersection (an "independent learner" with a
cooperative, fairness-aware reward and a neighbour-queue observation).

The multi-agent algorithm of this project is therefore: Independent DQN (IDQN) with
parameter-compatible agents, trained on the shared fairness-aware reward. See
docs/ALGORITHMS.md for why this was chosen (and its limitations: no centralised
critic / mixing network as in QMIX; non-stationarity is not explicitly handled).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from agents.model import QNetwork, Weights, build_qnet
from agents.replay_buffer import ReplayBuffer
from common.config import RLConfig


class DQNAgent:
    def __init__(self, agent_id: str, obs_dim: int, n_actions: int, rl: RLConfig, seed: int,
                 eps_decay_scale: float = 1.0):
        self.agent_id = agent_id
        self.rl = rl
        self.n_actions = n_actions
        mk = lambda s: build_qnet(obs_dim, n_actions, rl.hidden, rl.lr, rl.grad_clip, s, rl.nn_backend)
        self.online: QNetwork = mk(seed)
        self.target: QNetwork = mk(seed + 1)
        self.target.set_weights(self.online.get_weights())
        self.buffer = ReplayBuffer(rl.buffer_size, obs_dim, n_actions, seed + 2)
        self._rng = np.random.default_rng(seed + 3)
        self._eps_decay = max(1, int(rl.eps_decay_steps * eps_decay_scale))
        self.steps = 0  # decisions taken in training mode
        self.learn_steps = 0
        self.samples_round = 0  # transitions collected since the last federated round
        self.losses_round: list[float] = []
        self.version = 0  # model/checkpoint version label shown in the dashboard

    # ------------------------------------------------------------------ acting
    def epsilon(self) -> float:
        frac = min(self.steps / self._eps_decay, 1.0)
        return self.rl.eps_start + frac * (self.rl.eps_end - self.rl.eps_start)

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        return self.online.predict(obs[None, :])[0]

    def act(self, obs: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        """Pick an action among valid (masked-in) ones. greedy=True -> eval mode (no exploration,
        no step counting)."""
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            raise ValueError("action mask has no valid action")
        if not greedy:
            self.steps += 1
            if self._rng.random() < self.epsilon():
                return int(self._rng.choice(valid))
        q = self.q_values(obs)
        q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    # ------------------------------------------------------------------ learning
    def remember(self, obs, action, reward, next_obs, done, next_mask) -> None:
        self.buffer.add(obs, action, reward, next_obs, done, next_mask)
        self.samples_round += 1

    def learn(self) -> Optional[float]:
        if len(self.buffer) < max(self.rl.learn_start, self.rl.batch_size):
            return None
        if self.steps % self.rl.train_freq != 0:
            return None
        o, a, r, no, d, nm = self.buffer.sample(self.rl.batch_size)
        q_next = np.where(nm, self.target.predict(no), -np.inf).max(axis=1)
        q_next = np.where(np.isfinite(q_next), q_next, 0.0)
        targets = r + self.rl.gamma * (1.0 - d) * q_next
        loss = self.online.train_on_batch(o, a, targets.astype(np.float32))
        self.learn_steps += 1
        self.losses_round.append(loss)
        if self.learn_steps % self.rl.target_update_every == 0:
            self.target.set_weights(self.online.get_weights())
        return loss

    # ------------------------------------------------------------------ weights
    def get_weights(self) -> Weights:
        return self.online.get_weights()

    def set_weights(self, w: Weights, sync_target: bool = True) -> None:
        """Load weights into the online network (and, by default, the target network).
        The local optimiser state (Adam moments) and replay buffer are kept."""
        self.online.set_weights(w)
        if sync_target:
            self.target.set_weights(w)
