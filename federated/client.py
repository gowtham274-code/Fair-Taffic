"""Federated client = one intersection agent.

What a client SENDS to the server: its model weights, the number of transitions it
collected this round, and its mean local training loss.
What it NEVER sends: observations, rewards, replay-buffer contents, vehicle data.
What it keeps locally: replay buffer, optimiser state, exploration schedule.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from agents.marl_agent import DQNAgent
from federated.aggregation import ClientUpdate, Weights


class FederatedClient:
    def __init__(self, client_id: str, agent: DQNAgent):
        self.client_id = client_id
        self.agent = agent
        self.global_version = 0

    def receive_global(self, weights: Weights, version: int) -> None:
        """Replace the local model by the global one (local buffer/optimiser are kept)."""
        self.agent.set_weights(weights, sync_target=True)
        self.agent.version = version
        self.global_version = version

    def produce_update(self) -> ClientUpdate:
        a = self.agent
        losses = a.losses_round
        update = ClientUpdate(
            client_id=self.client_id,
            weights=a.get_weights(),
            n_samples=int(a.samples_round),
            metrics={"mean_loss": float(np.mean(losses)) if losses else float("nan"), "learn_steps": float(len(losses))},
        )
        a.samples_round = 0
        a.losses_round = []
        return update
