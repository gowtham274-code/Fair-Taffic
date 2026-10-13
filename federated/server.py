"""Federated coordinator server: holds the global model, aggregates client updates.

The server only ever receives ``ClientUpdate`` objects (weights, sample count, mean loss).
This is a *logical* federation on one machine (the clients are the intersection agents of
one simulation); no network transport is used and nothing is encrypted. Model updates can
still leak information about the local data - FL does not by itself guarantee privacy.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from common.logging_setup import get_logger
from federated.aggregation import ClientUpdate, UpdateError, Weights, fedavg, validate_update, weights_nbytes

log = get_logger(__name__)


@dataclass
class RoundRecord:
    round: int
    status: str  # "ok" | "no_valid_updates"
    global_version: int
    participants: List[str]
    rejected: Dict[str, str] = field(default_factory=dict)
    n_samples: Dict[str, int] = field(default_factory=dict)
    aggregation_weights: Dict[str, float] = field(default_factory=dict)
    mean_update_l2: float = 0.0
    bytes_up: int = 0
    bytes_down: int = 0
    local_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    validation: Optional[Dict[str, float]] = None
    timestamp: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FederatedServer:
    def __init__(self, initial_weights: Weights, weight_by: str = "samples"):
        self.global_weights: Weights = {k: v.copy() for k, v in initial_weights.items()}
        self.version = 0
        self.weight_by = weight_by
        self.clients: List[str] = []
        self.history: List[RoundRecord] = []

    def register_client(self, client_id: str) -> None:
        if client_id in self.clients:
            raise ValueError(f"client {client_id} already registered")
        self.clients.append(client_id)

    def aggregate(self, round_idx: int, updates: Sequence[ClientUpdate]) -> RoundRecord:
        """Validate, drop incompatible updates, FedAvg the rest, bump the model version."""
        valid: List[ClientUpdate] = []
        rejected: Dict[str, str] = {}
        for u in updates:
            if u.client_id not in self.clients:
                rejected[u.client_id] = "client not registered"
                continue
            try:
                validate_update(u, self.global_weights)
                valid.append(u)
            except UpdateError as exc:
                rejected[u.client_id] = str(exc)
                log.warning("round %d: rejected update: %s", round_idx, exc)
        rec = RoundRecord(round=round_idx, status="no_valid_updates", global_version=self.version,
                          participants=[u.client_id for u in valid], rejected=rejected, timestamp=time.time())
        rec.bytes_up = sum(weights_nbytes(u.weights) for u in valid)
        rec.n_samples = {u.client_id: u.n_samples for u in valid}
        rec.local_metrics = {u.client_id: dict(u.metrics) for u in valid}
        if not valid:
            self.history.append(rec)
            return rec
        new_w, alpha = fedavg(valid, self.weight_by, reference=self.global_weights)
        deltas = [np.sqrt(sum(float(np.sum((u.weights[k] - self.global_weights[k]) ** 2)) for k in new_w)) for u in valid]
        rec.mean_update_l2 = float(np.mean(deltas))
        rec.aggregation_weights = {u.client_id: float(a) for u, a in zip(valid, alpha)}
        self.global_weights = new_w
        self.version += 1
        rec.global_version = self.version
        rec.status = "ok"
        self.history.append(rec)
        return rec
