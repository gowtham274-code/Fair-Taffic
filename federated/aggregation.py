"""Federated Averaging (FedAvg, McMahan et al. 2017) for compatible Q-networks.

    w_global(t+1) = sum_k  (n_k / sum_j n_j) * w_k(t+1)

where w_k are the locally trained weights of client k after the round and n_k is the
number of transitions client k collected in that round (``weight_by="samples"``), or
1/K for every client (``weight_by="uniform"``).

Only agents with IDENTICAL network architectures can be averaged. Every update is
validated (same keys, same shapes, finite numbers) and incompatible updates are rejected
with a reason instead of being silently averaged.

Limitations (see docs/RESEARCH_LIMITATIONS.md): FedAvg assumes clients are statistically
similar; here the intersections see different traffic (non-IID), so a single averaged
policy is a compromise. Averaging weights of independently trained neural networks can
also hurt when the networks drift far apart within a round.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

Weights = Dict[str, np.ndarray]


class UpdateError(ValueError):
    """Raised for a missing, incompatible or non-finite client update."""


@dataclass
class ClientUpdate:
    client_id: str
    weights: Weights
    n_samples: int
    metrics: Dict[str, float] = field(default_factory=dict)  # e.g. mean local loss (no raw traffic data)


def weights_nbytes(w: Mapping[str, np.ndarray]) -> int:
    return int(sum(a.nbytes for a in w.values()))


def validate_update(update: ClientUpdate, reference: Mapping[str, np.ndarray]) -> None:
    if update.weights is None or len(update.weights) == 0:
        raise UpdateError(f"{update.client_id}: empty update")
    if set(update.weights) != set(reference):
        raise UpdateError(f"{update.client_id}: parameter names differ from the global model")
    for k, ref in reference.items():
        a = update.weights[k]
        if a.shape != ref.shape:
            raise UpdateError(f"{update.client_id}: shape mismatch for {k}: {a.shape} vs {ref.shape}")
        if not np.all(np.isfinite(a)):
            raise UpdateError(f"{update.client_id}: non-finite values in {k}")
    if update.n_samples < 0:
        raise UpdateError(f"{update.client_id}: negative n_samples")


def aggregation_weights(updates: Sequence[ClientUpdate], weight_by: str = "samples") -> np.ndarray:
    if not updates:
        raise UpdateError("no updates to aggregate")
    if weight_by == "uniform":
        return np.full(len(updates), 1.0 / len(updates))
    n = np.array([u.n_samples for u in updates], dtype=float)
    if n.sum() <= 0:  # nobody collected data: fall back to uniform
        return np.full(len(updates), 1.0 / len(updates))
    return n / n.sum()


def fedavg(updates: Sequence[ClientUpdate], weight_by: str = "samples",
           reference: Optional[Mapping[str, np.ndarray]] = None) -> Tuple[Weights, np.ndarray]:
    """Return (averaged weights, per-client aggregation weights).

    All updates must be mutually compatible (and compatible with ``reference`` when given).
    """
    if not updates:
        raise UpdateError("no updates to aggregate")
    ref = reference if reference is not None else updates[0].weights
    for u in updates:
        validate_update(u, ref)
    alpha = aggregation_weights(updates, weight_by)
    out: Weights = {}
    for k in ref:
        out[k] = np.sum([a * u.weights[k].astype(np.float64) for a, u in zip(alpha, updates)], axis=0).astype(np.float32)
    return out, alpha
