"""Generalized Gini Welfare Function (GGF).

Definition used in this project (Weymark 1981; used for MARL traffic fairness by
the paper cited in the project documents):

    Given utilities u_1..u_n (HIGHER IS BETTER), sort them in ascending order
    u_(1) <= u_(2) <= ... <= u_(n)  (worst-off group first) and compute

        GGF(u) = sum_k  w_k * u_(k)        with  w_1 >= w_2 >= ... >= w_n >= 0.

    Because the largest weight multiplies the worst-off utility, GGF rewards
    improvements for the worst-off groups more than improvements for the
    best-off ones. With equal weights it collapses to the plain average
    (purely "efficiency"); with all weight on w_1 it is the max-min criterion.

For traffic we use utility = -(mean waiting time of an approach), so GGF is a
(negative) number: closer to zero is better.

Default weights: geometric, w_k proportional to ``decay ** (k-1)``, normalised to sum
to 1. ``decay=1`` -> equal weights (no fairness emphasis); smaller decay -> stronger
emphasis on the worst-off. This weight family is an engineering decision (the source
documents do not fix the weights) and is configurable.

GGF is NOT the variance of waiting times; variance is reported separately.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def ggf_weights(n: int, decay: float = 0.5, weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """Return a non-increasing weight vector of length ``n`` that sums to 1.

    Args:
        n: number of groups (>= 0).
        decay: geometric decay in (0, 1]; ignored if ``weights`` is given.
        weights: explicit weights (length must equal ``n``). They must be
            non-negative and non-increasing; they are normalised to sum to 1.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return np.zeros(0)
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,):
            raise ValueError(f"weights must have length {n}, got {w.shape}")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite and non-negative")
        if np.any(np.diff(w) > 1e-12):
            raise ValueError("GGF weights must be non-increasing (largest weight on the worst-off)")
        total = w.sum()
        if total <= 0:
            raise ValueError("weights must not all be zero")
        return w / total
    if not (0.0 < decay <= 1.0):
        raise ValueError("decay must be in (0, 1]")
    w = decay ** np.arange(n, dtype=float)
    return w / w.sum()


def generalized_gini(
    utilities: Sequence[float],
    decay: float = 0.5,
    weights: Optional[Sequence[float]] = None,
) -> float:
    """GGF welfare of ``utilities`` (higher utility = better). Empty input -> 0.0.

    Raises ValueError for NaN/inf values.
    """
    u = np.asarray(list(utilities), dtype=float)
    if u.size == 0:
        return 0.0
    if not np.all(np.isfinite(u)):
        raise ValueError("utilities must be finite")
    w = ggf_weights(u.size, decay=decay, weights=weights)
    return float(np.dot(w, np.sort(u)))  # ascending: worst-off first


def ggf_waiting_time_score(
    waiting_times: Sequence[float],
    decay: float = 0.5,
    weights: Optional[Sequence[float]] = None,
) -> float:
    """GGF welfare of the utilities ``-waiting_time``. Closer to 0 is better.

    Equals minus the weighted sum of waiting times sorted from the longest wait
    to the shortest, so long waits are penalised most. Negative waits are invalid.
    """
    x = np.asarray(list(waiting_times), dtype=float)
    if x.size and np.any(x < 0):
        raise ValueError("waiting times must be non-negative")
    return generalized_gini(-x, decay=decay, weights=weights)
