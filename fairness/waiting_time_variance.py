"""Waiting-time variance (population variance, ddof = 0). Smaller is better.

This is an *additional* evaluation metric. It is mathematically different from the
Generalized Gini Welfare Function (see generalized_gini.py).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def waiting_time_variance(waiting_times: Sequence[float]) -> float:
    """Population variance in seconds^2. Empty input -> 0.0."""
    x = np.asarray(list(waiting_times), dtype=float)
    if x.size == 0:
        return 0.0
    if not np.all(np.isfinite(x)):
        raise ValueError("waiting times must be finite")
    return float(np.var(x))
