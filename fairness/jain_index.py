"""Jain's Fairness Index.

    J(x) = (sum x_i)^2 / (n * sum x_i^2),   x_i >= 0

J is in [1/n, 1]; 1 means all groups have identical values (perfectly fair),
1/n means one group has everything. Here x_i is the mean waiting time of group i
(an approach). Larger is better.

Edge cases: empty input and all-zero input (nobody waited, hence everyone equal)
both return 1.0.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def jain_index(values: Sequence[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return 1.0
    if not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("values must be finite and non-negative")
    denom = x.size * float(np.sum(x**2))
    if denom == 0.0:
        return 1.0
    return float(np.sum(x) ** 2 / denom)
