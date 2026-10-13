"""Fairness-aware reward for each intersection agent.

Per decision step, agent i receives

    r_i = w_wait * r_wait_i + w_queue * r_queue_i + w_thr * r_thr_i + w_fair_eff * r_fair

    r_wait_i  = - (vehicle-weighted mean local waiting time on i's approaches) / wait_ref
    r_queue_i = - (total halting vehicles on i's approaches) / queue_ref
    r_thr_i   = + (vehicles that crossed i's stop line this interval) / thr_ref
    r_fair    =   GGF( -mean_wait_g / wait_ref  for every approach g of the network ) (shared)

* "Local waiting time" of a vehicle on an approach = time it has accumulated waiting
  since it ENTERED that approach, so one vehicle's delay is counted by exactly one
  approach at a time (no double counting across intersections).
* All terms are divided by reference constants so they are dimensionless and of
  comparable scale; they are also clipped to [-clip, +clip].
* r_fair is the same for all agents (a shared team fairness signal). It depends on the
  *distribution* of mean waits across all approaches (GGF puts most weight on the worst
  approaches). Setting w_fair = 0 gives a delay-minimising ("conventional RL") reward,
  which is the non-fair RL baseline.
* Optional extension (NOT part of the baseline GGF in the source documents):
  ``dynamic_fairness`` scales w_fair by (1 + kappa * congestion), congestion in [0, 1].

Components are returned separately so the dashboard can show them.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

from fairness.generalized_gini import generalized_gini


class RewardConfig(BaseModel):
    w_wait: float = Field(1.0, ge=0)
    w_queue: float = Field(0.25, ge=0)
    w_throughput: float = Field(0.1, ge=0)
    w_fairness: float = Field(1.0, ge=0, description="0 disables the GGF term (conventional RL baseline)")
    wait_ref_s: float = Field(60.0, gt=0, description="seconds; normalises waiting times")
    queue_ref: float = Field(20.0, gt=0, description="vehicles per agent; normalises queues")
    throughput_ref: float = Field(5.0, gt=0, description="vehicles per decision interval")
    clip: float = Field(5.0, gt=0)
    ggf_decay: float = Field(0.5, gt=0, le=1.0)
    dynamic_fairness: bool = False
    dynamic_kappa: float = Field(1.0, ge=0)


@dataclass
class RewardComponents:
    wait: float
    queue: float
    throughput: float
    fairness: float
    fairness_weight: float
    total: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class AgentLocalStats:
    """Inputs the reward needs from one agent's approaches (only approaches that exist)."""

    n_vehicles: Sequence[float]
    sum_wait_s: Sequence[float]
    queue: Sequence[float]
    served: float


def mean_wait(n_vehicles: Sequence[float], sum_wait: Sequence[float]) -> float:
    n = float(np.sum(n_vehicles))
    return float(np.sum(sum_wait)) / n if n > 0 else 0.0


def _clip(x: float, c: float) -> float:
    return float(np.clip(x, -c, c))


def compute_rewards(
    local: Dict[str, AgentLocalStats],
    group_mean_waits: Sequence[float],
    cfg: RewardConfig,
) -> Dict[str, RewardComponents]:
    """Compute reward components for every agent.

    Args:
        local: per-agent local statistics.
        group_mean_waits: mean local waiting time (seconds) of every approach in the
            network (one entry per approach, 0 when an approach is empty).
        cfg: reward configuration.
    """
    waits = np.asarray(list(group_mean_waits), dtype=float)
    if waits.size and (np.any(waits < 0) or not np.all(np.isfinite(waits))):
        raise ValueError("group_mean_waits must be finite and non-negative")
    r_fair_raw = generalized_gini(-waits / cfg.wait_ref_s, decay=cfg.ggf_decay) if waits.size else 0.0
    r_fair = _clip(r_fair_raw, cfg.clip)

    total_queue = float(sum(np.sum(s.queue) for s in local.values()))
    n_agents = max(len(local), 1)
    congestion = float(np.clip(total_queue / (cfg.queue_ref * n_agents), 0.0, 1.0))
    w_fair = cfg.w_fairness * ((1.0 + cfg.dynamic_kappa * congestion) if cfg.dynamic_fairness else 1.0)

    out: Dict[str, RewardComponents] = {}
    for aid, s in local.items():
        r_wait = _clip(-mean_wait(s.n_vehicles, s.sum_wait_s) / cfg.wait_ref_s, cfg.clip)
        r_queue = _clip(-float(np.sum(s.queue)) / cfg.queue_ref, cfg.clip)
        r_thr = _clip(float(s.served) / cfg.throughput_ref, cfg.clip)
        total = cfg.w_wait * r_wait + cfg.w_queue * r_queue + cfg.w_throughput * r_thr + w_fair * r_fair
        out[aid] = RewardComponents(r_wait, r_queue, r_thr, r_fair, w_fair, float(total))
    return out
