"""Episode-level evaluation metrics (efficiency + fairness).

Definitions (all documented in docs/EXPERIMENTS.md):

* completed_vehicles      trips that reached their destination before the episode ended.
* unfinished_vehicles     vehicles still in the network OR still waiting to be inserted
                          when the episode ends. Reported so that a controller cannot
                          look efficient by leaving vehicles unfinished.
* throughput_veh_per_h    completed_vehicles / duration_s * 3600.
* avg_wait_s              mean accumulated waiting time per vehicle over ALL vehicles that
                          entered the network (finished + unfinished at episode end).
* max_wait_s              the largest accumulated waiting time of any such vehicle.
* wait_variance           population variance of those per-vehicle waiting times (s^2).
* approach groups         fairness is measured over approaches (junction x compass side);
                          group value = mean *local* waiting time of vehicles on that
                          approach (time accrued while on it; vehicles still on the
                          approach at the end are included).
* jain_index              Jain's index over the approach means (1 = perfectly equal).
* ggf_score               GGF over utilities -(approach mean wait) in seconds; closer to 0
                          is better; puts most weight on the worst approaches.
* avg_queue               time-average of halting vehicles per approach.
* fuel_g                  only with SUMO (default emission class); None otherwise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from fairness.generalized_gini import generalized_gini
from fairness.jain_index import jain_index
from fairness.waiting_time_variance import waiting_time_variance
from simulation.backend import ApproachState, VehicleSummary

Key = Tuple[str, str]


@dataclass
class EpisodeMetrics:
    completed_vehicles: int = 0
    departed_vehicles: int = 0
    unfinished_vehicles: int = 0
    throughput_veh_per_h: float = 0.0
    avg_wait_s: float = 0.0
    avg_wait_completed_s: float = 0.0
    max_wait_s: float = 0.0
    wait_variance: float = 0.0
    jain_index: float = 1.0
    ggf_score: float = 0.0
    avg_queue: float = 0.0
    fuel_g: Optional[float] = None
    sim_duration_s: float = 0.0
    wall_time_s: float = 0.0
    approach_mean_wait_s: Dict[str, float] = field(default_factory=dict)
    intersection_mean_wait_s: Dict[str, float] = field(default_factory=dict)
    intersection_avg_queue: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return asdict(self)

    def scalar_dict(self) -> Dict[str, float]:
        return {k: v for k, v in asdict(self).items() if isinstance(v, (int, float)) or v is None}


def approach_mean_waits(
    totals: Dict[Key, Tuple[float, int]], states: Dict[Key, ApproachState]
) -> Dict[Key, float]:
    """Mean local wait per approach: vehicles that left it plus vehicles still on it."""
    out: Dict[Key, float] = {}
    for key, (tw, tc) in totals.items():
        st = states.get(key, ApproachState())
        n = tc + st.n_vehicles
        out[key] = (tw + st.sum_wait) / n if n > 0 else 0.0
    return out


def compute_episode_metrics(
    summary: VehicleSummary,
    totals: Dict[Key, Tuple[float, int]],
    states: Dict[Key, ApproachState],
    queue_samples: List[Dict[Key, int]],
    duration_s: float,
    wall_time_s: float,
    ggf_decay: float = 0.5,
) -> EpisodeMetrics:
    all_waits = np.asarray(summary.arrived_waits + summary.active_waits, dtype=float)
    means = approach_mean_waits(totals, states)
    vals = list(means.values())
    inter: Dict[str, List[float]] = {}
    for (tls, _s), w in means.items():
        inter.setdefault(tls, []).append(w)
    iq: Dict[str, List[float]] = {}
    for sample in queue_samples:
        for (tls, _s), q in sample.items():
            iq.setdefault(tls, []).append(q)
    n_app_by_tls: Dict[str, int] = {}
    for (tls, _s) in totals:
        n_app_by_tls[tls] = n_app_by_tls.get(tls, 0) + 1
    # avg queue per approach per decision step
    avg_queue = 0.0
    if queue_samples:
        avg_queue = float(np.mean([np.mean(list(s.values())) for s in queue_samples if s]))
    completed = summary.arrived
    return EpisodeMetrics(
        completed_vehicles=completed,
        departed_vehicles=summary.departed,
        unfinished_vehicles=max(summary.departed - completed, 0) + summary.pending,
        throughput_veh_per_h=completed / duration_s * 3600.0 if duration_s > 0 else 0.0,
        avg_wait_s=float(all_waits.mean()) if all_waits.size else 0.0,
        avg_wait_completed_s=float(np.mean(summary.arrived_waits)) if summary.arrived_waits else 0.0,
        max_wait_s=float(all_waits.max()) if all_waits.size else 0.0,
        wait_variance=waiting_time_variance(all_waits),
        jain_index=jain_index(vals),
        ggf_score=generalized_gini([-v for v in vals], decay=ggf_decay),
        avg_queue=avg_queue,
        fuel_g=summary.fuel_g,
        sim_duration_s=float(duration_s),
        wall_time_s=float(wall_time_s),
        approach_mean_wait_s={f"{t}:{s}": float(w) for (t, s), w in sorted(means.items())},
        intersection_mean_wait_s={t: float(np.mean(v)) for t, v in sorted(inter.items())},
        intersection_avg_queue={t: float(np.sum(v) / max(len(queue_samples), 1) / n_app_by_tls.get(t, 1))
                                for t, v in sorted(iq.items())},
    )
