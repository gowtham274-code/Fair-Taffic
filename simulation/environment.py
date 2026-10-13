"""Multi-agent traffic-signal environment (one agent per signalised intersection).

Observation (per agent, 18 floats, all roughly in [0, 3]):
    for each compass slot N, E, S, W (zeros when the approach does not exist):
        queue / OBS_QUEUE_REF, mean local waiting / OBS_WAIT_REF, vehicles on approach / OBS_QUEUE_REF
    one-hot of the current green phase (padded to MAX_PHASES = 4)
    time spent in the current phase / OBS_TIME_REF
    mean queue of neighbouring signalised junctions / OBS_QUEUE_REF   (coordination signal)

Action: index of the desired GREEN phase (0 .. MAX_PHASES-1). Invalid actions (phase does
not exist, or a switch is requested before ``min_green`` seconds have elapsed) are masked
by ``action_mask``; if an invalid action is still passed it is ignored and the current
phase is kept. A real switch inserts ``yellow_time`` seconds of yellow first (derived from
the signal's own states), so the signal logic is never violated.

Decision interval: ``delta_time`` seconds. Episode ends after ``duration_s`` seconds.
Reward: see fairness/reward.py (components are returned in ``info['components']``).

All agents have identical observation / action dimensions, which is what makes federated
averaging of their networks meaningful.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from common.config import SimConfig
from fairness.reward import AgentLocalStats, RewardConfig, compute_rewards
from simulation.backend import ApproachState, TrafficBackend
from simulation.metrics import EpisodeMetrics, approach_mean_waits, compute_episode_metrics
from simulation.network.topology import SLOTS, NetTopology
from simulation.scenarios.demand import generate_demand

MAX_PHASES = 4
OBS_DIM = len(SLOTS) * 3 + MAX_PHASES + 1 + 1
OBS_QUEUE_REF = 10.0
OBS_WAIT_REF = 60.0
OBS_TIME_REF = 60.0
OBS_CLIP = 3.0


def make_backend(cfg: SimConfig) -> TrafficBackend:
    """Create the simulator backend named in the config."""
    if cfg.backend == "mock":
        from simulation.mock_backend import MockBackend

        return MockBackend(NetTopology.grid(cfg.grid_x, cfg.grid_y, cfg.edge_length_m), cfg.yellow_time)
    from simulation.network.builder import build_network
    from simulation.sumo_backend import SumoBackend

    net_dir = Path(cfg.work_dir) / f"net_{cfg.grid_x}x{cfg.grid_y}_{int(cfg.edge_length_m)}"
    net = build_network(net_dir, cfg.grid_x, cfg.grid_y, cfg.edge_length_m, cfg.netconvert_binary)
    return SumoBackend(cfg, net)


@dataclass
class _AgentSignal:
    green: int = 0
    pending: int = 0
    yellow_left: int = 0
    time_in_phase: int = 0


class TrafficEnv:
    def __init__(self, sim: SimConfig, reward: Optional[RewardConfig] = None,
                 backend: Optional[TrafficBackend] = None):
        self.sim = sim
        self.reward_cfg = reward or RewardConfig()
        self.backend = backend or make_backend(sim)
        self.topology = self.backend.topology
        self.agent_ids: List[str] = list(self.backend.tls_ids)
        self.obs_dim = OBS_DIM
        self.n_actions = MAX_PHASES
        self._sig: Dict[str, _AgentSignal] = {}
        self._episode_active = False
        self._slots = {a: self.topology.incoming_slots(a) for a in self.agent_ids}
        self._neighbors = {a: self.topology.neighbors(a) for a in self.agent_ids}

    # ------------------------------------------------------------------ helpers
    @property
    def sim_time(self) -> float:
        return self.backend.sim_time

    def n_green(self, aid: str) -> int:
        return min(self.backend.n_green_phases(aid), MAX_PHASES)

    def current_phase(self, aid: str) -> int:
        return self._sig[aid].green

    def phase_slots(self, aid: str, idx: int) -> FrozenSet[str]:
        return self.backend.phase_slots(aid, idx)

    def approach(self, aid: str, slot: str) -> ApproachState:
        return self.backend.approach_state(aid, slot)

    def action_mask(self, aid: str) -> np.ndarray:
        s = self._sig[aid]
        mask = np.zeros(MAX_PHASES, dtype=bool)
        if s.yellow_left > 0 or s.time_in_phase < self.sim.min_green:
            mask[s.green] = True
        else:
            mask[: self.n_green(aid)] = True
        return mask

    # ------------------------------------------------------------------ gym-like API
    def reset(self, seed: int) -> Dict[str, np.ndarray]:
        vehicles = generate_demand(self.topology, self.sim.scenario, self.sim.duration_s, seed,
                                   self.sim.demand_scale)
        self.backend.reset(seed, vehicles, self.sim.duration_s)
        self._sig = {a: _AgentSignal() for a in self.agent_ids}
        for a in self.agent_ids:
            self.backend.set_green(a, 0)
        self._episode_active = True
        self._wall0 = time.time()
        self._queue_samples: List[Dict[Tuple[str, str], int]] = []
        self._prev_left = {(a, s): 0 for a in self.agent_ids for s in self._slots[a]}
        self.n_vehicles_scheduled = len(vehicles)
        return self._observe()

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict[str, Any]]:
        if not self._episode_active:
            raise RuntimeError("call reset() before step()")
        applied: Dict[str, int] = {}
        for a in self.agent_ids:
            act = int(actions.get(a, self._sig[a].green))
            s = self._sig[a]
            mask = self.action_mask(a)
            if 0 <= act < MAX_PHASES and mask[act] and act != s.green:
                self.backend.set_yellow(a, s.green, act)
                s.pending, s.yellow_left = act, self.sim.yellow_time
                applied[a] = act
            else:
                applied[a] = s.green
        for _ in range(self.sim.delta_time):
            self.backend.step()
            for a in self.agent_ids:
                s = self._sig[a]
                if s.yellow_left > 0:
                    s.yellow_left -= 1
                    if s.yellow_left == 0:
                        self.backend.set_green(a, s.pending)
                        s.green, s.time_in_phase = s.pending, 0
                else:
                    s.time_in_phase += 1
        done = self.backend.sim_time >= self.sim.duration_s
        states = {(a, sl): self.backend.approach_state(a, sl) for a in self.agent_ids for sl in self._slots[a]}
        self._queue_samples.append({k: st.queue for k, st in states.items()})
        local: Dict[str, AgentLocalStats] = {}
        for a in self.agent_ids:
            sts = [(sl, states[(a, sl)]) for sl in self._slots[a]]
            served = sum(st.left_total - self._prev_left[(a, sl)] for sl, st in sts)
            local[a] = AgentLocalStats([st.n_vehicles for _, st in sts], [st.sum_wait for _, st in sts],
                                       [st.queue for _, st in sts], served)
        self._prev_left = {k: st.left_total for k, st in states.items()}
        group_waits = [(st.sum_wait / st.n_vehicles) if st.n_vehicles else 0.0 for st in states.values()]
        comps = compute_rewards(local, group_waits, self.reward_cfg)
        info = {
            "components": {a: c.as_dict() for a, c in comps.items()},
            "applied_phase": applied,
            "sim_time": self.backend.sim_time,
            "queues": {a: float(sum(st.queue for (aa, _s), st in states.items() if aa == a)) for a in self.agent_ids},
        }
        return self._observe(), {a: c.total for a, c in comps.items()}, done, info

    def _observe(self) -> Dict[str, np.ndarray]:
        total_q = {a: sum(self.backend.approach_state(a, s).queue for s in self._slots[a]) for a in self.agent_ids}
        out: Dict[str, np.ndarray] = {}
        for a in self.agent_ids:
            o = np.zeros(OBS_DIM, dtype=np.float32)
            for i, slot in enumerate(SLOTS):
                if slot in self._slots[a]:
                    st = self.backend.approach_state(a, slot)
                    mw = st.sum_wait / st.n_vehicles if st.n_vehicles else 0.0
                    o[3 * i: 3 * i + 3] = (st.queue / OBS_QUEUE_REF, mw / OBS_WAIT_REF, st.n_vehicles / OBS_QUEUE_REF)
            base = len(SLOTS) * 3
            o[base + self._sig[a].green] = 1.0
            o[base + MAX_PHASES] = self._sig[a].time_in_phase / OBS_TIME_REF
            nb = self._neighbors[a]
            o[base + MAX_PHASES + 1] = (np.mean([total_q[n] for n in nb]) / OBS_QUEUE_REF) if nb else 0.0
            out[a] = np.clip(o, 0.0, OBS_CLIP)
        return out

    # ------------------------------------------------------------------ metrics
    def episode_metrics(self, ggf_decay: float = 0.5) -> EpisodeMetrics:
        states = {(a, s): self.backend.approach_state(a, s) for a in self.agent_ids for s in self._slots[a]}
        return compute_episode_metrics(
            self.backend.vehicle_summary(), self.backend.approach_totals(), states,
            self._queue_samples, self.sim.duration_s, time.time() - self._wall0, ggf_decay,
        )

    def close(self) -> None:
        self._episode_active = False
        self.backend.close()

    def __enter__(self) -> "TrafficEnv":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
