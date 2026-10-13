"""MockBackend: a minimal queue-based traffic model (TEST DOUBLE, not SUMO).

Purpose: let unit/integration tests, CI and the demo pipeline run on machines
without SUMO. It follows the same vehicle list (demand) and topology as the SUMO
backend, but the traffic physics are crude: vehicles travel an edge in
length/13.89 s, queue at the stop line of the next signalised junction, and are
discharged at a saturation flow of 0.5 veh/s per approach during green.
Two green phases: N+S and E+W. Do NOT report numbers from this backend as research
results.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, FrozenSet, List, Sequence, Tuple

from simulation.backend import ApproachState, TrafficBackend, VehicleSummary
from simulation.network.topology import NetTopology
from simulation.scenarios.demand import Vehicle

SAT_FLOW = 0.5  # vehicles per second per approach while green
SPEED = 13.89


class _V:
    __slots__ = ("vid", "route", "i", "wait", "base")

    def __init__(self, vid: str, route: Tuple[str, ...]):
        self.vid, self.route, self.i, self.wait, self.base = vid, route, 0, 0.0, 0.0


class MockBackend(TrafficBackend):
    name = "mock"
    _PHASES: Tuple[FrozenSet[str], ...] = (frozenset({"N", "S"}), frozenset({"E", "W"}))

    def __init__(self, topology: NetTopology, yellow_time: int = 3):
        self.topology = topology
        self._slots = {t: topology.incoming_slots(t) for t in topology.tls_ids}
        self._edge_slot: Dict[str, Tuple[str, str]] = {
            e: (t, s) for t, d in self._slots.items() for s, e in d.items()
        }
        self._t = 0.0
        self._duration = 0

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int, vehicles: Sequence[Vehicle], duration_s: int) -> None:
        self._t = 0.0
        self._duration = duration_s
        self._todo: Deque[Vehicle] = deque(sorted(vehicles, key=lambda v: v.depart))
        self._queues: Dict[Tuple[str, str], Deque[_V]] = {
            (t, s): deque() for t, d in self._slots.items() for s in d
        }
        self._transit: List[Tuple[float, _V]] = []  # (arrival time at end of current edge, vehicle)
        self._toward: Dict[Tuple[str, str], int] = {k: 0 for k in self._queues}
        self._green: Dict[str, int] = {t: 0 for t in self._slots}
        self._yellow: Dict[str, bool] = {t: False for t in self._slots}
        self._credit: Dict[Tuple[str, str], float] = {k: 0.0 for k in self._queues}
        self._left: Dict[Tuple[str, str], int] = {k: 0 for k in self._queues}
        self._tot_wait: Dict[Tuple[str, str], float] = {k: 0.0 for k in self._queues}
        self._tot_cnt: Dict[Tuple[str, str], int] = {k: 0 for k in self._queues}
        self._arrived_waits: List[float] = []
        self._departed = 0

    def close(self) -> None:
        pass

    @property
    def sim_time(self) -> float:
        return self._t

    # ------------------------------------------------------------------ signals
    def n_green_phases(self, tls: str) -> int:
        return len(self._PHASES)

    def phase_slots(self, tls: str, idx: int) -> FrozenSet[str]:
        return self._PHASES[idx]

    def set_green(self, tls: str, idx: int) -> None:
        if not 0 <= idx < len(self._PHASES):
            raise ValueError(f"invalid green phase {idx}")
        self._green[tls] = idx
        self._yellow[tls] = False

    def set_yellow(self, tls: str, from_idx: int, to_idx: int) -> None:
        self._yellow[tls] = True

    # ------------------------------------------------------------------ dynamics
    def _enter_edge(self, v: _V) -> None:
        edge = v.route[v.i]
        arrive = self._t + self.topology.edge_length.get(edge, 200.0) / SPEED
        self._transit.append((arrive, v))
        key = self._edge_slot.get(edge)
        if key is not None:
            self._toward[key] += 1

    def step(self) -> None:
        # 1. insert vehicles whose departure time has come
        while self._todo and self._todo[0].depart <= self._t:
            veh = self._todo.popleft()
            self._departed += 1
            self._enter_edge(_V(veh.vid, veh.route))
        # 2. vehicles reaching the end of their edge
        still: List[Tuple[float, _V]] = []
        for arrive, v in self._transit:
            if arrive > self._t:
                still.append((arrive, v))
                continue
            key = self._edge_slot.get(v.route[v.i])
            if key is None:  # edge ends at a boundary junction -> trip finished
                self._arrived_waits.append(v.wait)
            else:
                self._toward[key] -= 1
                v.base = v.wait  # waiting accrued before this approach is not charged to it
                self._queues[key].append(v)
        self._transit = still
        # 3. discharge during green
        discharged = set()
        for key, q in self._queues.items():
            tls, slot = key
            self._credit[key] = min(self._credit[key] + SAT_FLOW, 1.0)
            if self._yellow[tls] or slot not in self._PHASES[self._green[tls]]:
                continue
            if q and self._credit[key] >= 1.0:
                v = q.popleft()
                self._credit[key] -= 1.0
                self._tot_wait[key] += v.wait - v.base
                self._tot_cnt[key] += 1
                self._left[key] += 1
                discharged.add(v.vid)
                v.i += 1
                self._enter_edge(v)
        # 4. everyone still queued waits one more second
        for q in self._queues.values():
            for v in q:
                if v.vid not in discharged:
                    v.wait += 1.0
        self._t += 1.0

    # ------------------------------------------------------------------ observations
    def approach_state(self, tls: str, slot: str) -> ApproachState:
        key = (tls, slot)
        if key not in self._queues:
            return ApproachState(present=False)
        q = self._queues[key]
        waits = [v.wait - v.base for v in q]
        return ApproachState(
            present=True,
            n_vehicles=len(q) + self._toward[key],
            queue=len(q),
            sum_wait=float(sum(waits)),
            max_wait=float(max(waits)) if waits else 0.0,
            left_total=self._left[key],
        )

    def approach_totals(self) -> Dict[Tuple[str, str], Tuple[float, int]]:
        return {k: (self._tot_wait[k], self._tot_cnt[k]) for k in self._queues}

    def vehicle_summary(self) -> VehicleSummary:
        active = [v.wait for q in self._queues.values() for v in q] + [v.wait for _a, v in self._transit]
        return VehicleSummary(
            arrived_waits=list(self._arrived_waits),
            active_waits=active,
            departed=self._departed,
            arrived=len(self._arrived_waits),
            pending=len(self._todo),
            fuel_g=None,
        )
