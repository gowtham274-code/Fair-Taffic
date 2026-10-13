"""SUMO / TraCI backend (the research backend).

NOTE ON VERIFICATION: this module was written without access to a SUMO installation
(the development sandbox had none). It follows the documented TraCI API but has NOT
been executed against SUMO yet. Run ``python scripts/validate_sumo.py`` and
``python -m unittest tests.integration.test_sumo_backend`` on a machine with SUMO.

Design:
  * Static information (junctions, edges, approach slots, neighbours) comes from the
    parsed ``.net.xml`` (NetTopology); TraCI is used only for dynamic state and for
    setting signal states.
  * Signals are controlled with setRedYellowGreenState: green states are read from
    the junction's own SUMO program (states containing 'G'/'g'), yellow states are
    derived (greens that turn red become 'y').
  * Each simulated second, per-vehicle accumulated waiting time is polled so that
    the final waiting time of a vehicle is known when it arrives. ``--waiting-time-memory``
    is set above the episode length so SUMO does not forget old waiting time.
  * "Local waiting" of a vehicle on an approach = accumulated waiting now minus its
    accumulated waiting when it entered that approach (no double counting).
  * Fuel (g) is integrated from traci.vehicle.getFuelConsumption (mg/s) with SUMO's
    default emission class; treat as indicative only.
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from common.config import SimConfig
from common.logging_setup import get_logger
from simulation.backend import ApproachState, TrafficBackend, VehicleSummary
from simulation.network.topology import NetTopology
from simulation.scenarios.demand import Vehicle, write_route_file

log = get_logger(__name__)


def _is_green(ch: str) -> bool:
    return ch in "Gg"


class SumoBackend(TrafficBackend):
    name = "sumo"

    def __init__(self, cfg: SimConfig, net_file: str | Path, topology: Optional[NetTopology] = None):
        self.cfg = cfg
        self.net_file = Path(net_file)
        self.topology = topology or NetTopology.from_net_xml(self.net_file)
        self._conn: Any = None
        self._label = f"ft_{uuid.uuid4().hex[:8]}"
        self._slots = {t: self.topology.incoming_slots(t) for t in self.topology.tls_ids}
        self._greens: Dict[str, List[str]] = {}
        self._phase_slots: Dict[str, List[FrozenSet[str]]] = {}
        self._lanes: Dict[Tuple[str, str], List[str]] = {}

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int, vehicles: Sequence[Vehicle], duration_s: int) -> None:
        self.close()
        try:
            import traci  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("the 'traci' package is not installed (pip install eclipse-sumo traci)") from exc
        binary = self.cfg.sumo_binary
        if self.cfg.gui and binary == "sumo":
            binary = "sumo-gui"
        if shutil.which(binary) is None and not os.environ.get("SUMO_HOME"):
            raise RuntimeError(f"SUMO binary '{binary}' not found on PATH (install SUMO or set SUMO_HOME)")
        work = Path(self.cfg.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        route_file = write_route_file(list(vehicles), work / f"{self._label}.rou.xml")
        cmd = [
            binary, "-n", str(self.net_file), "-r", str(route_file),
            "--seed", str(seed), "--begin", "0", "--end", str(duration_s + 1),
            "--step-length", "1", "--waiting-time-memory", str(duration_s + 100),
            "--time-to-teleport", "-1", "--no-step-log", "true", "--no-warnings", "true",
            "--duration-log.disable", "true",
        ]
        if self.cfg.gui:
            cmd += ["--start", "true", "--quit-on-end", "true"]
        traci.start(cmd, label=self._label)
        self._traci = traci
        self._conn = traci.getConnection(self._label)
        self._route_file = route_file
        self._total_scheduled = len(vehicles)
        self._init_signals()
        self._init_tracking()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as exc:  # pragma: no cover
                log.warning("error while closing TraCI connection: %s", exc)
            self._conn = None
        rf = getattr(self, "_route_file", None)
        if rf is not None:
            try:
                Path(rf).unlink(missing_ok=True)
            except OSError:
                pass
            self._route_file = None

    @property
    def sim_time(self) -> float:
        return float(self._conn.simulation.getTime())

    # ------------------------------------------------------------------ signals
    def _init_signals(self) -> None:
        c = self._conn
        sumo_tls = set(c.trafficlight.getIDList())
        missing = set(self.topology.tls_ids) - sumo_tls
        if missing:
            raise RuntimeError(f"traffic lights {sorted(missing)} from the network are not controllable in SUMO")
        for tls in self.topology.tls_ids:
            logic = c.trafficlight.getAllProgramLogics(tls)[0]
            greens: List[str] = []
            for ph in logic.phases:
                if any(_is_green(ch) for ch in ph.state) and ph.state not in greens:
                    greens.append(ph.state)
            if not greens:
                raise RuntimeError(f"no green phase found in the program of {tls}")
            self._greens[tls] = greens
            links = c.trafficlight.getControlledLinks(tls)  # list (per signal index) of [(in, out, via)]
            edge_to_slot = {e: s for s, e in self._slots[tls].items()}
            idx_slot: Dict[int, str] = {}
            for i, group in enumerate(links):
                if group:
                    edge = group[0][0].rsplit("_", 1)[0]
                    if edge in edge_to_slot:
                        idx_slot[i] = edge_to_slot[edge]
            self._phase_slots[tls] = [
                frozenset(idx_slot[i] for i, ch in enumerate(st) if _is_green(ch) and i in idx_slot)
                for st in greens
            ]
            for slot, edge in self._slots[tls].items():
                n_lanes = c.edge.getLaneNumber(edge)
                self._lanes[(tls, slot)] = [f"{edge}_{k}" for k in range(n_lanes)]
            c.trafficlight.setRedYellowGreenState(tls, greens[0])

    def n_green_phases(self, tls: str) -> int:
        if tls not in self._greens:
            raise RuntimeError("SUMO signal programs are read when the simulation starts; call reset() before "
                               "asking for phase counts")
        return len(self._greens[tls])

    def phase_slots(self, tls: str, idx: int) -> FrozenSet[str]:
        return self._phase_slots[tls][idx]

    def set_green(self, tls: str, idx: int) -> None:
        self._conn.trafficlight.setRedYellowGreenState(tls, self._greens[tls][idx])

    def set_yellow(self, tls: str, from_idx: int, to_idx: int) -> None:
        a, b = self._greens[tls][from_idx], self._greens[tls][to_idx]
        state = "".join("y" if (_is_green(ca) and not _is_green(cb)) else ca for ca, cb in zip(a, b))
        self._conn.trafficlight.setRedYellowGreenState(tls, state)

    # ------------------------------------------------------------------ tracking
    def _init_tracking(self) -> None:
        self._acc: Dict[str, float] = {}  # latest accumulated waiting per active vehicle
        self._where: Dict[str, Tuple[Tuple[str, str], float]] = {}  # vid -> (approach key, baseline)
        self._tot_wait: Dict[Tuple[str, str], float] = {k: 0.0 for k in self._lanes}
        self._tot_cnt: Dict[Tuple[str, str], int] = {k: 0 for k in self._lanes}
        self._left: Dict[Tuple[str, str], int] = {k: 0 for k in self._lanes}
        self._state: Dict[Tuple[str, str], ApproachState] = {k: ApproachState(present=True) for k in self._lanes}
        self._arrived_waits: List[float] = []
        self._departed = 0
        self._fuel_mg = 0.0

    def step(self) -> None:
        c = self._conn
        c.simulationStep()
        self._departed += c.simulation.getDepartedNumber()
        for vid in c.simulation.getArrivedIDList():
            self._arrived_waits.append(self._acc.pop(vid, 0.0))
            self._finalize(vid)
        for vid in c.vehicle.getIDList():
            self._acc[vid] = c.vehicle.getAccumulatedWaitingTime(vid)
            self._fuel_mg += c.vehicle.getFuelConsumption(vid)  # mg emitted during the last 1 s step
        present: Set[str] = set()
        for key, lanes in self._lanes.items():
            ids: List[str] = []
            halting = 0
            for ln in lanes:
                ids += list(c.lane.getLastStepVehicleIDs(ln))
                halting += c.lane.getLastStepHaltingNumber(ln)
            waits: List[float] = []
            for vid in ids:
                present.add(vid)
                cur = self._where.get(vid)
                acc = self._acc.get(vid, 0.0)
                if cur is None or cur[0] != key:
                    if cur is not None:
                        self._finalize(vid)
                    self._where[vid] = (key, acc)
                    cur = self._where[vid]
                waits.append(max(acc - cur[1], 0.0))
            self._state[key] = ApproachState(
                present=True, n_vehicles=len(ids), queue=halting,
                sum_wait=float(sum(waits)), max_wait=float(max(waits)) if waits else 0.0,
                left_total=self._left[key],
            )
        for vid in [v for v in self._where if v not in present]:
            self._finalize(vid)
        for key in self._state:  # refresh left_total after finalisations
            self._state[key].left_total = self._left[key]

    def _finalize(self, vid: str) -> None:
        cur = self._where.pop(vid, None)
        if cur is None:
            return
        key, base = cur
        self._tot_wait[key] += max(self._acc.get(vid, base) - base, 0.0)
        self._tot_cnt[key] += 1
        self._left[key] += 1

    # ------------------------------------------------------------------ observations
    def approach_state(self, tls: str, slot: str) -> ApproachState:
        return self._state.get((tls, slot), ApproachState(present=False))

    def approach_totals(self) -> Dict[Tuple[str, str], Tuple[float, int]]:
        return {k: (self._tot_wait[k], self._tot_cnt[k]) for k in self._lanes}

    def vehicle_summary(self) -> VehicleSummary:
        expected = int(self._conn.simulation.getMinExpectedNumber())
        active = list(self._acc.values())
        return VehicleSummary(
            arrived_waits=list(self._arrived_waits),
            active_waits=active,
            departed=self._departed,
            arrived=len(self._arrived_waits),
            pending=max(expected - len(active), 0),
            fuel_g=self._fuel_mg / 1000.0,
        )
