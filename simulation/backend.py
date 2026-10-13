"""Abstract traffic-simulator interface used by the environment.

Two implementations exist:
  * SumoBackend  (simulation/sumo_backend.py)  - the real research backend (SUMO + TraCI).
  * MockBackend  (simulation/mock_backend.py)  - a tiny pure-Python queue model that is a
    TEST DOUBLE for unit/integration tests and CI when SUMO is not installed. Results
    obtained with it are NOT SUMO results and must never be reported as such.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from simulation.network.topology import NetTopology
from simulation.scenarios.demand import Vehicle


@dataclass
class ApproachState:
    """Snapshot of one approach (all lanes of the incoming edge of one junction)."""

    present: bool = False
    n_vehicles: int = 0  # vehicles currently on the approach edge
    queue: int = 0  # halting vehicles (speed < 0.1 m/s)
    sum_wait: float = 0.0  # sum over vehicles of waiting accrued since entering this approach (s)
    max_wait: float = 0.0
    left_total: int = 0  # cumulative number of vehicles that have left the approach


@dataclass
class VehicleSummary:
    arrived_waits: List[float] = field(default_factory=list)  # accumulated waiting time of finished trips (s)
    active_waits: List[float] = field(default_factory=list)  # same, for vehicles still in the network
    departed: int = 0
    arrived: int = 0
    pending: int = 0  # scheduled vehicles not yet inserted into the network
    fuel_g: Optional[float] = None  # None when the backend has no emission model


class TrafficBackend(ABC):
    topology: NetTopology
    name: str = "abstract"

    @abstractmethod
    def reset(self, seed: int, vehicles: Sequence[Vehicle], duration_s: int) -> None: ...

    @abstractmethod
    def step(self) -> None:
        """Advance the simulation by exactly one second."""

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def sim_time(self) -> float: ...

    @property
    def tls_ids(self) -> List[str]:
        return self.topology.tls_ids

    @abstractmethod
    def n_green_phases(self, tls: str) -> int: ...

    @abstractmethod
    def phase_slots(self, tls: str, idx: int) -> FrozenSet[str]:
        """Slots (N/E/S/W) that get a green light in green phase ``idx``."""

    @abstractmethod
    def set_green(self, tls: str, idx: int) -> None: ...

    @abstractmethod
    def set_yellow(self, tls: str, from_idx: int, to_idx: int) -> None: ...

    @abstractmethod
    def approach_state(self, tls: str, slot: str) -> ApproachState: ...

    @abstractmethod
    def approach_totals(self) -> Dict[Tuple[str, str], Tuple[float, int]]:
        """(tls, slot) -> (total local wait seconds of vehicles that have LEFT it, their count)."""

    @abstractmethod
    def vehicle_summary(self) -> VehicleSummary: ...
