"""Road-network topology shared by the SUMO and the mock backends.

A NetTopology is a small directed graph (junctions, edges, allowed edge->edge
transitions) plus the geometry needed to label the four approaches of every
signalised junction as compass slots N/E/S/W ("vehicles arriving FROM that side").

It can be built either from a SUMO ``.net.xml`` file (``from_net_xml``) or synthetically
for the regular grid used in this project (``grid``). Both produce the same ids
(``J{ix}{iy}`` for signalised junctions, ``W0, E1, N0, S1 ...`` for boundary junctions,
``A_B`` for the edge from A to B), so demand generation is identical for both backends.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

SLOTS: Tuple[str, ...] = ("N", "E", "S", "W")


@dataclass
class NetTopology:
    junctions: Dict[str, Tuple[float, float, str]] = field(default_factory=dict)  # id -> (x, y, type)
    edges: Dict[str, Tuple[str, str]] = field(default_factory=dict)  # id -> (from, to)
    edge_length: Dict[str, float] = field(default_factory=dict)
    connections: Dict[str, List[str]] = field(default_factory=dict)  # edge -> next edges

    # ------------------------------------------------------------------ builders
    @classmethod
    def grid(cls, nx: int = 2, ny: int = 2, length: float = 200.0) -> "NetTopology":
        """Synthetic nx*ny grid with one boundary junction at each outer end."""
        t = cls()
        for ix in range(nx):
            for iy in range(ny):
                t.junctions[f"J{ix}{iy}"] = (ix * length, iy * length, "traffic_light")
        for iy in range(ny):
            t.junctions[f"W{iy}"] = (-length, iy * length, "dead_end")
            t.junctions[f"E{iy}"] = (nx * length, iy * length, "dead_end")
        for ix in range(nx):
            t.junctions[f"S{ix}"] = (ix * length, -length, "dead_end")
            t.junctions[f"N{ix}"] = (ix * length, ny * length, "dead_end")
        pairs: List[Tuple[str, str]] = []
        for ix in range(nx):
            for iy in range(ny):
                if ix + 1 < nx:
                    pairs.append((f"J{ix}{iy}", f"J{ix + 1}{iy}"))
                if iy + 1 < ny:
                    pairs.append((f"J{ix}{iy}", f"J{ix}{iy + 1}"))
        for iy in range(ny):
            pairs.append((f"W{iy}", f"J0{iy}"))
            pairs.append((f"J{nx - 1}{iy}", f"E{iy}"))
        for ix in range(nx):
            pairs.append((f"S{ix}", f"J{ix}0"))
            pairs.append((f"J{ix}{ny - 1}", f"N{ix}"))
        for a, b in pairs:
            for f, to in ((a, b), (b, a)):
                eid = f"{f}_{to}"
                t.edges[eid] = (f, to)
                t.edge_length[eid] = length
        t._build_connections_no_uturn()
        return t

    def _build_connections_no_uturn(self) -> None:
        out_by_from: Dict[str, List[str]] = {}
        for eid, (f, _t) in self.edges.items():
            out_by_from.setdefault(f, []).append(eid)
        self.connections = {}
        for eid, (f, to) in self.edges.items():
            self.connections[eid] = sorted(e for e in out_by_from.get(to, []) if self.edges[e][1] != f)

    @classmethod
    def from_net_xml(cls, path: str | Path) -> "NetTopology":
        """Parse a SUMO network file (plain XML parsing, no sumolib needed)."""
        root = ET.parse(str(path)).getroot()
        t = cls()
        for j in root.iter("junction"):
            jid = j.get("id", "")
            if jid.startswith(":"):
                continue
            t.junctions[jid] = (float(j.get("x", 0)), float(j.get("y", 0)), j.get("type", ""))
        for e in root.iter("edge"):
            eid = e.get("id", "")
            if eid.startswith(":") or e.get("function") == "internal":
                continue
            t.edges[eid] = (e.get("from", ""), e.get("to", ""))
            lane = e.find("lane")
            t.edge_length[eid] = float(lane.get("length", 100.0)) if lane is not None else 100.0
        conns: Dict[str, Set[str]] = {e: set() for e in t.edges}
        for c in root.iter("connection"):
            f, to = c.get("from", ""), c.get("to", "")
            if f in conns and to in t.edges:
                conns[f].add(to)
        t.connections = {e: sorted(v) for e, v in conns.items()}
        return t

    # ------------------------------------------------------------------ queries
    @property
    def tls_ids(self) -> List[str]:
        return sorted(j for j, (_x, _y, ty) in self.junctions.items() if ty == "traffic_light")

    def incoming_slots(self, tls: str) -> Dict[str, str]:
        """slot (N/E/S/W = side vehicles come FROM) -> incoming edge id."""
        jx, jy, _ = self.junctions[tls]
        res: Dict[str, str] = {}
        for eid, (f, to) in self.edges.items():
            if to != tls:
                continue
            fx, fy, _ = self.junctions[f]
            dx, dy = jx - fx, jy - fy
            if abs(dx) >= abs(dy):
                slot = "W" if dx > 0 else "E"
            else:
                slot = "S" if dy > 0 else "N"
            res[slot] = eid
        return res

    def neighbors(self, tls: str) -> List[str]:
        tl = set(self.tls_ids)
        nb = {f for (f, to) in self.edges.values() if to == tls and f in tl}
        nb |= {to for (f, to) in self.edges.values() if f == tls and to in tl}
        return sorted(nb)

    def _boundary(self) -> Set[str]:
        tl = self.tls_ids
        return {j for j in self.junctions if j not in tl}

    def side_of_boundary(self, jid: str) -> str:
        xs = [self.junctions[t][0] for t in self.tls_ids]
        ys = [self.junctions[t][1] for t in self.tls_ids]
        x, y, _ = self.junctions[jid]
        if x < min(xs):
            return "W"
        if x > max(xs):
            return "E"
        if y < min(ys):
            return "S"
        return "N"

    def entry_edges(self) -> List[Tuple[str, str]]:
        """(edge id, side) for edges that start at a boundary junction and end at a signal."""
        b = self._boundary()
        tl = set(self.tls_ids)
        return sorted((e, self.side_of_boundary(f)) for e, (f, to) in self.edges.items() if f in b and to in tl)

    def exit_edges(self) -> List[str]:
        b = self._boundary()
        tl = set(self.tls_ids)
        return sorted(e for e, (f, to) in self.edges.items() if to in b and f in tl)

    def route(self, entry: str, exit_: str) -> Optional[List[str]]:
        """Shortest edge path (fewest edges) from entry to exit following allowed connections."""
        if entry not in self.edges or exit_ not in self.edges:
            return None
        prev: Dict[str, Optional[str]] = {entry: None}
        dq: Deque[str] = deque([entry])
        while dq:
            cur = dq.popleft()
            if cur == exit_:
                path = []
                node: Optional[str] = cur
                while node is not None:
                    path.append(node)
                    node = prev[node]
                return path[::-1]
            for nxt in self.connections.get(cur, []):
                if nxt not in prev:
                    prev[nxt] = cur
                    dq.append(nxt)
        return None

    def validate(self, expect_tls: Optional[int] = None) -> List[str]:
        """Return a list of problems (empty list = OK)."""
        problems: List[str] = []
        if expect_tls is not None and len(self.tls_ids) != expect_tls:
            problems.append(f"expected {expect_tls} traffic-light junctions, found {len(self.tls_ids)}")
        for tls in self.tls_ids:
            if len(self.incoming_slots(tls)) < 3:
                problems.append(f"{tls} has fewer than 3 incoming approaches")
        ents, exits = self.entry_edges(), self.exit_edges()
        if not ents or not exits:
            problems.append("no entry or exit edges found")
        reachable = sum(1 for e, _ in ents for x in exits if self.route(e, x))
        if reachable == 0:
            problems.append("no entry edge can reach any exit edge")
        return problems
