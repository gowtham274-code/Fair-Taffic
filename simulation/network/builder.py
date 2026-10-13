"""Generate the SUMO network for the 2x2 (or nx x ny) signalised grid.

We write SUMO *plain XML* node/edge files and call ``netconvert``; junctions marked
``traffic_light`` get SUMO's default static signal program. No hand-written .net.xml.

    python -m simulation.network.builder --out simulation/network/grid2x2

Requires the SUMO binary ``netconvert`` (not available in every environment; see docs).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict

from common.logging_setup import get_logger
from simulation.network.topology import NetTopology

log = get_logger(__name__)


def write_plain_files(out_dir: str | Path, nx: int = 2, ny: int = 2, length: float = 200.0,
                      speed: float = 13.89, lanes: int = 1) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    topo = NetTopology.grid(nx, ny, length)
    nod = ['<?xml version="1.0" encoding="UTF-8"?>', "<nodes>"]
    for jid, (x, y, ty) in sorted(topo.junctions.items()):
        sumo_type = "traffic_light" if ty == "traffic_light" else "priority"
        nod.append(f'  <node id="{jid}" x="{x:.1f}" y="{y:.1f}" type="{sumo_type}"/>')
    nod.append("</nodes>")
    edg = ['<?xml version="1.0" encoding="UTF-8"?>', "<edges>"]
    for eid, (f, t) in sorted(topo.edges.items()):
        edg.append(f'  <edge id="{eid}" from="{f}" to="{t}" numLanes="{lanes}" speed="{speed}"/>')
    edg.append("</edges>")
    nod_p, edg_p = out / "grid.nod.xml", out / "grid.edg.xml"
    nod_p.write_text("\n".join(nod) + "\n")
    edg_p.write_text("\n".join(edg) + "\n")
    return {"nod": nod_p, "edg": edg_p}


def build_network(out_dir: str | Path, nx: int = 2, ny: int = 2, length: float = 200.0,
                  netconvert: str = "netconvert", force: bool = False) -> Path:
    """Create ``grid.net.xml`` in out_dir (reuses an existing file unless force)."""
    out = Path(out_dir)
    net = out / "grid.net.xml"
    files = write_plain_files(out, nx, ny, length)
    if net.exists() and not force:
        return net
    exe = shutil.which(netconvert)
    if exe is None:
        raise RuntimeError(
            "netconvert not found. Install SUMO (https://sumo.dlr.de/docs/Installing/) and make sure "
            "'netconvert' is on PATH, or set sim.netconvert_binary / SUMO_HOME. "
            "Use backend 'mock' only for tests."
        )
    cmd = [exe, "--node-files", str(files["nod"]), "--edge-files", str(files["edg"]),
           "--output-file", str(net), "--tls.default-type", "static", "--no-warnings", "true"]
    log.info("running %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0 or not net.exists():
        raise RuntimeError(f"netconvert failed: {res.stderr[-800:]}")
    problems = NetTopology.from_net_xml(net).validate(expect_tls=nx * ny)
    if problems:
        raise RuntimeError("generated network failed validation: " + "; ".join(problems))
    return net


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="simulation/network/grid2x2")
    ap.add_argument("--nx", type=int, default=2)
    ap.add_argument("--ny", type=int, default=2)
    ap.add_argument("--length", type=float, default=200.0)
    ap.add_argument("--netconvert", default="netconvert")
    ap.add_argument("--plain-only", action="store_true", help="only write .nod/.edg files")
    a = ap.parse_args()
    if a.plain_only:
        write_plain_files(a.out, a.nx, a.ny, a.length)
    else:
        print(build_network(a.out, a.nx, a.ny, a.length, a.netconvert, force=True))


if __name__ == "__main__":
    main()
