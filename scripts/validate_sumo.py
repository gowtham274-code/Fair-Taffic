"""Validate the SUMO integration on a machine with SUMO installed.

    python scripts/validate_sumo.py

Checks: binaries on PATH, TraCI import, network build, signal programs, one short episode with the
fixed-time controller (vehicles enter/leave, metrics finite, connection closed). Exits non-zero on failure.
"""
import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    for b in ("sumo", "netconvert"):
        if shutil.which(b) is None:
            print(f"FAIL: {b} not found on PATH (see docs/DEPLOYMENT.md)")
            return 1
    try:
        import traci  # noqa: F401
    except ImportError:
        print("FAIL: python package 'traci' missing (pip install traci sumolib eclipse-sumo)")
        return 1
    from common.config import ExperimentConfig
    from simulation.environment import TrafficEnv
    from simulation.traffic_controller import make_controller

    work = tempfile.mkdtemp()
    cfg = ExperimentConfig.model_validate({"sim": {"backend": "sumo", "duration_s": 300, "work_dir": work}})
    with TrafficEnv(cfg.sim, cfg.reward) as env:
        ctl = make_controller("fixed", 30)
        env.reset(101)  # starts SUMO; signal programs are only known after this
        print("signalised intersections:", env.agent_ids,
              {a: env.n_green(a) for a in env.agent_ids}, "green phases")
        done = False
        while not done:
            _, _, done, _ = env.step(ctl.act(env))
        m = env.episode_metrics().scalar_dict()
    print(m)
    ok = m["departed_vehicles"] > 0 and m["completed_vehicles"] > 0 and all(
        math.isfinite(v) for v in m.values() if isinstance(v, float))
    print("OK" if ok else "FAIL: no traffic moved or non-finite metrics")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
