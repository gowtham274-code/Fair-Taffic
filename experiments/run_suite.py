"""Run the full comparison: for every scenario and training seed run

    fixed_time, queue_greedy        (non-learning baselines)
    marl_conventional               (RL, delay-only reward: reward.w_fairness = 0)
    marl_fair                       (RL + GGF fairness reward)
    centralized_fair                (shared network, pooled experience - NOT federated)
    federated_fair                  (RL + GGF reward + FedAvg between intersections)

on identical demand (same evaluation seeds). Existing result directories are skipped, never
overwritten. Use --dry-run to list what would run.

    python -m experiments.run_suite --config experiments/configs/default.yaml \
        --scenarios low medium high uneven --seeds 0 1 2 --prefix suite
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common.config import deep_update, load_config
from experiments.cli import parse_overrides
from experiments.runner import run_experiment

VARIANTS: Dict[str, Dict[str, Any]] = {
    "fixed_time": {"kind": "baseline", "eval": {"controller": "fixed"}},
    "queue_greedy": {"kind": "baseline", "eval": {"controller": "queue_greedy"}},
    "marl_conventional": {"kind": "marl", "reward": {"w_fairness": 0.0}},
    "marl_fair": {"kind": "marl"},
    "centralized_fair": {"kind": "centralized"},
    "federated_fair": {"kind": "federated"},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="experiments/configs/default.yaml")
    ap.add_argument("--scenarios", nargs="+", default=["low", "medium", "high", "uneven"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--prefix", default="suite")
    ap.add_argument("--backend", choices=["sumo", "mock"], default=None)
    ap.add_argument("--set", dest="sets", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    common = parse_overrides(a.sets)
    if a.backend:
        common = deep_update(common, {"sim": {"backend": a.backend}})
    todo: List[Any] = []
    for sc in a.scenarios:
        for v in a.variants:
            # baselines are deterministic controllers: one run per scenario is enough
            seeds = [a.seeds[0]] if VARIANTS[v]["kind"] == "baseline" else a.seeds
            for s in seeds:
                name = f"{a.prefix}_{sc}_{v}_s{s}"
                ov = deep_update(deep_update(common, VARIANTS[v]),
                                 {"seed": s, "variant": v, "sim": {"scenario": sc}, "output": {"name": name}})
                todo.append((name, load_config(a.config, ov)))
    for name, cfg in todo:
        exists = (Path(cfg.output.root) / name).exists()
        print(("skip (exists) " if exists else "run  ") + name)
        if a.dry_run or exists:
            continue
        run_experiment(cfg)


if __name__ == "__main__":
    main()
