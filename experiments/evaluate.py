"""Evaluate a saved checkpoint (no training, greedy policy, no weight updates).

    python -m experiments.evaluate --run storage/experiments/marl_medium --scenario high \
        --seeds 201 202 203 --tag high_generalisation

Writes ``eval_<tag>.csv`` into the run directory (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from agents.checkpoints import load_checkpoint
from agents.marl_agent import DQNAgent
from common.config import ExperimentConfig, deep_update
from simulation.environment import TrafficEnv
from agents.trainer import evaluate_actors


def evaluate_run(run_dir: str | Path, scenario: Optional[str] = None, seeds: Optional[List[int]] = None,
                 version: Optional[int] = None):
    run = Path(run_dir)
    cfg = ExperimentConfig.model_validate(json.loads((run / "config.json").read_text()))
    if cfg.kind == "baseline":
        raise ValueError("baseline runs have no checkpoint; re-run the baseline instead")
    if scenario:
        cfg = ExperimentConfig.model_validate(deep_update(cfg.model_dump(mode="json"), {"sim": {"scenario": scenario}}))
    ck = run / "checkpoints"
    env = TrafficEnv(cfg.sim, cfg.reward)
    try:
        actors: Dict[str, DQNAgent] = {}
        if cfg.kind == "federated":
            models, _ = load_checkpoint(ck, "global", version)
            g = DQNAgent("global", env.obs_dim, env.n_actions, cfg.rl, seed=cfg.seed)
            g.set_weights(models["global"])
            actors = {a: g for a in env.agent_ids}
        else:
            models, _ = load_checkpoint(ck, "agents", version)
            for i, a in enumerate(env.agent_ids):
                ag = DQNAgent(a, env.obs_dim, env.n_actions, cfg.rl, seed=cfg.seed + i)
                ag.set_weights(models[f"agent_{a}"])
                actors[a] = ag
        return evaluate_actors(env, actors, seeds or cfg.eval.seeds, ggf_decay=cfg.reward.ggf_decay)[0]
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--tag", default="extra")
    a = ap.parse_args()
    out = Path(a.run) / f"eval_{a.tag}.csv"
    if out.exists():
        raise SystemExit(f"{out} exists; choose another --tag")
    rows = evaluate_run(a.run, a.scenario, a.seeds, a.version)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
