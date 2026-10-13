"""Run one experiment end-to-end and persist everything needed to reproduce it.

Result directory  <output.root>/<output.name>/ :
    config.json         full validated config (reproduce with --config config.json)
    status.json         live state: queued/running/completed/failed/stopped + progress
    train_log.csv       per-episode training log (marl / centralized / federated)
    fl_rounds.json      federated rounds (participants, samples, weights, bytes, validation)
    eval_results.csv    one row of metrics per evaluation seed
    eval_details.json   per-approach / per-intersection waits and queues per seed
    summary.json        mean and std over evaluation seeds + metadata
    agent_trace.csv     decision trace of the first evaluation episode
    checkpoints/        versioned model checkpoints
An existing directory is never overwritten unless output.overwrite is true.
"""
from __future__ import annotations

import csv
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agents.marl_agent import DQNAgent
from agents.trainer import MARLTrainer, evaluate_actors
from common.config import ExperimentConfig, config_hash
from common.logging_setup import get_logger
from common.seeding import set_global_seed
from experiments import registry
from simulation.environment import TrafficEnv
from simulation.traffic_controller import make_controller

log = get_logger(__name__)


class StopRequested(Exception):
    """Raised (from a signal handler) when the job manager asks the run to stop."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_status(run_dir: Path, state: str, **extra: Any) -> None:
    data = {"state": state, "updated": _now(), **extra}
    tmp = run_dir / "status.json.tmp"
    tmp.write_text(json.dumps(data))
    tmp.replace(run_dir / "status.json")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k in rows[0]:
        if k == "seed":
            continue
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if len(vals) == len(rows):
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return out


def prepare_run_dir(cfg: ExperimentConfig) -> Path:
    run_dir = Path(cfg.output.root) / cfg.output.name
    if run_dir.exists():
        if not cfg.output.overwrite:
            raise FileExistsError(f"{run_dir} already exists; choose another output.name or set output.overwrite=true")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    return run_dir


def run_experiment(cfg: ExperimentConfig) -> Path:
    """Run the experiment described by cfg. Returns the result directory."""
    run_dir = prepare_run_dir(cfg)
    root = cfg.output.root
    (run_dir / "config.json").write_text(json.dumps(cfg.model_dump(mode="json"), indent=2))
    started = _now()
    registry.upsert(root, cfg.output.name, kind=cfg.kind, variant=cfg.variant or cfg.kind, scenario=cfg.sim.scenario,
                    backend=cfg.sim.backend, seed=cfg.seed, status="running", started=started, path=str(run_dir))
    write_status(run_dir, "running", message="starting", started=started)
    set_global_seed(cfg.seed)
    t0 = time.time()
    try:
        summary = _run(cfg, run_dir)
        summary["meta"].update({"started": started, "ended": _now(), "total_seconds": time.time() - t0})
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        write_status(run_dir, "completed", message="done", started=started, ended=_now())
        registry.upsert(root, cfg.output.name, status="completed", ended=_now(), summary_json=json.dumps(summary))
    except (StopRequested, KeyboardInterrupt):
        write_status(run_dir, "stopped", message="stopped on request", started=started, ended=_now())
        registry.upsert(root, cfg.output.name, status="stopped", ended=_now())
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        (run_dir / "error.txt").write_text(tb)
        write_status(run_dir, "failed", message=str(exc)[:300], started=started, ended=_now())
        registry.upsert(root, cfg.output.name, status="failed", ended=_now(), error=str(exc)[:500])
        log.error("experiment %s failed: %s", cfg.output.name, exc)
        raise
    return run_dir


def _run(cfg: ExperimentConfig, run_dir: Path) -> Dict[str, Any]:
    def progress(d: Dict[str, Any]) -> None:
        write_status(run_dir, "running", **d)

    meta: Dict[str, Any] = {"name": cfg.output.name, "kind": cfg.kind, "variant": cfg.variant or cfg.kind,
                            "scenario": cfg.sim.scenario, "seed": cfg.seed, "sim_backend": cfg.sim.backend,
                            "config_hash": config_hash(cfg),
                            "is_test_double": cfg.sim.backend == "mock",
                            "eval_seeds": cfg.eval.seeds}
    if cfg.sim.backend == "mock":
        meta["warning"] = "Produced with the mock queue simulator (test double), NOT SUMO. Not a research result."
    train_seconds = 0.0
    trace: List[Dict[str, Any]] = []
    if cfg.kind == "baseline":
        controller = make_controller(cfg.eval.controller, cfg.eval.fixed_green_s)
        meta["controller"] = controller.name
        env = TrafficEnv(cfg.sim, cfg.reward)
        rows, details = [], []
        try:
            for i, s in enumerate(cfg.eval.seeds):
                obs = env.reset(s)
                done = False
                while not done:
                    obs, _r, done, info = env.step(controller.act(env))
                    if i == 0:
                        for a in env.agent_ids:
                            trace.append({"sim_time": info["sim_time"], "agent": a, "phase": info["applied_phase"][a],
                                          "queue": info["queues"][a],
                                          **{f"r_{k}": round(v, 5) for k, v in info["components"][a].items()}})
                m = env.episode_metrics(cfg.reward.ggf_decay)
                rows.append({"seed": s, **m.scalar_dict()})
                details.append({"seed": s, "approach_mean_wait_s": m.approach_mean_wait_s,
                                "intersection_mean_wait_s": m.intersection_mean_wait_s,
                                "intersection_avg_queue": m.intersection_avg_queue})
                progress({"episode": i + 1, "total_episodes": len(cfg.eval.seeds), "message": f"evaluation seed {s}"})
        finally:
            env.close()
    else:
        mode = {"marl": "independent", "centralized": "centralized", "federated": "federated"}[cfg.kind]
        trainer = MARLTrainer(cfg, mode, run_dir, progress)
        out = trainer.train()
        train_seconds = out.train_seconds
        _write_csv(run_dir / "train_log.csv", out.history)
        if out.fl_rounds:
            (run_dir / "fl_rounds.json").write_text(json.dumps(out.fl_rounds, indent=2))
        meta["nn_backend"] = next(iter(out.agents.values())).online.backend_name
        meta["train_seconds"] = train_seconds
        write_status(run_dir, "running", message="evaluating", episode=len(out.history),
                     total_episodes=len(out.history))
        env = TrafficEnv(cfg.sim, cfg.reward)
        try:
            actors = out.agents
            if cfg.kind == "federated" and out.global_weights is not None:
                # evaluate the GLOBAL model on every intersection (the federated result)
                g = DQNAgent("global", env.obs_dim, env.n_actions, cfg.rl, seed=cfg.seed)
                g.set_weights(out.global_weights)
                actors = {a: g for a in env.agent_ids}
            rows, details, trace = evaluate_actors(env, actors, cfg.eval.seeds, trace_first=True,
                                                   ggf_decay=cfg.reward.ggf_decay)
        finally:
            env.close()
    _write_csv(run_dir / "eval_results.csv", rows)
    (run_dir / "eval_details.json").write_text(json.dumps(details, indent=2))
    if trace:
        _write_csv(run_dir / "agent_trace.csv", trace)
    return {"meta": meta, "metrics": summarize(rows)}
