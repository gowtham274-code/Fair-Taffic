"""Read-only access to recorded experiment results (shared by the API and the dashboard).

Everything returned here comes from files written by ``experiments.runner``; nothing is generated.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.checkpoints import list_checkpoints
from experiments import registry
from experiments.compare_results import aggregate, load_runs

_NAME = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")


def _safe_dir(root: Path, name: str) -> Path:
    if not _NAME.match(name):
        raise ValueError("invalid experiment name")
    d = (root / name).resolve()
    if root.resolve() not in d.parents:
        raise ValueError("invalid experiment path")
    return d


def _csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({k: _num(v) for k, v in r.items()})
    return out


def _num(v: str) -> Any:
    try:
        return float(v) if v not in ("", None) else None
    except ValueError:
        return v


def _json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


class ResultsService:
    def __init__(self, experiments_root: Path):
        self.root = Path(experiments_root)

    def list_experiments(self) -> List[Dict[str, Any]]:
        rows = registry.list_experiments(self.root) if self.root.exists() else []
        for r in rows:
            r.pop("summary_json", None)
        return rows

    def status(self, name: str) -> Optional[Dict[str, Any]]:
        return _json(_safe_dir(self.root, name) / "status.json")

    def experiment(self, name: str) -> Optional[Dict[str, Any]]:
        d = _safe_dir(self.root, name)
        if not d.exists():
            return None
        return {
            "name": name,
            "config": _json(d / "config.json"),
            "status": _json(d / "status.json"),
            "summary": _json(d / "summary.json"),
            "eval_results": _csv(d / "eval_results.csv"),
            "eval_details": _json(d / "eval_details.json"),
            "train_log": _csv(d / "train_log.csv"),
            "fl_rounds": _json(d / "fl_rounds.json"),
            "agent_trace": _csv(d / "agent_trace.csv"),
            "checkpoints": {lbl: list_checkpoints(d / "checkpoints", lbl)
                            for lbl in (p.name for p in (d / "checkpoints").glob("*"))} if (d / "checkpoints").exists() else {},
            "error": (d / "error.txt").read_text()[-2000:] if (d / "error.txt").exists() else None,
        }

    def csv_path(self, name: str, file: str = "eval_results.csv") -> Optional[Path]:
        if file not in {"eval_results.csv", "train_log.csv", "agent_trace.csv"}:
            raise ValueError("file not exportable")
        p = _safe_dir(self.root, name) / file
        return p if p.exists() else None

    def comparison(self) -> List[Dict[str, Any]]:
        return aggregate(load_runs(self.root)) if self.root.exists() else []

    def latest_federated(self) -> Optional[Dict[str, Any]]:
        runs = [r for r in self.list_experiments() if r.get("kind") == "federated" and r.get("status") == "completed"]
        if not runs:
            return None
        name = runs[0]["name"]
        d = _safe_dir(self.root, name)
        rounds = _json(d / "fl_rounds.json") or []
        return {"experiment": name, "rounds": rounds, "n_rounds": len(rounds),
                "global_version": rounds[-1]["global_version"] if rounds else None,
                "checkpoints": list_checkpoints(d / "checkpoints", "global")}

    def intersections(self) -> Optional[Dict[str, Any]]:
        runs = [r for r in self.list_experiments() if r.get("status") == "completed"]
        if not runs:
            return None
        d = _safe_dir(self.root, runs[0]["name"])
        details = _json(d / "eval_details.json") or []
        return {"experiment": runs[0]["name"], "seed": details[0]["seed"] if details else None,
                "intersection_mean_wait_s": details[0]["intersection_mean_wait_s"] if details else {},
                "intersection_avg_queue": details[0]["intersection_avg_queue"] if details else {},
                "approach_mean_wait_s": details[0]["approach_mean_wait_s"] if details else {}}
