"""Shared command-line plumbing for the experiment scripts."""
from __future__ import annotations

import argparse
import ast
import signal
import sys
from typing import Any, Dict, List, Optional

from common.config import ExperimentConfig, deep_update, load_config
from experiments.runner import StopRequested, run_experiment


def parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """['sim.scenario=high', 'rl.episodes=5'] -> nested dict (values parsed as Python literals)."""
    out: Dict[str, Any] = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"bad --set {p!r}; expected key=value")
        key, val = p.split("=", 1)
        try:
            v = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            v = val
        d = out
        parts = key.split(".")
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = v
    return out


def build_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=None, help="YAML/JSON experiment config")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set sim.scenario=high (repeatable)")
    ap.add_argument("--name", default=None, help="result directory name (output.name)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--backend", choices=["sumo", "mock"], default=None,
                    help="'mock' is a test double (NOT SUMO); never use it for research numbers")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing result directory")
    return ap


def config_from_args(args: argparse.Namespace, forced: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    ov = parse_overrides(args.sets)
    if args.name:
        ov = deep_update(ov, {"output": {"name": args.name}})
    if args.seed is not None:
        ov = deep_update(ov, {"seed": args.seed})
    if args.backend:
        ov = deep_update(ov, {"sim": {"backend": args.backend}})
    if args.overwrite:
        ov = deep_update(ov, {"output": {"overwrite": True}})
    if forced:
        ov = deep_update(ov, forced)
    return load_config(args.config, ov)


def main_run(description: str, forced: Optional[Dict[str, Any]] = None) -> int:
    args = build_parser(description).parse_args()
    cfg = config_from_args(args, forced)

    def _term(_sig: int, _frm: Any) -> None:
        raise StopRequested()

    signal.signal(signal.SIGTERM, _term)
    try:
        run_dir = run_experiment(cfg)
    except StopRequested:
        print("stopped", file=sys.stderr)
        return 130
    print(f"results in {run_dir}")
    return 0
