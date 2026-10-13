"""Aggregate experiment results into comparison tables (mean +- std over runs and seeds).

Rows of every ``eval_results.csv`` are pooled per (backend, scenario, variant); a "variant" is
the ``variant`` label of the run (or its kind). Nothing is invented: if a group has a single
run the std only reflects the evaluation seeds.

    python -m experiments.compare_results --root storage/experiments --out reports/comparison
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

METRICS = ["avg_wait_s", "max_wait_s", "wait_variance", "jain_index", "ggf_score", "throughput_veh_per_h",
           "unfinished_vehicles", "avg_queue", "fuel_g"]
BETTER = {"avg_wait_s": "lower", "max_wait_s": "lower", "wait_variance": "lower", "jain_index": "higher",
          "ggf_score": "higher (closer to 0)", "throughput_veh_per_h": "higher", "unfinished_vehicles": "lower",
          "avg_queue": "lower", "fuel_g": "lower"}


def load_runs(root: str | Path) -> List[Dict[str, Any]]:
    runs = []
    for d in sorted(Path(root).glob("*")):
        sp, cp, ep = d / "summary.json", d / "config.json", d / "eval_results.csv"
        if not (sp.exists() and cp.exists() and ep.exists()):
            continue
        summ = json.loads(sp.read_text())
        with open(ep) as f:
            rows = list(csv.DictReader(f))
        runs.append({"name": d.name, "meta": summ["meta"], "rows": rows, "path": str(d)})
    return runs


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def aggregate(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in runs:
        m = r["meta"]
        groups.setdefault((m["sim_backend"], m["scenario"], m.get("variant") or m["kind"]), []).append(r)
    out = []
    for (backend, scenario, variant), rs in sorted(groups.items()):
        row: Dict[str, Any] = {"backend": backend, "scenario": scenario, "variant": variant,
                               "n_runs": len(rs), "n_eval_episodes": sum(len(r["rows"]) for r in rs),
                               "runs": ",".join(r["name"] for r in rs)}
        for k in METRICS:
            vals = np.array([_f(x.get(k, "nan")) for r in rs for x in r["rows"]], dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
            row[f"{k}_std"] = float(vals.std()) if vals.size else float("nan")
        out.append(row)
    return out


def to_markdown(rows: List[Dict[str, Any]]) -> str:
    cols = ["avg_wait_s", "max_wait_s", "jain_index", "ggf_score", "throughput_veh_per_h", "unfinished_vehicles"]
    lines = ["| backend | scenario | variant | runs | " + " | ".join(cols) + " |",
             "|---|---|---|---|" + "---|" * len(cols)]
    for r in rows:
        cells = [f"{r[c + '_mean']:.2f} ± {r[c + '_std']:.2f}" for c in cols]
        lines.append(f"| {r['backend']} | {r['scenario']} | {r['variant']} | {r['n_runs']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Better: " + "; ".join(f"{k} {v}" for k, v in BETTER.items() if k in cols))
    if any(r["backend"] == "mock" for r in rows):
        lines.append("\n> Rows with backend `mock` come from a test double, not SUMO, and are not research results.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="storage/experiments")
    ap.add_argument("--out", default="reports/comparison", help="output path prefix (.csv and .md are written)")
    a = ap.parse_args()
    rows = aggregate(load_runs(a.root))
    if not rows:
        raise SystemExit(f"no completed experiments with results under {a.root}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out.with_suffix(".md").write_text(to_markdown(rows))
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
