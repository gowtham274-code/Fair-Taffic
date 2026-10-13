"""SQLite registry of experiments (lightweight metadata; the result files stay on disk).

DB location: <experiments root>/../experiments.db  (e.g. storage/experiments.db).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments(
    name TEXT PRIMARY KEY, kind TEXT, variant TEXT, scenario TEXT, backend TEXT, seed INTEGER,
    status TEXT, started TEXT, ended TEXT, path TEXT, summary_json TEXT, error TEXT
)"""


def db_path(root: str | Path) -> Path:
    return Path(root).resolve().parent / "experiments.db"


def _connect(root: str | Path) -> sqlite3.Connection:
    p = db_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def upsert(root: str | Path, name: str, **fields: Any) -> None:
    allowed = {"kind", "variant", "scenario", "backend", "seed", "status", "started", "ended", "path",
               "summary_json", "error"}
    bad = set(fields) - allowed
    if bad:
        raise ValueError(f"unknown registry fields: {sorted(bad)}")
    with closing(_connect(root)) as conn, conn:
        conn.execute("INSERT OR IGNORE INTO experiments(name) VALUES (?)", (name,))
        if fields:
            sets = ", ".join(f"{k}=?" for k in fields)
            conn.execute(f"UPDATE experiments SET {sets} WHERE name=?", (*fields.values(), name))


def list_experiments(root: str | Path) -> List[Dict[str, Any]]:
    with closing(_connect(root)) as conn:
        rows = conn.execute("SELECT * FROM experiments ORDER BY started DESC, name").fetchall()
    return [dict(r) for r in rows]


def get_experiment(root: str | Path, name: str) -> Optional[Dict[str, Any]]:
    with closing(_connect(root)) as conn:
        r = conn.execute("SELECT * FROM experiments WHERE name=?", (name,)).fetchone()
    if r is None:
        return None
    d = dict(r)
    if d.get("summary_json"):
        d["summary"] = json.loads(d["summary_json"])
    return d
