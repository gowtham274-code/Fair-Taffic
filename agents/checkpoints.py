"""Versioned, safe model checkpoints.

Layout:  <root>/<label>/v0001/{meta.json, <name>.npz, ...}

* Weights are stored as ``.npz`` and loaded with ``allow_pickle=False`` (no pickle
  => loading a checkpoint cannot execute code).
* ``meta.json`` records a SHA-256 for every weight file; ``load_checkpoint`` verifies it,
  checks that all arrays are finite and (optionally) that shapes match the expected ones.
* Versions are never overwritten: every save creates the next version number.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

Weights = Dict[str, np.ndarray]
_VER = re.compile(r"^v(\d{4})$")
_SAFE = re.compile(r"^[A-Za-z0-9_\-]+$")


class CheckpointError(Exception):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions(label_dir: Path) -> List[int]:
    if not label_dir.exists():
        return []
    return sorted(int(m.group(1)) for d in label_dir.iterdir() if d.is_dir() and (m := _VER.match(d.name)))


def save_checkpoint(root: str | Path, label: str, models: Mapping[str, Weights], meta: Optional[Dict[str, Any]] = None) -> Path:
    """Save one or more named weight dicts as the next version of ``label``. Returns the directory."""
    if not _SAFE.match(label):
        raise CheckpointError(f"unsafe checkpoint label {label!r}")
    label_dir = Path(root) / label
    version = (_versions(label_dir) or [0])[-1] + 1
    d = label_dir / f"v{version:04d}"
    d.mkdir(parents=True, exist_ok=False)
    files: Dict[str, str] = {}
    for name, w in models.items():
        if not _SAFE.match(name):
            raise CheckpointError(f"unsafe model name {name!r}")
        for k, arr in w.items():
            if not np.all(np.isfinite(arr)):
                raise CheckpointError(f"non-finite values in {name}/{k}; refusing to save")
        f = d / f"{name}.npz"
        np.savez(f, **w)
        files[name] = _sha256(f)
    info = {"label": label, "version": version, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": files}
    info.update(meta or {})
    (d / "meta.json").write_text(json.dumps(info, indent=2, default=str))
    return d


def list_checkpoints(root: str | Path, label: str) -> List[Dict[str, Any]]:
    label_dir = Path(root) / label
    out = []
    for v in _versions(label_dir):
        mp = label_dir / f"v{v:04d}" / "meta.json"
        if mp.exists():
            out.append(json.loads(mp.read_text()))
    return out


def load_checkpoint(root: str | Path, label: str, version: Optional[int] = None,
                    expected_shapes: Optional[Mapping[str, Tuple[int, ...]]] = None) -> Tuple[Dict[str, Weights], Dict[str, Any]]:
    """Load (models, meta). ``version=None`` loads the latest. Verifies hashes/finite/shapes."""
    label_dir = Path(root) / label
    versions = _versions(label_dir)
    if not versions:
        raise CheckpointError(f"no checkpoint found for label {label!r} under {root}")
    v = version if version is not None else versions[-1]
    if v not in versions:
        raise CheckpointError(f"version {v} of {label!r} does not exist (have {versions})")
    d = label_dir / f"v{v:04d}"
    meta = json.loads((d / "meta.json").read_text())
    models: Dict[str, Weights] = {}
    for name, digest in meta.get("files", {}).items():
        f = d / f"{name}.npz"
        if not f.exists():
            raise CheckpointError(f"missing file {f}")
        if _sha256(f) != digest:
            raise CheckpointError(f"checksum mismatch for {f} (file corrupted or modified)")
        with np.load(f, allow_pickle=False) as z:
            w = {k: z[k] for k in z.files}
        for k, arr in w.items():
            if not np.all(np.isfinite(arr)):
                raise CheckpointError(f"non-finite values in {name}/{k}")
            if expected_shapes is not None and k in expected_shapes and tuple(arr.shape) != tuple(expected_shapes[k]):
                raise CheckpointError(f"shape mismatch for {name}/{k}: {arr.shape} vs {expected_shapes[k]}")
        models[name] = w
    return models, meta
