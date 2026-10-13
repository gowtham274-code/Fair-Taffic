"""Runtime settings from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(v: str | None, default: bool) -> bool:
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    storage_dir: Path
    api_key: str | None
    allow_mock: bool
    dev_open: bool
    project_root: Path

    @property
    def experiments_root(self) -> Path:
        return self.storage_dir / "experiments"

    @property
    def jobs_dir(self) -> Path:
        return self.storage_dir / "jobs"

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(__file__).resolve().parent.parent
        return cls(
            storage_dir=Path(os.environ.get("FAIRTRAFFIC_STORAGE", str(root / "storage"))).resolve(),
            api_key=os.environ.get("FAIRTRAFFIC_API_KEY") or None,
            allow_mock=_bool(os.environ.get("FAIRTRAFFIC_ALLOW_MOCK"), False),
            dev_open=_bool(os.environ.get("FAIRTRAFFIC_DEV_OPEN"), False),
            project_root=root,
        )
