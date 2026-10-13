"""Request / response models (pydantic) for the REST API."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from common.config import ExperimentConfig, deep_update

NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,60}$")


class ExperimentRequest(BaseModel):
    """Parameters of an experiment that may be started through the API (all bounded)."""

    kind: Literal["baseline", "marl", "centralized", "federated"] = "baseline"
    scenario: Literal["low", "medium", "high", "uneven"] = "medium"
    backend: Literal["sumo", "mock"] = "sumo"
    seed: int = Field(0, ge=0, le=1_000_000)
    duration_s: int = Field(600, ge=60, le=3600)
    controller: Literal["fixed", "queue_greedy"] = "fixed"
    episodes: int = Field(10, ge=1, le=300)
    rounds: int = Field(5, ge=1, le=50)
    episodes_per_round: int = Field(2, ge=1, le=10)
    participation_rate: float = Field(1.0, gt=0, le=1)
    w_fairness: float = Field(1.0, ge=0, le=5)
    eval_seeds: List[int] = Field(default_factory=lambda: [101, 102], min_length=1, max_length=10)
    name: Optional[str] = None
    variant: Optional[str] = Field(None, max_length=40)

    @field_validator("name")
    @classmethod
    def _name(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not NAME_RE.match(v):
            raise ValueError("name must match [A-Za-z0-9_-]{1,60}")
        return v

    @field_validator("duration_s")
    @classmethod
    def _dur(cls, v: int) -> int:
        if v % 5:
            raise ValueError("duration_s must be a multiple of 5 (the decision interval)")
        return v

    @model_validator(mode="after")
    def _eval_seeds(self) -> "ExperimentRequest":
        if any(s < 0 or s > 1_000_000 for s in self.eval_seeds):
            raise ValueError("eval_seeds must be in [0, 1000000]")
        return self

    def to_config(self, experiments_root: str, name: str) -> ExperimentConfig:
        variant = self.variant or {
            "baseline": "fixed_time" if self.controller == "fixed" else "queue_greedy",
            "marl": "marl_fair" if self.w_fairness > 0 else "marl_conventional",
            "centralized": "centralized_fair", "federated": "federated_fair"}[self.kind]
        ov: Dict[str, Any] = {
            "kind": self.kind, "seed": self.seed, "variant": variant,
            "sim": {"backend": self.backend, "scenario": self.scenario, "duration_s": self.duration_s},
            "rl": {"episodes": self.episodes},
            "reward": {"w_fairness": self.w_fairness},
            "federated": {"rounds": self.rounds, "episodes_per_round": self.episodes_per_round,
                          "participation_rate": self.participation_rate},
            "eval": {"seeds": self.eval_seeds, "controller": self.controller},
            "output": {"root": experiments_root, "name": name, "overwrite": False},
        }
        return ExperimentConfig.model_validate(ov)


class SimulationRequest(BaseModel):
    """Run one simulation with a non-learning controller (live-demo helper)."""

    scenario: Literal["low", "medium", "high", "uneven"] = "medium"
    controller: Literal["fixed", "queue_greedy"] = "fixed"
    backend: Literal["sumo", "mock"] = "sumo"
    duration_s: int = Field(600, ge=60, le=3600)
    seed: int = Field(101, ge=0, le=1_000_000)


class JobInfo(BaseModel):
    job_id: str
    experiment: str
    kind: str
    state: Literal["queued", "running", "completed", "failed", "stopped"]
    submitted: str
    started: Optional[str] = None
    ended: Optional[str] = None
    message: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None


class ErrorInfo(BaseModel):
    detail: str
