"""Central, validated configuration (pydantic). One ExperimentConfig describes a run
completely and is saved next to its results so the run can be reproduced."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from fairness.reward import RewardConfig

ScenarioName = Literal["low", "medium", "high", "uneven"]
ExperimentKind = Literal["baseline", "marl", "centralized", "federated"]


class SimConfig(BaseModel):
    backend: Literal["sumo", "mock"] = "sumo"
    scenario: ScenarioName = "medium"
    grid_x: int = Field(2, ge=2, le=4)
    grid_y: int = Field(2, ge=2, le=4)
    edge_length_m: float = Field(200.0, gt=50)
    duration_s: int = Field(1800, ge=60, le=7200)
    delta_time: int = Field(5, ge=1, le=60, description="seconds between agent decisions")
    yellow_time: int = Field(3, ge=1, le=10)
    min_green: int = Field(10, ge=1, le=120)
    demand_scale: float = Field(1.0, gt=0, le=5)
    gui: bool = False
    sumo_binary: str = "sumo"
    netconvert_binary: str = "netconvert"
    work_dir: str = "storage/sumo_work"

    @model_validator(mode="after")
    def _check(self) -> "SimConfig":
        if self.yellow_time > self.delta_time:
            raise ValueError("yellow_time must be <= delta_time so a phase change completes within one decision interval")
        if self.duration_s % self.delta_time != 0:
            raise ValueError("duration_s must be a multiple of delta_time")
        return self


class RLConfig(BaseModel):
    gamma: float = Field(0.95, ge=0, le=1)
    lr: float = Field(1e-3, gt=0)
    batch_size: int = Field(64, ge=1)
    buffer_size: int = Field(20000, ge=100)
    hidden: List[int] = Field(default_factory=lambda: [64, 64])
    eps_start: float = Field(1.0, ge=0, le=1)
    eps_end: float = Field(0.05, ge=0, le=1)
    eps_decay_steps: int = Field(3000, ge=1, description="agent decision steps over which epsilon decays linearly")
    target_update_every: int = Field(200, ge=1, description="learn steps between target-network syncs")
    learn_start: int = Field(256, ge=1)
    train_freq: int = Field(1, ge=1)
    grad_clip: float = Field(5.0, gt=0)
    episodes: int = Field(30, ge=1)
    nn_backend: Literal["auto", "torch", "numpy"] = "auto"


class FederatedConfig(BaseModel):
    rounds: int = Field(10, ge=1)
    episodes_per_round: int = Field(2, ge=1)
    participation_rate: float = Field(1.0, gt=0, le=1)
    aggregation: Literal["fedavg"] = "fedavg"
    weight_by: Literal["samples", "uniform"] = "samples"
    distribute_to: Literal["participants", "all"] = "participants"
    validate_every_rounds: int = Field(1, ge=0, description="0 disables server-side validation episodes")
    validation_seed: int = 9999


class EvalConfig(BaseModel):
    seeds: List[int] = Field(default_factory=lambda: [101, 102, 103])
    fixed_green_s: int = Field(30, ge=1, description="green time per phase of the fixed-time controller")
    controller: Literal["fixed", "queue_greedy"] = "fixed"


class OutputConfig(BaseModel):
    root: str = "storage/experiments"
    name: str = "run"
    overwrite: bool = False
    checkpoint_every_episodes: int = Field(10, ge=1)

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        if not v or any(c in v for c in "/\\.") or len(v) > 80:
            raise ValueError("name must be 1-80 chars without '/', '\\' or '.'")
        return v


class ExperimentConfig(BaseModel):
    kind: ExperimentKind = "marl"
    seed: int = Field(0, ge=0)
    sim: SimConfig = Field(default_factory=SimConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    federated: FederatedConfig = Field(default_factory=FederatedConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    variant: str = Field("", description="label used to group runs in comparisons, e.g. marl_fair")
    description: str = ""


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    """Load a YAML/JSON config file (optional), apply nested overrides, validate."""
    data: Dict[str, Any] = {}
    if path is not None:
        text = Path(path).read_text()
        data = yaml.safe_load(text) or {}
    if overrides:
        data = deep_update(data, overrides)
    return ExperimentConfig.model_validate(data)


def config_hash(cfg: ExperimentConfig) -> str:
    blob = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
