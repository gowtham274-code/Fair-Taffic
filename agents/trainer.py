"""Training loops for the three training architectures.

* ``independent``  - decentralised: every intersection trains its own DQN on its own
                     experience, no parameter sharing (non-federated MARL baseline).
* ``federated``    - every intersection trains locally; every ``episodes_per_round``
                     episodes the FederatedCoordinator averages the weights (FedAvg) and
                     redistributes the global model. Raw experience never leaves a client.
* ``centralized``  - ONE shared network (parameter sharing) trained on the POOLED experience
                     of all intersections. This needs the raw experience in one place, so it
                     is NOT federated; it is a separate baseline.

All three use the same environment, reward and DQN hyper-parameters.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from agents.checkpoints import save_checkpoint
from agents.marl_agent import DQNAgent
from common.config import ExperimentConfig, config_hash
from common.logging_setup import get_logger
from federated.client import FederatedClient
from federated.rounds import FederatedCoordinator
from simulation.environment import TrafficEnv

log = get_logger(__name__)
Progress = Callable[[Dict[str, Any]], None]


@dataclass
class TrainOutcome:
    agents: Dict[str, DQNAgent]  # actor per intersection (shared object in centralized mode)
    history: List[Dict[str, Any]] = field(default_factory=list)
    fl_rounds: List[Dict[str, Any]] = field(default_factory=list)
    global_weights: Optional[Dict[str, np.ndarray]] = None
    train_seconds: float = 0.0


def run_episode(env: TrafficEnv, actors: Dict[str, DQNAgent], seed: int, train: bool,
                trace: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Run one episode. train=False => greedy actions, no replay, no weight updates."""
    obs = env.reset(seed)
    done = False
    comp_sum = {"wait": 0.0, "queue": 0.0, "throughput": 0.0, "fairness": 0.0, "total": 0.0}
    n_comp = 0
    losses: List[float] = []
    while not done:
        masks = {a: env.action_mask(a) for a in env.agent_ids}
        acts = {a: actors[a].act(obs[a], masks[a], greedy=not train) for a in env.agent_ids}
        nxt, rew, done, info = env.step(acts)
        if train:
            nmasks = {a: env.action_mask(a) for a in env.agent_ids}
            for a in env.agent_ids:
                # the episode ends by time limit (truncation), not by a real terminal state,
                # so we still bootstrap from the next state: done=False in the replay buffer
                actors[a].remember(obs[a], acts[a], rew[a], nxt[a], False, nmasks[a])
                loss = actors[a].learn()
                if loss is not None:
                    losses.append(loss)
        for a in env.agent_ids:
            for k in comp_sum:
                comp_sum[k] += info["components"][a][k]
            n_comp += 1
        if trace is not None:
            for a in env.agent_ids:
                trace.append({"sim_time": info["sim_time"], "agent": a, "action": acts[a],
                              "phase": info["applied_phase"][a], "queue": info["queues"][a],
                              "obs": [round(float(x), 4) for x in obs[a]],
                              **{f"r_{k}": round(v, 5) for k, v in info["components"][a].items()}})
        obs = nxt
    metrics = env.episode_metrics()
    return {"metrics": metrics, "components": {k: v / max(n_comp, 1) for k, v in comp_sum.items()},
            "mean_loss": float(np.mean(losses)) if losses else float("nan")}


def evaluate_actors(env: TrafficEnv, actors: Dict[str, DQNAgent], seeds: List[int],
                    trace_first: bool = False, ggf_decay: float = 0.5):
    rows, details, trace = [], [], []
    for i, s in enumerate(seeds):
        res = run_episode(env, actors, s, train=False, trace=trace if (trace_first and i == 0) else None)
        m = res["metrics"]
        rows.append({"seed": s, **m.scalar_dict()})
        details.append({"seed": s, "approach_mean_wait_s": m.approach_mean_wait_s,
                        "intersection_mean_wait_s": m.intersection_mean_wait_s,
                        "intersection_avg_queue": m.intersection_avg_queue})
    return rows, details, trace


class MARLTrainer:
    def __init__(self, cfg: ExperimentConfig, mode: str, run_dir: str | Path, progress: Optional[Progress] = None):
        if mode not in ("independent", "federated", "centralized"):
            raise ValueError(f"unknown training mode {mode!r}")
        self.cfg, self.mode, self.run_dir = cfg, mode, Path(run_dir)
        self.progress = progress or (lambda _d: None)
        self.ckpt_root = self.run_dir / "checkpoints"
        self.cfg_hash = config_hash(cfg)

    # ------------------------------------------------------------------ construction
    def _build(self, env: TrafficEnv) -> Dict[str, DQNAgent]:
        c = self.cfg
        if self.mode == "centralized":
            shared = DQNAgent("shared", env.obs_dim, env.n_actions, c.rl, seed=c.seed * 1000,
                              eps_decay_scale=len(env.agent_ids))
            return {a: shared for a in env.agent_ids}
        return {a: DQNAgent(a, env.obs_dim, env.n_actions, c.rl, seed=c.seed * 1000 + 10 * i)
                for i, a in enumerate(env.agent_ids)}

    def _meta(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        return {"kind": self.mode, "config_hash": self.cfg_hash, "obs_dim": 18, "n_actions": 4,
                "hidden": self.cfg.rl.hidden, "sim_backend": self.cfg.sim.backend, **extra}

    # ------------------------------------------------------------------ main loop
    def train(self) -> TrainOutcome:
        c = self.cfg
        env = TrafficEnv(c.sim, c.reward)
        t0 = time.time()
        try:
            actors = self._build(env)
            nn_backend = next(iter(actors.values())).online.backend_name
            log.info("training mode=%s nn_backend=%s backend=%s", self.mode, nn_backend, c.sim.backend)
            out = TrainOutcome(agents=actors)
            coordinator: Optional[FederatedCoordinator] = None
            if self.mode == "federated":
                clients = [FederatedClient(a, actors[a]) for a in env.agent_ids]
                val_agent = DQNAgent("validator", env.obs_dim, env.n_actions, c.rl, seed=c.seed * 1000 + 999)

                def validator(w: Dict[str, np.ndarray]) -> Dict[str, float]:
                    val_agent.set_weights(w)
                    res = run_episode(env, {a: val_agent for a in env.agent_ids}, c.federated.validation_seed, train=False)
                    m = res["metrics"]
                    return {"avg_wait_s": m.avg_wait_s, "max_wait_s": m.max_wait_s, "jain_index": m.jain_index,
                            "ggf_score": m.ggf_score, "throughput_veh_per_h": m.throughput_veh_per_h}

                coordinator = FederatedCoordinator(clients, c.federated, seed=c.seed, validator=validator)
                total_eps = c.federated.rounds * c.federated.episodes_per_round
            else:
                total_eps = c.rl.episodes
            for ep in range(total_eps):
                seed = c.seed * 100000 + ep  # training seeds differ from evaluation seeds (eval.seeds)
                res = run_episode(env, actors, seed, train=True)
                m = res["metrics"]
                eps = float(np.mean([actors[a].epsilon() for a in env.agent_ids]))
                row = {"episode": ep + 1, "train_seed": seed, "epsilon": round(eps, 4),
                       "mean_loss": res["mean_loss"], "reward_total": res["components"]["total"],
                       "reward_wait": res["components"]["wait"], "reward_queue": res["components"]["queue"],
                       "reward_throughput": res["components"]["throughput"],
                       "reward_fairness": res["components"]["fairness"],
                       **{k: v for k, v in m.scalar_dict().items() if k != "wall_time_s"},
                       "fl_round": None}
                out.history.append(row)
                prog = {"episode": ep + 1, "total_episodes": total_eps, "message": f"episode {ep + 1}/{total_eps}"}
                if coordinator is not None and (ep + 1) % c.federated.episodes_per_round == 0:
                    rnd = (ep + 1) // c.federated.episodes_per_round
                    rec = coordinator.communication_round(rnd)
                    out.fl_rounds.append(rec.as_dict())
                    row["fl_round"] = rnd
                    save_checkpoint(self.ckpt_root, "global", {"global": coordinator.server.global_weights},
                                    self._meta({"round": rnd, "episode": ep + 1, "global_version": rec.global_version}))
                    prog.update({"round": rnd, "total_rounds": c.federated.rounds, "global_version": rec.global_version})
                    (self.run_dir / "fl_rounds.json").write_text(json.dumps(out.fl_rounds, indent=2))
                self.progress(prog)
                if self.mode != "federated" and ((ep + 1) % c.output.checkpoint_every_episodes == 0 or ep + 1 == total_eps):
                    self._save_agents(actors, ep + 1)
            if self.mode == "federated" and coordinator is not None:
                out.global_weights = {k: v.copy() for k, v in coordinator.server.global_weights.items()}
                self._save_agents(actors, total_eps, label="local_agents")
            out.train_seconds = time.time() - t0
            return out
        finally:
            env.close()

    def _save_agents(self, actors: Dict[str, DQNAgent], episode: int, label: str = "agents") -> None:
        models = {f"agent_{a}": ag.get_weights() for a, ag in actors.items()}
        d = save_checkpoint(self.ckpt_root, label, models,
                            self._meta({"episode": episode, "shared": self.mode == "centralized"}))
        if label == "agents":
            version = int(d.name[1:])
            for ag in actors.values():
                ag.version = version
