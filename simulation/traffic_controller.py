"""Non-learning baseline controllers that act through the same environment.

Using the same environment (same yellow-time handling, min-green, decision interval)
makes the comparison with the learned policies fair.

* FixedTimeController   : cycles through the green phases, ``green_s`` seconds each
                          (pre-timed control; ignores traffic).
* QueueGreedyController : simple actuated/adaptive rule: serve the phase whose approaches
                          hold the most halting vehicles; switch only if the other phase has
                          clearly more queue (hysteresis) and min-green allows it.
                          It is a conventional (non-learning) adaptive baseline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from simulation.environment import TrafficEnv


class Controller(ABC):
    name = "controller"

    @abstractmethod
    def act(self, env: TrafficEnv) -> Dict[str, int]: ...


class FixedTimeController(Controller):
    name = "fixed_time"

    def __init__(self, green_s: int = 30):
        if green_s < 1:
            raise ValueError("green_s must be >= 1")
        self.green_s = green_s

    def act(self, env: TrafficEnv) -> Dict[str, int]:
        step = int(env.sim_time // self.green_s)
        return {a: step % env.n_green(a) for a in env.agent_ids}


class QueueGreedyController(Controller):
    name = "queue_greedy"

    def __init__(self, hysteresis: float = 1.0):
        self.hysteresis = hysteresis

    def act(self, env: TrafficEnv) -> Dict[str, int]:
        actions: Dict[str, int] = {}
        for a in env.agent_ids:
            cur = env.current_phase(a)
            mask = env.action_mask(a)
            scores = []
            for p in range(env.n_green(a)):
                scores.append(sum(env.approach(a, s).queue for s in env.phase_slots(a, p)))
            best = max((p for p in range(len(scores)) if mask[p]), key=lambda p: scores[p], default=cur)
            actions[a] = best if scores[best] > scores[cur] + self.hysteresis else cur
        return actions


def make_controller(kind: str, fixed_green_s: int = 30) -> Controller:
    if kind == "fixed":
        return FixedTimeController(fixed_green_s)
    if kind == "queue_greedy":
        return QueueGreedyController()
    raise ValueError(f"unknown controller {kind!r}")
