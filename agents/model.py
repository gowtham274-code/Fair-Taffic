"""Q-networks used by the intersection agents.

Two interchangeable implementations with the SAME weight layout (a dict of NumPy
arrays ``W0, b0, W1, b1, ...`` with W of shape (in, out)):

  * TorchQNet  - PyTorch MLP (the project's intended backend).
  * NumpyQNet  - pure-NumPy MLP with manual back-propagation + Adam. It exists so the
    full pipeline (training, federated averaging, tests) can run where PyTorch is not
    installed (e.g. CI, restricted sandboxes). It is mathematically the same model
    (ReLU MLP, Huber loss, Adam, gradient clipping) but is slower.

``build_qnet(..., backend="auto")`` uses PyTorch if importable, otherwise NumPy, and
logs which one is used. Because weights are exchanged as plain NumPy dicts, federated
averaging and checkpoints do not depend on the backend.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

import numpy as np

from common.logging_setup import get_logger

log = get_logger(__name__)
Weights = Dict[str, np.ndarray]


class QNetwork(ABC):
    backend_name = "abstract"

    @abstractmethod
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Q-values, shape (batch, n_actions). No gradient / no update."""

    @abstractmethod
    def train_on_batch(self, obs: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        """One Adam step on Huber(Q(s,a), target). Returns the loss."""

    @abstractmethod
    def get_weights(self) -> Weights: ...

    @abstractmethod
    def set_weights(self, w: Weights) -> None: ...


def layer_sizes(obs_dim: int, hidden: Sequence[int], n_actions: int) -> List[int]:
    return [obs_dim, *hidden, n_actions]


# ---------------------------------------------------------------------------- NumPy
class NumpyQNet(QNetwork):
    backend_name = "numpy"

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int], lr: float, grad_clip: float, seed: int):
        rng = np.random.default_rng(seed)
        sizes = layer_sizes(obs_dim, hidden, n_actions)
        self.n_layers = len(sizes) - 1
        self.p: Weights = {}
        for i in range(self.n_layers):
            self.p[f"W{i}"] = (rng.standard_normal((sizes[i], sizes[i + 1])) * np.sqrt(2.0 / sizes[i])).astype(np.float32)
            self.p[f"b{i}"] = np.zeros(sizes[i + 1], dtype=np.float32)
        self.lr, self.grad_clip = lr, grad_clip
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def _forward(self, x: np.ndarray):
        acts = [x]
        for i in range(self.n_layers):
            z = acts[-1] @ self.p[f"W{i}"] + self.p[f"b{i}"]
            acts.append(np.maximum(z, 0.0) if i < self.n_layers - 1 else z)
        return acts

    def predict(self, obs: np.ndarray) -> np.ndarray:
        return self._forward(np.asarray(obs, dtype=np.float32))[-1]

    def train_on_batch(self, obs: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        x = np.asarray(obs, dtype=np.float32)
        acts = self._forward(x)
        q = acts[-1]
        b = x.shape[0]
        idx = np.arange(b)
        delta = q[idx, actions] - targets
        loss = float(np.mean(np.where(np.abs(delta) <= 1.0, 0.5 * delta**2, np.abs(delta) - 0.5)))
        g_out = np.zeros_like(q)
        g_out[idx, actions] = np.clip(delta, -1.0, 1.0) / b  # d Huber / d q
        grads: Weights = {}
        g = g_out
        for i in reversed(range(self.n_layers)):
            grads[f"W{i}"] = acts[i].T @ g
            grads[f"b{i}"] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.p[f"W{i}"].T) * (acts[i] > 0)
        norm = float(np.sqrt(sum(float(np.sum(gr**2)) for gr in grads.values())))
        scale = min(1.0, self.grad_clip / (norm + 1e-8))
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, gr in grads.items():
            gr = gr * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * gr
            self.v[k] = b2 * self.v[k] + (1 - b2) * gr**2
            mh = self.m[k] / (1 - b1**self.t)
            vh = self.v[k] / (1 - b2**self.t)
            self.p[k] = (self.p[k] - self.lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
        return loss

    def get_weights(self) -> Weights:
        return {k: v.copy() for k, v in self.p.items()}

    def set_weights(self, w: Weights) -> None:
        for k in self.p:
            if k not in w or w[k].shape != self.p[k].shape:
                raise ValueError(f"incompatible weights for {k}")
            self.p[k] = np.asarray(w[k], dtype=np.float32).copy()


# ---------------------------------------------------------------------------- PyTorch
class TorchQNet(QNetwork):  # pragma: no cover - requires torch (not installed in the dev sandbox)
    backend_name = "torch"

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int], lr: float, grad_clip: float, seed: int):
        import torch
        from torch import nn

        self.torch = torch
        torch.manual_seed(seed)
        sizes = layer_sizes(obs_dim, hidden, n_actions)
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.linears = [m for m in self.net if isinstance(m, nn.Linear)]
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.grad_clip = grad_clip

    def predict(self, obs: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            return self.net(torch.as_tensor(np.asarray(obs), dtype=torch.float32)).cpu().numpy()

    def train_on_batch(self, obs: np.ndarray, actions: np.ndarray, targets: np.ndarray) -> float:
        torch = self.torch
        o = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        a = torch.as_tensor(np.asarray(actions), dtype=torch.int64).unsqueeze(1)
        t = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
        q = self.net(o).gather(1, a).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(q, t)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.opt.step()
        return float(loss.item())

    def get_weights(self) -> Weights:
        out: Weights = {}
        for i, lin in enumerate(self.linears):
            out[f"W{i}"] = lin.weight.detach().cpu().numpy().T.astype(np.float32).copy()
            out[f"b{i}"] = lin.bias.detach().cpu().numpy().astype(np.float32).copy()
        return out

    def set_weights(self, w: Weights) -> None:
        torch = self.torch
        with torch.no_grad():
            for i, lin in enumerate(self.linears):
                if w[f"W{i}"].T.shape != tuple(lin.weight.shape):
                    raise ValueError(f"incompatible weights for W{i}")
                lin.weight.copy_(torch.as_tensor(w[f"W{i}"].T.copy()))
                lin.bias.copy_(torch.as_tensor(w[f"b{i}"].copy()))


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def build_qnet(obs_dim: int, n_actions: int, hidden: Sequence[int], lr: float, grad_clip: float,
               seed: int, backend: str = "auto") -> QNetwork:
    if backend == "auto":
        backend = "torch" if torch_available() else "numpy"
        if backend == "numpy":
            log.warning("PyTorch not installed: using the NumPy Q-network fallback")
    if backend == "torch":
        if not torch_available():
            raise RuntimeError("nn_backend='torch' requested but PyTorch is not installed")
        return TorchQNet(obs_dim, n_actions, hidden, lr, grad_clip, seed)
    if backend == "numpy":
        return NumpyQNet(obs_dim, n_actions, hidden, lr, grad_clip, seed)
    raise ValueError(f"unknown nn backend {backend!r}")
