import tempfile
import unittest
from pathlib import Path

import numpy as np

from agents.checkpoints import CheckpointError, list_checkpoints, load_checkpoint, save_checkpoint
from agents.marl_agent import DQNAgent
from agents.model import NumpyQNet, build_qnet, torch_available
from agents.replay_buffer import ReplayBuffer
from common.config import RLConfig


def tiny_cfg(**kw):
    return RLConfig(hidden=[8, 8], batch_size=16, learn_start=16, buffer_size=500, **kw)


class TestNumpyNet(unittest.TestCase):
    def test_gradient_matches_finite_difference(self):
        net = NumpyQNet(5, 3, [6], lr=0.0, grad_clip=1e9, seed=0)  # lr=0 -> weights unchanged
        rng = np.random.default_rng(1)
        x = rng.standard_normal((4, 5)).astype(np.float32)
        a = np.array([0, 1, 2, 1])
        tgt = rng.standard_normal(4).astype(np.float32) * 0.3  # small residual: stays in the quadratic Huber zone

        def loss():
            q = net.predict(x)[np.arange(4), a]
            d = q - tgt
            return float(np.mean(np.where(np.abs(d) <= 1, 0.5 * d**2, np.abs(d) - 0.5)))

        # analytic grad via train_on_batch internals: replicate by calling with lr tiny and measuring Adam's first step
        net2 = NumpyQNet(5, 3, [6], lr=1e-3, grad_clip=1e9, seed=0)
        w0 = net2.get_weights()
        net2.train_on_batch(x, a, tgt)
        # Adam's first step = lr * sign(grad) (m/sqrt(v) = +-1) -> compare sign with finite differences
        w1 = net2.get_weights()
        eps = 1e-3
        checked = 0
        for k in ("W0", "W1"):
            idx = (1, 1)
            base = net.get_weights()
            wp = {kk: v.copy() for kk, v in base.items()}
            wm = {kk: v.copy() for kk, v in base.items()}
            wp[k][idx] += eps
            wm[k][idx] -= eps
            net.set_weights(wp); lp = loss()
            net.set_weights(wm); lm = loss()
            net.set_weights(base)
            fd = (lp - lm) / (2 * eps)
            step = w1[k][idx] - w0[k][idx]
            if abs(fd) > 1e-6:
                self.assertEqual(np.sign(step), -np.sign(fd), msg=k)
                checked += 1
        self.assertGreater(checked, 0)

    def test_learns_simple_regression(self):
        net = NumpyQNet(2, 2, [16], lr=5e-3, grad_clip=10, seed=0)
        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, (256, 2)).astype(np.float32)
        a = rng.integers(0, 2, 256)
        t = np.where(a == 0, x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]).astype(np.float32)
        first = net.train_on_batch(x, a, t)
        for _ in range(600):
            last = net.train_on_batch(x, a, t)
        self.assertLess(last, 0.2 * first)

    def test_weight_roundtrip_and_incompatible_rejected(self):
        a = NumpyQNet(4, 2, [5], 1e-3, 5, seed=1)
        b = NumpyQNet(4, 2, [5], 1e-3, 5, seed=2)
        b.set_weights(a.get_weights())
        x = np.ones((1, 4), dtype=np.float32)
        np.testing.assert_allclose(a.predict(x), b.predict(x))
        c = NumpyQNet(4, 2, [7], 1e-3, 5, seed=3)
        with self.assertRaises(ValueError):
            c.set_weights(a.get_weights())

    def test_get_weights_returns_copies(self):
        a = NumpyQNet(3, 2, [4], 1e-3, 5, seed=1)
        w = a.get_weights()
        w["W0"][:] = 0
        self.assertTrue(np.any(a.get_weights()["W0"] != 0))

    def test_build_auto(self):
        net = build_qnet(3, 2, [4], 1e-3, 5, 0, "auto")
        self.assertEqual(net.backend_name, "torch" if torch_available() else "numpy")
        with self.assertRaises(ValueError):
            build_qnet(3, 2, [4], 1e-3, 5, 0, "tf")


@unittest.skipUnless(torch_available(), "PyTorch not installed")
class TestTorchParity(unittest.TestCase):  # pragma: no cover
    def test_same_interface_and_weight_layout(self):
        t = build_qnet(4, 3, [5], 1e-3, 5, 0, "torch")
        n = build_qnet(4, 3, [5], 1e-3, 5, 0, "numpy")
        n.set_weights(t.get_weights())
        x = np.random.default_rng(0).standard_normal((6, 4)).astype(np.float32)
        np.testing.assert_allclose(t.predict(x), n.predict(x), atol=1e-5)
        loss = t.train_on_batch(x, np.zeros(6, dtype=int), np.ones(6, dtype=np.float32))
        self.assertTrue(np.isfinite(loss))


class TestReplayBuffer(unittest.TestCase):
    def test_ring_and_sample(self):
        b = ReplayBuffer(3, 2, 2)
        for i in range(5):
            b.add(np.full(2, i), i % 2, float(i), np.full(2, i + 1), False, np.ones(2, bool))
        self.assertEqual(len(b), 3)
        self.assertEqual(set(b.rewards.tolist()), {2.0, 3.0, 4.0})
        o, a, r, no, d, m = b.sample(8)
        self.assertEqual(o.shape, (8, 2))
        with self.assertRaises(ValueError):
            ReplayBuffer(3, 2, 2).sample(1)


class TestAgent(unittest.TestCase):
    def test_mask_respected_in_both_modes(self):
        ag = DQNAgent("a", 6, 4, tiny_cfg(nn_backend="numpy"), seed=0)
        mask = np.array([False, True, False, False])
        obs = np.zeros(6, dtype=np.float32)
        self.assertTrue(all(ag.act(obs, mask) == 1 for _ in range(50)))  # exploring
        self.assertTrue(all(ag.act(obs, mask, greedy=True) == 1 for _ in range(10)))
        with self.assertRaises(ValueError):
            ag.act(obs, np.zeros(4, bool))

    def test_eval_mode_does_not_learn_or_count_steps(self):
        ag = DQNAgent("a", 6, 4, tiny_cfg(nn_backend="numpy"), seed=0)
        w0 = ag.get_weights()
        for _ in range(30):
            ag.act(np.zeros(6, np.float32), np.ones(4, bool), greedy=True)
        self.assertEqual(ag.steps, 0)
        self.assertEqual(len(ag.buffer), 0)
        for k, v in ag.get_weights().items():
            np.testing.assert_array_equal(v, w0[k])

    def test_epsilon_schedule(self):
        ag = DQNAgent("a", 6, 4, tiny_cfg(nn_backend="numpy", eps_start=1.0, eps_end=0.1, eps_decay_steps=100), seed=0)
        self.assertAlmostEqual(ag.epsilon(), 1.0)
        ag.steps = 50
        self.assertAlmostEqual(ag.epsilon(), 0.55)
        ag.steps = 10_000
        self.assertAlmostEqual(ag.epsilon(), 0.1)

    def test_learning_updates_weights_after_warmup(self):
        ag = DQNAgent("a", 6, 4, tiny_cfg(nn_backend="numpy"), seed=0)
        rng = np.random.default_rng(0)
        losses = []
        for i in range(40):
            ag.steps += 1
            ag.remember(rng.random(6).astype(np.float32), int(rng.integers(4)), 1.0, rng.random(6).astype(np.float32), False, np.ones(4, bool))
            l = ag.learn()
            if l is not None:
                losses.append(l)
        self.assertGreater(len(losses), 0)
        self.assertEqual(ag.samples_round, 40)

    def test_set_weights_syncs_target(self):
        a = DQNAgent("a", 6, 4, tiny_cfg(nn_backend="numpy"), seed=0)
        b = DQNAgent("b", 6, 4, tiny_cfg(nn_backend="numpy"), seed=5)
        b.set_weights(a.get_weights())
        x = np.ones((1, 6), np.float32)
        np.testing.assert_allclose(a.online.predict(x), b.target.predict(x))


class TestCheckpoints(unittest.TestCase):
    def w(self, v=1.0):
        return {"W0": np.full((2, 2), v, np.float32), "b0": np.zeros(2, np.float32)}

    def test_versioning_roundtrip_hash_and_shapes(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = save_checkpoint(d, "agents", {"agent_a": self.w(1), "agent_b": self.w(2)}, {"episode": 1})
            p2 = save_checkpoint(d, "agents", {"agent_a": self.w(3)}, {"episode": 2})
            self.assertEqual((p1.name, p2.name), ("v0001", "v0002"))
            m, meta = load_checkpoint(d, "agents")
            self.assertEqual(meta["version"], 2)
            np.testing.assert_array_equal(m["agent_a"]["W0"], self.w(3)["W0"])
            m1, _ = load_checkpoint(d, "agents", version=1)
            self.assertEqual(set(m1), {"agent_a", "agent_b"})
            self.assertEqual([c["version"] for c in list_checkpoints(d, "agents")], [1, 2])
            with self.assertRaises(CheckpointError):
                load_checkpoint(d, "agents", expected_shapes={"W0": (3, 3)})
            with self.assertRaises(CheckpointError):
                load_checkpoint(d, "agents", version=9)
            with self.assertRaises(CheckpointError):
                load_checkpoint(d, "nothing")

    def test_tampering_detected_and_nan_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = save_checkpoint(d, "g", {"global": self.w()})
            f = Path(p) / "global.npz"
            np.savez(f, W0=np.zeros((2, 2), np.float32), b0=np.zeros(2, np.float32))
            with self.assertRaises(CheckpointError):
                load_checkpoint(d, "g")
            bad = self.w(); bad["W0"][0, 0] = np.nan
            with self.assertRaises(CheckpointError):
                save_checkpoint(d, "g2", {"global": bad})

    def test_unsafe_labels(self):
        with tempfile.TemporaryDirectory() as d:
            for lab in ("../x", "a/b", ""):
                with self.assertRaises(CheckpointError):
                    save_checkpoint(d, lab, {"m": self.w()})

    def test_agent_save_load_gives_same_policy(self):
        cfg = tiny_cfg(nn_backend="numpy")
        a = DQNAgent("a", 6, 4, cfg, seed=0)
        with tempfile.TemporaryDirectory() as d:
            save_checkpoint(d, "agents", {"agent_a": a.get_weights()})
            m, _ = load_checkpoint(d, "agents")
        b = DQNAgent("a", 6, 4, cfg, seed=99)
        b.set_weights(m["agent_a"])
        x = np.random.default_rng(0).random(6).astype(np.float32)
        np.testing.assert_allclose(a.q_values(x), b.q_values(x))


if __name__ == "__main__":
    unittest.main()
