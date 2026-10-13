import unittest

import numpy as np

from agents.marl_agent import DQNAgent
from common.config import FederatedConfig, RLConfig
from federated.aggregation import ClientUpdate, UpdateError, aggregation_weights, fedavg, validate_update
from federated.client import FederatedClient
from federated.rounds import FederatedCoordinator
from federated.server import FederatedServer


def W(v, shape=(2, 3)):
    return {"W0": np.full(shape, v, np.float32), "b0": np.full(shape[1], v, np.float32)}


class TestFedAvg(unittest.TestCase):
    def test_weighted_average_by_samples(self):
        ups = [ClientUpdate("a", W(1.0), 10), ClientUpdate("b", W(4.0), 30)]
        g, alpha = fedavg(ups, "samples")
        np.testing.assert_allclose(alpha, [0.25, 0.75])
        np.testing.assert_allclose(g["W0"], 1.0 * 0.25 + 4.0 * 0.75)

    def test_uniform_and_zero_samples_fallback(self):
        ups = [ClientUpdate("a", W(1.0), 10), ClientUpdate("b", W(3.0), 30)]
        np.testing.assert_allclose(fedavg(ups, "uniform")[0]["b0"], 2.0)
        zero = [ClientUpdate("a", W(1.0), 0), ClientUpdate("b", W(3.0), 0)]
        np.testing.assert_allclose(aggregation_weights(zero), [0.5, 0.5])

    def test_single_client_returns_its_weights(self):
        g, _ = fedavg([ClientUpdate("a", W(2.5), 5)])
        np.testing.assert_allclose(g["W0"], 2.5)

    def test_incompatible_updates_rejected(self):
        ref = W(0.0)
        bad_shape = ClientUpdate("x", W(1.0, (3, 3)), 1)
        bad_keys = ClientUpdate("y", {"W9": np.zeros((2, 3), np.float32)}, 1)
        nan = W(1.0); nan["W0"][0, 0] = np.nan
        for u in (bad_shape, bad_keys, ClientUpdate("z", nan, 1), ClientUpdate("e", {}, 1), ClientUpdate("n", W(1.0), -1)):
            with self.assertRaises(UpdateError):
                validate_update(u, ref)
        with self.assertRaises(UpdateError):
            fedavg([ClientUpdate("a", W(1.0), 1), bad_shape])
        with self.assertRaises(UpdateError):
            fedavg([])

    def test_inputs_not_modified(self):
        u = ClientUpdate("a", W(1.0), 1)
        fedavg([u, ClientUpdate("b", W(3.0), 1)])
        np.testing.assert_allclose(u.weights["W0"], 1.0)


class TestServer(unittest.TestCase):
    def test_aggregate_version_and_record(self):
        s = FederatedServer(W(0.0))
        for c in "ab":
            s.register_client(c)
        rec = s.aggregate(1, [ClientUpdate("a", W(1.0), 10, {"mean_loss": 0.5}), ClientUpdate("b", W(3.0), 10)])
        self.assertEqual((rec.status, rec.global_version, s.version), ("ok", 1, 1))
        np.testing.assert_allclose(s.global_weights["W0"], 2.0)
        self.assertEqual(rec.participants, ["a", "b"])
        self.assertGreater(rec.bytes_up, 0)
        self.assertGreater(rec.mean_update_l2, 0)
        self.assertEqual(rec.local_metrics["a"]["mean_loss"], 0.5)

    def test_rejects_bad_and_unregistered_keeps_good(self):
        s = FederatedServer(W(0.0))
        s.register_client("a"); s.register_client("b")
        rec = s.aggregate(1, [ClientUpdate("a", W(2.0), 5), ClientUpdate("b", W(1.0, (9, 9)), 5), ClientUpdate("ghost", W(1.0), 5)])
        self.assertEqual(rec.participants, ["a"])
        self.assertEqual(set(rec.rejected), {"b", "ghost"})
        np.testing.assert_allclose(s.global_weights["W0"], 2.0)

    def test_no_valid_updates_keeps_global(self):
        s = FederatedServer(W(7.0))
        s.register_client("a")
        rec = s.aggregate(1, [ClientUpdate("a", W(1.0, (9, 9)), 5)])
        self.assertEqual((rec.status, s.version), ("no_valid_updates", 0))
        np.testing.assert_allclose(s.global_weights["W0"], 7.0)

    def test_duplicate_registration(self):
        s = FederatedServer(W(0.0))
        s.register_client("a")
        with self.assertRaises(ValueError):
            s.register_client("a")


def make_clients(n=4):
    rl = RLConfig(hidden=[8], nn_backend="numpy", learn_start=8, batch_size=8, buffer_size=200)
    return [FederatedClient(f"c{i}", DQNAgent(f"c{i}", 6, 4, rl, seed=10 * i)) for i in range(n)]


class TestCoordinator(unittest.TestCase):
    def test_all_start_from_same_global_model(self):
        cl = make_clients()
        FederatedCoordinator(cl, FederatedConfig())
        ref = cl[0].agent.get_weights()
        for c in cl[1:]:
            for k, v in c.agent.get_weights().items():
                np.testing.assert_array_equal(v, ref[k])

    def test_round_averages_and_distributes(self):
        cl = make_clients()
        co = FederatedCoordinator(cl, FederatedConfig(participation_rate=1.0))
        for i, c in enumerate(cl):  # simulate divergent local training
            w = c.agent.get_weights()
            c.agent.set_weights({k: v + i for k, v in w.items()})
            c.agent.samples_round = 10
        before = [c.agent.get_weights() for c in cl]
        rec = co.communication_round(1)
        expected = {k: np.mean([b[k] for b in before], axis=0) for k in before[0]}
        for c in cl:
            for k, v in c.agent.get_weights().items():
                np.testing.assert_allclose(v, expected[k], atol=1e-5)
        self.assertEqual(rec.global_version, 1)
        self.assertEqual(len(rec.participants), 4)
        self.assertGreater(rec.bytes_down, 0)
        self.assertEqual(sum(c.agent.samples_round for c in cl), 0)  # counters reset

    def test_partial_participation_only_participants_receive(self):
        cl = make_clients(4)
        co = FederatedCoordinator(cl, FederatedConfig(participation_rate=0.5, distribute_to="participants"), seed=1)
        for i, c in enumerate(cl):
            c.agent.set_weights({k: v + i for k, v in c.agent.get_weights().items()})
            c.agent.samples_round = 5
        before = {c.client_id: c.agent.get_weights() for c in cl}
        rec = co.communication_round(1)
        self.assertEqual(len(rec.participants), 2)
        for c in cl:
            same = all(np.array_equal(v, before[c.client_id][k]) for k, v in c.agent.get_weights().items())
            self.assertEqual(same, c.client_id not in rec.participants)

    def test_validator_called_per_schedule(self):
        cl = make_clients(2)
        calls = []
        co = FederatedCoordinator(cl, FederatedConfig(validate_every_rounds=2), validator=lambda w: calls.append(1) or {"x": 1.0})
        for r in (1, 2, 3, 4):
            co.communication_round(r)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
