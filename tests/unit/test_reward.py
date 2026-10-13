import unittest

import numpy as np

from fairness.generalized_gini import generalized_gini
from fairness.reward import AgentLocalStats, RewardConfig, compute_rewards


def stats(n=(2, 2), w=(20.0, 40.0), q=(1, 3), served=2):
    return AgentLocalStats(list(n), list(w), list(q), served)


class TestReward(unittest.TestCase):
    def test_components_hand_computed(self):
        cfg = RewardConfig(w_wait=1, w_queue=0.5, w_throughput=0.1, w_fairness=1, wait_ref_s=60, queue_ref=20,
                           throughput_ref=5, ggf_decay=0.5)
        r = compute_rewards({"a": stats()}, [10.0, 30.0], cfg)["a"]
        self.assertAlmostEqual(r.wait, -(60 / 4) / 60)           # mean wait 15 s
        self.assertAlmostEqual(r.queue, -4 / 20)
        self.assertAlmostEqual(r.throughput, 2 / 5)
        self.assertAlmostEqual(r.fairness, generalized_gini([-10 / 60, -30 / 60], decay=0.5))
        self.assertAlmostEqual(r.total, r.wait + 0.5 * r.queue + 0.1 * r.throughput + r.fairness)

    def test_fairness_term_zero_weight_gives_conventional_reward(self):
        cfg = RewardConfig(w_fairness=0.0)
        r = compute_rewards({"a": stats()}, [10.0, 30.0], cfg)["a"]
        self.assertAlmostEqual(r.total, cfg.w_wait * r.wait + cfg.w_queue * r.queue + cfg.w_throughput * r.throughput)

    def test_fairness_shared_and_prefers_equal(self):
        cfg = RewardConfig()
        local = {"a": stats(), "b": stats()}
        r_eq = compute_rewards(local, [20, 20, 20, 20], cfg)
        r_un = compute_rewards(local, [5, 5, 5, 65], cfg)
        self.assertEqual(r_eq["a"].fairness, r_eq["b"].fairness)
        self.assertGreater(r_eq["a"].fairness, r_un["a"].fairness)

    def test_empty_and_zero(self):
        cfg = RewardConfig()
        r = compute_rewards({"a": AgentLocalStats([0, 0], [0, 0], [0, 0], 0)}, [], cfg)["a"]
        self.assertEqual(r.total, 0.0)
        self.assertEqual(compute_rewards({}, [], cfg), {})

    def test_clipping(self):
        cfg = RewardConfig(clip=1.0)
        r = compute_rewards({"a": AgentLocalStats([1], [1e6], [1000], 0)}, [1e6], cfg)["a"]
        self.assertEqual(r.wait, -1.0)
        self.assertEqual(r.queue, -1.0)
        self.assertEqual(r.fairness, -1.0)

    def test_invalid_waits_rejected(self):
        with self.assertRaises(ValueError):
            compute_rewards({"a": stats()}, [-1.0], RewardConfig())

    def test_dynamic_weight_increases_with_congestion(self):
        cfg = RewardConfig(dynamic_fairness=True, dynamic_kappa=1.0, queue_ref=10)
        low = compute_rewards({"a": stats(q=(0, 0))}, [10, 20], cfg)["a"].fairness_weight
        high = compute_rewards({"a": stats(q=(10, 10))}, [10, 20], cfg)["a"].fairness_weight
        self.assertAlmostEqual(low, 1.0)
        self.assertAlmostEqual(high, 2.0)
        off = compute_rewards({"a": stats(q=(10, 10))}, [10, 20], RewardConfig())["a"].fairness_weight
        self.assertAlmostEqual(off, 1.0)


if __name__ == "__main__":
    unittest.main()
