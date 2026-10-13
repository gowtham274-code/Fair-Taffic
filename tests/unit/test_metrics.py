import unittest

import numpy as np

from fairness.generalized_gini import generalized_gini
from simulation.backend import ApproachState, VehicleSummary
from simulation.metrics import approach_mean_waits, compute_episode_metrics


class TestMetrics(unittest.TestCase):
    def test_hand_computed_episode(self):
        summary = VehicleSummary(arrived_waits=[10.0, 30.0], active_waits=[20.0], departed=4, arrived=2, pending=1, fuel_g=None)
        totals = {("J", "N"): (40.0, 4), ("J", "S"): (0.0, 0)}
        states = {("J", "N"): ApproachState(True, 1, 1, 10.0, 10.0, 4), ("J", "S"): ApproachState(True, 0, 0, 0.0, 0.0, 0)}
        samples = [{("J", "N"): 2, ("J", "S"): 0}, {("J", "N"): 4, ("J", "S"): 0}]
        m = compute_episode_metrics(summary, totals, states, samples, duration_s=3600, wall_time_s=1.0, ggf_decay=0.5)
        self.assertEqual(m.completed_vehicles, 2)
        self.assertEqual(m.unfinished_vehicles, (4 - 2) + 1)
        self.assertAlmostEqual(m.throughput_veh_per_h, 2.0)
        self.assertAlmostEqual(m.avg_wait_s, 20.0)
        self.assertAlmostEqual(m.avg_wait_completed_s, 20.0)
        self.assertEqual(m.max_wait_s, 30.0)
        self.assertAlmostEqual(m.wait_variance, np.var([10, 30, 20]))
        self.assertAlmostEqual(m.approach_mean_wait_s["J:N"], (40 + 10) / (4 + 1))  # vehicles that left + still on it
        self.assertEqual(m.approach_mean_wait_s["J:S"], 0.0)
        self.assertAlmostEqual(m.jain_index, 0.5)  # [10, 0] -> 100 / (2*100)
        self.assertAlmostEqual(m.ggf_score, generalized_gini([-10.0, 0.0], decay=0.5))
        self.assertAlmostEqual(m.avg_queue, np.mean([1.0, 2.0]))

    def test_empty_episode_is_safe(self):
        m = compute_episode_metrics(VehicleSummary(), {}, {}, [], 600, 0.1)
        self.assertEqual((m.completed_vehicles, m.avg_wait_s, m.max_wait_s, m.jain_index, m.ggf_score), (0, 0, 0, 1.0, 0.0))

    def test_unfinished_vehicles_visible(self):
        s = VehicleSummary(arrived_waits=[], active_waits=[50.0, 60.0], departed=2, arrived=0, pending=3)
        m = compute_episode_metrics(s, {}, {}, [], 600, 0.1)
        self.assertEqual(m.unfinished_vehicles, 5)
        self.assertEqual(m.avg_wait_s, 55.0)  # unfinished vehicles still count

    def test_mean_waits_without_vehicles(self):
        self.assertEqual(approach_mean_waits({("J", "N"): (0.0, 0)}, {})[("J", "N")], 0.0)


if __name__ == "__main__":
    unittest.main()
