import math
import unittest

import numpy as np

from fairness.generalized_gini import generalized_gini, ggf_weights, ggf_waiting_time_score
from fairness.jain_index import jain_index
from fairness.waiting_time_variance import waiting_time_variance


class TestGGF(unittest.TestCase):
    def test_weights_geometric_sum_to_one_and_decreasing(self):
        w = ggf_weights(3, decay=0.5)
        np.testing.assert_allclose(w, np.array([1, 0.5, 0.25]) / 1.75)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertTrue(np.all(np.diff(w) <= 0))

    def test_manual_example(self):
        # utilities -1,-3,-2 -> sorted asc -3,-2,-1 ; weights 1,.5,.25 /1.75
        self.assertAlmostEqual(generalized_gini([-1, -3, -2], decay=0.5), -4.25 / 1.75, places=9)

    def test_order_invariant(self):
        self.assertAlmostEqual(generalized_gini([-1, -2, -3]), generalized_gini([-3, -1, -2]))

    def test_equal_weights_is_mean(self):
        self.assertAlmostEqual(generalized_gini([1, 2, 6], decay=1.0), 3.0)

    def test_explicit_weights(self):
        self.assertAlmostEqual(generalized_gini([4, 2], weights=[2, 1]), (2 * 2 + 1 * 4) / 3)

    def test_prefers_equal_distribution_with_same_mean(self):
        self.assertGreater(ggf_waiting_time_score([10, 10, 10, 10]), ggf_waiting_time_score([1, 1, 1, 37]))

    def test_not_variance(self):
        a, b = [0, 10], [5, 5]
        self.assertGreater(waiting_time_variance(a), waiting_time_variance(b))
        self.assertGreater(ggf_waiting_time_score(b), ggf_waiting_time_score(a))

    def test_edge_cases(self):
        self.assertEqual(generalized_gini([]), 0.0)
        self.assertEqual(ggf_waiting_time_score([0, 0, 0]), 0.0)
        self.assertEqual(generalized_gini([-5]), -5.0)
        with self.assertRaises(ValueError):
            generalized_gini([1, float("nan")])
        with self.assertRaises(ValueError):
            ggf_waiting_time_score([-1.0])
        with self.assertRaises(ValueError):
            ggf_weights(3, weights=[1, 2, 3])  # increasing
        with self.assertRaises(ValueError):
            ggf_weights(3, weights=[1, 1])  # wrong length
        with self.assertRaises(ValueError):
            ggf_weights(2, decay=1.5)


class TestJainVariance(unittest.TestCase):
    def test_jain(self):
        self.assertAlmostEqual(jain_index([5, 5, 5]), 1.0)
        self.assertAlmostEqual(jain_index([1, 0, 0, 0]), 0.25)
        self.assertAlmostEqual(jain_index([1, 2, 3]), 36 / (3 * 14))

    def test_jain_edges(self):
        self.assertEqual(jain_index([]), 1.0)
        self.assertEqual(jain_index([0, 0]), 1.0)
        with self.assertRaises(ValueError):
            jain_index([-1, 2])

    def test_variance(self):
        self.assertAlmostEqual(waiting_time_variance([1, 2, 3, 4]), 1.25)
        self.assertEqual(waiting_time_variance([]), 0.0)
        self.assertEqual(waiting_time_variance([7]), 0.0)
        with self.assertRaises(ValueError):
            waiting_time_variance([1, math.inf])


if __name__ == "__main__":
    unittest.main()
