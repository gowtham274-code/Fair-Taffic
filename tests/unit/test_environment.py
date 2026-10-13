import unittest

import numpy as np

from common.config import SimConfig
from simulation.environment import MAX_PHASES, OBS_DIM, TrafficEnv
from simulation.network.topology import NetTopology
from simulation.mock_backend import MockBackend
from simulation.traffic_controller import FixedTimeController, QueueGreedyController, make_controller


def mock_env(**kw):
    kw = {"backend": "mock", "duration_s": 300, "scenario": "medium", **kw}
    return TrafficEnv(SimConfig(**kw))


class TestMockBackend(unittest.TestCase):
    def test_vehicles_flow_through_and_waits_accumulate(self):
        env = mock_env()
        env.reset(1)
        done = False
        while not done:
            _o, _r, done, _i = env.step({})  # keep current phase everywhere
        s = env.backend.vehicle_summary()
        self.assertGreater(s.departed, 0)
        self.assertGreater(s.arrived, 0)
        self.assertLessEqual(s.arrived, s.departed)
        self.assertTrue(all(w >= 0 for w in s.arrived_waits + s.active_waits))
        env.close()

    def test_no_discharge_on_red_or_yellow(self):
        t = NetTopology.grid(2, 2, 200)
        b = MockBackend(t)
        b.reset(0, [], 100)
        from simulation.scenarios.demand import Vehicle
        route = tuple(t.route("W0_J00", "J10_E0") or t.route("W0_J00", "J11_E1"))
        b.reset(0, [Vehicle("v0", 0, route)], 100)
        b.set_green("J00", 1)       # E/W green (W approach is green)
        b.set_yellow("J00", 1, 0)   # ...but yellow now
        for _ in range(40):
            b.step()
        st = b.approach_state("J00", "W")
        self.assertEqual(st.left_total, 0)
        self.assertEqual(st.queue, 1)
        self.assertGreater(st.sum_wait, 0)


class TestEnv(unittest.TestCase):
    def test_reset_obs_shape_and_bounds(self):
        env = mock_env()
        obs = env.reset(1)
        self.assertEqual(set(obs), set(env.agent_ids))
        self.assertEqual(len(env.agent_ids), 4)
        for o in obs.values():
            self.assertEqual(o.shape, (OBS_DIM,))
            self.assertTrue(np.all((o >= 0) & (o <= 3)))
        env.close()

    def test_step_before_reset_raises(self):
        with self.assertRaises(RuntimeError):
            mock_env().step({})

    def test_min_green_mask_and_yellow_logic(self):
        env = mock_env(min_green=10, delta_time=5, yellow_time=3)
        env.reset(1)
        a = env.agent_ids[0]
        self.assertEqual(env.action_mask(a).tolist(), [True, False, False, False])  # only current before min_green
        env.step({x: 1 for x in env.agent_ids})  # request switch too early -> ignored
        self.assertEqual(env.current_phase(a), 0)
        env.step({x: 1 for x in env.agent_ids})  # now time_in_phase = 10 -> allowed
        self.assertTrue(env.action_mask(a)[1])  # min_green (10 s) elapsed -> switching is allowed now
        obs, r, done, info = env.step({x: 1 for x in env.agent_ids})
        self.assertEqual(info["applied_phase"][a], 1)
        self.assertEqual(env.current_phase(a), 1)  # yellow (3 s) completed inside the 5 s interval
        env.close()

    def test_invalid_action_is_ignored(self):
        env = mock_env(min_green=1, delta_time=5)
        env.reset(1)
        a = env.agent_ids[0]
        env.step({})
        _o, _r, _d, info = env.step({x: 3 for x in env.agent_ids})  # phase 3 does not exist (2 greens)
        self.assertEqual(info["applied_phase"][a], 0)
        env.close()

    def test_episode_terminates_and_rewards_are_finite(self):
        env = mock_env(duration_s=120)
        env.reset(2)
        done, n = False, 0
        while not done:
            _o, rew, done, info = env.step({})
            n += 1
            self.assertTrue(all(np.isfinite(v) for v in rew.values()))
            for c in info["components"].values():
                self.assertTrue(set(c) >= {"wait", "queue", "throughput", "fairness", "total"})
        self.assertEqual(n, 120 // 5)
        env.close()

    def test_same_seed_same_result_different_seed_differs(self):
        def run(seed):
            env = mock_env(duration_s=300)
            env.reset(seed)
            c = FixedTimeController(20)
            done = False
            while not done:
                _o, _r, done, _i = env.step(c.act(env))
            m = env.episode_metrics()
            env.close()
            return m.completed_vehicles, round(m.avg_wait_s, 6)
        self.assertEqual(run(5), run(5))
        self.assertNotEqual(run(5), run(6))

    def test_reset_clears_state(self):
        env = mock_env(duration_s=60)
        env.reset(1)
        for _ in range(12):
            env.step({})
        env.reset(1)
        self.assertEqual(env.sim_time, 0.0)
        self.assertEqual(env.backend.vehicle_summary().arrived, 0)
        env.close()

    def test_metrics_after_episode(self):
        env = mock_env(duration_s=300)
        env.reset(1)
        c = make_controller("fixed", 20)
        done = False
        while not done:
            _o, _r, done, _i = env.step(c.act(env))
        m = env.episode_metrics()
        self.assertGreater(m.completed_vehicles, 0)
        self.assertTrue(0 < m.jain_index <= 1)
        self.assertLessEqual(m.ggf_score, 0)
        self.assertEqual(len(m.approach_mean_wait_s), 16)
        self.assertEqual(set(m.intersection_mean_wait_s), set(env.agent_ids))
        self.assertGreaterEqual(m.max_wait_s, m.avg_wait_s)
        env.close()


class TestControllers(unittest.TestCase):
    def test_fixed_time_cycles(self):
        env = mock_env(min_green=1)
        env.reset(1)
        c = FixedTimeController(10)
        seq = []
        for _ in range(8):
            seq.append(c.act(env)[env.agent_ids[0]])
            env.step(c.act(env))
        self.assertEqual(set(seq), {0, 1})
        with self.assertRaises(ValueError):
            FixedTimeController(0)

    def test_queue_greedy_serves_longer_queue(self):
        env = mock_env(min_green=1, scenario="uneven")
        env.reset(1)
        c = QueueGreedyController()
        for _ in range(40):
            env.step(c.act(env))
        a = env.agent_ids[0]
        acts = c.act(env)
        scores = [sum(env.approach(a, s).queue for s in env.phase_slots(a, p)) for p in range(2)]
        if abs(scores[0] - scores[1]) > 1 and env.action_mask(a).all():
            self.assertEqual(acts[a], int(np.argmax(scores)))

    def test_factory(self):
        with self.assertRaises(ValueError):
            make_controller("nope")


if __name__ == "__main__":
    unittest.main()
