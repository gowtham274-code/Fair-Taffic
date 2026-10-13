"""Integration tests that need a real SUMO installation (sumo, netconvert, python 'traci').
They are SKIPPED automatically when SUMO is missing.  Run them on a machine with SUMO:

    python -m unittest tests.integration.test_sumo_backend -v
"""
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import traci  # noqa: F401
    HAVE_TRACI = True
except ImportError:
    HAVE_TRACI = False

HAVE_SUMO = bool(shutil.which("sumo") and shutil.which("netconvert")) and HAVE_TRACI


@unittest.skipUnless(HAVE_SUMO, "SUMO / netconvert / traci not installed")
class TestSumo(unittest.TestCase):  # pragma: no cover - not runnable in the dev sandbox
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def sim(self, **kw):
        from common.config import SimConfig
        return SimConfig(backend="sumo", duration_s=300, work_dir=str(Path(self.tmp.name) / "work"), **kw)

    def test_network_builds_and_validates(self):
        from simulation.network.builder import build_network
        from simulation.network.topology import NetTopology
        net = build_network(Path(self.tmp.name) / "net", 2, 2, 200.0, force=True)
        topo = NetTopology.from_net_xml(net)
        self.assertEqual(topo.validate(expect_tls=4), [])

    def test_env_runs_with_traci_and_closes(self):
        from simulation.environment import TrafficEnv
        from simulation.traffic_controller import FixedTimeController
        env = TrafficEnv(self.sim())
        obs = env.reset(1)
        self.assertEqual(len(obs), 4)
        c, done = FixedTimeController(30), False
        while not done:
            _o, rew, done, info = env.step(c.act(env))
        m = env.episode_metrics()
        self.assertGreater(m.completed_vehicles, 0)
        self.assertIsNotNone(m.fuel_g)
        env.close()

    def test_same_seed_is_deterministic(self):
        from simulation.environment import TrafficEnv
        from simulation.traffic_controller import FixedTimeController

        def run():
            env = TrafficEnv(self.sim())
            env.reset(3)
            c, done = FixedTimeController(30), False
            while not done:
                _o, _r, done, _i = env.step(c.act(env))
            m = env.episode_metrics()
            env.close()
            return m.completed_vehicles, round(m.avg_wait_s, 6)
        self.assertEqual(run(), run())

    def test_phases_and_slots_discovered(self):
        from simulation.environment import TrafficEnv
        env = TrafficEnv(self.sim())
        env.reset(1)
        for a in env.agent_ids:
            self.assertGreaterEqual(env.n_green(a), 2)
            self.assertTrue(all(env.phase_slots(a, p) for p in range(env.n_green(a))))
        env.close()


if __name__ == "__main__":
    unittest.main()
