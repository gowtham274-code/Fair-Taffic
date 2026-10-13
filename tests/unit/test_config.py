import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from common.config import ExperimentConfig, SimConfig, config_hash, load_config


class TestConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = ExperimentConfig()
        self.assertEqual(cfg.sim.scenario, "medium")
        self.assertEqual(len(config_hash(cfg)), 12)

    def test_yaml_and_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.yaml"
            p.write_text("seed: 3\nsim:\n  scenario: high\n  duration_s: 600\n")
            cfg = load_config(p, {"sim": {"duration_s": 300}, "rl": {"episodes": 2}})
            self.assertEqual((cfg.seed, cfg.sim.scenario, cfg.sim.duration_s, cfg.rl.episodes), (3, "high", 300, 2))

    def test_invalid_values(self):
        for bad in ({"sim": {"scenario": "weird"}}, {"sim": {"duration_s": 10}}, {"sim": {"yellow_time": 9, "delta_time": 5}},
                    {"sim": {"duration_s": 1801}}, {"reward": {"w_wait": -1}}, {"output": {"name": "../x"}},
                    {"federated": {"participation_rate": 0}}, {"kind": "nope"}):
            with self.assertRaises(ValidationError, msg=str(bad)):
                ExperimentConfig.model_validate(bad)

    def test_hash_changes_with_config(self):
        a, b = ExperimentConfig(), ExperimentConfig(seed=1)
        self.assertNotEqual(config_hash(a), config_hash(b))
        self.assertEqual(config_hash(a), config_hash(ExperimentConfig()))

    def test_roundtrip_json(self):
        cfg = ExperimentConfig(seed=5)
        again = ExperimentConfig.model_validate(json.loads(json.dumps(cfg.model_dump(mode="json"))))
        self.assertEqual(config_hash(cfg), config_hash(again))

    def test_shipped_configs_validate(self):
        for p in Path("experiments/configs").glob("*.yaml"):
            load_config(p)


if __name__ == "__main__":
    unittest.main()
