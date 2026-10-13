"""End-to-end tests on the MOCK backend (no SUMO needed). They test the software pipeline
(training loop, federated rounds, persistence, checkpoints), not traffic realism."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agents.checkpoints import list_checkpoints, load_checkpoint
from common.config import load_config
from experiments import registry
from experiments.compare_results import aggregate, load_runs, to_markdown
from experiments.evaluate import evaluate_run
from experiments.runner import run_experiment


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def cfg(root, name, kind, **extra):
    ov = {"kind": kind, "seed": 1, "sim": {"backend": "mock", "duration_s": 120, "scenario": "uneven"},
          "rl": {"episodes": 4, "hidden": [8], "learn_start": 16, "batch_size": 16, "eps_decay_steps": 100,
                 "nn_backend": "numpy"},
          "federated": {"rounds": 2, "episodes_per_round": 2},
          "eval": {"seeds": [11, 12]}, "output": {"root": str(root), "name": name, "checkpoint_every_episodes": 2}}
    for k, v in extra.items():
        ov.setdefault(k, {}).update(v) if isinstance(v, dict) else ov.update({k: v})
    return load_config(None, ov)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "experiments"

    def tearDown(self):
        self.tmp.cleanup()

    def test_baseline_run_persists_everything(self):
        d = run_experiment(cfg(self.root, "fixed", "baseline"))
        for f in ("config.json", "status.json", "summary.json", "eval_results.csv", "eval_details.json", "agent_trace.csv"):
            self.assertTrue((d / f).exists(), f)
        self.assertEqual(json.loads((d / "status.json").read_text())["state"], "completed")
        summ = json.loads((d / "summary.json").read_text())
        self.assertTrue(summ["meta"]["is_test_double"])
        self.assertIn("warning", summ["meta"])
        rows = read_csv(d / "eval_results.csv")
        self.assertEqual([r["seed"] for r in rows], ["11", "12"])
        self.assertEqual(registry.get_experiment(self.root, "fixed")["status"], "completed")

    def test_marl_trains_checkpoints_and_evaluates(self):
        d = run_experiment(cfg(self.root, "marl", "marl"))
        log = read_csv(d / "train_log.csv")
        self.assertEqual(len(log), 4)
        self.assertTrue(all(np.isfinite(float(r["reward_total"])) for r in log))
        cps = list_checkpoints(d / "checkpoints", "agents")
        self.assertEqual(len(cps), 2)  # episodes 2 and 4
        models, meta = load_checkpoint(d / "checkpoints", "agents")
        self.assertEqual(set(models), {f"agent_J{x}{y}" for x in (0, 1) for y in (0, 1)})
        rows = evaluate_run(d, scenario="low", seeds=[21])  # loads checkpoint, no training
        self.assertEqual(len(rows), 1)

    def test_federated_rounds_logged_and_global_checkpointed(self):
        d = run_experiment(cfg(self.root, "fed", "federated"))
        rounds = json.loads((d / "fl_rounds.json").read_text())
        self.assertEqual([r["round"] for r in rounds], [1, 2])
        self.assertEqual([r["global_version"] for r in rounds], [1, 2])
        self.assertEqual(len(rounds[0]["participants"]), 4)
        self.assertIsNotNone(rounds[0]["validation"])
        self.assertAlmostEqual(sum(rounds[0]["aggregation_weights"].values()), 1.0, places=6)
        self.assertEqual(len(list_checkpoints(d / "checkpoints", "global")), 2)
        self.assertEqual(len(list_checkpoints(d / "checkpoints", "local_agents")), 1)
        # the final global weights are the FedAvg of what was exchanged: all locals equal the global after the round
        g, _ = load_checkpoint(d / "checkpoints", "global")
        loc, _ = load_checkpoint(d / "checkpoints", "local_agents")
        for m in loc.values():
            for k, v in m.items():
                np.testing.assert_allclose(v, g["global"][k], atol=1e-6)
        self.assertEqual(len(evaluate_run(d, seeds=[31])), 1)

    def test_centralized_runs_and_is_marked(self):
        d = run_experiment(cfg(self.root, "cen", "centralized"))
        _m, meta = load_checkpoint(d / "checkpoints", "agents")
        self.assertTrue(meta["shared"])
        self.assertEqual(meta["kind"], "centralized")

    def test_never_overwrites_without_flag(self):
        run_experiment(cfg(self.root, "x", "baseline"))
        with self.assertRaises(FileExistsError):
            run_experiment(cfg(self.root, "x", "baseline"))
        run_experiment(cfg(self.root, "x", "baseline", output={"overwrite": True}))

    def test_failed_run_is_recorded_as_failed(self):
        bad = cfg(self.root, "bad", "baseline", sim={"backend": "sumo", "sumo_binary": "definitely-not-sumo",
                                                      "netconvert_binary": "definitely-not-netconvert",
                                                      "work_dir": str(Path(self.tmp.name) / "w")})
        with self.assertRaises(RuntimeError):
            run_experiment(bad)
        self.assertEqual(json.loads((self.root / "bad" / "status.json").read_text())["state"], "failed")
        self.assertEqual(registry.get_experiment(self.root, "bad")["status"], "failed")
        self.assertTrue((self.root / "bad" / "error.txt").exists())

    def test_same_config_and_seed_reproduce_identical_metrics(self):
        a = run_experiment(cfg(self.root, "r1", "marl"))
        b = run_experiment(cfg(self.root, "r2", "marl"))
        ra = read_csv(a / "eval_results.csv")
        rb = read_csv(b / "eval_results.csv")
        for x, y in zip(ra, rb):
            for k in ("avg_wait_s", "completed_vehicles", "jain_index", "ggf_score"):
                self.assertEqual(x[k], y[k], k)

    def test_compare_results_aggregates_and_flags_mock(self):
        run_experiment(cfg(self.root, "f1", "baseline", variant="fixed_time"))
        run_experiment(cfg(self.root, "f2", "baseline", variant="fixed_time", seed=2))
        rows = aggregate(load_runs(self.root))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n_runs"], 2)
        self.assertEqual(rows[0]["n_eval_episodes"], 4)
        self.assertIn("test double", to_markdown(rows))

    def test_registry_lists(self):
        run_experiment(cfg(self.root, "a", "baseline"))
        run_experiment(cfg(self.root, "b", "baseline"))
        self.assertEqual({e["name"] for e in registry.list_experiments(self.root)}, {"a", "b"})
        self.assertIsNone(registry.get_experiment(self.root, "zzz"))
        with self.assertRaises(ValueError):
            registry.upsert(self.root, "a", evil="x")


if __name__ == "__main__":
    unittest.main()
