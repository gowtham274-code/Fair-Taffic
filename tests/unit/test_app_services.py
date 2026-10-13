import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from app.schemas.api import ExperimentRequest
from app.services.job_manager import Conflict, JobManager
from app.services.results import ResultsService

ROOT = Path(__file__).resolve().parents[2]


class TestSchemas(unittest.TestCase):
    def test_rejects_bad_input(self):
        for bad in ({"name": "../x"}, {"duration_s": 61}, {"duration_s": 99999}, {"episodes": 0},
                    {"kind": "rm -rf"}, {"eval_seeds": []}):
            with self.assertRaises(ValidationError, msg=str(bad)):
                ExperimentRequest(**bad)

    def test_to_config(self):
        cfg = ExperimentRequest(kind="federated", backend="mock", scenario="uneven", duration_s=60, episodes=2,
                                rounds=1, name="a1").to_config("/tmp/x", "a1")
        self.assertEqual((cfg.kind, cfg.sim.scenario, cfg.output.overwrite), ("federated", "uneven", False))


class TestJobsAndResults(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "experiments"
        self.jm = JobManager(self.root, self.tmp / "jobs", ROOT)
        self.addCleanup(self.jm.shutdown)

    def req(self, name, **kw):
        base = dict(kind="baseline", backend="mock", duration_s=60, name=name, eval_seeds=[1])
        base.update(kw)
        return ExperimentRequest(**base).to_config(str(self.root), name)

    def test_job_runs_to_completed_and_results_readable(self):
        job = self.jm.submit(self.req("j1"))
        self.assertIn(job.state, ("queued", "running"))
        done = self.jm.wait(job.job_id, 120)
        self.assertEqual(done.state, "completed", done.message)
        svc = ResultsService(self.root)
        self.assertEqual([e["name"] for e in svc.list_experiments()], ["j1"])
        exp = svc.experiment("j1")
        self.assertEqual(len(exp["eval_results"]), 1)
        self.assertEqual(svc.comparison()[0]["variant"], "fixed_time")

    def test_duplicate_name_conflicts_and_failed_job_reports_failure(self):
        j = self.jm.submit(self.req("dup"))
        with self.assertRaises(Conflict):
            self.jm.submit(self.req("dup"))
        self.jm.wait(j.job_id, 120)
        cfg = self.req("bad", backend="sumo")
        cfg.sim.netconvert_binary = "definitely-not-netconvert"  # fails whether or not SUMO is installed
        cfg.sim.work_dir = str(self.tmp / "empty_work")  # no cached network to reuse
        bad = self.jm.submit(cfg)
        done = self.jm.wait(bad.job_id, 120)
        self.assertEqual(done.state, "failed")
        self.assertTrue(done.message)

    def test_queued_job_can_be_cancelled(self):
        a = self.jm.submit(self.req("s1", duration_s=600))
        b = self.jm.submit(self.req("s2"))
        self.assertEqual(self.jm.stop(b.job_id).state, "stopped")
        self.jm.stop(a.job_id)
        self.assertIn(self.jm.wait(a.job_id, 60).state, ("stopped", "completed"))

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            ResultsService(self.root).experiment("../etc")


if __name__ == "__main__":
    unittest.main()
