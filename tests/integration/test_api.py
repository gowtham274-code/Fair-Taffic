"""API tests; skipped when FastAPI is not installed (it could not be installed in the dev sandbox)."""
import os
import tempfile
import time
import unittest

try:
    from fastapi.testclient import TestClient
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "fastapi not installed")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.update(FAIRTRAFFIC_STORAGE=tempfile.mkdtemp(), FAIRTRAFFIC_API_KEY="k", FAIRTRAFFIC_ALLOW_MOCK="1")
        from app.api import main
        cls.ctx = TestClient(main.app)
        cls.client = cls.ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def test_health_and_auth(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        self.assertEqual(self.client.post("/api/experiments/run", json={}).status_code, 401)
        self.assertEqual(self.client.post("/api/experiments/run", json={"episodes": 0},
                                          headers={"X-API-Key": "k"}).status_code, 422)

    def test_run_and_poll(self):
        h = {"X-API-Key": "k"}
        r = self.client.post("/api/experiments/run", headers=h,
                             json={"backend": "mock", "duration_s": 60, "eval_seeds": [1], "name": "apit"})
        self.assertEqual(r.status_code, 202)
        jid = r.json()["job_id"]
        for _ in range(200):
            st = self.client.get(f"/api/jobs/{jid}").json()["state"]
            if st in ("completed", "failed"):
                break
            time.sleep(0.2)
        self.assertEqual(st, "completed")
        self.assertEqual(self.client.get("/api/experiments/apit").status_code, 200)
        self.assertEqual(self.client.get("/api/experiments/nope").status_code, 404)
