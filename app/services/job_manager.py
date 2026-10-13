"""Single-instance background job manager.

* One job runs at a time (the simulator and model files are not safe to share between
  concurrent experiments); further jobs wait in a FIFO queue.
* Every job is a separate OS process (``python -m experiments.run_config``), so a crash
  or a long run never blocks or corrupts the API process, and "stop" is a real SIGTERM.
* The reported state is derived from facts: queued (in our queue), running (process alive),
  completed (exit code 0 AND the run's status.json says completed), failed, stopped.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from common.config import ExperimentConfig
from common.logging_setup import get_logger

log = get_logger(__name__)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class QueueFull(Exception):
    pass


class Conflict(Exception):
    pass


@dataclass
class Job:
    job_id: str
    experiment: str
    kind: str
    config_path: Path
    log_path: Path
    state: str = "queued"
    submitted: str = field(default_factory=_now)
    started: Optional[str] = None
    ended: Optional[str] = None
    message: Optional[str] = None
    stop_requested: bool = False
    proc: Optional[subprocess.Popen] = None
    simulation: bool = False  # created through /api/simulation/start

    def info(self, experiments_root: Path) -> Dict[str, Any]:
        progress = None
        sp = experiments_root / self.experiment / "status.json"
        if self.state == "running" and sp.exists():
            try:
                progress = json.loads(sp.read_text())
            except (OSError, ValueError):
                progress = None
        return {"job_id": self.job_id, "experiment": self.experiment, "kind": self.kind, "state": self.state,
                "submitted": self.submitted, "started": self.started, "ended": self.ended,
                "message": self.message, "progress": progress}


class JobManager:
    def __init__(self, experiments_root: Path, jobs_dir: Path, project_root: Path,
                 python: str = sys.executable, max_queue: int = 20, autostart: bool = True):
        self.root = Path(experiments_root)
        self.jobs_dir = Path(jobs_dir)
        self.project_root = Path(project_root)
        self.python = python
        self.max_queue = max_queue
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, Job] = {}
        self._queue: Deque[str] = deque()
        self._cv = threading.Condition()
        self._current: Optional[str] = None
        self._closing = False
        self._thread = threading.Thread(target=self._worker, name="job-worker", daemon=True)
        if autostart:
            self._thread.start()

    # ------------------------------------------------------------------ public API
    def submit(self, cfg: ExperimentConfig, simulation: bool = False) -> Job:
        name = cfg.output.name
        with self._cv:
            if (self.root / name).exists() or any(j.experiment == name and j.state in ("queued", "running")
                                                   for j in self._jobs.values()):
                raise Conflict(f"experiment name {name!r} already exists or is queued")
            if len(self._queue) >= self.max_queue:
                raise QueueFull(f"queue is full ({self.max_queue} jobs waiting)")
            jid = uuid.uuid4().hex[:10]
            cpath = self.jobs_dir / f"{jid}.json"
            cpath.write_text(json.dumps(cfg.model_dump(mode="json"), indent=2))
            job = Job(jid, name, cfg.kind, cpath, self.jobs_dir / f"{jid}.log", simulation=simulation)
            self._jobs[jid] = job
            self._queue.append(jid)
            self._cv.notify_all()
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._cv:
            return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        with self._cv:
            return sorted(self._jobs.values(), key=lambda j: j.submitted, reverse=True)

    def current(self) -> Optional[Job]:
        with self._cv:
            return self._jobs.get(self._current) if self._current else None

    def active_experiments(self) -> List[str]:
        with self._cv:
            return [j.experiment for j in self._jobs.values() if j.state in ("queued", "running")]

    def queue_length(self) -> int:
        with self._cv:
            return len(self._queue)

    def stop(self, job_id: str) -> Job:
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.state == "queued" and job_id in self._queue:
                self._queue.remove(job_id)
                job.state, job.ended, job.message = "stopped", _now(), "cancelled before start"
                return job
            if job.state not in ("queued", "running"):
                return job
            # running, or just dequeued by the worker and about to start: _run honours stop_requested
            job.stop_requested = True
            proc = job.proc
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
        return job

    def shutdown(self, timeout: float = 15.0) -> None:
        with self._cv:
            self._closing = True
            cur = self._jobs.get(self._current) if self._current else None
            self._cv.notify_all()
        if cur is not None:
            self.stop(cur.job_id)
        if self._thread.is_alive():
            self._thread.join(timeout)

    def wait(self, job_id: str, timeout: float = 120.0) -> Job:
        """Block until the job reaches a final state (for tests / scripts)."""
        end = time.time() + timeout
        while time.time() < end:
            j = self.get(job_id)
            if j is not None and j.state in ("completed", "failed", "stopped"):
                return j
            time.sleep(0.05)
        raise TimeoutError(job_id)

    # ------------------------------------------------------------------ worker
    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._closing:
                    self._cv.wait()
                if self._closing:
                    return
                jid = self._queue.popleft()
                job = self._jobs[jid]
                self._current = jid
            try:
                self._run(job)
            except Exception as exc:  # never let the worker die
                log.exception("job %s crashed the manager: %s", jid, exc)
                with self._cv:
                    job.state, job.ended, job.message = "failed", _now(), f"manager error: {exc}"
            finally:
                with self._cv:
                    self._current = None

    def _run(self, job: Job) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.project_root) + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [self.python, "-m", "experiments.run_config", "--config", str(job.config_path)]
        with open(job.log_path, "wb") as lf:
            try:
                proc = subprocess.Popen(cmd, cwd=self.project_root, env=env, stdout=lf, stderr=subprocess.STDOUT)
            except OSError as exc:
                with self._cv:
                    job.state, job.ended, job.message = "failed", _now(), f"could not start process: {exc}"
                return
            with self._cv:
                job.proc, job.state, job.started = proc, "running", _now()
                stop_now = job.stop_requested
            if stop_now:
                proc.send_signal(signal.SIGTERM)
            rc = proc.wait()
        status = self._read_status(job)
        with self._cv:
            job.ended = _now()
            if job.stop_requested or rc == 130 or status == "stopped":
                job.state, job.message = "stopped", "stopped on request"
            elif rc == 0 and status == "completed":
                job.state, job.message = "completed", None
            else:
                job.state = "failed"
                job.message = self._tail(job.log_path) or f"process exited with code {rc}"

    def _read_status(self, job: Job) -> Optional[str]:
        sp = self.root / job.experiment / "status.json"
        try:
            return json.loads(sp.read_text()).get("state")
        except (OSError, ValueError):
            return None

    @staticmethod
    def _tail(path: Path, n: int = 400) -> str:
        try:
            return path.read_text(errors="replace").strip()[-n:]
        except OSError:
            return ""
