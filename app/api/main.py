"""FastAPI application.

STATUS: written for FastAPI but NOT executed in the development sandbox (FastAPI could not be
installed there); the service layer it calls (job manager, results, schemas) is tested separately.
Run it with:  uvicorn app.api.main:app --host 0.0.0.0 --port 8000

Security: read endpoints are open. Endpoints that start/stop jobs require the header
``X-API-Key: $FAIRTRAFFIC_API_KEY``; if no key is configured they are refused (503) unless
FAIRTRAFFIC_DEV_OPEN=1 is set for local development. All job parameters are bounded by pydantic
models (app/schemas/api.py); clients cannot pass paths, commands or arbitrary config.
"""
from __future__ import annotations

import hmac
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from app.schemas.api import ExperimentRequest, JobInfo, SimulationRequest
from app.services.job_manager import Conflict, JobManager, QueueFull
from app.services.results import ResultsService
from app.settings import Settings

settings = Settings.from_env()
state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["jobs"] = JobManager(settings.experiments_root, settings.jobs_dir, settings.project_root)
    state["results"] = ResultsService(settings.experiments_root)
    state["started"] = time.time()
    yield
    state["jobs"].shutdown()


app = FastAPI(title="Fair Traffic Signal Control via MARL", version="1.0", lifespan=lifespan)


def jobs() -> JobManager:
    return state["jobs"]


def results() -> ResultsService:
    return state["results"]


def require_admin(x_api_key: Optional[str] = Header(default=None)) -> None:
    if settings.api_key is None:
        if settings.dev_open:
            return
        raise HTTPException(503, "job control is disabled: set FAIRTRAFFIC_API_KEY (or FAIRTRAFFIC_DEV_OPEN=1 locally)")
    if x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(401, "missing or invalid X-API-Key")


def _check_backend(backend: str) -> None:
    if backend == "mock" and not settings.allow_mock:
        raise HTTPException(400, "backend 'mock' (test double) is disabled; set FAIRTRAFFIC_ALLOW_MOCK=1 to allow it")


def _submit(cfg, simulation: bool = False):
    try:
        return jobs().submit(cfg, simulation=simulation)
    except Conflict as e:
        raise HTTPException(409, str(e))
    except QueueFull as e:
        raise HTTPException(429, str(e))


def _info(job) -> Dict[str, Any]:
    return job.info(settings.experiments_root)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "uptime_s": round(time.time() - state.get("started", time.time()), 1)}


@app.get("/api/status")
def status() -> Dict[str, Any]:
    cur = jobs().current()
    return {"running_job": _info(cur) if cur else None, "queued": jobs().queue_length(),
            "experiments_recorded": len(results().list_experiments())}


@app.get("/api/metrics")
def metrics() -> List[Dict[str, Any]]:
    """Aggregated comparison (mean ± std over recorded evaluation episodes)."""
    return results().comparison()


@app.get("/api/intersections")
def intersections() -> Dict[str, Any]:
    out = results().intersections()
    if out is None:
        raise HTTPException(404, "no completed experiment yet")
    return out


@app.get("/api/experiments")
def experiments() -> List[Dict[str, Any]]:
    return results().list_experiments()


@app.get("/api/experiments/{name}")
def experiment(name: str) -> Dict[str, Any]:
    try:
        out = results().experiment(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if out is None:
        raise HTTPException(404, f"experiment {name!r} not found")
    return out


@app.get("/api/experiments/{name}/export/{file}")
def export(name: str, file: str):
    try:
        p = results().csv_path(name, file)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if p is None:
        raise HTTPException(404, "file not found")
    return FileResponse(p, media_type="text/csv", filename=f"{name}_{file}")


@app.post("/api/experiments/run", response_model=JobInfo, dependencies=[Depends(require_admin)], status_code=202)
def run_experiment(req: ExperimentRequest) -> Dict[str, Any]:
    _check_backend(req.backend)
    name = req.name or f"api_{req.kind}_{req.scenario}_{int(time.time())}"
    return _info(_submit(req.to_config(str(settings.experiments_root), name)))


@app.post("/api/simulation/start", response_model=JobInfo, dependencies=[Depends(require_admin)], status_code=202)
def simulation_start(req: SimulationRequest) -> Dict[str, Any]:
    _check_backend(req.backend)
    ereq = ExperimentRequest(kind="baseline", scenario=req.scenario, backend=req.backend, controller=req.controller,
                             duration_s=req.duration_s, eval_seeds=[req.seed])
    return _info(_submit(ereq.to_config(str(settings.experiments_root), f"sim_{req.controller}_{req.scenario}_{int(time.time())}"),
                         simulation=True))


@app.post("/api/simulation/stop", dependencies=[Depends(require_admin)])
def simulation_stop() -> Dict[str, Any]:
    cur = jobs().current()
    if cur is None:
        raise HTTPException(409, "no job is running")
    return _info(jobs().stop(cur.job_id))


@app.get("/api/jobs", response_model=List[JobInfo])
def list_jobs() -> List[Dict[str, Any]]:
    return [_info(j) for j in jobs().list()]


@app.get("/api/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: str) -> Dict[str, Any]:
    j = jobs().get(job_id)
    if j is None:
        raise HTTPException(404, "unknown job")
    return _info(j)


@app.post("/api/jobs/{job_id}/stop", response_model=JobInfo, dependencies=[Depends(require_admin)])
def stop_job(job_id: str) -> Dict[str, Any]:
    try:
        return _info(jobs().stop(job_id))
    except KeyError:
        raise HTTPException(404, "unknown job")


@app.get("/api/federated/status")
def federated_status() -> Dict[str, Any]:
    out = results().latest_federated()
    if out is None:
        raise HTTPException(404, "no completed federated experiment yet")
    return out
