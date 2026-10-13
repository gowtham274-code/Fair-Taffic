# API (FastAPI; written but **not executed** in the sandbox — service layer is tested)
Read endpoints are open; job endpoints need header `X-API-Key` (503 if no key is configured unless `FAIRTRAFFIC_DEV_OPEN=1`). Parameters are bounded (see `app/schemas/api.py`); the `mock` backend is refused unless `FAIRTRAFFIC_ALLOW_MOCK=1`.

| Method | Path | Purpose |
|---|---|---|
| GET | /health | liveness |
| GET | /api/status | running job, queue length, #experiments |
| GET | /api/metrics | aggregated comparison from recorded runs |
| GET | /api/intersections | per-intersection wait/queue of latest completed run |
| GET | /api/experiments, /api/experiments/{name} | list / full record |
| GET | /api/experiments/{name}/export/{file} | CSV export |
| POST | /api/experiments/run | queue an experiment (202) |
| POST | /api/simulation/start, /stop | run a baseline simulation / stop current job |
| GET | /api/jobs, /api/jobs/{id}; POST /api/jobs/{id}/stop | job states |
| GET | /api/federated/status | rounds, global version, checkpoints |

Job states: `queued → running → completed | failed | stopped`. A job is `completed` only if its process exited 0 **and** its `status.json` says completed. One job runs at a time (separate OS process); duplicates of an experiment name get 409.
