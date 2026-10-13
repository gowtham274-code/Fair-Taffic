# Deployment

**Nothing here has been deployed or verified on a live server.** The Dockerfile/compose were not built in the sandbox.

## A. Local development
```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # SUMO via the eclipse-sumo wheel
python scripts/validate_sumo.py            # validates SUMO/TraCI integration
bash scripts/run_tests.sh
export FAIRTRAFFIC_DEV_OPEN=1              # local only: job control without key
uvicorn app.api.main:app --port 8000 &
streamlit run app/dashboard/app.py
```
Without SUMO: tests still run (SUMO/torch/FastAPI tests skip). Optional GUI demo: install `sumo-gui` and set `sim.gui: true`.

## B. Docker
```
cp .env.example .env   # set FAIRTRAFFIC_API_KEY
docker compose up --build
```
API :8000 (`/health`), dashboard :8501, volume `fairtraffic-data` holds results/checkpoints.

## C. Linux VM
Ubuntu 22.04+, ≥4 GB RAM, Docker or Python 3.11. Open 80/443 only; keep 8000/8501 behind a reverse proxy (e.g. Caddy: `example.com { reverse_proxy /api/* localhost:8000; reverse_proxy localhost:8501 }` gives automatic HTTPS). Restart policy `unless-stopped`. Set a long random `FAIRTRAFFIC_API_KEY`; do not publish job endpoints without it. Free PaaS tiers often lack SUMO/PyTorch RAM — use a VM.

## D. Update
`git pull && docker compose up -d --build` (volume persists).
## E. Backup
`bash scripts/backup.sh /data/storage backups`
## F. Troubleshooting
* `netconvert not found` → SUMO not on PATH; set `SUMO_HOME`, or use the wheel (`pip install eclipse-sumo`).
* `traci` import error → `pip install traci sumolib`.
* Experiment `failed` in the jobs list → see `storage/jobs/<id>.log` and `<run>/error.txt`.
* Port in use → change the published port in compose.
* Out of memory → shorten `sim.duration_s`, run fewer episodes; one job runs at a time by design.
* API refuses to start jobs (503/401) → configure `FAIRTRAFFIC_API_KEY` and send `X-API-Key`.
