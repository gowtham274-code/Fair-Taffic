# NOT built or run in the development sandbox (no Docker daemon / package access there).
# Verify with:  docker build -t fairtraffic . && docker run --rm fairtraffic python scripts/validate_sumo.py
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app FAIRTRAFFIC_STORAGE=/data/storage
WORKDIR /app
COPY requirements.txt .
# eclipse-sumo ships the sumo/netconvert binaries inside the wheel
RUN pip install --no-cache-dir -r requirements.txt \
 && python - <<'PY'
import os, eclipse_sumo
print("SUMO_HOME", eclipse_sumo.SUMO_HOME)
PY
ENV SUMO_HOME=/usr/local/lib/python3.11/site-packages/sumo
ENV PATH="${SUMO_HOME}/bin:${PATH}"
COPY . .
RUN useradd -m app && mkdir -p /data/storage && chown -R app /data /app
USER app
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
