#!/usr/bin/env bash
# Unit + integration tests (SUMO/torch/fastapi tests skip themselves when the dependency is missing).
set -e
cd "$(dirname "$0")/.."
PYTHONPATH=. python3 -m unittest discover -s tests -t . -v
