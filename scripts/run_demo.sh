#!/usr/bin/env bash
# Deterministic quick demo: fixed-time baseline, then federated fairness-aware MARL, then a comparison table.
# Add BACKEND=mock to rehearse without SUMO (pipeline only; mock numbers are not research results).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
B=${BACKEND:-sumo}
python3 -m experiments.run_baselines --config experiments/configs/demo_quick.yaml --backend $B --name demo_fixed --set eval.controller=fixed --set variant=fixed_time
python3 -m experiments.run_federated --config experiments/configs/demo_quick.yaml --backend $B --name demo_federated --set variant=federated_fair
python3 -m experiments.compare_results --root storage/experiments --out reports/demo_comparison
