# Faculty demonstration guide
Prepare: SUMO machine, `python scripts/validate_sumo.py`, then pre-train once (`python -m experiments.run_federated --config experiments/configs/demo_quick.yaml --name demo_federated`) so no training is needed live. An untrained model must never be presented as trained.

Script (≈15 min): 1 problem & existing-system limits (slides 5, 7) → 2 pick scenario *uneven* → 3 run fixed-time baseline (`scripts/run_demo.sh`) → 4 Traffic Simulation page: queues, waits, throughput → 5 fairness-aware MARL: AI Agent Monitoring (observation, action, reward components) → 6 explain GGF (METHODOLOGY.md) → 7 Federated page: rounds, participants, global version; say clearly *weights, not raw observations* → 8 Performance Comparison page (only real recorded runs) → 9 export CSV → 10 limitations (RESEARCH_LIMITATIONS.md) and future work.
Rehearse without SUMO: `BACKEND=mock bash scripts/run_demo.sh` — but the dashboard then shows the mock warning, and numbers are not results.
