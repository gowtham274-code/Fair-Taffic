# Fair Traffic Signal Control via Multi-Agent Reinforcement Learning
*AI-Driven Fair & Adaptive Urban Traffic Management System* — B.Tech AI & DS final-year project.

**Abstract.** Each of four interconnected intersections is a Double-DQN agent that observes local (and neighbour) queues and waits and picks a green phase. The reward combines waiting time, queues, throughput and a Generalized Gini Welfare (GGF) fairness term. Agents can train federated (FedAvg of model weights, no raw observations shared). Evaluation uses SUMO with waiting time, throughput, Jain's index, variance, GGF and queue length, comparing fixed-time, a queue-based adaptive rule, conventional RL, fair MARL, and fair MARL + FL.

## Status — read this first
| Part | State |
|---|---|
| Fairness (GGF, Jain, variance), reward, environment, DQN agents, FedAvg, checkpoints, experiment runner, job manager, results service | **Implemented and tested** (unit + integration, 94+ tests, run with the mock simulator and NumPy network) |
| SUMO/TraCI backend, network builder | **Written, never run** (no SUMO in the dev sandbox). Validate: `python scripts/validate_sumo.py` |
| PyTorch network | **Written, never run**; NumPy equivalent is tested |
| FastAPI app, Streamlit dashboard | **Written, never run** (packages unavailable); API logic below them is tested; `tests/integration/test_api.py` runs once FastAPI is installed |
| Dockerfile / compose | **Unverified**; **not deployed** |
| Research results on SUMO | **None yet**. `reports/pipeline_check/` is mock-simulator output proving the pipeline only |
Not implemented: encryption, differential privacy, pedestrians/cyclists, QMIX. See `docs/RESEARCH_LIMITATIONS.md`.

## Quick start
```
pip install -r requirements.txt
python scripts/validate_sumo.py                       # SUMO check
bash scripts/run_tests.sh                             # tests
python -m experiments.run_baselines --config experiments/configs/default.yaml --name fixed_medium
python -m experiments.run_marl      --config experiments/configs/default.yaml --name marl_medium
python -m experiments.run_marl      --config experiments/configs/default.yaml --name rl_conv --set reward.w_fairness=0 --set variant=marl_conventional
python -m experiments.run_federated --config experiments/configs/default.yaml --name fed_medium
python -m experiments.run_suite --seeds 0 1 2         # full comparison
python -m experiments.compare_results                 # tables
uvicorn app.api.main:app --port 8000 ; streamlit run app/dashboard/app.py
```
No SUMO? add `--backend mock` (test double; **not** research results). Quick demo: `bash scripts/run_demo.sh`.

## Layout
`common/` config & seeding · `fairness/` GGF, Jain, variance, reward · `simulation/` env, SUMO & mock backends, network builder, demand, metrics, baselines · `agents/` Q-nets, DQN agent, replay, trainer, checkpoints · `federated/` client, server, FedAvg, rounds · `experiments/` runner, CLI scripts, configs, comparison · `app/` API, dashboard, job manager, results · `tests/` · `scripts/` · `docs/` (ARCHITECTURE, METHODOLOGY, ALGORITHMS, EXPERIMENTS, DEPLOYMENT, API, DEMO_GUIDE, FACULTY_QA, RESEARCH_LIMITATIONS).

## Reproducibility
Every run saves `config.json`; seeds fix Python/NumPy/torch and demand; rerun with `--config <run>/config.json`. Existing result folders are never overwritten (unless `--overwrite`).

## Screenshots
None yet — add real dashboard screenshots after running on SUMO.

## Source notes
References are taken from your slides. Please verify: the literature table credits "Alia et al., 2025" for the paper that reference [1] lists as Yu, Siddique, Sinha & Cao (MALTA @ AAAI 2025), and the doc cites "Raeis & Leon-Garcia (2021)" / "Siddique et al. (2024)", which are not in the slide references.
# Fair-Taffic
