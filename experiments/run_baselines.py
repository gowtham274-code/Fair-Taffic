"""Run a non-learning baseline (fixed-time or queue-greedy) on one scenario.

    python -m experiments.run_baselines --config experiments/configs/default.yaml \
        --set sim.scenario=high --set eval.controller=fixed --name fixed_high
"""
from experiments.cli import main_run

if __name__ == "__main__":
    raise SystemExit(main_run(__doc__, forced={"kind": "baseline"}))
