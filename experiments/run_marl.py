"""Train + evaluate fairness-aware MARL (independent DQN agents, no federation).

    python -m experiments.run_marl --config experiments/configs/default.yaml --name marl_medium
    # conventional (delay-only) RL baseline: --set reward.w_fairness=0 --set variant=marl_conventional
    # centralised parameter-sharing baseline (NOT federated): use run_marl with --centralized
"""
import sys

from experiments.cli import main_run

if __name__ == "__main__":
    forced = {"kind": "centralized"} if "--centralized" in sys.argv else {"kind": "marl"}
    if "--centralized" in sys.argv:
        sys.argv.remove("--centralized")
    raise SystemExit(main_run(__doc__, forced=forced))
