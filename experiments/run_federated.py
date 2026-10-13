"""Train + evaluate fairness-aware MARL with federated averaging between intersections.

    python -m experiments.run_federated --config experiments/configs/default.yaml --name fed_medium
"""
from experiments.cli import main_run

if __name__ == "__main__":
    raise SystemExit(main_run(__doc__, forced={"kind": "federated"}))
