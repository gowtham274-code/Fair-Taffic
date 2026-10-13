"""Run whatever experiment a config file describes (kind is read from the file).
Used by the API job manager:  python -m experiments.run_config --config job.json"""
from experiments.cli import main_run

if __name__ == "__main__":
    raise SystemExit(main_run(__doc__))
