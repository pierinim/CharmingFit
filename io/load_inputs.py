import yaml
from pathlib import Path

def load_observables(path="data/observables.yaml"):
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["observables"]
