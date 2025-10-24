# io/load_inputs.py
import yaml
from pathlib import Path

def load_yaml(path, key=None):
    """Generic safe YAML loader. If key is provided, return that subkey."""
    data = yaml.safe_load(Path(path).read_text())
    return data[key] if key else data

def load_all_configs(base_dir="config"):
    """Load all config YAML files into a single dictionary."""
    base = Path(base_dir)
    cfg = {}

    # Main inputs
    cfg["modes"]        = load_yaml(base / "modes.yaml", "modes")
    cfg["observables"]  = load_yaml(base / "observables.yaml", "observables")
    cfg["ckm_priors"]   = load_yaml(base / "ckm.yaml", "ckm_priors")
    cfg["hadronic_priors"] = load_yaml(base / "priors.yaml", "hadronic")
    cfg["su3_priors"]      = load_yaml(base / "priors.yaml", "su3_breaking")

    # Sanity checks
    _check_consistency(cfg)
    return cfg

def _check_consistency(cfg):
    """Basic consistency checks: every observable's mode exists in modes.yaml."""
    obs_modes = {o["mode"] for o in cfg["observables"]}
    mode_keys = set(cfg["modes"].keys())
    missing = obs_modes - mode_keys
    if missing:
        raise ValueError(f"Modes in observables.yaml not defined in modes.yaml: {missing}")
