# scripts/check_mode_consistency.py
import yaml
from amplitudes.builder import build_amplitudes
from priors.generate_point import draw_point

obs = yaml.safe_load(open("data/observables.yaml"))["observables"]
A, _ = build_amplitudes(draw_point())
obs_modes = {o["mode"] for o in obs}
missing = obs_modes - A.keys()
if missing:
    print("❌ Missing amplitudes:", missing)
else:
    print("✅ All observables have matching amplitudes.")
