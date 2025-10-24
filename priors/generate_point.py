import numpy as np
import yaml
from pathlib import Path
from amplitudes.ckm import ckm_from_wolfenstein

# -----------------------------
# Utility: read YAML once
# -----------------------------
def load_priors_config(path="config/priors.yaml"):
    """Load the prior configuration YAML."""
    return yaml.safe_load(Path(path).read_text())

# -----------------------------
# Sampling helpers
# -----------------------------
def sample_uniform(p):
    return np.random.uniform(p["min"], p["max"])

def sample_normal(p):
    val = np.random.normal(p["mean"], p["sigma"])
    # Optional truncation to avoid negative magnitudes
    if "min" in p and val < p["min"]:
        val = p["min"]
    return val

def draw_value(entry):
    """Dispatch by distribution type."""
    dist = entry.get("dist", "uniform")
    if dist == "uniform":
        return sample_uniform(entry)
    elif dist == "normal":
        return sample_normal(entry)
    elif dist == "scaled_uniform":
        # For hadronic amplitudes, scaling handled separately
        return np.random.uniform(entry["min"], entry["max"])
    elif dist == "fixed" or "value" in entry:
        return entry["value"]
    else:
        raise ValueError(f"Unknown prior type: {dist}")

# -----------------------------
# Main drawing routines
# -----------------------------
def draw_ckm(priors):
    """Draw CKM parameters (λ, A, ρ̄, η̄) from priors.yaml."""
    pckm = priors["ckm"]
    lam = draw_value(pckm["lambda"])
    A = draw_value(pckm["A"])
    rhobar = draw_value(pckm["rhobar"])
    etabar = draw_value(pckm["etabar"])
    ckm = ckm_from_wolfenstein(A, lam, rhobar, etabar)
    return ckm

def draw_hadronic(ckm, priors):
    """Draw topological amplitudes relative to T0."""
    phad = priors["hadronic"]
    BR_pipi = 5.36e-6
    lam_u_d = abs(ckm["lam_u_d"])
    T0 = np.sqrt(BR_pipi) / lam_u_d
    had = {}
    for k in ["T", "C", "E", "A", "P", "PA"]:
        rng = phad[k]
        had[k] = draw_value(rng) * T0  # scale factor
    return had

def draw_su3_breaking(priors):
    """Draw SU(3)-breaking factors in polar form."""
    pmag = priors["su3_breaking"]["magnitude"]
    pphi = priors["su3_breaking"]["phase"]

    def one_factor():
        r = draw_value(pmag)
        phi = draw_value(pphi)
        return r, phi

    def make_group(keys):
        out = {}
        for k in keys:
            r, phi = one_factor()
            out[f"{k}_abs"] = r
            out[f"{k}_phase"] = phi
        return out

    pi = make_group(["T", "C", "E", "A", "P", "PA"])
    kappa = make_group(["P", "A", "E", "PA"])
    rho = make_group(["P", "T", "C"])
    sigma = make_group(["T", "E", "P", "PA"])
    return pi, kappa, rho, sigma

def draw_norms(priors):
    """Return normalization factors (typically fixed at 1)."""
    pnorm = priors.get("norm", {})
    return dict(
        N_B0=pnorm.get("N_B0", {}).get("value", 1.0),
        N_Bp=pnorm.get("N_Bp", {}).get("value", 1.0),
        N_Bs=pnorm.get("N_Bs", {}).get("value", 1.0),
    )

def draw_point(priors_path="config/priors.yaml"):
    """Generate one random parameter point consistent with priors.yaml."""
    priors = load_priors_config(priors_path)
    ckm = draw_ckm(priors)
    had = draw_hadronic(ckm, priors)
    piF, kap, rhoF, sig = draw_su3_breaking(priors)
    norms = draw_norms(priors)

    theta = dict(
        ckm=dict(
            gamma=ckm["gamma"],
            lam_u_d_abs=abs(ckm["lam_u_d"]),
            lam_c_d_abs=abs(ckm["lam_c_d"]),
            lam_u_s_abs=abs(ckm["lam_u_s"]),
            lam_c_s_abs=abs(ckm["lam_c_s"]),
        ),
        mixing=dict(beta=ckm["beta"], beta_s=ckm["beta_s"]),
        norm=norms,
        hadronic=had,
        pi=piF, kappa=kap, rho=rhoF, sigma=sig
    )
    return theta

# -----------------------------
# CLI test
# -----------------------------
if __name__ == "__main__":
    th = draw_point()
    print("Random draw:")
    print("gamma =", th["ckm"]["gamma"])
    print("lam_u_d =", th["ckm"]["lam_u_d_abs"])
    print("T =", th["hadronic"]["T"])
