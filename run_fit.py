#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, cmath, numpy as np
from pathlib import Path

from fitio.load_inputs import load_all_configs
from fitio.write_root import write_root
from amplitudes.builder import build_amplitudes, exp_i


import emcee

# ----------------- MCMC setup -----------------
param_names = [
    # Hadronic
    "T_abs", "C_abs", "E_abs", "A_abs", "P_abs", "PA_abs",
    "C_phase", "E_phase", "A_phase", "P_phase", "PA_phase",
    # pi
    "pi_T_abs", "pi_C_abs", "pi_E_abs", "pi_A_abs", "pi_P_abs", "pi_PA_abs",
    "pi_T_phase", "pi_C_phase", "pi_E_phase", "pi_A_phase", "pi_P_phase", "pi_PA_phase",
    # kappa
    "kappa_P_abs", "kappa_A_abs", "kappa_E_abs", "kappa_PA_abs",
    "kappa_P_phase", "kappa_A_phase", "kappa_E_phase", "kappa_PA_phase",
    # rho
    "rho_P_abs", "rho_T_abs", "rho_C_abs",
    "rho_P_phase", "rho_T_phase", "rho_C_phase",
    # sigma
    "sigma_T_abs", "sigma_E_abs", "sigma_P_abs", "sigma_PA_abs",
    "sigma_T_phase", "sigma_E_phase", "sigma_P_phase", "sigma_PA_phase",
    # CKM
    "lambda", "A", "rhobar", "etabar",
    # global
    "global_scale"
]

n_params = len(param_names)

# Bounds for uniform priors
bounds = {}
for name in param_names:
    if name.endswith("_abs"):
        if "pi_" in name or "kappa_" in name or "rho_" in name or "sigma_" in name:
            bounds[name] = (0, 10)  # wide for SU(3)
        else:
            bounds[name] = (0, 1.5)  # hadronic
    elif name.endswith("_phase"):
        bounds[name] = (-math.pi, math.pi)
    elif name == "global_scale":
        bounds[name] = (1e-2, 1)
    elif name == "lambda":
        bounds[name] = (0.22, 0.23)
    elif name == "A":
        bounds[name] = (0.75, 0.85)
    elif name == "rhobar":
        bounds[name] = (0.1, 0.2)
    elif name == "etabar":
        bounds[name] = (0.3, 0.4)

def log_prior(theta, ckm_priors):
    lp = 0.0
    for i, name in enumerate(param_names):
        val = theta[i]
        if name.endswith("_abs") and ("pi_" in name or "kappa_" in name or "rho_" in name or "sigma_" in name):
            # Gaussian for SU(3) mag
            lp += -0.5 * ((val - 1.0) / 0.2)**2
        elif name in ["lambda", "A", "rhobar", "etabar"]:
            # Gaussian priors for CKM parameters (from config)
            prior = ckm_priors[name]
            lp += -0.5 * ((val - prior["mean"]) / prior["sigma"])**2
        elif name in bounds:
            if not (bounds[name][0] <= val <= bounds[name][1]):
                return -np.inf
        else:
            pass  # uniform
    return lp

def theta_to_dict(theta, ckm_priors):
    d = {}
    idx = 0
    # hadronic
    had = {}
    for k in ["T", "C", "E", "A", "P", "PA"]:
        phase = 0 if k == "T" else theta[idx + 6]
        had[k] = theta[idx] * exp_i(phase)
        idx += 1
    idx += 5  # skip phases
    d["hadronic"] = had
    # pi
    pi = {}
    for k in ["T", "C", "E", "A", "P", "PA"]:
        pi[f"{k}_abs"] = theta[idx]
        pi[f"{k}_phase"] = theta[idx + 6]
        idx += 1
    idx += 6
    d["pi"] = pi
    # kappa
    kappa = {}
    for k in ["P", "A", "E", "PA"]:
        kappa[f"{k}_abs"] = theta[idx]
        kappa[f"{k}_phase"] = theta[idx + 4]
        idx += 1
    idx += 4
    d["kappa"] = kappa
    # rho
    rho = {}
    for k in ["P", "T", "C"]:
        rho[f"{k}_abs"] = theta[idx]
        rho[f"{k}_phase"] = theta[idx + 3]
        idx += 1
    idx += 3
    d["rho"] = rho
    # sigma
    sigma = {}
    for k in ["T", "E", "P", "PA"]:
        sigma[f"{k}_abs"] = theta[idx]
        sigma[f"{k}_phase"] = theta[idx + 4]
        idx += 1
    idx += 4
    d["sigma"] = sigma
    # CKM parameters
    lam, A, rhobar, etabar = theta[idx], theta[idx+1], theta[idx+2], theta[idx+3]
    idx += 4
    ckm = ckm_from_wolfenstein(A, lam, rhobar, etabar)
    d["ckm"] = dict(
        gamma=ckm["gamma"],
        lam_u_d_abs=abs(ckm["lam_u_d"]),
        lam_c_d_abs=abs(ckm["lam_c_d"]),
        lam_u_s_abs=abs(ckm["lam_u_s"]),
        lam_c_s_abs=abs(ckm["lam_c_s"]),
    )
    d["mixing"] = {"beta": ckm["beta"], "beta_s": ckm["beta_s"]}
    # global_scale
    d["global_scale"] = theta[idx]
    idx += 1
    d["norm"] = {"N_B0": 1.0, "N_Bp": 1.0, "N_Bs": 1.0, "global_scale": d["global_scale"]}
    return d

def log_probability(theta, obs, modes, ckm_priors, obs_filter=None):
    """
    obs_filter: optional function(obs_row) -> bool to selectively include observables
    """
    lp = log_prior(theta, ckm_priors)
    if not np.isfinite(lp):
        return -np.inf
    try:
        theta_dict = theta_to_dict(theta, ckm_priors)
        pred = predict_all(theta_dict, obs, modes)
        
        # Filter observables if requested
        filtered_obs = obs if obs_filter is None else [r for r in obs if obs_filter(r)]
        
        ll = logL_gaussian(pred, filtered_obs)
        return lp + ll
    except:
        return -np.inf


# ----------------- CKM functions -----------------
def ckm_from_wolfenstein(A, lam, rhobar, etabar):
    rho = rhobar / (1 - lam**2 / 2)
    eta = etabar / (1 - lam**2 / 2)

    Vud = 1 - lam**2 / 2
    Vus = lam
    Vub = A * lam**3 * (rho - 1j * eta)
    Vcb = A * lam**2
    Vcs = 1 - lam**2 / 2
    Vcd = -lam + 0.5 * A**2 * lam**5 * (1 - 2 * (rho + 1j * eta))
    Vtd = A * lam**3 * (1 - rho - 1j * eta)
    Vts = -A * lam**2 + 0.5 * A * lam**4 * (1 - 2 * (rho + 1j * eta))
    Vtb = 1 + 0j

    lam_u_d = Vub * np.conj(Vud)
    lam_c_d = Vcb * np.conj(Vcd)
    lam_u_s = Vub * np.conj(Vus)
    lam_c_s = Vcb * np.conj(Vcs)

    gamma = np.angle(-Vud * Vub.conjugate() / (Vcd * Vcb.conjugate()))
    beta = np.angle(-Vcd * Vcb.conjugate() / (Vtd * Vtb.conjugate()))
    beta_s = np.angle(-Vts * Vtb.conjugate() / (Vcs * Vcb.conjugate()))

    return dict(
        lam_u_d=lam_u_d, lam_c_d=lam_c_d,
        lam_u_s=lam_u_s, lam_c_s=lam_c_s,
        gamma=gamma, beta=beta, beta_s=beta_s
    )


def draw_ckm(priors):
    """Gaussian CKM priors (from config/ckm.yaml)"""
    def g(x): return np.random.normal(x["mean"], x["sigma"])
    lam = g(priors["lambda"]); A = g(priors["A"])
    rhobar = g(priors["rhobar"]); etabar = g(priors["etabar"])
    return ckm_from_wolfenstein(A, lam, rhobar, etabar)


# ----------------- Hadronic priors -----------------
def _draw_phase(cfg):
    t = cfg.get("type", "uniform")
    if t == "fixed":
        return float(cfg["value"])
    elif t == "uniform":
        return np.random.uniform(cfg["min"], cfg["max"])
    elif t == "gaussian":
        return np.random.normal(cfg["mean"], cfg["sigma"])
    else:
        raise ValueError(f"Unknown phase type: {t}")


def draw_hadronic_scaled(hadronic_priors):
    """Draw hadronic topologies as complex amplitudes."""
    mags = hadronic_priors["magnitudes"]
    phs = hadronic_priors["phases"]
    ref = hadronic_priors.get("reference_phase", "T")

    had = {}
    for k in ["T", "C", "E", "A", "P", "PA"]:
        r = np.random.uniform(mags[k]["min"], mags[k]["max"])
        phi = 0.0 if k == ref else _draw_phase(phs[k])
        had[k] = r * exp_i(phi)
    return had


def draw_su3_breaking(cfg):
    """Draw SU(3)-breaking multiplicative factors."""
    mag_cfg = cfg["magnitude"]
    ph_cfg = cfg["phase"]

    def rphi():
        r = max(mag_cfg["min"], np.random.normal(mag_cfg["mean"], mag_cfg["sigma"]))
        phi = np.random.uniform(ph_cfg["min"], ph_cfg["max"])
        return r, phi

    def fill(dic, keys):
        for k in keys:
            r, phi = rphi()
            dic[f"{k}_abs"] = r
            dic[f"{k}_phase"] = phi

    pi, kappa, rho, sigma = {}, {}, {}, {}
    fill(pi, ["T", "C", "E", "A", "P", "PA"])
    fill(kappa, ["P", "A", "E", "PA"])
    fill(rho, ["P", "T", "C"])
    fill(sigma, ["T", "E", "P", "PA"])

    return pi, kappa, rho, sigma


# ----------------- Observable calculations -----------------
def br_from_amp(A, meson, N_B0=1.0, N_Bp=1.0, N_Bs=1.0, global_scale=1.0):
    N = {"B0": N_B0, "B+": N_Bp, "Bs": N_Bs}[meson]
    return N * global_scale * abs(A)**2


def direct_acp(A, Abar, eps=1e-16):
    """
    Compute direct CP asymmetry safely.
    Returns 0.0 when both amplitudes are too small.
    """
    aa = abs(A)**2
    bb = abs(Abar)**2
    denom = aa + bb
    if denom < eps:
        return 0.0
    return (bb - aa) / denom


def q_over_p(meson, beta, beta_s):
    phi = beta if meson == "B0" else (beta_s if meson == "Bs" else None)
    if phi is None:
        raise ValueError("q/p only defined for B0 or Bs")
    return exp_i(-2 * phi)


def lambdaf(qop, Abar, A): return qop * (Abar / A)


def S_C_from_lambda(lmbd):
    abs2 = abs(lmbd)**2
    denom = 1 + abs2
    C = (1 - abs2) / denom
    S = (2 * lmbd.imag) / denom
    return S, C


# ----------------- Likelihood -----------------
def logL_gaussian(pred, obs_rows):
    ll = 0.0
    for r in obs_rows:
        key = (r["mode"], r["type"])
        if key not in pred or not math.isfinite(pred[key]):
            continue
        d = pred[key] - r["value"]
        ll += -0.5 * (d*d) / (r["sigma"]**2)
    return ll


# ----------------- Prediction wrapper -----------------
def predict_all(theta, obs_rows, mode_map):
    A, Abar = build_amplitudes(theta)
    beta, beta_s = theta["mixing"]["beta"], theta["mixing"]["beta_s"]
    N_B0, N_Bp, N_Bs, global_scale = theta["norm"]["N_B0"], theta["norm"]["N_Bp"], theta["norm"]["N_Bs"], theta["norm"]["global_scale"]

    pred = {}
    for r in obs_rows:
        mode, typ = r["mode"], r["type"]
        meson = mode_map[mode]["meson"]
        if typ == "BR":
            pred[(mode, "BR")] = global_scale * (abs(A[mode])**2 + abs(Abar[mode])**2) / 2
        elif typ == "ACP":
            acp = direct_acp(A[mode], Abar[mode])
            # Flip sign for CP eigenstates (to match experimental convention)
            eta_f = mode_map[mode].get("cp_eigen", 1.0)
            if eta_f != 1.0 and mode_map[mode].get("has_mixing", False):
                acp *= -eta_f
            pred[(mode, "ACP")] = acp
        elif typ == "S":
            qop = q_over_p(meson, beta, beta_s)
            lam = lambdaf(qop, Abar[mode], A[mode])
            # Apply CP eigenvalue for modes measured as CP eigenstates (e.g., K_S π0)
            eta = mode_map[mode].get("cp_eigen", 1.0)
            lam *= eta
            S, C = S_C_from_lambda(lam)
            pred[(mode, "S")] = S
            pred[(mode, "C")] = C
    return pred


# ----------------- Main driver -----------------
def main():
    ap = argparse.ArgumentParser(description="CharmingFit MCMC sampling")
    ap.add_argument("--config", default="config", help="Base directory for all YAML configs")
    ap.add_argument("--output", default="CharmingFit.root", help="Output ROOT file")
    ap.add_argument("--nwalkers", type=int, default=100, help="Number of MCMC walkers")
    ap.add_argument("--nsteps", type=int, default=10000, help="Number of MCMC steps per walker")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--obs-filter", help="Python expression to filter observables (e.g., 'r[\"mode\"].startswith(\"B0_\")')")
    
    # Observable block flags
    ap.add_argument("--Bpipi", action="store_true", help="Include B→ππ observables")
    ap.add_argument("--BKpi", action="store_true", help="Include B→Kπ observables") 
    ap.add_argument("--BsKpi", action="store_true", help="Include Bs→Kπ observables")
    ap.add_argument("--BKK", action="store_true", help="Include B→KK observables")
    ap.add_argument("--BsKK", action="store_true", help="Include Bs→KK observables")
    ap.add_argument("--all-obs", action="store_true", help="Include all observables (default)")
    ap.add_argument("--nevents", type=int, default=0, help="Number of importance sampling events (0 to skip)")
    
    args = ap.parse_args()

    np.random.seed(args.seed)
    cfg = load_all_configs(args.config)

    obs = cfg["observables"]
    modes = cfg["modes"]
    ckm_priors = cfg["ckm_priors"]

    # Define observable blocks
    obs_blocks = {
        "Bpipi": ["B0_pi+pi-", "B+_pi+pi0", "B0_pi0pi0"],
        "BKpi": ["B0_K+pi-", "B0_K0pi0", "B+_K+pi0", "B+_K0barpi+"],
        "BsKpi": ["Bs_K-pi+"],
        "BKK": ["B+_K0K+", "B0_K+K-", "B0_K0K0bar"],
        "BsKK": ["Bs_K+K-", "Bs_K0K0bar"]
    }

    # Create observable filter based on flags
    obs_filter = None
    if args.obs_filter:
        # Custom filter expression
        def obs_filter(r):
            try:
                return eval(args.obs_filter, {"__builtins__": {}}, {"r": r})
            except:
                return True  # include by default on error
        
        filtered_obs = [r for r in obs if obs_filter(r)]
        print(f"📋 Using custom filter: {args.obs_filter}")
        print(f"📊 Using {len(filtered_obs)} observables")
        
    elif any([args.Bpipi, args.BKpi, args.BsKpi, args.BKK, args.BsKK]):
        # Block-based filtering
        enabled_blocks = []
        if args.Bpipi: enabled_blocks.append("Bpipi")
        if args.BKpi: enabled_blocks.append("BKpi")
        if args.BsKpi: enabled_blocks.append("BsKpi")
        if args.BKK: enabled_blocks.append("BKK")
        if args.BsKK: enabled_blocks.append("BsKK")
        
        enabled_modes = set()
        for block in enabled_blocks:
            enabled_modes.update(obs_blocks[block])
        
        def obs_filter(r):
            return r["mode"] in enabled_modes
        
        print(f"📋 Enabled observable blocks: {enabled_blocks}")
        print(f"📋 Enabled modes: {sorted(enabled_modes)}")
        
        # Count observables by type
        filtered_obs = [r for r in obs if obs_filter(r)]
        obs_by_type = {}
        for r in filtered_obs:
            key = f"{r['mode']} ({r['type']})"
            obs_by_type[key] = r['value']
        print(f"📊 Using {len(filtered_obs)} observables:")
        for key, value in sorted(obs_by_type.items()):
            print(f"   {key}: {value}")
    else:
        # Default: include all observables
        print(f"📋 Using all {len(obs)} observables (no filtering)")

    # Fix CKM to central values for MCMC
    ckm_fixed = ckm_from_wolfenstein(ckm_priors["A"]["mean"], ckm_priors["lambda"]["mean"], 
                                     ckm_priors["rhobar"]["mean"], ckm_priors["etabar"]["mean"])

    # Initial positions for walkers
    pos = []
    for i in range(args.nwalkers):
        np.random.seed(args.seed + i)  # different seed per walker
        theta = []
        for name in param_names:
            if name.endswith("_abs"):
                if "pi_" in name or "kappa_" in name or "rho_" in name or "sigma_" in name:
                    val = np.random.normal(1.0, 0.2)
                else:
                    val = np.random.uniform(bounds[name][0], bounds[name][1])
                theta.append(val + np.random.normal(0, 0.01))  # add noise
            elif name.endswith("_phase"):
                theta.append(np.random.uniform(bounds[name][0], bounds[name][1]) + np.random.normal(0, 0.01))
            elif name in ["lambda", "A", "rhobar", "etabar"]:
                # Start near the prior means
                priors = {
                    "lambda": 0.22650, "A": 0.790, "rhobar": 0.145, "etabar": 0.343
                }
                theta.append(priors[name] + np.random.normal(0, 0.01))
            elif name == "global_scale":
                theta.append(np.random.uniform(bounds[name][0], bounds[name][1]) + np.random.normal(0, 0.001))
        pos.append(theta)
    pos = np.array(pos)

    # Run MCMC
    sampler = emcee.EnsembleSampler(args.nwalkers, n_params, log_probability, args=(obs, modes, ckm_priors, obs_filter))
    sampler.run_mcmc(pos, args.nsteps, progress=True)

    # Get samples
    flat_samples = sampler.get_chain(discard=100, thin=10, flat=True)

    # Prepare buffer for ROOT output
    buf = {"weight": [], "logL": [], "gamma": [], "beta": [], "beta_s": []}
    pred_keys = sorted(set((r["mode"], r["type"]) for r in obs))
    pred_names = {k: f"pred__{k[0]}__{k[1]}".replace("+","p").replace("-","m") for k in pred_keys}
    for name in pred_names.values(): buf[name] = []

    # Hadronic topologies
    for k in ["T","C","E","A","P","PA"]:
        buf[f"{k}_abs"] = []
        buf[f"{k}_phase"] = []

    # SU(3)-breaking factors
    for fam, keys in zip(["pi","kappa","rho","sigma"],
                         [["T","C","E","A","P","PA"],
                          ["P","A","E","PA"],
                          ["P","T","C"],
                          ["T","E","P","PA"]]):
        for k in keys:
            buf[f"{fam}_{k}_abs"] = []
            buf[f"{fam}_{k}_phase"] = []

    # CKM parameters
    buf["lambda"] = []
    buf["A"] = []
    buf["rhobar"] = []
    buf["etabar"] = []
    
    buf["global_scale"] = []

    for theta in flat_samples:
        theta_dict = theta_to_dict(theta, ckm_priors)
        pred = predict_all(theta_dict, obs, modes)
        ll = logL_gaussian(pred, obs)
        buf["weight"].append(1.0)  # uniform weight for MCMC
        buf["logL"].append(ll)
        buf["gamma"].append(theta_dict["mixing"]["beta"] * 180/np.pi)  # Convert to degrees
        buf["beta"].append(theta_dict["mixing"]["beta"])
        buf["beta_s"].append(theta_dict["mixing"]["beta_s"])        
        # Extract CKM parameters from theta (positions 45-48 in 0-based indexing)
        ckm_start_idx = 45
        buf["lambda"].append(theta[ckm_start_idx])
        buf["A"].append(theta[ckm_start_idx+1])
        buf["rhobar"].append(theta[ckm_start_idx+2])
        buf["etabar"].append(theta[ckm_start_idx+3])
        for k, name in pred_names.items():
            buf[name].append(pred.get(k, float("nan")))

        # Hadronic topologies normalized by T
        Tref = theta_dict["hadronic"]["T"]
        for k in ["T","C","E","A","P","PA"]:
            amp = theta_dict["hadronic"][k] / Tref
            buf[f"{k}_abs"].append(abs(amp))
            buf[f"{k}_phase"].append(np.angle(amp))

        # SU(3)-breaking factors
        for fam, block in zip(["pi","kappa","rho","sigma"],
                              [theta_dict["pi"], theta_dict["kappa"], theta_dict["rho"], theta_dict["sigma"]]):
            for kk in [x[:-4] for x in block.keys() if x.endswith("_abs")]:
                buf[f"{fam}_{kk}_abs"].append(block[f"{kk}_abs"])
                buf[f"{fam}_{kk}_phase"].append(block[f"{kk}_phase"])

        buf["global_scale"].append(theta_dict["global_scale"])

    write_root(args.output, buf)

    print(f"Wrote: {args.output}")
    print(f"Samples: {len(flat_samples)}")

    # Importance sampling (optional)
    if args.nevents > 0:
        obs = cfg["observables"]
        modes = cfg["modes"]
        ckm_priors = cfg["ckm_priors"]
        had_priors = cfg["hadronic_priors"]
        su3_priors = cfg["su3_priors"]

        buf = {"weight": [], "logL": [], "gamma": [], "beta": [], "beta_s": [], "global_scale": []}
        pred_keys = sorted(set((r["mode"], r["type"]) for r in obs))
        pred_names = {k: f"pred__{k[0]}__{k[1]}".replace("+","p").replace("-","m") for k in pred_keys}
        for name in pred_names.values(): buf[name] = []

        # Hadronic topologies
        for k in ["T","C","E","A","P","PA"]:
            buf[f"{k}_abs"] = []
            buf[f"{k}_phase"] = []

        # SU(3)-breaking factors
        for fam, keys in zip(["pi","kappa","rho","sigma"],
                             [["T","C","E","A","P","PA"],
                              ["P","A","E","PA"],
                              ["P","T","C"],
                              ["T","E","P","PA"]]):
            for k in keys:
                buf[f"{fam}_{k}_abs"] = []
                buf[f"{fam}_{k}_phase"] = []

        N_B0 = N_Bp = N_Bs = 1.0

        for _ in range(args.nevents):
            ckm = draw_ckm(ckm_priors)
            had = draw_hadronic_scaled(had_priors)
            piF, kap, rhoF, sig = draw_su3_breaking(su3_priors)
            global_scale = np.random.uniform(1e-2, 1)

            theta = dict(
                ckm=dict(
                    gamma=ckm["gamma"],
                    lam_u_d_abs=abs(ckm["lam_u_d"]),
                    lam_c_d_abs=abs(ckm["lam_c_d"]),
                    lam_u_s_abs=abs(ckm["lam_u_s"]),
                    lam_c_s_abs=abs(ckm["lam_c_s"]),
                ),
                mixing=dict(beta=ckm["beta"], beta_s=ckm["beta_s"]),
                norm=dict(N_B0=N_B0, N_Bp=N_Bp, N_Bs=N_Bs, global_scale=global_scale),
                hadronic=had, pi=piF, kappa=kap, rho=rhoF, sigma=sig
            )

            pred = predict_all(theta, obs, modes)
            ll = logL_gaussian(pred, obs)
            # Store the log-likelihood now; stable weights are computed after the loop.
            buf["weight"].append(0.0)
            buf["logL"].append(ll)
            buf["gamma"].append(theta["ckm"]["gamma"])
            buf["beta"].append(theta["mixing"]["beta"])
            buf["beta_s"].append(theta["mixing"]["beta_s"])
            buf["global_scale"].append(global_scale)
            for k, name in pred_names.items():
                buf[name].append(pred.get(k, float("nan")))

            # Hadronic topologies normalized by T
            Tref = theta["hadronic"]["T"]
            for k in ["T","C","E","A","P","PA"]:
                amp = theta["hadronic"][k] / Tref
                buf[f"{k}_abs"].append(abs(amp))
                buf[f"{k}_phase"].append(np.angle(amp))

            # SU(3)-breaking factors
            for fam, block in zip(["pi","kappa","rho","sigma"],
                                  [theta["pi"], theta["kappa"], theta["rho"], theta["sigma"]]):
                for kk in [x[:-4] for x in block.keys() if x.endswith("_abs")]:
                    buf[f"{fam}_{kk}_abs"].append(block[f"{kk}_abs"])
                    buf[f"{fam}_{kk}_phase"].append(block[f"{kk}_phase"])

        # Compute in log-space to avoid underflow
        logw = np.array(buf["logL"])
        logw -= np.max(logw)             # rescale for numerical stability
        w = np.exp(logw)
        buf["weight"] = w.tolist()

        write_root(args.output, buf)

        ess = (w.sum()**2) / (w*w).sum() if np.isfinite(w).all() and w.sum() > 0 else float("nan")

        print(f"Wrote: {args.output}")
        print(f"Draws: {len(w)},  ESS ≈ {ess:.1f}  (ESS/N ≈ {ess/len(w):.3f})")

        # ESS is useful to know whether your likelihood is too sharp:
        # ESS/N ≈ 1 → very flat posterior (weights uniform)
        # ESS/N ≪ 1 → very peaked posterior (you may need MCMC rather than pure sampling)
    
if __name__ == "__main__":
    main()
