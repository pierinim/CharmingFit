#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, cmath, numpy as np
from pathlib import Path

from fitio.load_inputs import load_all_configs
from fitio.write_root import write_root

from amplitudes.builder import build_amplitudes

# ----------------- Utility functions -----------------
def exp_i(phi): return cmath.cos(phi) + 1j * cmath.sin(phi)
def polar(mag, phase): return mag * exp_i(phase)


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


def draw_hadronic_scaled(lam_u_d_abs, hadronic_priors, BR_pipi=5.36e-6):
    """Draw hadronic topologies as complex amplitudes."""
    T0 = math.sqrt(BR_pipi) / lam_u_d_abs
    mags = hadronic_priors["magnitudes"]
    phs = hadronic_priors["phases"]
    ref = hadronic_priors.get("reference_phase", "T")

    had = {}
    for k in ["T", "C", "E", "A", "P", "PA"]:
        r = np.random.uniform(mags[k]["min"], mags[k]["max"]) * T0
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
def br_from_amp(A, meson, N_B0=1.0, N_Bp=1.0, N_Bs=1.0):
    N = {"B0": N_B0, "B+": N_Bp, "Bs": N_Bs}[meson]
    return N * abs(A)**2


def direct_acp(A, Abar):
    return (abs(Abar)**2 - abs(A)**2) / (abs(Abar)**2 + abs(A)**2)


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
    N_B0, N_Bp, N_Bs = theta["norm"]["N_B0"], theta["norm"]["N_Bp"], theta["norm"]["N_Bs"]

    pred = {}
    for r in obs_rows:
        mode, typ = r["mode"], r["type"]
        meson = mode_map[mode]["meson"]
        if typ == "BR":
            pred[(mode, "BR")] = br_from_amp(A[mode], meson, N_B0, N_Bp, N_Bs)
        elif typ == "ACP":
            pred[(mode, "ACP")] = direct_acp(A[mode], Abar[mode])
        elif typ == "S":
            qop = q_over_p(meson, beta, beta_s)
            lam = lambdaf(qop, Abar[mode], A[mode])
            S, C = S_C_from_lambda(lam)
            pred[(mode, "S")] = S
            pred[(mode, "C")] = C
    return pred


# ----------------- Main driver -----------------
def main():
    ap = argparse.ArgumentParser(description="CharmingFit Bayesian sampling for charmless B decays")
    ap.add_argument("--config", default="config", help="Base directory for all YAML configs")
    ap.add_argument("--out", default="CharmingFit.root", help="Output ROOT file")
    ap.add_argument("-N", type=int, default=200000, help="Number of random draws")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    np.random.seed(args.seed)
    cfg = load_all_configs(args.config)

    obs = cfg["observables"]
    modes = cfg["modes"]
    ckm_priors = cfg["ckm_priors"]
    had_priors = cfg["hadronic_priors"]
    su3_priors = cfg["su3_priors"]

    buf = {"weight": [], "logL": [], "gamma": [], "beta": [], "beta_s": []}
    pred_keys = sorted(set((r["mode"], r["type"]) for r in obs))
    pred_names = {k: f"pred__{k[0]}__{k[1]}".replace("+","p").replace("-","m") for k in pred_keys}
    for name in pred_names.values(): buf[name] = []

    N_B0 = N_Bp = N_Bs = 1.0

    for _ in range(args.N):
        ckm = draw_ckm(ckm_priors)
        had = draw_hadronic_scaled(abs(ckm["lam_u_d"]), had_priors)
        piF, kap, rhoF, sig = draw_su3_breaking(su3_priors)

        theta = dict(
            ckm=dict(
                gamma=ckm["gamma"],
                lam_u_d_abs=abs(ckm["lam_u_d"]),
                lam_c_d_abs=abs(ckm["lam_c_d"]),
                lam_u_s_abs=abs(ckm["lam_u_s"]),
                lam_c_s_abs=abs(ckm["lam_c_s"]),
            ),
            mixing=dict(beta=ckm["beta"], beta_s=ckm["beta_s"]),
            norm=dict(N_B0=N_B0, N_Bp=N_Bp, N_Bs=N_Bs),
            hadronic=had, pi=piF, kappa=kap, rho=rhoF, sigma=sig
        )

        pred = predict_all(theta, obs, modes)
        ll = logL_gaussian(pred, obs)
        w = math.exp(ll)

        buf["weight"].append(w)
        buf["logL"].append(ll)
        buf["gamma"].append(theta["ckm"]["gamma"])
        buf["beta"].append(theta["mixing"]["beta"])
        buf["beta_s"].append(theta["mixing"]["beta_s"])
        for k, name in pred_names.items():
            buf[name].append(pred.get(k, float("nan")))

    write_root(args.out, buf)

    w = np.array(buf["weight"])
    ess = (w.sum()**2) / (w*w).sum()
    print(f"Wrote: {args.out}")
    print(f"Draws: {len(w)},  ESS ≈ {ess:.1f}  (ESS/N ≈ {ess/len(w):.3f})")


if __name__ == "__main__":
    main()
