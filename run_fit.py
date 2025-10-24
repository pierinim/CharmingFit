#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, cmath, numpy as np, yaml, uproot
from pathlib import Path

# ---------- Inputs ----------
# UTfit 2022 (arXiv:2212.03894) flat ranges
UTFIT_RANGES = {
    "lambda": (0.2255, 0.2275),
    "A":      (0.76,   0.82),
    "rhobar": (0.11,   0.17),
    "etabar": (0.33,   0.38),
}

# ---------- Utilities ----------
def exp_i(phi): return cmath.cos(phi) + 1j*cmath.sin(phi)
def polar(mag, phase): return mag * exp_i(phase)

def ckm_from_wolfenstein(A, lam, rhobar, etabar):
    # Buras: convert to (rho, eta)
    rho = rhobar / (1 - lam**2/2)
    eta = etabar / (1 - lam**2/2)
    # CKM elements to O(λ^5) where needed
    Vud = 1 - lam**2/2
    Vus = lam
    Vub = A * lam**3 * (rho - 1j*eta)
    Vcb = A * lam**2
    Vcs = 1 - lam**2/2
    Vcd = -lam + 0.5*A**2*lam**5*(1 - 2*(rho + 1j*eta))
    Vtd = A * lam**3 * (1 - rho - 1j*eta)
    Vts = -A * lam**2 + 0.5*A*lam**4*(1 - 2*(rho + 1j*eta))
    Vtb = 1 + 0j

    lam_u_d = Vub * np.conj(Vud)
    lam_c_d = Vcb * np.conj(Vcd)
    lam_u_s = Vub * np.conj(Vus)
    lam_c_s = Vcb * np.conj(Vcs)

    gamma  = np.angle(-Vud*Vub.conjugate()/(Vcd*Vcb.conjugate()))
    beta   = np.angle(-Vcd*Vcb.conjugate()/(Vtd*Vtb.conjugate()))
    beta_s = np.angle(-Vts*Vtb.conjugate()/(Vcs*Vcb.conjugate()))

    return dict(
        lam_u_d=lam_u_d, lam_c_d=lam_c_d,
        lam_u_s=lam_u_s, lam_c_s=lam_c_s,
        gamma=gamma, beta=beta, beta_s=beta_s
    )

# ---------- Priors: CKM + hadronic ----------
def draw_ckm():
    lam = np.random.uniform(*UTFIT_RANGES["lambda"])
    A   = np.random.uniform(*UTFIT_RANGES["A"])
    rhobar = np.random.uniform(*UTFIT_RANGES["rhobar"])
    etabar = np.random.uniform(*UTFIT_RANGES["etabar"])
    return ckm_from_wolfenstein(A, lam, rhobar, etabar)

def draw_hadronic_scaled(lam_u_d_abs, BR_pipi=5.36e-6):
    # T0 ~ sqrt(BR(pi+pi-)) / |λ_u^(d)|
    T0 = math.sqrt(BR_pipi) / lam_u_d_abs
    # All reference topologies real, positive, ~ U(0,2) * T0
    had = {k: np.random.uniform(0, 2) * T0 for k in ["T","C","E","A","P","PA"]}
    return had

def draw_su3_breaking():
    # Each factor ~ magnitude N(1,0.3) truncated at >0; phase U(-π,π)
    def rphi():
        r = max(1e-6, np.random.normal(1.0, 0.3))
        phi = np.random.uniform(-math.pi, math.pi)
        return r, phi
    # π-set (B->ππ)
    pi = {}
    for k in ["T","C","E","A","P","PA"]:
        r,phi = rphi(); pi[f"{k}_abs"]=r; pi[f"{k}_phase"]=phi
    # κ-set (B->KK, ΔS=0)
    kappa = {}
    for k in ["P","A","E","PA"]:
        r,phi = rphi(); kappa[f"{k}_abs"]=r; kappa[f"{k}_phase"]=phi
    # ρ-set (Bs->Kπ)
    rho = {}
    for k in ["P","T","C"]:
        r,phi = rphi(); rho[f"{k}_abs"]=r; rho[f"{k}_phase"]=phi
    # σ-set (Bs->KK)
    sigma = {}
    for k in ["T","E","P","PA"]:
        r,phi = rphi(); sigma[f"{k}_abs"]=r; sigma[f"{k}_phase"]=phi
    return pi, kappa, rho, sigma

# ---------- Amplitudes (your LaTeX, exactly) ----------
def build_amplitudes(theta):
    ckm = theta["ckm"]; had = theta["hadronic"]
    piF, kap, rhoF, sig = theta["pi"], theta["kappa"], theta["rho"], theta["sigma"]
    gamma = ckm["gamma"]

    # CKM building blocks: |λ| * e^{-iγ} for u; |λ| for c (phases moved to γ)
    lu_d = ckm["lam_u_d_abs"] * exp_i(-gamma)
    lc_d = ckm["lam_c_d_abs"] + 0j
    lu_s = ckm["lam_u_s_abs"] * exp_i(-gamma)
    lc_s = ckm["lam_c_s_abs"] + 0j

    T,C,E,A,P,PA = (had[k] for k in ("T","C","E","A","P","PA"))
    # SU(3) factors
    piT=polar(piF["T_abs"],piF["T_phase"]);  piC=polar(piF["C_abs"],piF["C_phase"])
    piE=polar(piF["E_abs"],piF["E_phase"]);  piA=polar(piF["A_abs"],piF["A_phase"])
    piP=polar(piF["P_abs"],piF["P_phase"]);  piPA=polar(piF["PA_abs"],piF["PA_phase"])
    kP=polar(kap["P_abs"],kap["P_phase"]);   kA=polar(kap["A_abs"],kap["A_phase"])
    kE=polar(kap["E_abs"],kap["E_phase"]);   kPA=polar(kap["PA_abs"],kap["PA_phase"])
    rP=polar(rhoF["P_abs"],rhoF["P_phase"]); rT=polar(rhoF["T_abs"],rhoF["T_phase"])
    rC=polar(rhoF["C_abs"],rhoF["C_phase"])
    sT=polar(sig["T_abs"],sig["T_phase"]);   sE=polar(sig["E_abs"],sig["E_phase"])
    sP=polar(sig["P_abs"],sig["P_phase"]);   sPA=polar(sig["PA_abs"],sig["PA_phase"])

    Aamp = {}

    # B -> K pi (|ΔS|=1, q=s)
    Aamp["Bp_K0pi+"] =  (lc_s * P) + (lu_s * A)
    Aamp["B0_K+pi-"] = -(lc_s * P) - (lu_s * T)
    Aamp["Bp_K+pi0"] = (-(lc_s * P) - (lu_s * (T + C + A))) / cmath.sqrt(2)
    Aamp["B0_KSpi0"] = ( (lc_s * P) - (lu_s * C) ) / cmath.sqrt(2)

    # B -> π π (ΔS=0, q=d)
    Aamp["B0_pi+pi-"] = - ( lu_d * (T*piT + E*piE) + lc_d * (P*piP + PA*piPA) )
    Aamp["Bp_pi+pi0"] = - ( lu_d * (T*piT + C*piC + A*piA) ) / cmath.sqrt(2)
    Aamp["B0_pi0pi0"] = ( - ( lu_d * ( C*piC ) ) + ( lc_d * ( P*piP ) ) + ( lu_d * (E*piE) ) + ( lc_d * (PA*piPA) ) ) / cmath.sqrt(2)

    # B -> K K (ΔS=0, q=d)
    Aamp["Bp_KSK+"]   =  ( lc_d * (P*kP) ) + ( lu_d * (A*kA) )
    Aamp["B0_KSKS"]   =  ( lc_d * (P*kP + PA*kPA) )
    Aamp["B0_K+K-"]   = -( lu_d * (E*kE) + lc_d * (PA*kPA) )

    # Bs -> K π (q=d)
    Aamp["Bs_K-pi+"]  = -( lc_d * (P*rP) + lu_d * (T*rT) )
    Aamp["Bs_K0pi0"]  = ( lc_d * (P*rP) - lu_d * (C*rC) ) / cmath.sqrt(2)

    # Bs -> K K (q=s)
    Aamp["Bs_K+K-"]    = -( lu_s * (T*sT + E*sE) + lc_s * (P*sP + PA*sPA) )
    Aamp["Bs_K0K0bar"] =  ( lc_s * (P*sP + PA*sPA) )

    # CP-conjugate amplitudes: flip γ -> -γ
    lu_d_bar = ckm["lam_u_d_abs"] * exp_i(+gamma)
    lu_s_bar = ckm["lam_u_s_abs"] * exp_i(+gamma)
    lc_d_bar = lc_d; lc_s_bar = lc_s

    Abar = {}
    # Repeat with lu_{d,s} -> conjugated weak phase
    Abar["Bp_K0pi+"] =  (lc_s_bar * P) + (lu_s_bar * A)
    Abar["B0_K+pi-"] = -(lc_s_bar * P) - (lu_s_bar * T)
    Abar["Bp_K+pi0"] = (-(lc_s_bar * P) - (lu_s_bar * (T + C + A))) / cmath.sqrt(2)
    Abar["B0_KSpi0"] = ( (lc_s_bar * P) - (lu_s_bar * C) ) / cmath.sqrt(2)

    Abar["B0_pi+pi-"] = - ( lu_d_bar * (T*piT + E*piE) + lc_d_bar * (P*piP + PA*piPA) )
    Abar["Bp_pi+pi0"] = - ( lu_d_bar * (T*piT + C*piC + A*piA) ) / cmath.sqrt(2)
    Abar["B0_pi0pi0"] = ( - ( lu_d_bar * ( C*piC ) ) + ( lc_d_bar * ( P*piP ) ) + ( lu_d_bar * (E*piE) ) + ( lc_d_bar * (PA*piPA) ) ) / cmath.sqrt(2)

    Abar["Bp_KSK+"]   =  ( lc_d_bar * (P*kP) ) + ( lu_d_bar * (A*kA) )
    Abar["B0_KSKS"]   =  ( lc_d_bar * (P*kP + PA*kPA) )
    Abar["B0_K+K-"]   = -( lu_d_bar * (E*kE) + lc_d_bar * (PA*kPA) )

    Abar["Bs_K-pi+"]  = -( lc_d_bar * (P*rP) + lu_d_bar * (T*rT) )
    Abar["Bs_K0pi0"]  = ( lc_d_bar * (P*rP) - lu_d_bar * (C*rC) ) / cmath.sqrt(2)

    Abar["Bs_K+K-"]    = -( lu_s_bar * (T*sT + E*sE) + lc_s_bar * (P*sP + PA*sPA) )
    Abar["Bs_K0K0bar"] =  ( lc_s_bar * (P*sP + PA*sPA) )

    return Aamp, Abar

# ---------- Observables ----------
def br_from_amp(A, meson, N_B0=1.0, N_Bp=1.0, N_Bs=1.0):
    N = {"B0": N_B0, "Bp": N_Bp, "Bs": N_Bs}[meson]
    return N * (A.real*A.real + A.imag*A.imag)

def direct_acp(A, Abar):
    aa = (A.real*A.real + A.imag*A.imag)
    bb = (Abar.real*Abar.real + Abar.imag*Abar.imag)
    return (bb - aa) / (bb + aa)

def q_over_p(meson, beta, beta_s):
    phi = beta if meson=="B0" else (beta_s if meson=="Bs" else None)
    if phi is None: raise ValueError("q/p only for B0 or Bs")
    return cmath.exp(-2j*phi)

def lambdaf(qop, Abar, A):
    return qop * (Abar / A)

def S_C_from_lambda(lmbd):
    abs2 = (lmbd.real*lmbd.real + lmbd.imag*lmbd.imag)
    denom = 1.0 + abs2
    C = (1.0 - abs2) / denom
    S = (2.0 * lmbd.imag) / denom
    return S, C

# Which meson each mode belongs to
MODE_MESON = {
  "B0_pi+pi-":"B0", "Bp_pi+pi0":"Bp", "B0_pi0pi0":"B0",
  "B0_K+pi-":"B0", "B0_KSpi0":"B0", "Bp_K+pi0":"Bp", "Bp_K0pi+":"Bp",
  "Bp_KSK+":"Bp", "B0_K+K-":"B0", "B0_KSKS":"B0",
  "Bs_K+K-":"Bs", "Bs_KSKS":"Bs", "Bs_K-pi+":"Bs",
  # extra internal / not in input table:
  "Bs_K0pi0":"Bs", "Bs_K0K0bar":"Bs"
}

# ---------- Likelihood ----------
def logL_gaussian(pred, obs_rows):
    ll = 0.0
    for r in obs_rows:
        key = (r["mode"], r["type"])
        if key not in pred or not math.isfinite(pred[key]):  # skip missing/NaN
            continue
        d = pred[key] - r["value"]
        s = r["sigma"]
        ll += -0.5*(d*d)/(s*s)
    return ll

# ---------- Prediction wrapper ----------
def predict_all(theta, obs_rows):
    A, Abar = build_amplitudes(theta)
    beta, beta_s = theta["mixing"]["beta"], theta["mixing"]["beta_s"]
    N_B0, N_Bp, N_Bs = theta["norm"]["N_B0"], theta["norm"]["N_Bp"], theta["norm"]["N_Bs"]

    pred = {}
    for r in obs_rows:
        mode = r["mode"]; typ = r["type"]; meson = MODE_MESON[mode]
        if typ == "BR":
            pred[(mode,"BR")] = br_from_amp(A[mode], meson, N_B0, N_Bp, N_Bs)
        elif typ == "ACP":
            pred[(mode,"ACP")] = direct_acp(A[mode], Abar[mode])
        elif typ == "S":
            qop = q_over_p(meson, beta, beta_s)
            lam = lambdaf(qop, Abar[mode], A[mode])
            S, C = S_C_from_lambda(lam)
            pred[(mode,"S")] = S
            pred[(mode,"C")] = C  # stored even if not in obs; handy for diagnostics
    return pred

# ---------- Main sampling loop ----------
def main():
    ap = argparse.ArgumentParser(description="UTfit-style Bayesian sampling for charmless two-body B decays")
    ap.add_argument("--observables", default="data/observables.yaml", help="YAML with experimental inputs")
    ap.add_argument("--out", default="charmless_fit.root", help="Output ROOT file")
    ap.add_argument("-N", type=int, default=200000, help="Number of prior draws")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    np.random.seed(args.seed)
    obs = yaml.safe_load(Path(args.observables).read_text())["observables"]

    # Storage buffers (fill with python lists, then convert to numpy)
    buf = {
        "weight": [], "logL": [],
        "gamma": [], "beta": [], "beta_s": [],
        "lam_u_d_abs": [], "lam_c_d_abs": [], "lam_u_s_abs": [], "lam_c_s_abs": [],
        "T": [], "C": [], "E": [], "A": [], "P": [], "PA": [],
    }
    # Dynamically add predicted branches for every measured quantity
    pred_keys = sorted(set((r["mode"], r["type"]) for r in obs))
    pred_branch_names = {k: f"pred__{k[0]}__{k[1]}".replace("+","p").replace("-","m").replace("0","0").replace("K_S","KS").replace("*","") for k in pred_keys}
    for name in pred_branch_names.values():
        buf[name] = []

    # constants (normalizations can be fixed to 1.0)
    N_B0=N_Bp=N_Bs=1.0

    # Draw loop
    for _ in range(args.N):
        # CKM
        ckm = draw_ckm()
        # hadronic scale
        had = draw_hadronic_scaled(abs(ckm["lam_u_d"]))
        # SU(3)
        piF, kap, rhoF, sig = draw_su3_breaking()

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
            hadronic=had,
            pi=piF, kappa=kap, rho=rhoF, sigma=sig
        )

        pred = predict_all(theta, obs)
        ll = logL_gaussian(pred, obs)
        w = math.exp(ll)  # importance weight ∝ likelihood (priors are the draw density)

        buf["weight"].append(w)
        buf["logL"].append(ll)
        buf["gamma"].append(theta["ckm"]["gamma"])
        buf["beta"].append(theta["mixing"]["beta"])
        buf["beta_s"].append(theta["mixing"]["beta_s"])
        buf["lam_u_d_abs"].append(theta["ckm"]["lam_u_d_abs"])
        buf["lam_c_d_abs"].append(theta["ckm"]["lam_c_d_abs"])
        buf["lam_u_s_abs"].append(theta["ckm"]["lam_u_s_abs"])
        buf["lam_c_s_abs"].append(theta["ckm"]["lam_c_s_abs"])
        for k in ["T","C","E","A","P","PA"]:
            buf[k].append(theta["hadronic"][k])

        for k, name in pred_branch_names.items():
            val = pred.get(k, float("nan"))
            buf[name].append(val)

    # Convert to numpy arrays
    arrays = {k: np.asarray(v, dtype="f8") for k, v in buf.items()}

    # Write ROOT
    with uproot.recreate(args.out) as f:
        f["charmless_fit"] = arrays

    # Quick ESS report
    w = arrays["weight"]
    ess = (w.sum()**2) / (w*w).sum()
    print(f"Wrote: {args.out}")
    print(f"Draws: {len(w)},  ESS ≈ {ess:.1f}  (ESS/N ≈ {ess/len(w):.3f})")

if __name__ == "__main__":

    ##### DIAGNOSTIC
    from amplitudes.builder import build_amplitudes
    from priors.generate_point import draw_point
    
    theta = draw_point()
    A, Abar = build_amplitudes(theta)
    print(sorted(A.keys()))

    main()
