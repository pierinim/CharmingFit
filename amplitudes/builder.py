import cmath

def exp_i(phi): 
    return cmath.cos(phi) + 1j*cmath.sin(phi)

def polar(mag, phase): 
    return mag * exp_i(phase)

def build_amplitudes(theta):
    """Return dictionaries of A and Abar amplitudes for all decay modes."""
    ckm = theta["ckm"]
    had = theta["hadronic"]
    piF, kap, rhoF, sig = theta["pi"], theta["kappa"], theta["rho"], theta["sigma"]
    gamma = ckm["gamma"]

    # CKM factors
    lu_d = ckm["lam_u_d_abs"] * exp_i(-gamma)
    lc_d = ckm["lam_c_d_abs"]
    lu_s = ckm["lam_u_s_abs"] * exp_i(-gamma)
    lc_s = ckm["lam_c_s_abs"]

    # Hadronic amplitudes
    T, C, E, A, P, PA = (had[k] for k in ("T","C","E","A","P","PA"))

    # SU(3)-breaking helpers
    def pfx(fset, key): return polar(fset[key+"_abs"], fset[key+"_phase"])
    piT,piC,piE,piA,piP,piPA = [pfx(piF,k) for k in ("T","C","E","A","P","PA")]
    kP,kA,kE,kPA = [pfx(kap,k) for k in ("P","A","E","PA")]
    rP,rT,rC = [pfx(rhoF,k) for k in ("P","T","C")]
    sT,sE,sP,sPA = [pfx(sig,k) for k in ("T","E","P","PA")]

    Aamp = {}

    # |ΔS|=1 (B+ and B0 → Kπ)
    Aamp["B+_K0barpi+"] = (lc_s * P) + (lu_s * A)
    Aamp["B0_K+pi-"]    = -(lc_s * P) - (lu_s * T)
    Aamp["B+_K+pi0"]    = (-(lc_s * P) - (lu_s * (T + C + A))) / cmath.sqrt(2)
    Aamp["B0_K0pi0"]    = ( (lc_s * P) - (lu_s * C) ) / cmath.sqrt(2)

    # ΔS=0 (B → ππ)
    Aamp["B0_pi+pi-"] = - ( lu_d*(T*piT + E*piE) + lc_d*(P*piP + PA*piPA) )
    Aamp["B+_pi+pi0"] = - ( lu_d*(T*piT + C*piC + A*piA) ) / cmath.sqrt(2)
    Aamp["B0_pi0pi0"] = ( -lu_d*(C*piC) + lc_d*(P*piP) + lu_d*(E*piE) + lc_d*(PA*piPA) ) / cmath.sqrt(2)

    # ΔS=0 (B → KK)
    Aamp["B+_K0K+"]   = (lc_d*(P*kP)) + (lu_d*(A*kA))
    Aamp["B0_K0K0bar"]= (lc_d*(P*kP + PA*kPA))
    Aamp["B0_K+K-"]   = -(lu_d*(E*kE) + lc_d*(PA*kPA))

    # Bs → Kπ
    Aamp["Bs_K-pi+"]  = -(lc_d*(P*rP) + lu_d*(T*rT))
    Aamp["Bs_K0pi0"]  = (lc_d*(P*rP) - lu_d*(C*rC)) / cmath.sqrt(2)

    # Bs → KK
    Aamp["Bs_K+K-"]    = -(lu_s*(T*sT + E*sE) + lc_s*(P*sP + PA*sPA))
    Aamp["Bs_K0K0bar"] =  (lc_s*(P*sP + PA*sPA))

    # CP-conjugate amplitudes (placeholder: conjugate)
    Abar = {mode: A.conjugate() for mode, A in Aamp.items()}

    return Aamp, Abar
