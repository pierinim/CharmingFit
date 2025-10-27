# amplitudes/builder.py
import cmath

def exp_i(phi): 
    return cmath.cos(phi) + 1j*cmath.sin(phi)

def polar(mag, phase): 
    return mag * exp_i(phase)

def build_amplitudes(theta):
    """
    Build A and Abar for all modes, using complex hadronic topologies (T,C,E,A,P,PA)
    and complex SU(3)-breaking multipliers. CP conjugation flips only weak phases
    (γ → -γ), while keeping strong phases (hadronic and SU(3) factors) unchanged.
    """
    ckm = theta["ckm"]
    had = theta["hadronic"]   # COMPLEX: each of T,C,E,A,P,PA is complex already
    piF, kap, rhoF, sig = theta["pi"], theta["kappa"], theta["rho"], theta["sigma"]

    gamma = ckm["gamma"]

    # CKM building blocks: magnitudes are positive reals we pass in as *_abs,
    # weak phase e^{∓ iγ} carried only by λ_u. λ_c taken as +|λ_c|.
    lu_d     = ckm["lam_u_d_abs"] * exp_i(-gamma)
    lu_s     = ckm["lam_u_s_abs"] * exp_i(-gamma)
    lu_d_bar = ckm["lam_u_d_abs"] * exp_i(+gamma)
    lu_s_bar = ckm["lam_u_s_abs"] * exp_i(+gamma)
    lc_d = ckm["lam_c_d_abs"] + 0j
    lc_s = ckm["lam_c_s_abs"] + 0j

    # Hadronic topologies (already complex)
    T, C, E, A, P, PA = (had[k] for k in ("T","C","E","A","P","PA"))

    # SU(3)-breaking helpers (complex multipliers)
    def pfx(fset, key): 
        return polar(fset[key+"_abs"], fset[key+"_phase"])

    # B -> ππ (pi), B -> KK ΔS=0 (kappa), Bs -> Kπ (rho), Bs -> KK (sigma)
    piT,piC,piE,piA,piP,piPA = [pfx(piF,k) for k in ("T","C","E","A","P","PA")]
    kP,kA,kE,kPA             = [pfx(kap,k) for k in ("P","A","E","PA")]
    rP,rT,rC                 = [pfx(rhoF,k) for k in ("P","T","C")]
    sT,sE,sP,sPA             = [pfx(sig,k) for k in ("T","E","P","PA")]

    Aamp = {}

    # ------------------------------
    # |ΔS| = 1  (B → Kπ), reference (no extra SU(3) factors here)
    # ------------------------------
    Aamp["B+_K0barpi+"] = (lc_s * P) + (lu_s * A)
    Aamp["B0_K+pi-"]    = -(lc_s * P) - (lu_s * T)
    Aamp["B+_K+pi0"]    = (-(lc_s * P) - (lu_s * (T + C + A))) / cmath.sqrt(2)
    Aamp["B0_K0pi0"]    = ( (lc_s * P) - (lu_s * C) ) / cmath.sqrt(2)

    # ------------------------------
    # ΔS = 0  (B → ππ)  — SU(3): pi-factors
    # ------------------------------
    Aamp["B0_pi+pi-"] = - ( lu_d*(T*piT + E*piE) + lc_d*(P*piP + PA*piPA) )
    Aamp["B+_pi+pi0"] = - ( lu_d*(T*piT + C*piC + A*piA) ) / cmath.sqrt(2)
    Aamp["B0_pi0pi0"] = ( -lu_d*(C*piC) + lc_d*(P*piP) + lu_d*(E*piE) + lc_d*(PA*piPA) ) / cmath.sqrt(2)

    # ------------------------------
    # ΔS = 0  (B → KK)  — SU(3): kappa-factors
    # ------------------------------
    Aamp["B+_K0K+"]    = (lc_d*(P*kP)) + (lu_d*(A*kA))
    Aamp["B0_K0K0bar"] = (lc_d*(P*kP + PA*kPA))
    Aamp["B0_K+K-"]    = -(lu_d*(E*kE) + lc_d*(PA*kPA))

    # ------------------------------
    # Bs → Kπ  — SU(3): rho-factors
    # ------------------------------
    Aamp["Bs_K-pi+"]  = -(lc_d*(P*rP) + lu_d*(T*rT))
    Aamp["Bs_K0pi0"]  = ( lc_d*(P*rP) - lu_d*(C*rC) ) / cmath.sqrt(2)

    # ------------------------------
    # Bs → KK  — SU(3): sigma-factors
    # ------------------------------
    Aamp["Bs_K+K-"]    = -( lu_s*(T*sT + E*sE) + lc_s*(P*sP + PA*sPA) )
    Aamp["Bs_K0K0bar"] =  ( lc_s*(P*sP + PA*sPA) )

    # ------------------------------------------------------------
    # CP-conjugate amplitudes: flip weak phases ONLY (γ → -γ),
    # reuse the *same* complex hadronic topologies and SU(3) factors.
    # ------------------------------------------------------------
    Abar = {}
    # |ΔS| = 1 (B → Kπ)
    Abar["B+_K0barpi+"] = (lc_s * P) + (lu_s_bar * A)
    Abar["B0_K+pi-"]    = -(lc_s * P) - (lu_s_bar * T)
    Abar["B+_K+pi0"]    = (-(lc_s * P) - (lu_s_bar * (T + C + A))) / cmath.sqrt(2)
    Abar["B0_K0pi0"]    = ( (lc_s * P) - (lu_s_bar * C) ) / cmath.sqrt(2)

    # ΔS = 0 (B → ππ)
    Abar["B0_pi+pi-"] = - ( lu_d_bar*(T*piT + E*piE) + lc_d*(P*piP + PA*piPA) )
    Abar["B+_pi+pi0"] = - ( lu_d_bar*(T*piT + C*piC + A*piA) ) / cmath.sqrt(2)
    Abar["B0_pi0pi0"] = ( -lu_d_bar*(C*piC) + lc_d*(P*piP) + lu_d_bar*(E*piE) + lc_d*(PA*piPA) ) / cmath.sqrt(2)

    # ΔS = 0 (B → KK)
    Abar["B+_K0K+"]    = (lc_d*(P*kP)) + (lu_d_bar*(A*kA))
    Abar["B0_K0K0bar"] = (lc_d*(P*kP + PA*kPA))
    Abar["B0_K+K-"]    = -(lu_d_bar*(E*kE) + lc_d*(PA*kPA))

    # Bs → Kπ
    Abar["Bs_K-pi+"]  = -(lc_d*(P*rP) + lu_d_bar*(T*rT))
    Abar["Bs_K0pi0"]  = ( lc_d*(P*rP) - lu_d_bar*(C*rC) ) / cmath.sqrt(2)

    # Bs → KK
    Abar["Bs_K+K-"]    = -( lu_s_bar*(T*sT + E*sE) + lc_s*(P*sP + PA*sPA) )
    Abar["Bs_K0K0bar"] =  ( lc_s*(P*sP + PA*sPA) )

    return Aamp, Abar
