import numpy as np

def ckm_from_wolfenstein(A, lam, rhobar, etabar):
    """Compute CKM matrix elements and derived phases from Wolfenstein parameters."""
    rho = rhobar / (1 - lam**2/2)
    eta = etabar / (1 - lam**2/2)

    Vud = 1 - lam**2/2
    Vus = lam
    Vub = A * lam**3 * (rho - 1j * eta)
    Vcb = A * lam**2
    Vcs = 1 - lam**2/2
    Vcd = -lam + 0.5 * A**2 * lam**5 * (1 - 2*(rho + 1j*eta))
    Vtd = A * lam**3 * (1 - rho - 1j * eta)
    Vts = -A * lam**2 + 0.5 * A * lam**4 * (1 - 2*(rho + 1j*eta))
    Vtb = 1 + 0j

    lam_u_d = Vub * np.conj(Vud)
    lam_c_d = Vcb * np.conj(Vcd)
    lam_u_s = Vub * np.conj(Vus)
    lam_c_s = Vcb * np.conj(Vcs)

    gamma = np.angle(-Vud * np.conj(Vub) / (Vcd * np.conj(Vcb)))
    beta = np.angle(-Vcd * np.conj(Vcb) / (Vtd * np.conj(Vtb)))
    beta_s = np.angle(-Vts * np.conj(Vtb) / (Vcs * np.conj(Vcb)))

    return dict(
        lam_u_d=lam_u_d, lam_c_d=lam_c_d,
        lam_u_s=lam_u_s, lam_c_s=lam_c_s,
        gamma=gamma, beta=beta, beta_s=beta_s
    )
