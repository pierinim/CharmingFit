import cmath

def q_over_p(meson, beta, beta_s):
    """Return q/p for neutral B0 or Bs mesons."""
    if meson == "B0":
        phi = beta
    elif meson == "Bs":
        phi = beta_s
    else:
        raise ValueError("q/p only defined for B0 or Bs")
    return cmath.exp(-2j * phi)

def lambdaf(qop, Abar, A):
    return qop * (Abar / A)

def S_C_from_lambda(lmbd):
    abs2 = (lmbd.real*lmbd.real + lmbd.imag*lmbd.imag)
    denom = 1 + abs2
    C = (1 - abs2) / denom
    S = (2 * lmbd.imag) / denom
    return S, C
