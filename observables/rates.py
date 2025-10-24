def br_from_amp(A, norm=1.0):
    """Return |A|^2 * normalization."""
    return norm * (A.real*A.real + A.imag*A.imag)

def direct_acp(A, Abar):
    aa = (A.real*A.real + A.imag*A.imag)
    bb = (Abar.real*Abar.real + Abar.imag*Abar.imag)
    return (bb - aa) / (bb + aa)
