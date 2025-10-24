import math

def logL_gaussian(pred, obs_rows):
    """Compute log-likelihood as sum of Gaussian terms."""
    ll = 0.0
    for r in obs_rows:
        key = (r["mode"], r["type"])
        if key not in pred or not math.isfinite(pred[key]):
            continue
        delta = pred[key] - r["value"]
        sigma = r["sigma"]
        ll += -0.5 * (delta * delta) / (sigma * sigma)
    return ll
