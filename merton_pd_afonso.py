import numpy as np
import pandas as pd
from scipy.special import ndtr

# helpers
def _norm_cdf(x):
    return ndtr(np.asarray(x, dtype=float))

def merton_dd(V, B_T, drift, sigma_V, T=1.0):
    """
    Distance-to-default for horizon T under GBM:
      DD = [ ln(V/B_T) + (drift - 0.5*sigma^2)*T ] / (sigma*sqrt(T))
    """
    V = np.asarray(V, dtype=float)
    B_T = np.asarray(B_T, dtype=float)
    drift = np.asarray(drift, dtype=float)
    sigma_V = np.asarray(sigma_V, dtype=float)

    # hard validity (avoid silently masking bad inputs)
    if np.any(~np.isfinite(V)) or np.any(~np.isfinite(B_T)) or np.any(~np.isfinite(drift)) or np.any(~np.isfinite(sigma_V)):
        return np.nan
    if np.any(V <= 0) or np.any(B_T <= 0) or np.any(sigma_V <= 0):
        return np.nan
    
    T = float(T)
    if T <= 0:
        raise ValueError("T must be > 0 (years).")    

    eps = 1e-14
    V = np.maximum(V, eps)
    B_T = np.maximum(B_T, eps)
    T = float(T)
    if T <= 0:
        raise ValueError("T must be > 0 (years).")

    sig_sqrtT = np.maximum(sigma_V * np.sqrt(T), eps)
    return (np.log(V / B_T) + (drift - 0.5 * sigma_V**2) * T) / sig_sqrtT


def estimate_mu_from_weekly_implied_assets(weekly_df: pd.DataFrame, sigmaV: float, ann_factor: float = 52.0) -> float:
    """
    Given weekly implied assets (weekly_df must have column 'dlogV'),
    estimate annual mu via:
      mu ≈ mean(dlogV)*ann_factor + 0.5*sigma^2
    """
    dlogV = weekly_df["dlogV"].to_numpy(dtype=float)
    dlogV = dlogV[np.isfinite(dlogV)]
    if dlogV.size < 2 or not np.isfinite(sigmaV) or sigmaV <= 0:
        return np.nan
    return float(np.mean(dlogV) * ann_factor + 0.5 * sigmaV**2)


# PDs
def merton_pd_rn_1y(V, B_1y, r, sigma_V):
    """
    1y-ahead risk-neutral PD (default by t+1):
      PD_Q = Φ(-DD_Q), with drift = r - q
    """
    dd_q = merton_dd(V, B_1y, drift=(r), sigma_V=sigma_V, T=1.0)
    return _norm_cdf(-dd_q)


def merton_pd_physical_1y(V, B_1y, mu, sigma_V):
    """
    1y-ahead physical PD under P:
      PD_P = Φ(-DD_P), with drift = mu
    """
    dd_p = merton_dd(V, B_1y, drift=mu, sigma_V=sigma_V, T=1.0)
    return _norm_cdf(-dd_p)