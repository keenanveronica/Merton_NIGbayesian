import numpy as np
from scipy.special import ndtr

def _norm_cdf(x):
    return ndtr(np.asarray(x, dtype=float))

def merton_dd(V, B_T, drift, sigma_V, T=1.0):
    """
    Distance-to-default for horizon T under GBM:
      DD = [ ln(V/B_T) + (drift - 0.5*sigma^2)*T ] / (sigma*sqrt(T))
    where:
      - V      = asset value at time t
      - B_T    = default barrier / debt due at horizon t+T (face/value at T)
      - drift  = mu (physical) OR (r - q) (risk-neutral with payout q)
      - sigma_V is annualized asset vol
      - T in years
    """
    V = np.asarray(V, dtype=float)
    B_T = np.asarray(B_T, dtype=float)
    drift = np.asarray(drift, dtype=float)
    sigma_V = np.asarray(sigma_V, dtype=float)

    eps = 1e-14
    V = np.maximum(V, eps)
    B_T = np.maximum(B_T, eps)
    T = float(T)
    if T <= 0:
        raise ValueError("T must be > 0 (years).")

    sig_sqrtT = np.maximum(sigma_V * np.sqrt(T), eps)
    return (np.log(V / B_T) + (drift - 0.5 * sigma_V**2) * T) / sig_sqrtT


def merton_pd_rn_1y(V, B_1y, r, sigma_V, *, q=0.0):
    """
    1y-ahead risk-neutral PD (default by t+1):
      PD_Q = Φ(-DD_Q), with drift = r - q
    Inputs:
      V      : asset value today
      B_1y   : liability/barrier value at t+1y (face/proxy at horizon)
      r      : annual risk-free rate (continuous compounding recommended)
      sigma_V: annual asset volatility
      q      : payout/dividend yield on assets (default 0)
    """
    dd_q = merton_dd(V, B_1y, drift=(r - q), sigma_V=sigma_V, T=1.0)
    return _norm_cdf(-dd_q)
