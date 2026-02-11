"""
nig_em_paper.py

Compact, paper-aligned EM initialization for the NIG structural credit-risk model
(Jovan & Ahčan): equity is priced as a call on assets under an Esscher-transformed
(NIG) risk-neutral measure, while asset log-returns are fitted under the physical
measure via MLE.

EM loop (per Jovan & Ahčan):
  1) E-step: infer the asset path A_t by inverting the NIG call price each day.
  2) M-step: fit (alpha, beta, delta, mu) by maximizing the NIG log-likelihood of daily
             asset log-returns r_t = log(A_t/A_{t-1}).
  3) Update the Esscher tilt theta_t for each day using that day's risk-free rate r_f(t).

Time scaling:
  - Daily step h = 1/250 (trading days).
  - Remaining maturity uses the paper's convention "T - n = 250 days":
      T0 = (n-1)*h + 1.0
    so the *last* observation is priced with 1-year maturity (PD horizon fixed at 1 year).

This file is intentionally short and self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import norminvgauss


# -----------------------------
# NIG utilities
# -----------------------------

def _gamma(alpha: float, beta: float) -> float:
    """gamma = sqrt(alpha^2 - beta^2)."""
    return float(np.sqrt(max(alpha * alpha - beta * beta, 0.0)))


def update_theta(params: Dict[str, float], r_f: float) -> float:
    """
    Scalar Esscher tilt update (closed form).
    theta depends on the (annualized) risk-free rate r_f.

    Feasibility checks follow the paper's existence conditions.
    """
    alpha = float(params.get("alpha", 0.0))
    beta = float(params.get("beta1", 0.0))
    delta = float(params.get("delta", 0.0))
    mu = float(params.get("beta0", 0.0))

    if delta <= 0.0 or alpha <= 0.0 or abs(beta) >= alpha:
        raise ValueError("Invalid NIG params: require delta>0, alpha>0, and |beta|<alpha.")
    if alpha < 0.5:
        raise ValueError("Esscher tilt existence requires alpha >= 0.5.")
    bound = delta * np.sqrt(max(2.0 * alpha - 1.0, 0.0))
    if abs(mu) > bound:
        raise ValueError("Esscher tilt existence requires |mu| <= delta*sqrt(2*alpha-1).")

    num = mu - float(r_f)
    inside = 4.0 * (alpha ** 2) * (delta ** 2) * (num ** 2) + (delta ** 2)
    theta = -beta - 0.5 - (num / (2.0 * delta)) * (np.sqrt(inside) - 1.0)
    return float(theta)


def update_theta_series(params: Dict[str, float], r_f_series: np.ndarray) -> np.ndarray:
    """Vector theta_t computed day-by-day from update_theta(..., r_f[t])."""
    r_f_series = np.asarray(r_f_series, dtype=float)
    out = np.empty_like(r_f_series, dtype=float)
    for t in range(r_f_series.size):
        out[t] = update_theta(params, float(r_f_series[t]))
    return out


# -----------------------------
# Equity pricing under NIG
# -----------------------------

def nig_call_price(A: float, L: float, T: float, r: float, params: Dict[str, float]) -> float:
    """
    Equity-as-call price under NIG with Esscher tilt theta:
        E = A * P_+(X > x0) - L*exp(-rT) * P_-(X > x0)
    where x0 = log(L/A), and the two tail probs use:
        beta_plus  = beta + theta + 1
        beta_minus = beta + theta
    with loc = mu*T and scale = delta*T.
    """
    if A <= 0.0 or L <= 0.0 or T <= 0.0:
        return np.nan

    alpha = float(params.get("alpha", 0.0))
    beta = float(params.get("beta1", 0.0))
    delta = float(params.get("delta", 0.0))
    mu = float(params.get("beta0", 0.0))
    theta = float(params.get("theta", 0.0))

    if delta <= 0.0 or alpha <= 0.0 or abs(beta) >= alpha:
        return np.nan

    x0 = np.log(L / A)
    loc = mu * T
    scale = delta * T

    beta_plus = beta + theta + 1.0
    beta_minus = beta + theta

    # Tail probs enter the call pricing identity
    cdf_plus = norminvgauss.cdf(x0, a=alpha, b=beta_plus, loc=loc, scale=scale)
    cdf_minus = norminvgauss.cdf(x0, a=alpha, b=beta_minus, loc=loc, scale=scale)

    tail_plus = 1.0 - cdf_plus
    tail_minus = 1.0 - cdf_minus

    return float(A * tail_plus - L * tail_minus)


def invert_nig_call_price(
    E_obs: float,
    L: float,
    T: float,
    r: float,
    params: Dict[str, float],
    A_min_factor: float = 1e-6,
    A_max_factor: float = 50.0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """
    Solve for A such that nig_call_price(A,...) == E_obs (bracketing + Brent).
    Falls back to E_obs + L if bracketing fails.
    """
    if E_obs <= 0.0 or L <= 0.0 or T <= 0.0:
        raise ValueError("E_obs, L, T must be positive")

    A_min = max(E_obs, A_min_factor * L)
    A_max = A_max_factor * (E_obs + L)

    def f(A: float) -> float:
        return nig_call_price(A, L=L, T=T, r=r, params=params) - E_obs

    f_min, f_max = f(A_min), f(A_max)
    if not (np.isfinite(f_min) and np.isfinite(f_max)):
        return float(E_obs + L)

    if f_min * f_max > 0.0:
        # Expand bracket
        success = False
        for i in range(1, 18):
            factor = 2.0 ** i
            A_max_ext = A_max * factor
            A_min_ext = max(A_min / factor, 1e-12)
            f_min_ext, f_max_ext = f(A_min_ext), f(A_max_ext)
            if np.isfinite(f_min_ext) and np.isfinite(f_max_ext) and (f_min_ext * f_max_ext <= 0.0):
                A_min, A_max = A_min_ext, A_max_ext
                success = True
                break
        if not success:
            return float(E_obs + L)

    try:
        return float(brentq(f, A_min, A_max, xtol=tol, maxiter=max_iter))
    except Exception:
        return float(E_obs + L)


# -----------------------------
# M-step: NIG likelihood (paper)
# -----------------------------

def _nig_negloglik_daily(
    r: np.ndarray, alpha: float, beta: float, delta_annual: float, mu_annual: float, h: float
) -> float:
    """
    Negative log-likelihood used in the paper:
      r_t ~ NIG(alpha, beta, loc=mu*h, scale=delta*h)
    where (delta, mu) are annualized and r_t are daily log-returns.
    """
    if alpha <= 0.0 or delta_annual <= 0.0 or abs(beta) >= alpha:
        return 1e50
    loc = mu_annual * h
    scale = delta_annual * h
    ll = norminvgauss.logpdf(r, a=alpha, b=beta, loc=loc, scale=scale)
    if not np.all(np.isfinite(ll)):
        return 1e50
    return float(-np.sum(ll))


def _fit_nig_mle_daily(
    r: np.ndarray,
    x0: Tuple[float, float, float, float],
    h: float,
) -> Tuple[float, float, float, float]:
    """
    MLE for (alpha, beta, delta_annual, mu_annual) using a stable reparameterization:
      beta = b
      alpha = |b| + 0.500001 + exp(a_raw)   (enforces alpha>|beta| and alpha>=0.5)
      delta = exp(log_delta)
      mu = mu_raw
    """
    alpha0, beta0, delta0, mu0 = map(float, x0)
    beta_init = beta0
    alpha_gap = max(alpha0 - abs(beta_init) - 0.500001, 1e-6)
    a_raw0 = float(np.log(alpha_gap))
    log_delta0 = float(np.log(max(delta0, 1e-12)))
    mu_raw0 = mu0

    def unpack(u: np.ndarray) -> Tuple[float, float, float, float]:
        a_raw, b_raw, log_delta, mu_raw = map(float, u)
        beta = b_raw
        alpha = abs(beta) + 0.500001 + np.exp(a_raw)
        delta = np.exp(log_delta)
        mu = mu_raw
        return float(alpha), float(beta), float(delta), float(mu)

    def obj(u: np.ndarray) -> float:
        alpha, beta, delta, mu = unpack(u)
        return _nig_negloglik_daily(r, alpha, beta, delta, mu, h=h)

    u0 = np.array([a_raw0, beta_init, log_delta0, mu_raw0], dtype=float)
    res = minimize(obj, u0, method="L-BFGS-B")

    alpha_hat, beta_hat, delta_hat, mu_hat = unpack(res.x)
    return alpha_hat, beta_hat, delta_hat, mu_hat


# -----------------------------
# EM initialization (paper)
# -----------------------------

@dataclass(frozen=True)
class EMResult:
    params: Dict[str, float]
    asset_path: np.ndarray
    theta_series: np.ndarray
    converged: bool
    n_iter: int
    diff_last: np.ndarray
    theta_trace: np.ndarray  # (n_iter, n_days)


def em_init_nig_params(
    equity: np.ndarray,
    liabilities_L: float,
    rf: np.ndarray,
    start_params: Dict[str, float],
    *,
    trading_days: int = 250,
    pd_horizon_years: float = 1.0,   # fixed at 1 year by default
    min_iter: int = 3,
    max_iter: int = 10,
    tol: float = 1e-3,
) -> EMResult:
    """
    Paper-aligned EM initializer.

    Training window length n is inferred from equity/rf; maturity schedule is set so
    that the last observation is priced with 1-year maturity (pd_horizon_years).
    """
    E = np.asarray(equity, dtype=float)
    if E.ndim != 1 or E.size < 3:
        raise ValueError("equity must be a 1D array with length >= 3")
    if np.any(E <= 0.0):
        raise ValueError("All equity observations must be positive")

    rf = np.asarray(rf, dtype=float)
    if rf.shape != E.shape:
        raise ValueError("rf must have the same shape as equity")

    n = int(E.size)
    h = 1.0 / float(trading_days)

    # Paper convention: ensure last day has 1Y maturity
    T0 = (n - 1) * h + float(pd_horizon_years)

    alpha = float(start_params["alpha"])
    beta1 = float(start_params["beta1"])
    delta = float(start_params["delta"])
    beta0 = float(start_params["beta0"])

    asset_path = np.full(n, np.nan, dtype=float)
    theta_trace = np.full((max_iter, n), np.nan, dtype=float)

    converged = False
    diff_last = np.full(4, np.nan, dtype=float)
    n_done = 0

    for it in range(max_iter):
        # θ_t depends on r_f(t); compute a full vector per EM iteration
        theta_series = update_theta_series({"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0}, rf)
        theta_trace[it, :] = theta_series

        # E-step: invert equity call price each day to infer A_t
        for t in range(n):
            T_rem = T0 - t * h  # always >= pd_horizon_years > 0
            params_t = {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0, "theta": float(theta_series[t])}
            asset_path[t] = invert_nig_call_price(float(E[t]), liabilities_L, float(T_rem), float(rf[t]), params_t)

        rA = np.diff(np.log(asset_path))

        # M-step: maximize NIG likelihood for daily returns
        a_new, b_new, d_new, mu_new = _fit_nig_mle_daily(rA, (alpha, beta1, delta, beta0), h=h)

        theta_new = update_theta_series({"alpha": a_new, "beta1": b_new, "delta": d_new, "beta0": mu_new}, rf)

        alpha_floor = max(
            np.max(np.abs(b_new + theta_new)),
            np.max(np.abs(b_new + theta_new + 1.0))
        ) + 1e-6

        if a_new < alpha_floor:
            a_new = alpha_floor

        diff = np.array([abs(a_new - alpha), abs(b_new - beta1), abs(d_new - delta), abs(mu_new - beta0)], dtype=float)
        diff_last = diff
        n_done = it + 1

        if n_done >= min_iter and np.all(diff < tol):
            converged = True
            alpha, beta1, delta, beta0 = a_new, b_new, d_new, mu_new
            break

        alpha, beta1, delta, beta0 = a_new, b_new, d_new, mu_new

    # Final theta series under final parameters
    theta_series = update_theta_series({"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0}, rf)

    return EMResult(
        params={"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0},
        asset_path=asset_path,
        theta_series=theta_series,
        converged=converged,
        n_iter=n_done,
        diff_last=diff_last,
        theta_trace=theta_trace[:n_done, :],
    )
