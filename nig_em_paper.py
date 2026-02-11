from typing import Dict, Tuple, Optional

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import norminvgauss


# NIG utilities

def _gamma(alpha: float, beta: float) -> float:
    """gamma = sqrt(alpha^2 - beta^2)."""
    return float(np.sqrt(max(alpha * alpha - beta * beta, 0.0)))


def update_theta(params: Dict[str, float], r_f: float) -> float:
    """
    Scalar Esscher update (closed form).
    theta depends on the daily risk-free rate r_f.

    Feasibility checks follow the paper's existence conditions.
    """
    alpha = float(params.get("alpha", 0.0))
    beta = float(params.get("beta1", 0.0))
    delta = float(params.get("delta", 0.0))
    mu = float(params.get("beta0", 0.0))

    if delta <= 0.0 or alpha <= 0.0 or abs(beta) >= alpha:
        raise ValueError("Invalid NIG params: require delta>0, alpha>0, and |beta|<alpha.")
    if alpha < 0.5:
        raise ValueError("Theta existence requires alpha >= 0.5.")
    bound = delta * np.sqrt(max(2.0 * alpha - 1.0, 0.0))
    if abs(mu) > bound:
        raise ValueError("Theta existence requires |mu| <= delta*sqrt(2*alpha-1).")

    num = mu - float(r_f)
    inside = (4*(alpha ** 2)*(delta ** 2))/((num ** 2) + (delta ** 2))
    theta = -beta - 0.5 - (num/(2*delta))*(np.sqrt(inside) - 1.0)
    return float(theta)


def update_theta_series(params: Dict[str, float], r_f_series: np.ndarray) -> np.ndarray:
    """Vector theta_t computed day-by-day from update_theta(..., r_f[t])."""
    r_f_series = np.asarray(r_f_series, dtype=float)
    out = np.empty_like(r_f_series, dtype=float)
    for t in range(r_f_series.size):
        out[t] = update_theta(params, float(r_f_series[t]))
    return out


# Equity pricing under NIG - we can turn this into a series function

def nig_call_price(A: float, L_face: float, L_disc: float, T: float, params: Dict[str, float]) -> float:
    if A <= 0.0 or L_face <= 0.0 or L_disc <= 0.0 or T <= 0.0:
        return np.nan

    alpha = float(params.get("alpha", 0.0))
    beta  = float(params.get("beta1", 0.0))
    delta = float(params.get("delta", 0.0))
    mu    = float(params.get("beta0", 0.0))
    theta = float(params.get("theta", 0.0))

    if delta <= 0.0 or alpha <= 0.0 or abs(beta) >= alpha:
        return np.nan

    # threshold must use face value at maturity (paper)
    x0 = np.log(L_face / A)

    loc = mu * T
    scale = delta * T

    beta_plus  = beta + theta + 1.0
    beta_minus = beta + theta

    cdf_plus  = norminvgauss.cdf(x0, a=alpha, b=beta_plus,  loc=loc, scale=scale)
    cdf_minus = norminvgauss.cdf(x0, a=alpha, b=beta_minus, loc=loc, scale=scale)

    tail_plus  = 1.0 - cdf_plus
    tail_minus = 1.0 - cdf_minus

    return float(A * tail_plus - L_disc * tail_minus)



# CAN BE CONVERTED TO A "ROOT" FINDER ALSO - check if we should do this in series
def invert_nig_call_price(
    E_obs: float,
    L: float,
    L_face: float,
    T: float,
    params: Dict[str, float],
    A_min_factor: float = 1e-6,
    A_max_factor: float = 50,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """
    Solve for A such that nig_call_price(A,...) == E_obs (bracketing + Brent).
    Falls back to A = E_obs + L (discounted or not?) if bracketing fails.
    """
    if E_obs <= 0.0 or L <= 0.0 or T <= 0.0 or L_face <= 0.0:
        raise ValueError("E_obs, L, T, L_face must be positive")

    A_min = max(E_obs, A_min_factor * L_face)
    A_max = A_max_factor * (E_obs + L_face)

    def f(A: float) -> float:
        return nig_call_price(A, L_face=L_face, L_disc=L, T=T, params=params) - E_obs

    f_min, f_max = f(A_min), f(A_max)
    if not (np.isfinite(f_min) and np.isfinite(f_max)):
        return float(E_obs + L_face)

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
            return float(E_obs + L_face)

    try:
        return float(brentq(f, A_min, A_max, xtol=tol, maxiter=max_iter))
    except Exception:
        return float(E_obs + L_face)
#NEED TO MAKE PRINTS TO CHECK IF THIS IS BEING DONE CORRECTLY


def get_asset_path(
        params: Dict[str, float],
        theta_series: np.ndarray,
        dates: np.ndarray,
        E_series: np.ndarray,
        L_series: np.ndarray, #NEED TO BE DISCOUNTED
        start_date: None,
        end_date: None,
) -> np.ndarray:

    dates = np.asarray(dates)
    E_series = np.asarray(E_series, dtype=float)
    L_series = np.asarray(L_series, dtype=float)
    theta_series = np.asarray(theta_series, dtype=float)

    if not (len(dates) == len(E_series) == len(L_series) == len(theta_series)):
        raise ValueError("dates, E_series, L_series, theta_series must have the same length")

    # date slice (inclusive)
    mask = np.ones(len(dates), dtype=bool)
    if start_date is not None:
        mask &= (dates >= start_date)
    if end_date is not None:
        mask &= (dates <= end_date)

    E = E_series[mask]
    L = L_series[mask]
    th = theta_series[mask]

    n = len(E)
    h = 1/250
    T = 250 + n-1

    end_idx_full = np.where(mask)[0][-1]     # last index in full arrays that is inside training window
    face_idx_full = end_idx_full + 250       # 1 trading year ahead

    if face_idx_full >= len(L_series):
        raise ValueError("Need liabilities 1y after end_date (end training window earlier).")

    L_face = float(L_series[face_idx_full])
    A_path = np.empty(n, dtype=float)

    for t in range(n):
        T_rem = (T - t) * h  # convert remaining days to years

        params_t = dict(params)
        params_t["theta"] = float(th[t])

        A_path[t] = invert_nig_call_price(
            E_obs=float(E[t]),
            L=float(L[t]),
            L_face=L_face, # WE MIGHT NEED TO CHANGE THIS
            T=float(T_rem),
            params=params_t,
        )

    return A_path



# M-step: NIG likelihood (paper)

def _nig_negloglik_daily(r, alpha, beta, delta_annual, mu_annual):

    h = 1/250

    if alpha <= 0.0 or delta_annual <= 0.0 or abs(beta) >= alpha:
        return 1e50

    r = np.asarray(r, dtype=float)
    if r.ndim != 1:
        r = r.reshape(-1)
    if not np.all(np.isfinite(r)):
        return 1e50

    loc = float(mu_annual) * float(h)
    scale = float(delta_annual) * float(h)
    scale = max(scale, 1e-12)

    ll = norminvgauss.logpdf(r, a=float(alpha), b=float(beta), loc=loc, scale=scale)
    if not np.all(np.isfinite(ll)):
        return 1e50

    return float(-np.sum(ll))



def _fit_nig_mle_daily(
    r: np.ndarray,
    x0: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    MLE for (alpha, beta, delta_annual, mu_annual) using a stable reparameterization:
      beta = b
      alpha = |b| + 0.500001 + exp(a_raw)   (enforces alpha>|beta| and alpha>=0.5)
      delta = exp(log_delta)
      mu = mu_raw
    """
    h = 1/250,
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
        return _nig_negloglik_daily(r, alpha, beta, delta, mu)

    u0 = np.array([a_raw0, beta_init, log_delta0, mu_raw0], dtype=float)
    res = minimize(obj, u0, method="L-BFGS-B")

    alpha_hat, beta_hat, delta_hat, mu_hat = unpack(res.x)
    return alpha_hat, beta_hat, delta_hat, mu_hat


# EM initialization (paper)

def EM_algo(
    E_series: np.ndarray,
    L_series: np.ndarray,     # discounted liability series (your current setup)
    rf_series: np.ndarray,    # daily risk-free series (needed for theta)
    dates: np.ndarray,
    start_params: Dict[str, float],
    start_date=None,
    end_date=None,
    max_iter: int = 10,
    min_iter: int = 3,
    tol: float = 1e-3,
) -> Dict[str, object]:
    """
    EM loop (Jovan & Ahcan style):
      E-step: infer A_t by inverting the NIG call each day
      M-step: MLE fit NIG params to daily log-returns of A_t
      theta:  recompute daily theta_t from updated params + rf(t)

    Returns a dict with final params + traces.
    """

    E_series = np.asarray(E_series, dtype=float)
    L_series = np.asarray(L_series, dtype=float)
    rf_series = np.asarray(rf_series, dtype=float)
    dates = np.asarray(dates)

    if not (len(E_series) == len(L_series) == len(rf_series) == len(dates)):
        raise ValueError("E_series, L_series, rf_series, dates must have the same length")

    # current params
    alpha = float(start_params["alpha"])
    beta1 = float(start_params["beta1"])
    delta = float(start_params["delta"])
    beta0 = float(start_params["beta0"])

    theta_trace = []
    param_trace = []
    diff_trace = []
    converged = False

    for it in range(max_iter):
        # theta vector for this iteration (full series; get_asset_path slices internally)
        theta_series = update_theta_series(
            {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0},
            rf_series
        )
        theta_trace.append(theta_series)

        # E-step: infer asset path on the training window
        A_path = get_asset_path(
            params={"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0},
            theta_series=theta_series,
            dates=dates,
            E_series=E_series,
            L_series=L_series,
            start_date=start_date,
            end_date=end_date,
        )

        rA = np.diff(np.log(A_path))
        if not np.all(np.isfinite(rA)) or rA.size < 2:
            raise ValueError("Non-finite or too-short asset return series produced in E-step")

        # M-step: MLE update (annual params; likelihood uses daily scaling internally)
        a_new, b_new, d_new, mu_new = _fit_nig_mle_daily(
            rA, (alpha, beta1, delta, beta0)
        )

        # Recompute theta under proposed params to enforce pricing feasibility:
        theta_new = update_theta_series(
            {"alpha": a_new, "beta1": b_new, "delta": d_new, "beta0": mu_new},
            rf_series
        )

        # Ensure alpha is large enough so |beta+theta| < alpha and |beta+theta+1| < alpha for all days
        alpha_floor = max(
            np.max(np.abs(b_new + theta_new)),
            np.max(np.abs(b_new + theta_new + 1.0))
        ) + 1e-6
        if a_new < alpha_floor:
            a_new = float(alpha_floor)

        diff = np.array(
            [abs(a_new - alpha), abs(b_new - beta1), abs(d_new - delta), abs(mu_new - beta0)],
            dtype=float
        )

        param_trace.append((alpha, beta1, delta, beta0))
        diff_trace.append(diff)

        alpha, beta1, delta, beta0 = float(a_new), float(b_new), float(d_new), float(mu_new)

        if (it + 1) >= min_iter and np.all(diff < tol):
            converged = True
            break

    # final theta (full series)
    theta_final = update_theta_series(
        {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0},
        rf_series
    )

    return {
        "params": {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0},
        "converged": converged,
        "n_iter": it + 1,
        "diff_last": diff_trace[-1] if diff_trace else None,
        "param_trace": param_trace,
        "diff_trace": diff_trace,
        "theta_final": theta_final,
        "theta_trace": theta_trace,
    }