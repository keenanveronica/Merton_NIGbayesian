from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import norminvgauss


# NIG utilities
def update_theta(params: Dict[str, float], r_f: float) -> float:

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
    theta = -beta - 0.5 - (num/(2*delta))*np.sqrt(inside - 1.0)
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
    beta = float(params.get("beta1", 0.0))
    delta = float(params.get("delta", 0.0))
    mu = float(params.get("beta0", 0.0))
    theta = float(params.get("theta", 0.0))

    if delta <= 0.0 or alpha <= 0.0 or abs(beta) >= alpha:
        return np.nan

    # threshold must use face value at maturity (paper)
    x0 = np.log(L_face / A)

    loc = mu * T
    scale = delta * T

    beta_plus = beta + theta + 1.0
    beta_minus = beta + theta

    cdf_plus = norminvgauss.cdf(x0, a=alpha, b=beta_plus,  loc=loc, scale=scale)
    cdf_minus = norminvgauss.cdf(x0, a=alpha, b=beta_minus, loc=loc, scale=scale)

    tail_plus = 1.0 - cdf_plus
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


# NEED TO MAKE PRINTS TO CHECK IF THIS IS BEING DONE CORRECTLY
def get_asset_path(
    params: Dict[str, float],
    theta_series: np.ndarray,
    rf_series: np.ndarray,        # NEW: needed to compute Le^{-r tau}
    dates: np.ndarray,
    E_series: np.ndarray,
    L_face_series: np.ndarray,    # NEW meaning: FACE liabilities (strike proxy), NOT discounted
    start_date: None,
    end_date: None,
    *,
    daycount: int = 250,
    discounting: str = "continuous",  # "continuous" matches exp(-r*tau) in Eq (26)
) -> np.ndarray:

    dates = np.asarray(dates)
    E_series = np.asarray(E_series, dtype=float)
    L_face_series = np.asarray(L_face_series, dtype=float)
    theta_series = np.asarray(theta_series, dtype=float)
    rf_series = np.asarray(rf_series, dtype=float)

    if not (len(dates) == len(E_series) == len(L_face_series) == len(theta_series) == len(rf_series)):
        raise ValueError("dates, E_series, L_face_series, theta_series, rf_series must have the same length")

    # date slice (inclusive)
    mask = np.ones(len(dates), dtype=bool)
    if start_date is not None:
        mask &= (dates >= start_date)
    if end_date is not None:
        mask &= (dates <= end_date)

    E = E_series[mask]
    r = rf_series[mask]
    th = theta_series[mask]

    n = len(E)
    if n < 3:
        raise ValueError("Training window too short")

    h = 1.0 / daycount
    T_days = (daycount + (n - 1))  # same structure as your current code: end-of-window has tau=1y

    end_idx_full = np.where(mask)[0][-1]     # last index in full arrays inside training window
    L_face = float(L_face_series[end_idx_full])  # PAPER STYLE: constant L at evaluation date :contentReference[oaicite:4]{index=4}

    if not (np.isfinite(L_face) and L_face > 0):
        raise ValueError("Invalid L_face at end of window")

    A_path = np.empty(n, dtype=float)

    for t in range(n):
        tau = (T_days - t) * h  # in years

        # discounted strike in Eq (26): L * exp(-r * tau) :contentReference[oaicite:5]{index=5}
        r_t = float(r[t])
        if discounting == "continuous":
            L_disc = L_face * np.exp(-r_t * tau)
        elif discounting == "simple":
            L_disc = L_face / (1.0 + r_t * tau)
        else:
            raise ValueError("discounting must be 'continuous' or 'simple'")

        params_t = dict(params)
        params_t["theta"] = float(th[t])

        A_path[t] = invert_nig_call_price(
            E_obs=float(E[t]),
            L=float(L_disc),       # this is Le^{-r tau}
            L_face=float(L_face),  # this is face L in ln(L/A)
            T=float(tau),
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
    L_series: np.ndarray,     # discounted liability series
    rf_series: np.ndarray,    # daily risk-free series (needed for theta)
    dates: np.ndarray,
    start_params: Dict[str, float],
    start_date=None,
    end_date=None,
    max_iter: int = 10,
    min_iter: int = 3,
    tol: float = 1e-3,
) -> Dict[str, object]:

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

    converged = False
    diff_last = None

    for it in range(max_iter):
        # theta vector for this iteration (full series; get_asset_path slices internally)
        theta_full = update_theta_series(
            {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0},
            rf_series
        )

        # E-step: infer asset path on the training window
        A_path = get_asset_path(
            params={"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0},
            theta_series=theta_full,
            rf_series=rf_series,
            dates=dates,
            E_series=E_series,
            L_face_series=L_series,
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
        theta_new_full = update_theta_series(
            {"alpha": a_new, "beta1": b_new, "delta": d_new, "beta0": mu_new},
            rf_series
        )

        # Ensure alpha is large enough so |beta+theta| < alpha and |beta+theta+1| < alpha for all days
        alpha_floor = max(
            np.max(np.abs(b_new + theta_new_full)),
            np.max(np.abs(b_new + theta_new_full + 1.0))
        ) + 1e-6
        if a_new < alpha_floor:
            a_new = float(alpha_floor)

        diff_last = np.array(
            [abs(a_new - alpha), abs(b_new - beta1), abs(d_new - delta), abs(mu_new - beta0)],
            dtype=float
        )

        alpha, beta1, delta, beta0 = float(a_new), float(b_new), float(d_new), float(mu_new)

        if (it + 1) >= min_iter and np.all(diff_last < tol):
            converged = True
            break

    # --- build training-window outputs ---
    mask = np.ones(len(dates), dtype=bool)
    if start_date is not None:
        mask &= (dates >= start_date)
    if end_date is not None:
        mask &= (dates <= end_date)

    idx = np.where(mask)[0]
    dates_win = dates[mask]

    theta_full = update_theta_series({"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0}, rf_series)
    theta_win = theta_full[mask]

    # final asset path for window under final params/theta
    A_win = get_asset_path(
        params={"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0},
        theta_series=theta_full,
        rf_series=rf_series,
        dates=dates,
        E_series=E_series,
        L_face_series=L_series,
        start_date=start_date,
        end_date=end_date,
    )

    return {
        "params": {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0},
        "converged": converged,
        "n_iter": it + 1,
        "dates_win": dates_win,
        "idx_win": idx,          # indices in the FULL firm series
        "A_win": A_win,          # same length as dates_win
        "theta_win": theta_win,  # same length as dates_win
    }


def _compute_pd_with_beta(
    A0: float,
    L: float,
    T: float,
    alpha: float,
    beta: float,
    beta0: float,
    delta: float,
) -> float:
    """
    Internal helper: PD = P(log(A_T/A0) < log(L/A0)) when log-return ~ NIG(alpha, beta, loc=beta0*T, scale=delta*T).
    """
    if A0 <= 0.0 or L <= 0.0 or T <= 0.0:
        raise ValueError("A0, L and T must be positive")
    if delta <= 0.0:
        raise ValueError("delta must be > 0")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")
    if abs(beta) >= alpha:
        raise ValueError("|beta| must be < alpha for a valid NIG distribution")

    delta_T = delta * T
    loc_T = beta0 * T
    x_thresh = np.log(L / A0)
    pd = norminvgauss.cdf(x_thresh, alpha, beta, loc=loc_T, scale=delta_T)
    return float(pd)


# Physical-measure PD:
# uses beta = beta1 (no risk-neutral tilt).
def compute_pd_physical(A0: float, L: float, T: float, params: Dict[str, float]) -> float:
    """
    Physical-measure PD: uses beta = beta1.
    """
    alpha = float(params.get("alpha", 0.0))
    beta1 = float(params.get("beta1", 0.0))
    beta0 = float(params.get("beta0", 0.0))
    delta = float(params.get("delta", 0.0))

    return _compute_pd_with_beta(
        A0=A0, L=L, T=T,
        alpha=alpha, beta=beta1,
        beta0=beta0, delta=delta
    )


# Risk-neutral PD:
# uses beta = beta1 + theta (theta must already be present in params).
def compute_pd_risk_neutral(A0: float, L: float, T: float, params: Dict[str, float]) -> float:
    """
    Risk-neutral PD: uses beta = beta1 + theta.
    Assumes params['theta'] has already been computed (e.g., via update_theta(params, r_f)).
    """
    alpha = float(params.get("alpha", 0.0))
    beta1 = float(params.get("beta1", 0.0))
    theta = float(params.get("theta", 0.0))
    beta0 = float(params.get("beta0", 0.0))
    delta = float(params.get("delta", 0.0))

    return _compute_pd_with_beta(
        A0=A0, L=L, T=T,
        alpha=alpha, beta=(beta1 + theta),
        beta0=beta0, delta=delta
    )


def one_year_pd_timeseries(
        out: dict,
        L_face_series_full: np.ndarray) -> pd.DataFrame:
    params = out["params"]
    dates_win = np.asarray(out["dates_win"])
    idx_win = np.asarray(out["idx_win"], dtype=int)
    A_win = np.asarray(out["A_win"], dtype=float)
    theta_win = np.asarray(out["theta_win"], dtype=float)

    L_face_series_full = np.asarray(L_face_series_full, dtype=float)

    # Use L at time t (no look-ahead)
    L_t = np.full(len(idx_win), np.nan, dtype=float)
    ok = idx_win < len(L_face_series_full)
    L_t[ok] = L_face_series_full[idx_win[ok]]

    pd_p = np.full(len(idx_win), np.nan, dtype=float)
    pd_q = np.full(len(idx_win), np.nan, dtype=float)

    for i in range(len(idx_win)):
        A0 = float(A_win[i])
        L0 = float(L_t[i])
        if not (np.isfinite(A0) and np.isfinite(L0) and A0 > 0.0 and L0 > 0.0):
            continue

        pd_p[i] = compute_pd_physical(A0=A0, L=L0, T=1.0, params=params)

        params_t = dict(params)
        params_t["theta"] = float(theta_win[i])
        pd_q[i] = compute_pd_risk_neutral(A0=A0, L=L0, T=1.0, params=params_t)

    return pd.DataFrame({
        "date": dates_win,
        "A_hat": A_win,
        "theta": theta_win,
        "L_proxy": L_t,
        "PD_physical": pd_p,
        "PD_risk_neutral": pd_q,
    })
