from typing import Dict
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


# Equity as call-option formula (as in Merton Model)
def nig_call_price(
    A: float,
    L_face: float,
    r: float,
    T: float,
    params: Dict[str, float],
    *,
    discounting: str = "continuous",
) -> float:
    """
    NIG equity price per Jovan & Ahčan Eq. (26):
      E_t = A_t * P^{β+θ+1}(X >= ln(L/A)) - L*e^{-rT} * P^{β+θ}(X >= ln(L/A))
    where X ~ NIG(α, β_shift, δ*T, μ*T).  (Eq. 21 scaling)
    """
    if A <= 0.0 or L_face <= 0.0 or T <= 0.0:
        return np.nan

    alpha = float(params.get("alpha", 0.0))
    beta = float(params.get("beta1", 0.0))
    delta = float(params.get("delta", 0.0))
    mu = float(params.get("beta0", 0.0))
    theta = float(params.get("theta", 0.0))

    if delta <= 0.0 or alpha <= 0.0:
        return np.nan

    x0 = np.log(L_face / A)

    # Lévy scaling (Eq. 21): delta_T, mu_T
    scale = float(delta) * float(T)
    loc = float(mu) * float(T)

    beta_plus = beta + theta + 1.0
    beta_minus = beta + theta

    # NIG validity in paper parametrisation: |beta_shift| < alpha
    if abs(beta_plus) >= alpha or abs(beta_minus) >= alpha:
        return np.nan

    # SciPy parametrisation: a = alpha*scale, b = beta*scale
    a = float(alpha) * scale
    b_plus = float(beta_plus) * scale
    b_minus = float(beta_minus) * scale

    tail_plus = norminvgauss.sf(x0, a=a, b=b_plus,  loc=loc, scale=scale)
    tail_minus = norminvgauss.sf(x0, a=a, b=b_minus, loc=loc, scale=scale)

    if discounting == "continuous":
        L_disc = float(L_face) * np.exp(-float(r) * float(T))
    elif discounting == "simple":
        L_disc = float(L_face) / (1.0 + float(r) * float(T))
    else:
        raise ValueError("discounting must be 'continuous' or 'simple'")

    return float(A * tail_plus - L_disc * tail_minus)


def invert_nig_call_price(
    E_obs: float,
    L_face: float,
    r: float,
    T: float,
    params: Dict[str, float],
    *,
    discounting: str = "continuous",
    A_min_factor: float = 1e-6,
    A_max_factor: float = 50.0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    if E_obs <= 0.0 or L_face <= 0.0 or T <= 0.0:
        raise ValueError("E_obs, L_face, T must be positive")

    A_min = max(E_obs, A_min_factor * L_face)
    A_max = A_max_factor * (E_obs + L_face)

    def f(A: float) -> float:
        return nig_call_price(
            A=A,
            L_face=L_face,
            r=r,
            T=T,
            params=params,
            discounting=discounting,
        ) - E_obs

    f_min, f_max = f(A_min), f(A_max)
    if not (np.isfinite(f_min) and np.isfinite(f_max)):
        return float(E_obs + L_face)

    if f_min * f_max > 0.0:
        for i in range(1, 18):
            factor = 2.0 ** i
            A_max_ext = A_max * factor
            A_min_ext = max(A_min / factor, 1e-12)
            f_min_ext, f_max_ext = f(A_min_ext), f(A_max_ext)
            if np.isfinite(f_min_ext) and np.isfinite(f_max_ext) and (f_min_ext * f_max_ext <= 0.0):
                A_min, A_max = A_min_ext, A_max_ext
                break
        else:
            return float(E_obs + L_face)

    return float(brentq(f, A_min, A_max, xtol=tol, maxiter=max_iter))


# E-step: infer asset path on the training window by inverting the NIG call price formula day-by-day
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
    horizon_days = daycount    # 1yr in trading years (as illustrated in paper)
    horizon_years = horizon_days * h

    end_idx_full = np.where(mask)[0][-1]     # last index in full arrays inside training window
    L_face = float(L_face_series[end_idx_full])  # PAPER STYLE: constant L at evaluation date

    if not (np.isfinite(L_face) and L_face > 0):
        raise ValueError("Invalid L_face at end of window")

    A_path = np.empty(n, dtype=float)

    for t in range(n):
        tau = horizon_years + (n - 1 - t) * h   # end has tau=1y, earlier have >1y (paper’s T-n=250 convention) :contentReference[oaicite:4]{index=4}

        r_t = float(r[t])  # KEEP: needed because discounting happens inside nig_call_price via exp(-r_t * tau)

        params_t = dict(params)
        params_t["theta"] = float(th[t])

        A_path[t] = invert_nig_call_price(
            E_obs=float(E[t]),
            L_face=float(L_face),
            r=r_t,
            T=float(tau),
            params=params_t,
            discounting=discounting,
        )

    return A_path


# M-step: NIG likelihood (paper)
def _nig_negloglik_daily(r, alpha, beta, delta_annual, mu_annual, *, daycount=250):
    h = 1.0 / daycount

    if alpha <= 0.0 or delta_annual <= 0.0 or abs(beta) >= alpha:
        return 1e50

    r = np.asarray(r, dtype=float).reshape(-1)
    if not np.all(np.isfinite(r)):
        return 1e50

    loc = float(mu_annual) * h
    scale = max(float(delta_annual) * h, 1e-12)

    # SciPy mapping: a = alpha*scale, b = beta*scale
    a = float(alpha) * scale
    b = float(beta) * scale

    ll = norminvgauss.logpdf(r, a=a, b=b, loc=loc, scale=scale)
    if not np.all(np.isfinite(ll)):
        return 1e50

    return float(-np.sum(ll))


def _fit_nig_mle_daily(r, x0, *, daycount=250):
    alpha0, beta0, delta0, mu0 = map(float, x0)

    # Paper-like lower bounds
    alpha_lb = 0.51
    delta_lb = 1e-9
    mu_bound = 1000.0
    eps = 1e-8

    # Reparam:
    # alpha = alpha_lb + exp(a_raw)
    # delta = delta_lb + exp(log_delta)
    # beta  = (alpha - eps) * tanh(b_raw)      -> guarantees |beta| < alpha
    # mu    = mu_bound * tanh(mu_raw / mu_bound)  -> keeps mu in [-1000,1000]
    def unpack(u):
        a_raw, b_raw, log_delta, mu_raw = map(float, u)
        alpha = alpha_lb + np.exp(a_raw)
        delta = delta_lb + np.exp(log_delta)
        beta = (alpha - eps) * np.tanh(b_raw)          # ensures |beta|<alpha
        mu = mu_bound * np.tanh(mu_raw / mu_bound)
        return float(alpha), float(beta), float(delta), float(mu)

    def obj(u):
        alpha, beta, delta, mu = unpack(u)
        return _nig_negloglik_daily(r, alpha, beta, delta, mu, daycount=daycount)

    # initialise u0 from x0
    u0 = np.array([
        np.log(max(alpha0 - alpha_lb, 1e-6)),
        np.arctanh(np.clip(beta0 / max(alpha0, 1e-6), -0.999, 0.999)),
        np.log(max(delta0 - delta_lb, 1e-12)),
        mu0,
    ], dtype=float)

    res = minimize(obj, u0, method="L-BFGS-B")

    if not res.success:
        # EM-friendly fallback: return old params
        return alpha0, beta0, delta0, mu0

    return unpack(res.x)


# EM initialization (paper)
def EM_algo(
    E_series: np.ndarray,
    L_face_series: np.ndarray,   # FACE liabilities (strike proxy); discounting handled inside Eq (26)
    rf_series: np.ndarray,       # daily risk-free series (needed for theta and discounting inside pricing)
    dates: np.ndarray,
    start_params: Dict[str, float],
    start_date=None,
    end_date=None,
    max_iter: int = 10,
    min_iter: int = 3,
    tol: float = 1e-3,
) -> Dict[str, object]:

    E_series = np.asarray(E_series, dtype=float)
    L_face_series = np.asarray(L_face_series, dtype=float)
    rf_series = np.asarray(rf_series, dtype=float)
    dates = np.asarray(dates)

    if not (len(E_series) == len(L_face_series) == len(rf_series) == len(dates)):
        raise ValueError("E_series, L_face_series, rf_series, dates must have the same length")

    # build training-window mask ONCE
    mask = np.ones(len(dates), dtype=bool)
    if start_date is not None:
        mask &= (dates >= start_date)
    if end_date is not None:
        mask &= (dates <= end_date)

    idx = np.where(mask)[0]
    if idx.size < 5:
        raise ValueError("Training window too short after applying start_date/end_date")

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

        # basic safety: theta must exist / be finite on the window (Eq. 24 feasibility) :contentReference[oaicite:2]{index=2}
        if not np.all(np.isfinite(theta_full[mask])):
            # fail fast or fallback: here we fail fast to avoid silent bad fits
            raise ValueError("Non-finite theta produced on training window; parameters infeasible for Eq. (24).")

        # E-step: infer asset path on the training window
        A_path = get_asset_path(
            params={"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0},
            theta_series=theta_full,
            rf_series=rf_series,
            dates=dates,
            E_series=E_series,
            L_face_series=L_face_series,
            start_date=start_date,
            end_date=end_date,
        )

        # daily log-returns of inferred assets
        rA = np.diff(np.log(A_path))
        if rA.size < 2 or (not np.all(np.isfinite(rA))):
            raise ValueError("Non-finite or too-short asset return series produced in E-step")

        # M-step: MLE update (annual params; likelihood uses daily scaling internally)
        a_new, b_new, d_new, mu_new = _fit_nig_mle_daily(
            rA, (alpha, beta1, delta, beta0)
        )

        # Recompute theta under proposed params
        theta_new_full = update_theta_series(
            {"alpha": a_new, "beta1": b_new, "delta": d_new, "beta0": mu_new},
            rf_series
        )

        if not np.all(np.isfinite(theta_new_full[mask])):
            # fallback: reject update (keeps EM stable)
            a_new, b_new, d_new, mu_new = alpha, beta1, delta, beta0
            theta_new_full = theta_full

        # Ensure pricing feasibility for Eq. (26) within the WINDOW:
        # need |beta+theta|<alpha and |beta+theta+1|<alpha (both shifts appear in Eq. 26)
        theta_new_win = theta_new_full[mask]
        alpha_floor = max(
            float(np.max(np.abs(b_new + theta_new_win))),
            float(np.max(np.abs(b_new + theta_new_win + 1.0))),
        ) + 1e-6

        if a_new < alpha_floor:
            a_new = float(alpha_floor)
            # theta depends on alpha (Eq. 24), so recompute after changing alpha :contentReference[oaicite:4]{index=4}
            theta_new_full = update_theta_series(
                {"alpha": a_new, "beta1": b_new, "delta": d_new, "beta0": mu_new},
                rf_series
            )
            if not np.all(np.isfinite(theta_new_full[mask])):
                # fallback again if bumping alpha breaks theta feasibility
                a_new, b_new, d_new, mu_new = alpha, beta1, delta, beta0
                theta_new_full = theta_full

        diff_last = np.array(
            [abs(a_new - alpha), abs(b_new - beta1), abs(d_new - delta), abs(mu_new - beta0)],
            dtype=float
        )

        alpha, beta1, delta, beta0 = float(a_new), float(b_new), float(d_new), float(mu_new)

        if (it + 1) >= min_iter and np.all(diff_last < tol):
            converged = True
            break

    # build training-window outputs
    dates_win = dates[mask]

    theta_full = update_theta_series(
        {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0},
        rf_series
    )
    theta_win = theta_full[mask]

    # final asset path for window under final params/theta
    A_win = get_asset_path(
        params={"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0},
        theta_series=theta_full,
        rf_series=rf_series,
        dates=dates,
        E_series=E_series,
        L_face_series=L_face_series,
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
        "diff_last": diff_last,  # optional but useful for debugging
    }


from scipy.stats import norminvgauss
import numpy as np

def _compute_pd_with_beta(
    A0: float,
    L: float,
    T: float,
    alpha: float,
    beta: float,
    beta0: float,   # your beta0 is μ
    delta: float,
) -> float:
    """
    PD = P( log(A_T/A0) < log(L/A0) )
    when log-return ~ NIG(alpha, beta, delta*T, mu*T).

    SciPy mapping for norminvgauss:
      scale = delta*T
      loc   = mu*T
      a     = alpha*scale
      b     = beta*scale
    """
    if not (A0 > 0.0 and L > 0.0 and T > 0.0):
        raise ValueError("A0, L and T must be positive")
    if delta <= 0.0:
        raise ValueError("delta must be > 0")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")
    if abs(beta) >= alpha:
        raise ValueError("|beta| must be < alpha for a valid NIG distribution")

    x_thresh = np.log(L / A0)

    # Lévy scaling over horizon T
    scale = float(delta) * float(T)      # δ_T
    loc   = float(beta0) * float(T)      # μ_T

    # SciPy NIG shape parameters
    a = float(alpha) * scale
    b = float(beta)  * scale

    pd = norminvgauss.cdf(x_thresh, a=a, b=b, loc=loc, scale=scale)
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
# uses beta = beta1 + theta
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

    # Use L at time t
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
