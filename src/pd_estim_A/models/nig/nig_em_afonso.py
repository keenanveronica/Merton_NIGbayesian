"""
Frequentist weekly NIG structural model utilities.

This module implements a paper-style EM routine for the Merton-NIG model,
adapted from the daily setup in Jovan & Ahčan to weekly Friday-aligned data.

Core workflow for one rolling window:
1) infer the in-sample weekly asset path by inverting the NIG call-price formula,
2) estimate annual NIG parameters with an EM loop,
3) use the final annual parameters to invert the OOS weekly asset path,
4) compute 1-year physical and risk-neutral PDs for each OOS week.

Conventions
-----------
- Parameters (alpha, beta1, delta, beta0) are annual parameters.
- Weekly scaling uses h = 1 / ann_factor, with ann_factor defaulting to 52.
- The training-window E-step follows the paper logic and keeps the strike
  proxy (liabilities) fixed at the end-of-window level while the time-to-maturity
  decreases backward through the window.
- The OOS inversion uses tau = forecast_horizon_years (default 1.0) at each OOS
  week and the liabilities observed at that week.
- The risk-free rate is assumed to be an annual continuously-compounded rate
  because the pricing formula discounts with exp(-r * tau).

Expected input columns for the window runners
---------------------------------------------
- date      : weekly observation date
- market_cap: equity market value E_t
- debt_face : debt / strike proxy L_t
- rf        : annual risk-free rate r_t
- gvkey     : optional firm identifier
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import norminvgauss


EPS = 1e-12


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _as_ts(x: Any) -> Optional[pd.Timestamp]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return pd.Timestamp(x)


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _finite_positive(x: float) -> bool:
    return np.isfinite(x) and x > 0.0


def daily_to_weekly_nig_panel(
    df_firm_daily: pd.DataFrame,
    *,
    date_col: str = "date",
    equity_col: str = "market_cap",
    debt_col: str = "debt_face",
    rf_col: str = "rf",
    gvkey_col: str = "gvkey",
    week_freq: str = "W-FRI",
    keep_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Convert a daily firm panel into a Friday-aligned weekly panel.

    For each calendar week ending on Friday, keep the LAST available daily
    observation within that week. If Friday exists, this is the Friday row;
    otherwise it is usually the Thursday (or earlier) row from that week.

    This keeps the NIG math unchanged and only changes the calendar from daily
    inputs to the weekly Friday-aligned panel used downstream.
    """
    _require_columns(df_firm_daily, [date_col, equity_col, debt_col, rf_col])

    df = df_firm_daily.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")

    base_cols = [date_col, equity_col, debt_col, rf_col]
    if gvkey_col in df.columns:
        base_cols.append(gvkey_col)
    if keep_cols is not None:
        for c in keep_cols:
            if c in df.columns and c not in base_cols:
                base_cols.append(c)

    df = df[base_cols].copy()
    for c in [equity_col, debt_col, rf_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.set_index(date_col)
    agg_map = {c: "last" for c in df.columns}
    weekly = df.resample(week_freq, label="right", closed="right").agg(agg_map).reset_index()

    weekly = weekly[
        np.isfinite(weekly[equity_col])
        & np.isfinite(weekly[debt_col])
        & np.isfinite(weekly[rf_col])
    ].copy()
    weekly = weekly[(weekly[equity_col] > 0.0) & (weekly[debt_col] > 0.0)].copy()
    weekly = weekly.sort_values(date_col).reset_index(drop=True)
    return weekly


def _clean_weekly_panel(
    df_firm: pd.DataFrame,
    *,
    date_col: str,
    equity_col: str,
    debt_col: str,
    rf_col: str,
) -> pd.DataFrame:
    out = df_firm.copy()
    _require_columns(out, [date_col, equity_col, debt_col, rf_col])

    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")

    for c in [equity_col, debt_col, rf_col]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out[np.isfinite(out[equity_col]) & np.isfinite(out[debt_col]) & np.isfinite(out[rf_col])].copy()
    out = out[out[equity_col] > 0.0].copy()
    out = out[out[debt_col] > 0.0].copy()
    out = out.reset_index(drop=True)
    return out


def _slice_window(
    df: pd.DataFrame,
    *,
    date_col: str,
    start_date: Any,
    end_date: Any,
) -> pd.DataFrame:
    start_ts = _as_ts(start_date)
    end_ts = _as_ts(end_date)

    mask = np.ones(len(df), dtype=bool)
    if start_ts is not None:
        mask &= df[date_col] >= start_ts
    if end_ts is not None:
        mask &= df[date_col] <= end_ts
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# NIG parameter handling
# ---------------------------------------------------------------------------

def validate_nig_params(alpha: float, beta: float, delta: float) -> None:
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and > 0.")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be finite and > 0.")
    if not np.isfinite(beta) or abs(beta) >= alpha:
        raise ValueError("Need |beta| < alpha.")


def _minimum_alpha_for_theta_existence(mu: float, delta: float) -> float:
    """
    Paper-style sufficient condition:
      alpha >= 0.5
      |mu| <= delta * sqrt(2 alpha - 1)

    Rearranged as a lower bound on alpha:
      alpha >= (1 + (mu / delta)^2) / 2
    """
    if not _finite_positive(delta):
        raise ValueError("delta must be > 0 for theta existence.")
    return 0.5 * (1.0 + (float(mu) / float(delta)) ** 2)


def update_theta(params: Dict[str, float], r_f: float) -> float:
    """
    Compute the Esscher tilt theta from Jovan & Ahčan Eq. (24).

    params keys:
      alpha, beta1, delta, beta0
    where beta1 is the NIG skewness parameter beta and beta0 is mu.
    """
    alpha = float(params["alpha"])
    beta = float(params["beta1"])
    delta = float(params["delta"])
    mu = float(params["beta0"])
    r_f = float(r_f)

    validate_nig_params(alpha, beta, delta)

    alpha_lb_theta = max(0.5, _minimum_alpha_for_theta_existence(mu, delta))
    if alpha < alpha_lb_theta:
        raise ValueError(
            "Theta existence failed: alpha too small relative to mu and delta."
        )

    inside = (4.0 * alpha * alpha * delta * delta) / (((mu - r_f) ** 2) + delta * delta) - 1.0
    if not np.isfinite(inside) or inside < 0.0:
        raise ValueError("Theta formula is not real for these parameters and rf.")

    theta = -beta - 0.5 - ((mu - r_f) / (2.0 * delta)) * np.sqrt(inside)
    if not np.isfinite(theta):
        raise ValueError("Non-finite theta.")
    return float(theta)


def update_theta_series(params: Dict[str, float], r_f_series: Iterable[float]) -> np.ndarray:
    r_f = np.asarray(list(r_f_series), dtype=float)
    out = np.empty_like(r_f, dtype=float)
    for i, rf_i in enumerate(r_f):
        out[i] = update_theta(params, float(rf_i))
    return out


def _make_params_window_feasible(
    params: Dict[str, float],
    r_f_series: Sequence[float],
    *,
    max_iter: int = 12,
    buffer: float = 1e-6,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Small practical guardrail so the EM iterations remain numerically feasible.

    We keep the paper-style parameters, but if the current alpha is too small to
    support theta or the pricing shifts (beta + theta, beta + theta + 1), we
    increase alpha minimally until the window becomes valid.
    """
    alpha = float(params["alpha"])
    beta1 = float(params["beta1"])
    delta = float(params["delta"])
    beta0 = float(params["beta0"])
    r_f = np.asarray(r_f_series, dtype=float)

    validate_nig_params(alpha, beta1, delta)

    for _ in range(max_iter):
        alpha_floor_theta = _minimum_alpha_for_theta_existence(beta0, delta) + buffer
        alpha_floor_real = np.max(np.sqrt(((beta0 - r_f) ** 2 + delta ** 2)) / (2.0 * delta)) + buffer
        alpha_floor = max(0.51, float(alpha_floor_theta), float(alpha_floor_real))
        alpha = max(alpha, alpha_floor)

        trial = {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0}
        try:
            theta = update_theta_series(trial, r_f)
        except Exception:
            alpha = alpha * 1.10 + buffer
            continue

        alpha_needed = max(
            np.max(np.abs(beta1 + theta)),
            np.max(np.abs(beta1 + theta + 1.0)),
        ) + buffer

        if alpha >= alpha_needed:
            return trial, theta

        alpha = alpha_needed

    raise ValueError("Could not make (alpha, beta, delta, mu) feasible on the window.")


# ---------------------------------------------------------------------------
# SciPy NIG mapping
# ---------------------------------------------------------------------------

def _scipy_nig_args(
    alpha: float,
    beta: float,
    delta_h: float,
    mu_h: float,
) -> Dict[str, float]:
    """
    Map the paper parametrization (alpha, beta, delta_h, mu_h) to SciPy's
    norminvgauss(a, b, loc, scale).

    SciPy documentation states:
      a = alpha * delta_h
      b = beta  * delta_h
      loc = mu_h
      scale = delta_h
    """
    validate_nig_params(alpha, beta, delta_h if delta_h > 0.0 else np.nan)
    if not np.isfinite(mu_h):
        raise ValueError("mu_h must be finite.")

    a = float(alpha) * float(delta_h)
    b = float(beta) * float(delta_h)
    if abs(b) >= a:
        raise ValueError("SciPy requires |b| < a.")
    return {"a": a, "b": b, "loc": float(mu_h), "scale": float(delta_h)}


# ---------------------------------------------------------------------------
# NIG pricing and inversion
# ---------------------------------------------------------------------------

def nig_call_price(
    A: float,
    L_face: float,
    r: float,
    tau: float,
    params: Dict[str, float],
    *,
    theta: Optional[float] = None,
    discounting: str = "continuous",
) -> float:
    """
    Jovan & Ahčan Eq. (26), implemented via NIG survival probabilities.

    E_t = A_t * P^{beta+theta+1}(X >= ln(L/A))
          - L * exp(-r tau) * P^{beta+theta}(X >= ln(L/A))
    """
    if not (_finite_positive(A) and _finite_positive(L_face) and _finite_positive(tau)):
        return np.nan

    alpha = float(params["alpha"])
    beta1 = float(params["beta1"])
    delta = float(params["delta"])
    beta0 = float(params["beta0"])

    validate_nig_params(alpha, beta1, delta)

    if theta is None:
        theta = update_theta(params, float(r))
    theta = float(theta)

    beta_plus = beta1 + theta + 1.0
    beta_minus = beta1 + theta

    if abs(beta_plus) >= alpha or abs(beta_minus) >= alpha:
        return np.nan

    x0 = np.log(float(L_face) / float(A))
    delta_tau = float(delta) * float(tau)
    mu_tau = float(beta0) * float(tau)

    args_plus = _scipy_nig_args(alpha, beta_plus, delta_tau, mu_tau)
    args_minus = _scipy_nig_args(alpha, beta_minus, delta_tau, mu_tau)

    tail_plus = norminvgauss.sf(x0, **args_plus)
    tail_minus = norminvgauss.sf(x0, **args_minus)

    if not (np.isfinite(tail_plus) and np.isfinite(tail_minus)):
        return np.nan

    if discounting == "continuous":
        L_disc = float(L_face) * np.exp(-float(r) * float(tau))
    elif discounting == "simple":
        L_disc = float(L_face) / (1.0 + float(r) * float(tau))
    else:
        raise ValueError("discounting must be 'continuous' or 'simple'.")

    price = float(A) * float(tail_plus) - float(L_disc) * float(tail_minus)
    return float(price)


def invert_asset_one_week_nig(
    E_obs: float,
    L_face: float,
    r: float,
    tau: float,
    params: Dict[str, float],
    *,
    theta: Optional[float] = None,
    discounting: str = "continuous",
    A_min_factor: float = 1e-8,
    A_max_factor: float = 50.0,
    xtol: float = 1e-10,
    maxiter: int = 300,
) -> float:
    """
    Invert the paper's NIG equity price numerically for a single observation.
    """
    if not (_finite_positive(E_obs) and _finite_positive(L_face) and _finite_positive(tau)):
        return np.nan

    if theta is None:
        try:
            theta = update_theta(params, float(r))
        except Exception:
            return np.nan

    def f(A: float) -> float:
        return nig_call_price(
            A=A,
            L_face=L_face,
            r=r,
            tau=tau,
            params=params,
            theta=theta,
            discounting=discounting,
        ) - float(E_obs)

    A_min = max(float(E_obs), float(A_min_factor) * float(L_face), EPS)
    A_max = max(float(A_max_factor) * (float(E_obs) + float(L_face)), A_min * 2.0)

    f_min = f(A_min)
    f_max = f(A_max)

    if not (np.isfinite(f_min) and np.isfinite(f_max)):
        return np.nan

    if f_min * f_max > 0.0:
        bracket_found = False
        for i in range(1, 22):
            factor = 2.0 ** i
            A_lo = max(A_min / factor, EPS)
            A_hi = A_max * factor
            f_lo = f(A_lo)
            f_hi = f(A_hi)
            if np.isfinite(f_lo) and np.isfinite(f_hi) and (f_lo * f_hi <= 0.0):
                A_min, A_max = A_lo, A_hi
                bracket_found = True
                break
        if not bracket_found:
            return np.nan

    try:
        return float(brentq(f, A_min, A_max, xtol=xtol, maxiter=maxiter))
    except Exception:
        return np.nan


def infer_training_asset_path_nig(
    E_series: Sequence[float],
    L_face_series: Sequence[float],
    rf_series: Sequence[float],
    params: Dict[str, float],
    *,
    ann_factor: float = 52.0,
    forecast_horizon_years: float = 1.0,
    discounting: str = "continuous",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Paper-style E-step over the training window.

    The strike proxy L is fixed at the end-of-window value, while the maturity
    decreases backward through the sample:
      tau_t = forecast_horizon_years + (n - 1 - t) / ann_factor
    """
    E = np.asarray(E_series, dtype=float)
    L_face = np.asarray(L_face_series, dtype=float)
    rf = np.asarray(rf_series, dtype=float)

    if not (len(E) == len(L_face) == len(rf)):
        raise ValueError("E_series, L_face_series, rf_series must have the same length.")
    if len(E) < 3:
        raise ValueError("Training window too short.")
    if not np.all(np.isfinite(E)) or not np.all(np.isfinite(L_face)) or not np.all(np.isfinite(rf)):
        raise ValueError("Non-finite values inside the training window.")
    if not np.all(E > 0.0) or not np.all(L_face > 0.0):
        raise ValueError("E and L_face must be positive.")

    params_feas, theta = _make_params_window_feasible(params, rf)
    n = len(E)
    L_end = float(L_face[-1])

    A = np.full(n, np.nan, dtype=float)
    for t in range(n):
        tau_t = float(forecast_horizon_years) + float(n - 1 - t) / float(ann_factor)
        A[t] = invert_asset_one_week_nig(
            E_obs=float(E[t]),
            L_face=L_end,
            r=float(rf[t]),
            tau=tau_t,
            params=params_feas,
            theta=float(theta[t]),
            discounting=discounting,
        )
    return A, theta


def infer_oos_asset_path_nig(
    E_series: Sequence[float],
    L_face_series: Sequence[float],
    rf_series: Sequence[float],
    params: Dict[str, float],
    *,
    tau_years: float = 1.0,
    discounting: str = "continuous",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Invert the asset path week-by-week in the OOS window using fixed annual
    parameters and current weekly liabilities.
    """
    E = np.asarray(E_series, dtype=float)
    L_face = np.asarray(L_face_series, dtype=float)
    rf = np.asarray(rf_series, dtype=float)

    if not (len(E) == len(L_face) == len(rf)):
        raise ValueError("E_series, L_face_series, rf_series must have the same length.")
    if len(E) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    params_feas, theta = _make_params_window_feasible(params, rf)

    A = np.full(len(E), np.nan, dtype=float)
    for t in range(len(E)):
        A[t] = invert_asset_one_week_nig(
            E_obs=float(E[t]),
            L_face=float(L_face[t]),
            r=float(rf[t]),
            tau=float(tau_years),
            params=params_feas,
            theta=float(theta[t]),
            discounting=discounting,
        )
    return A, theta


# ---------------------------------------------------------------------------
# NIG likelihood and M-step
# ---------------------------------------------------------------------------

def _nig_negloglik_from_returns(
    r: Sequence[float],
    alpha: float,
    beta1: float,
    delta_annual: float,
    beta0_annual: float,
    *,
    ann_factor: float = 52.0,
) -> float:
    """
    Negative log-likelihood of weekly log-asset returns under annual NIG params.
    """
    validate_nig_params(alpha, beta1, delta_annual)

    x = np.asarray(r, dtype=float).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return 1e100

    h = 1.0 / float(ann_factor)
    delta_h = float(delta_annual) * h
    beta0_h = float(beta0_annual) * h

    try:
        args = _scipy_nig_args(alpha, beta1, delta_h, beta0_h)
        ll = norminvgauss.logpdf(x, **args)
    except Exception:
        return 1e100

    if not np.all(np.isfinite(ll)):
        return 1e100

    return float(-np.sum(ll))


def fit_nig_mle_from_weekly_returns(
    r: Sequence[float],
    x0: Tuple[float, float, float, float],
    *,
    ann_factor: float = 52.0,
) -> Tuple[float, float, float, float]:
    """
    M-step optimizer for annual NIG parameters.

    We optimize over unconstrained transformed variables, then map them back to:
      alpha > 0.51
      delta > 1e-9
      |beta| < alpha
      mu in [-1000, 1000]
    """
    alpha0, beta10, delta0, beta00 = map(float, x0)

    alpha_lb = 0.51
    delta_lb = 1e-9
    mu_bound = 1000.0
    eps = 1e-8

    def unpack(u: np.ndarray) -> Tuple[float, float, float, float]:
        a_raw, b_raw, log_delta_raw, mu_raw = map(float, u)
        alpha = alpha_lb + np.exp(a_raw)
        delta = delta_lb + np.exp(log_delta_raw)
        beta1 = (alpha - eps) * np.tanh(b_raw)
        beta0 = mu_bound * np.tanh(mu_raw / mu_bound)
        return float(alpha), float(beta1), float(delta), float(beta0)

    def obj(u: np.ndarray) -> float:
        alpha, beta1, delta, beta0 = unpack(u)
        return _nig_negloglik_from_returns(
            r,
            alpha=alpha,
            beta1=beta1,
            delta_annual=delta,
            beta0_annual=beta0,
            ann_factor=ann_factor,
        )

    u0 = np.array(
        [
            np.log(max(alpha0 - alpha_lb, 1e-6)),
            np.arctanh(np.clip(beta10 / max(alpha0, 1e-6), -0.999999, 0.999999)),
            np.log(max(delta0 - delta_lb, 1e-12)),
            beta00,
        ],
        dtype=float,
    )

    best = None
    methods = ["L-BFGS-B", "Powell"]

    for method in methods:
        try:
            res = minimize(obj, u0, method=method)
        except Exception:
            continue
        if best is None or (res.fun < best.fun):
            best = res

    if best is None or not np.isfinite(best.fun):
        return alpha0, beta10, delta0, beta00

    if not best.success and best.fun > obj(u0):
        return alpha0, beta10, delta0, beta00

    return unpack(best.x)


# ---------------------------------------------------------------------------
# EM estimation for one training window
# ---------------------------------------------------------------------------

def em_nig_weekly_window(
    train_df: pd.DataFrame,
    *,
    date_col: str = "date",
    equity_col: str = "market_cap",
    debt_col: str = "debt_face",
    rf_col: str = "rf",
    start_params: Optional[Dict[str, float]] = None,
    ann_factor: float = 52.0,
    forecast_horizon_years: float = 1.0,
    max_iter: int = 10,
    min_iter: int = 3,
    tol: float = 1e-1,
    discounting: str = "continuous",
) -> Dict[str, Any]:
    """
    Paper-style EM estimation on a weekly training window.

    Returns the final annual parameters, the in-sample inferred asset path, the
    theta series on the training window, and diagnostics.
    """
    if start_params is None:
        start_params = {
            "alpha": 10.0,
            "beta1": 0.0,
            "delta": 1.0,
            "beta0": 0.0,
        }

    _require_columns(train_df, [date_col, equity_col, debt_col, rf_col])
    w = train_df.copy().sort_values(date_col).reset_index(drop=True)

    E = w[equity_col].to_numpy(dtype=float)
    L = w[debt_col].to_numpy(dtype=float)
    rf = w[rf_col].to_numpy(dtype=float)

    if len(w) < 5:
        raise ValueError("Training window too short for EM estimation.")

    params = {
        "alpha": float(start_params["alpha"]),
        "beta1": float(start_params["beta1"]),
        "delta": float(start_params["delta"]),
        "beta0": float(start_params["beta0"]),
    }

    diff_last = np.full(4, np.nan, dtype=float)
    converged = False
    A_last = None
    theta_last = None
    n_iter = 0

    for it in range(max_iter):
        params_feas, theta = _make_params_window_feasible(params, rf)

        A_hat, theta_used = infer_training_asset_path_nig(
            E_series=E,
            L_face_series=L,
            rf_series=rf,
            params=params_feas,
            ann_factor=ann_factor,
            forecast_horizon_years=forecast_horizon_years,
            discounting=discounting,
        )

        if not np.all(np.isfinite(A_hat)) or np.any(A_hat <= 0.0):
            raise ValueError("E-step failed: could not infer a valid training asset path.")

        logret = np.diff(np.log(A_hat))
        if len(logret) < 2 or not np.all(np.isfinite(logret)):
            raise ValueError("Invalid asset returns generated in the E-step.")

        alpha_new, beta1_new, delta_new, beta0_new = fit_nig_mle_from_weekly_returns(
            logret,
            (params_feas["alpha"], params_feas["beta1"], params_feas["delta"], params_feas["beta0"]),
            ann_factor=ann_factor,
        )

        params_new_raw = {
            "alpha": float(alpha_new),
            "beta1": float(beta1_new),
            "delta": float(delta_new),
            "beta0": float(beta0_new),
        }
        params_new, _ = _make_params_window_feasible(params_new_raw, rf)

        diff_last = np.array(
            [
                abs(params_new["alpha"] - params["alpha"]),
                abs(params_new["beta1"] - params["beta1"]),
                abs(params_new["delta"] - params["delta"]),
                abs(params_new["beta0"] - params["beta0"]),
            ],
            dtype=float,
        )

        params = params_new
        A_last = A_hat
        theta_last = theta_used
        n_iter = it + 1

        if (it + 1) >= min_iter and np.all(diff_last < float(tol)):
            converged = True
            break

    if A_last is None or theta_last is None:
        raise RuntimeError("EM finished without producing outputs.")

    out_train = w[[date_col]].copy()
    out_train["A_hat_train"] = A_last
    out_train["theta_train"] = theta_last
    out_train["L_train_used"] = float(L[-1])

    return {
        "params": params,
        "converged": bool(converged),
        "n_iter": int(n_iter),
        "diff_last": diff_last,
        "train_df": out_train,
    }


# ---------------------------------------------------------------------------
# PD calculation
# ---------------------------------------------------------------------------

def _compute_pd_with_beta(
    A0: float,
    L: float,
    T: float,
    alpha: float,
    beta: float,
    delta: float,
    beta0: float,
) -> float:
    """
    PD = P[ log(A_T / A_0) <= log(L / A_0) ]
    when the log-return over horizon T follows an NIG law.
    """
    if not (_finite_positive(A0) and _finite_positive(L) and _finite_positive(T)):
        return np.nan
    validate_nig_params(alpha, beta, delta)

    x_thr = np.log(float(L) / float(A0))
    delta_T = float(delta) * float(T)
    beta0_T = float(beta0) * float(T)

    try:
        args = _scipy_nig_args(alpha, beta, delta_T, beta0_T)
    except Exception:
        return np.nan

    pd_val = norminvgauss.cdf(x_thr, **args)
    return float(pd_val) if np.isfinite(pd_val) else np.nan


def compute_pd_physical(
    A0: float,
    L: float,
    T: float,
    params: Dict[str, float],
) -> float:
    return _compute_pd_with_beta(
        A0=A0,
        L=L,
        T=T,
        alpha=float(params["alpha"]),
        beta=float(params["beta1"]),
        delta=float(params["delta"]),
        beta0=float(params["beta0"]),
    )


def compute_pd_risk_neutral(
    A0: float,
    L: float,
    T: float,
    params: Dict[str, float],
    *,
    r_f: Optional[float] = None,
    theta: Optional[float] = None,
) -> float:
    if theta is None:
        if r_f is None:
            raise ValueError("Need either theta or r_f for risk-neutral PD.")
        theta = update_theta(params, float(r_f))

    return _compute_pd_with_beta(
        A0=A0,
        L=L,
        T=T,
        alpha=float(params["alpha"]),
        beta=float(params["beta1"]) + float(theta),
        delta=float(params["delta"]),
        beta0=float(params["beta0"]),
    )


# ---------------------------------------------------------------------------
# OOS forecasting for one window
# ---------------------------------------------------------------------------

def forecast_nig_oos_window(
    oos_df: pd.DataFrame,
    params: Dict[str, float],
    *,
    date_col: str = "date",
    equity_col: str = "market_cap",
    debt_col: str = "debt_face",
    rf_col: str = "rf",
    pd_horizon_years: float = 1.0,
    inversion_tau_years: float = 1.0,
    discounting: str = "continuous",
) -> pd.DataFrame:
    _require_columns(oos_df, [date_col, equity_col, debt_col, rf_col])

    if len(oos_df) == 0:
        return oos_df[[date_col]].copy()

    w = oos_df.copy().sort_values(date_col).reset_index(drop=True)

    E = w[equity_col].to_numpy(dtype=float)
    L = w[debt_col].to_numpy(dtype=float)
    rf = w[rf_col].to_numpy(dtype=float)

    A_hat, theta = infer_oos_asset_path_nig(
        E_series=E,
        L_face_series=L,
        rf_series=rf,
        params=params,
        tau_years=inversion_tau_years,
        discounting=discounting,
    )

    pd_p = np.full(len(w), np.nan, dtype=float)
    pd_q = np.full(len(w), np.nan, dtype=float)

    for i in range(len(w)):
        if not (_finite_positive(A_hat[i]) and _finite_positive(L[i])):
            continue
        pd_p[i] = compute_pd_physical(
            A0=float(A_hat[i]),
            L=float(L[i]),
            T=float(pd_horizon_years),
            params=params,
        )
        pd_q[i] = compute_pd_risk_neutral(
            A0=float(A_hat[i]),
            L=float(L[i]),
            T=float(pd_horizon_years),
            params=params,
            theta=float(theta[i]),
        )

    out = w[[date_col]].copy()
    out["A_hat_oos"] = A_hat
    out["theta_oos"] = theta
    out["L_proxy"] = L
    out["PD_P"] = pd_p
    out["PD_Q"] = pd_q
    return out


# ---------------------------------------------------------------------------
# High-level runners
# ---------------------------------------------------------------------------

def run_nig_window_for_firm(
    df_firm: pd.DataFrame,
    *,
    train_start: Any,
    train_end: Any,
    oos_start: Any,
    oos_end: Any,
    gvkey_col: str = "gvkey",
    input_frequency: str = "weekly",
    week_freq: str = "W-FRI",
    date_col: str = "date",
    equity_col: str = "market_cap",
    debt_col: str = "debt_face",
    rf_col: str = "rf",
    start_params: Optional[Dict[str, float]] = None,
    ann_factor: float = 52.0,
    forecast_horizon_years: float = 1.0,
    pd_horizon_years: float = 1.0,
    max_iter: int = 10,
    min_iter: int = 3,
    tol: float = 1e-1,
    discounting: str = "continuous",
) -> Dict[str, Any]:
    """
    Full estimation + OOS forecasting for one firm and one rolling window.

    input_frequency:
      - "daily"  -> first convert to a Friday-aligned weekly panel
      - "weekly" -> assume df_firm is already weekly
    """
    if input_frequency not in {"daily", "weekly"}:
        raise ValueError("input_frequency must be 'daily' or 'weekly'.")

    if input_frequency == "daily":
        df_weekly = daily_to_weekly_nig_panel(
            df_firm,
            date_col=date_col,
            equity_col=equity_col,
            debt_col=debt_col,
            rf_col=rf_col,
            gvkey_col=gvkey_col,
            week_freq=week_freq,
        )
    else:
        df_weekly = df_firm.copy()

    df0 = _clean_weekly_panel(
        df_weekly,
        date_col=date_col,
        equity_col=equity_col,
        debt_col=debt_col,
        rf_col=rf_col,
    )

    gvkey_val = None
    if gvkey_col in df_weekly.columns and len(df_weekly) > 0:
        vals = df_weekly[gvkey_col].dropna().astype(str).unique().tolist()
        gvkey_val = vals[0] if vals else None

    train_df = _slice_window(df0, date_col=date_col, start_date=train_start, end_date=train_end)
    oos_df = _slice_window(df0, date_col=date_col, start_date=oos_start, end_date=oos_end)

    meta = {
        "gvkey": gvkey_val,
        "train_start_req": _as_ts(train_start),
        "train_end_req": _as_ts(train_end),
        "oos_start_req": _as_ts(oos_start),
        "oos_end_req": _as_ts(oos_end),
        "train_start_used_weekly": train_df[date_col].min() if len(train_df) else pd.NaT,
        "train_end_used_weekly": train_df[date_col].max() if len(train_df) else pd.NaT,
        "oos_start_used_weekly": oos_df[date_col].min() if len(oos_df) else pd.NaT,
        "oos_end_used_weekly": oos_df[date_col].max() if len(oos_df) else pd.NaT,
    }

    if len(train_df) < 5:
        return {**meta, "ok": False, "msg": "training_window_too_short", "oos_df": pd.DataFrame(), "train_df": pd.DataFrame()}
    if len(oos_df) == 0:
        return {**meta, "ok": False, "msg": "empty_oos_window", "oos_df": pd.DataFrame(), "train_df": pd.DataFrame()}

    try:
        em_out = em_nig_weekly_window(
            train_df,
            date_col=date_col,
            equity_col=equity_col,
            debt_col=debt_col,
            rf_col=rf_col,
            start_params=start_params,
            ann_factor=ann_factor,
            forecast_horizon_years=forecast_horizon_years,
            max_iter=max_iter,
            min_iter=min_iter,
            tol=tol,
            discounting=discounting,
        )
        params = em_out["params"]

        oos_forecast = forecast_nig_oos_window(
            oos_df,
            params,
            date_col=date_col,
            equity_col=equity_col,
            debt_col=debt_col,
            rf_col=rf_col,
            pd_horizon_years=pd_horizon_years,
            inversion_tau_years=forecast_horizon_years,
            discounting=discounting,
        )

        for k, v in params.items():
            oos_forecast[k] = float(v)

        oos_forecast["em_converged"] = bool(em_out["converged"])
        oos_forecast["em_n_iter"] = int(em_out["n_iter"])
        oos_forecast["window_train_start"] = meta["train_start_used_weekly"]
        oos_forecast["window_train_end"] = meta["train_end_used_weekly"]
        oos_forecast["window_oos_start"] = meta["oos_start_used_weekly"]
        oos_forecast["window_oos_end"] = meta["oos_end_used_weekly"]
        if gvkey_val is not None:
            oos_forecast[gvkey_col] = gvkey_val

        train_panel = em_out["train_df"].copy()
        for k, v in params.items():
            train_panel[k] = float(v)
        train_panel["em_converged"] = bool(em_out["converged"])
        train_panel["em_n_iter"] = int(em_out["n_iter"])
        if gvkey_val is not None:
            train_panel[gvkey_col] = gvkey_val

        return {
            **meta,
            "ok": True,
            "msg": "ok",
            "params": params,
            "em_converged": bool(em_out["converged"]),
            "em_n_iter": int(em_out["n_iter"]),
            "diff_last": em_out["diff_last"],
            "train_df": train_panel,
            "oos_df": oos_forecast,
        }

    except Exception as exc:
        return {
            **meta,
            "ok": False,
            "msg": str(exc),
            "oos_df": pd.DataFrame(),
            "train_df": pd.DataFrame(),
        }


def process_one_firm_nig(
    df_firm: pd.DataFrame,
    window_plan_df: pd.DataFrame,
    *,
    train_start_col: str = "train_start",
    train_end_col: str = "train_end",
    oos_start_col: str = "oos_start",
    oos_end_col: str = "oos_end",
    gvkey_col: str = "gvkey",
    input_frequency: str = "weekly",
    week_freq: str = "W-FRI",
    date_col: str = "date",
    equity_col: str = "market_cap",
    debt_col: str = "debt_face",
    rf_col: str = "rf",
    start_params: Optional[Dict[str, float]] = None,
    ann_factor: float = 52.0,
    forecast_horizon_years: float = 1.0,
    pd_horizon_years: float = 1.0,
    max_iter: int = 10,
    min_iter: int = 3,
    tol: float = 1e-1,
    discounting: str = "continuous",
) -> pd.DataFrame:
    """
    Run all requested windows for one firm and return one stacked OOS panel.
    """
    _require_columns(window_plan_df, [train_start_col, train_end_col, oos_start_col, oos_end_col])

    frames: List[pd.DataFrame] = []

    for window_idx, row in window_plan_df.reset_index(drop=True).iterrows():
        out = run_nig_window_for_firm(
            df_firm,
            train_start=row[train_start_col],
            train_end=row[train_end_col],
            oos_start=row[oos_start_col],
            oos_end=row[oos_end_col],
            gvkey_col=gvkey_col,
            input_frequency=input_frequency,
            week_freq=week_freq,
            date_col=date_col,
            equity_col=equity_col,
            debt_col=debt_col,
            rf_col=rf_col,
            start_params=start_params,
            ann_factor=ann_factor,
            forecast_horizon_years=forecast_horizon_years,
            pd_horizon_years=pd_horizon_years,
            max_iter=max_iter,
            min_iter=min_iter,
            tol=tol,
            discounting=discounting,
        )

        if out["ok"]:
            df_win = out["oos_df"].copy()
            df_win["window_idx"] = int(window_idx)
            df_win["ok"] = True
            df_win["msg"] = "ok"
            frames.append(df_win)
        else:
            gvkey_val = None
            if gvkey_col in df_firm.columns and len(df_firm) > 0:
                vals = df_firm[gvkey_col].dropna().astype(str).unique().tolist()
                gvkey_val = vals[0] if vals else None
            fail = pd.DataFrame(
                {
                    "date": [pd.NaT],
                    "window_idx": [int(window_idx)],
                    "ok": [False],
                    "msg": [out["msg"]],
                    "window_train_start": [out["train_start_used_weekly"]],
                    "window_train_end": [out["train_end_used_weekly"]],
                    "window_oos_start": [out["oos_start_used_weekly"]],
                    "window_oos_end": [out["oos_end_used_weekly"]],
                }
            )
            if gvkey_val is not None:
                fail[gvkey_col] = gvkey_val
            frames.append(fail)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "update_theta",
    "update_theta_series",
    "nig_call_price",
    "invert_asset_one_week_nig",
    "infer_training_asset_path_nig",
    "infer_oos_asset_path_nig",
    "fit_nig_mle_from_weekly_returns",
    "em_nig_weekly_window",
    "compute_pd_physical",
    "compute_pd_risk_neutral",
    "forecast_nig_oos_window",
    "run_nig_window_for_firm",
    "process_one_firm_nig",
    "daily_to_weekly_nig_panel",
]