import numpy as np
import warnings
from typing import Dict, Union, Optional, Literal
from scipy.optimize import brentq
from scipy.stats import norminvgauss


def update_theta(
    params: Dict[str, float],
    r_f: Union[float, np.ndarray],
    *,
    enforce_pricing: bool = True,
    warn: bool = True,
) -> Union[float, np.ndarray]:
    """
    Compute Esscher tilt theta. Vectorized:
      - r_f can be a scalar or a 1D array.
      - returns scalar float if r_f is scalar, else ndarray of theta_t.
    Any infeasible points (complex theta / pricing infeasible) are set to np.nan.
    """

    alpha = float(params.get("alpha", 0.0))
    beta = float(params.get("beta1", 0.0))
    delta = float(params.get("delta", 0.0))
    mu = float(params.get("beta0", 0.0))

    # parameter-level checks
    if delta <= 0.0 or alpha <= 0.0 or abs(beta) >= alpha:
        raise ValueError("Invalid NIG params: require delta>0, alpha>0, and |beta|<alpha.")
    if alpha < 0.5:
        raise ValueError("Theta existence requires alpha >= 0.5.")
    bound = delta * np.sqrt(max(2.0 * alpha - 1.0, 0.0))
    if abs(mu) > bound:
        raise ValueError("Theta existence requires |mu| <= delta*sqrt(2*alpha-1).")

    # vectorized theta computation over r_f
    r_f_arr = np.asarray(r_f, dtype=float)
    num = mu - r_f_arr

    inside = (4.0 * (alpha ** 2) * (delta ** 2)) / ((num ** 2) + (delta ** 2))
    rad = inside - 1.0

    theta = np.full_like(r_f_arr, np.nan, dtype=float)

    ok = np.isfinite(r_f_arr) & np.isfinite(rad) & (rad >= 0.0)
    theta[ok] = -beta - 0.5 - (num[ok] / (2.0 * delta)) * np.sqrt(rad[ok])

    # nig_call_price uses beta_minus = beta + theta and beta_plus = beta + theta + 1
    # already present downstream
    if enforce_pricing:
        ok2 = ok & (np.abs(beta + theta) < alpha) & (np.abs(beta + theta + 1.0) < alpha)
        theta[~ok2] = np.nan
        ok = ok2

    if warn and (not np.all(ok)):
        n_bad = int(np.size(ok) - np.count_nonzero(ok))
        warnings.warn(
            f"update_theta: produced {n_bad} NaN(s) due to infeasible theta (rate/path or pricing constraints).",
            RuntimeWarning,
            stacklevel=2,
        )

    # return scalar if scalar input
    if np.isscalar(r_f):
        return float(theta.reshape(-1)[0])
    return theta


# Equity as call-option formula (as in Merton Model)
def nig_call_price(
    A: float,
    L_face: float,
    r: float,
    T: float,
    params: Dict[str, float],
    *,
    discounting: str = "continuous",
    theta_override: Optional[float] = None,
) -> float:
    """
    NIG equity price per Jovan & Ahčan Eq. (26):

      E_t = A_t * P^{β+θ+1}(X >= ln(L/A)) - L*e^{-rT} * P^{β+θ}(X >= ln(L/A))

    where X ~ NIG(α, β_shift, δ*T, μ*T). (Eq. 21 scaling)

    Notes for A+C(+triggers):
      - Prefer passing theta_override (computed once per day) for speed/stability.
      - Returns np.nan if parameters/feasibility do not support pricing.
    """
    if A <= 0.0 or L_face <= 0.0 or T <= 0.0:
        return np.nan

    alpha = float(params.get("alpha", 0.0))
    beta = float(params.get("beta1", 0.0))
    delta = float(params.get("delta", 0.0))
    mu = float(params.get("beta0", 0.0))

    if delta <= 0.0 or alpha <= 0.0:
        return np.nan

    theta = float(theta_override) if theta_override is not None else float(params.get("theta", np.nan))
    if not np.isfinite(theta):
        return np.nan

    x0 = np.log(L_face / A)

    # Lévy scaling (Eq. 21): delta_T, mu_T
    scale = delta * T
    loc = mu * T

    beta_plus = beta + theta + 1.0
    beta_minus = beta + theta

    # NIG validity: |beta_shift| < alpha
    if abs(beta_plus) >= alpha or abs(beta_minus) >= alpha:
        return np.nan

    a = alpha * scale
    b_plus = beta_plus * scale
    b_minus = beta_minus * scale

    tail_plus = norminvgauss.sf(x0, a=a, b=b_plus, loc=loc, scale=scale)
    tail_minus = norminvgauss.sf(x0, a=a, b=b_minus, loc=loc, scale=scale)

    if discounting == "continuous":
        L_disc = L_face * np.exp(-r * T)
    elif discounting == "simple":
        L_disc = L_face / (1.0 + r * T)
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
    A_prev: Optional[float] = None,
    bracket_mult: float = 2.5,
    A_min_factor: float = 1e-6,
    A_max_factor: float = 50.0,
    tol: float = 1e-8,
    max_iter: int = 80,
    on_fail: str = "nan",  # "nan", "fallback", "raise"
    warn: bool = True,
) -> float:
    """
    Solve E_obs = nig_call_price(A, ...) for A.

      - supports warm-starting via A_prev
      - returns NaN on infeasibility (default) so triggers can react
    """

    # Input sanity
    if E_obs <= 0.0 or L_face <= 0.0 or T <= 0.0:
        msg = "invert_nig_call_price: E_obs, L_face, T must be positive."
        if on_fail == "raise":
            raise ValueError(msg)
        if warn:
            warnings.warn(msg + " Returning NaN.", RuntimeWarning, stacklevel=2)
        return np.nan

    # Compute theta once
    theta = update_theta(params, float(r), warn=False)  # warn handled here
    if not np.isfinite(theta):
        msg = "invert_nig_call_price: theta infeasible -> cannot price/invert. Returning NaN."
        if on_fail == "raise":
            raise ValueError(msg)
        if warn:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return np.nan

    def f(A: float) -> float:
        val = nig_call_price(
            A=A,
            L_face=L_face,
            r=r,
            T=T,
            params=params,
            discounting=discounting,
            theta_override=float(theta),
        )
        if not np.isfinite(val):
            return np.nan
        return val - E_obs

    # Initial bracket
    if A_prev is not None and np.isfinite(A_prev) and A_prev > 0.0:
        A_min = max(1e-12, A_prev / bracket_mult)
        A_max = A_prev * bracket_mult
    else:
        A_min = max(E_obs, A_min_factor * L_face)
        A_max = A_max_factor * (E_obs + L_face)

    # Try to get finite endpoint values
    f_min, f_max = f(A_min), f(A_max)

    # If endpoints not finite, expand a bit
    if not (np.isfinite(f_min) and np.isfinite(f_max)):
        ok = False
        for i in range(1, 10):
            factor = 2.0 ** i
            A_min_try = max(1e-12, A_min / factor)
            A_max_try = A_max * factor
            f_min_try, f_max_try = f(A_min_try), f(A_max_try)
            if np.isfinite(f_min_try) and np.isfinite(f_max_try):
                A_min, A_max = A_min_try, A_max_try
                f_min, f_max = f_min_try, f_max_try
                ok = True
                break
        if not ok:
            msg = "invert_nig_call_price: could not find finite bracket endpoints. Returning NaN."
            if on_fail == "raise":
                raise ValueError(msg)
            if warn:
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            return np.nan if on_fail == "nan" else float(E_obs + L_face)

    # If sign does not straddle root, expand bracket
    if f_min * f_max > 0.0:
        for i in range(1, 18):
            factor = 2.0 ** i
            A_min_ext = max(1e-12, A_min / factor)
            A_max_ext = A_max * factor
            f_min_ext, f_max_ext = f(A_min_ext), f(A_max_ext)
            if np.isfinite(f_min_ext) and np.isfinite(f_max_ext) and (f_min_ext * f_max_ext <= 0.0):
                A_min, A_max = A_min_ext, A_max_ext
                f_min, f_max = f_min_ext, f_max_ext
                break
        else:
            msg = "invert_nig_call_price: failed to bracket root (no sign change). Returning NaN."
            if on_fail == "raise":
                raise ValueError(msg)
            if warn:
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            return np.nan if on_fail == "nan" else float(E_obs + L_face)

    # Root solve
    try:
        return float(brentq(f, A_min, A_max, xtol=tol, maxiter=max_iter))
    except Exception as e:
        msg = f"invert_nig_call_price: brentq failed ({type(e).__name__}: {e}). Returning NaN."
        if on_fail == "raise":
            raise
        if warn:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return np.nan if on_fail == "nan" else float(E_obs + L_face)


# E-step: infer asset path on the training window by inverting the NIG call price formula day-by-day
def get_asset_path(
    params: Dict[str, float],
    rf_series: np.ndarray,         # daily risk-free rates aligned with E/L/dates
    dates: np.ndarray,
    E_series: np.ndarray,
    L_face_series: np.ndarray,     # liabilities proxy (strike), can be time-varying
    start_date=None,
    end_date=None,
    *,
    daycount: int = 250,
    discounting: str = "continuous",
    # --- mode switches ---
    tau_mode: Literal["one_year", "paper"] = "one_year",
    liability_mode: Literal["timevarying", "paper_const_end"] = "timevarying",
    # --- speed/stability ---
    warm_start: bool = True,
    A0: Optional[float] = None,     # optional warm start for the first point in the slice
    warn: bool = True,
) -> np.ndarray:
    """
    Implied asset path by inverting the NIG equity-call relation day-by-day.

    Default (A+C(+triggers)) behaviour:
      - tau_mode="one_year": uses T=1.0 for every day (1Y-consistent state A^(1Y)_t)
      - liability_mode="timevarying": uses L_t for every day

    Paper replication behaviour:
      - tau_mode="paper": uses tau_t = 1 + (n-1-t)/daycount
      - liability_mode="paper_const_end": uses constant L = L_end (window end)

    Returns:
      - A_path aligned to the selected date slice, with np.nan where inversion infeasible.
    """

    dates = np.asarray(dates)
    E_series = np.asarray(E_series, dtype=float)
    L_face_series = np.asarray(L_face_series, dtype=float)
    rf_series = np.asarray(rf_series, dtype=float)

    if not (len(dates) == len(E_series) == len(L_face_series) == len(rf_series)):
        raise ValueError("dates, E_series, L_face_series, rf_series must have the same length")

    # inclusive date slice
    mask = np.ones(len(dates), dtype=bool)
    if start_date is not None:
        mask &= (dates >= start_date)
    if end_date is not None:
        mask &= (dates <= end_date)

    E = E_series[mask]
    r = rf_series[mask]
    L = L_face_series[mask]
    n = len(E)

    if n < 3:
        raise ValueError("Training window too short")

    # --- define tau_t ---
    h = 1.0 / daycount
    if tau_mode == "one_year":
        tau_vec = np.full(n, 1.0, dtype=float)
    elif tau_mode == "paper":
        # end has tau=1y, earlier have >1y
        tau_vec = 1.0 + (np.arange(n - 1, -1, -1) * h)
    else:
        raise ValueError("tau_mode must be 'one_year' or 'paper'")

    # --- define liabilities used in inversion ---
    if liability_mode == "timevarying":
        L_used = L
    elif liability_mode == "paper_const_end":
        # constant L at evaluation date (window end)
        L_end = float(L[-1])
        if not (np.isfinite(L_end) and L_end > 0):
            raise ValueError("Invalid L_face at end of window")
        L_used = np.full(n, L_end, dtype=float)
    else:
        raise ValueError("liability_mode must be 'timevarying' or 'paper_const_end'")

    A_path = np.full(n, np.nan, dtype=float)

    # warm start state
    A_prev = float(A0) if (A0 is not None and np.isfinite(A0) and A0 > 0.0) else None

    # counters for a single summary warning
    n_theta_nan = 0
    n_inv_nan = 0

    for t in range(n):
        E_t = float(E[t])
        L_t = float(L_used[t])
        r_t = float(r[t])
        T_t = float(tau_vec[t])

        # compute theta on the fly (vectorized update_theta exists, but here we want per-day control)
        theta_t = update_theta(params, r_t, warn=False)  # returns float; may be nan
        if not np.isfinite(theta_t):
            n_theta_nan += 1
            continue

        # pass theta via params (or use theta_override in nig_call_price; either is fine)
        params_t = dict(params)
        params_t["theta"] = float(theta_t)

        A_t = invert_nig_call_price(
            E_obs=E_t,
            L_face=L_t,
            r=r_t,
            T=T_t,
            params=params_t,
            discounting=discounting,
            A_prev=(A_prev if warm_start else None),
            warn=False,          # we warn once at the end
            on_fail="nan",       # return np.nan on infeasibility (good for triggers)
        )

        if np.isfinite(A_t) and A_t > 0.0:
            A_path[t] = float(A_t)
            A_prev = float(A_t)  # only update warm start when we have a valid point
        else:
            n_inv_nan += 1

    if warn and (n_theta_nan + n_inv_nan) > 0:
        warnings.warn(
            f"get_asset_path: produced NaNs (theta infeasible: {n_theta_nan}, inversion failed: {n_inv_nan}) "
            f"over {n} points. Consider triggering recalibration when this rate increases.",
            RuntimeWarning,
            stacklevel=2,
        )

    return A_path
