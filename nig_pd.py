import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, TypeAlias
from scipy.stats import norminvgauss
import warnings

ArrayLike: TypeAlias = Union[float, np.ndarray]


def _compute_pd_with_beta(
    A0: ArrayLike,
    L: ArrayLike,
    T: ArrayLike,
    alpha: float,
    beta: ArrayLike,     # <-- allow vector beta (e.g., beta1 + theta_t)
    beta0: float,        # mu
    delta: float,
    *,
    on_fail: str = "nan",   # "nan" | "raise"
    warn: bool = False,
    clip01: bool = True,
) -> ArrayLike:
    """
    Computes PD = P(AT <= L) by evaluating the NIG CDF at x = log(L/A0),
    where log(AT/A0) ~ NIG(alpha, beta, delta*T, mu*T) with Lévy scaling. :contentReference[oaicite:2]{index=2}
    """

    # scalar parameter checks (alpha, delta, mu)
    if not (np.isfinite(alpha) and np.isfinite(delta) and np.isfinite(beta0)):
        msg = "_compute_pd_with_beta: non-finite scalar parameter(s) alpha/delta/mu."
        if on_fail == "raise":
            raise ValueError(msg)
        if warn:
            warnings.warn(msg + " Returning NaN(s).", RuntimeWarning, stacklevel=2)
        return np.nan

    if (alpha <= 0.0) or (delta <= 0.0):
        msg = "_compute_pd_with_beta: invalid scalar params (need alpha>0, delta>0)."
        if on_fail == "raise":
            raise ValueError(msg)
        if warn:
            warnings.warn(msg + " Returning NaN(s).", RuntimeWarning, stacklevel=2)
        return np.nan

    # broadcast all inputs INCLUDING beta
    A0 = np.asarray(A0, dtype=float)
    L = np.asarray(L, dtype=float)
    T = np.asarray(T, dtype=float)
    beta_arr = np.asarray(beta, dtype=float)

    out_shape = np.broadcast(A0, L, T, beta_arr).shape
    A0b = np.broadcast_to(A0, out_shape)
    Lb  = np.broadcast_to(L, out_shape)
    Tb  = np.broadcast_to(T, out_shape)
    betab = np.broadcast_to(beta_arr, out_shape)

    pd_out = np.full(out_shape, np.nan, dtype=float)

    # elementwise feasibility:
    # - inputs positive/finite
    # - NIG existence: |beta| < alpha (must hold pointwise if beta is time-varying)
    ok = (
        np.isfinite(A0b) & np.isfinite(Lb) & np.isfinite(Tb) & np.isfinite(betab) &
        (A0b > 0.0) & (Lb > 0.0) & (Tb > 0.0) &
        (np.abs(betab) < alpha)
    )

    if not np.any(ok):
        if warn:
            warnings.warn("_compute_pd_with_beta: no valid points after feasibility checks.", RuntimeWarning, stacklevel=2)
        return float(pd_out) if pd_out.size == 1 else pd_out

    x_thresh = np.log(Lb[ok] / A0b[ok])

    # Lévy scaling over horizon T: delta_T = delta*T, mu_T = mu*T :contentReference[oaicite:3]{index=3}
    scale = np.maximum(delta * Tb[ok], 1e-12)
    loc = beta0 * Tb[ok]

    a = alpha * scale
    b = betab[ok] * scale

    pd_vals = norminvgauss.cdf(x_thresh, a=a, b=b, loc=loc, scale=scale)
    if clip01:
        pd_vals = np.clip(pd_vals, 0.0, 1.0)

    pd_out[ok] = pd_vals
    return float(pd_out) if pd_out.size == 1 else pd_out


ArrayLike: TypeAlias = Union[float, np.ndarray]


# Physical-measure PD: uses beta = beta1 (no Esscher tilt)
def compute_pd_physical(
    A0: ArrayLike,
    L: ArrayLike,
    T: ArrayLike,
    params: Dict[str, float],
    *,
    warn: bool = False,
) -> ArrayLike:
    alpha = float(params.get("alpha", np.nan))
    beta1 = float(params.get("beta1", np.nan))
    beta0 = float(params.get("beta0", np.nan))  # mu
    delta = float(params.get("delta", np.nan))

    return _compute_pd_with_beta(
        A0=A0, L=L, T=T,
        alpha=alpha, beta=beta1,
        beta0=beta0, delta=delta,
        on_fail="nan",
        warn=warn,
    )


# Risk-neutral PD: uses beta = beta1 + theta (Esscher tilt)
def compute_pd_risk_neutral(
    A0: ArrayLike,
    L: ArrayLike,
    T: ArrayLike,
    params: Dict[str, float],
    *,
    theta: Optional[float] = None,
    warn: bool = False,
) -> ArrayLike:
    """
    Risk-neutral PD: beta is shifted by the Esscher tilt theta. :contentReference[oaicite:1]{index=1}
    You must provide theta (argument) or have it in params["theta"].
    """

    alpha = float(params.get("alpha", np.nan))
    beta1 = float(params.get("beta1", np.nan))
    beta0 = float(params.get("beta0", np.nan))  # mu
    delta = float(params.get("delta", np.nan))

    theta_val = float(params.get("theta", np.nan)) if theta is None else float(theta)

    if not np.isfinite(theta_val):
        if warn:
            warnings.warn(
                "compute_pd_risk_neutral: theta missing/non-finite; returning NaN (do not default to 0).",
                RuntimeWarning,
                stacklevel=2,
            )
        return np.nan if np.isscalar(A0) and np.isscalar(L) and np.isscalar(T) else np.full(np.broadcast(A0, L, T).shape, np.nan)

    return _compute_pd_with_beta(
        A0=A0, L=L, T=T,
        alpha=alpha, beta=(beta1 + theta_val),
        beta0=beta0, delta=delta,
        on_fail="nan",
        warn=warn,
    )


def one_year_pd_timeseries(out: dict, L_face_series_full: np.ndarray) -> pd.DataFrame:
    params = out["params"]
    dates_win = np.asarray(out["dates_win"])
    idx_win = np.asarray(out["idx_win"], dtype=int)
    A_win = np.asarray(out["A_win"], dtype=float)
    theta_win = np.asarray(out["theta_win"], dtype=float)

    L_face_series_full = np.asarray(L_face_series_full, dtype=float)

    # align L_t to the window via idx_win (idx_win are indices in the full firm series)
    L_t = np.full(len(idx_win), np.nan, dtype=float)
    ok_idx = (idx_win >= 0) & (idx_win < len(L_face_series_full))
    L_t[ok_idx] = L_face_series_full[idx_win[ok_idx]]

    # extract params once
    alpha = float(params.get("alpha", np.nan))
    beta1 = float(params.get("beta1", np.nan))
    beta0 = float(params.get("beta0", np.nan))   # mu
    delta = float(params.get("delta", np.nan))

    # valid days mask (asset inversion may produce NaNs)
    valid = (
        np.isfinite(A_win) & (A_win > 0.0) &
        np.isfinite(L_t)  & (L_t  > 0.0)
    )

    pd_p = np.full(len(idx_win), np.nan, dtype=float)
    pd_q = np.full(len(idx_win), np.nan, dtype=float)

    if np.any(valid):
        # Physical PD uses beta = beta1 (no Esscher tilt)
        pd_p[valid] = _compute_pd_with_beta(
            A0=A_win[valid], L=L_t[valid], T=1.0,
            alpha=alpha, beta=beta1,
            beta0=beta0, delta=delta,
            on_fail="nan",
        )

        # Risk-neutral PD uses beta = beta1 + theta_t
        beta_q = beta1 + theta_win
        pd_q[valid] = _compute_pd_with_beta(
            A0=A_win[valid], L=L_t[valid], T=1.0,
            alpha=alpha, beta=beta_q[valid],
            beta0=beta0, delta=delta,
            on_fail="nan",
        )

    return pd.DataFrame({
        "date": dates_win,
        "A_hat": A_win,
        "theta": theta_win,
        "L_proxy": L_t,
        "PD_physical": pd_p,
        "PD_risk_neutral": pd_q,
    })


def compute_pd_physical_vec(A0, L, T, alpha, beta1, delta, beta0, warn=False):
    """
    Vectorized physical PD under NIG.
    Inputs can be arrays (same length).
    Uses the same mapping as the scalar compute_pd_physical:
      X_T ~ NIG(alpha, beta1, delta*T, beta0*T)
      PD = P( X_T <= log(L/A0) )
    """
    A0 = np.asarray(A0, dtype=float)
    L = np.asarray(L, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    delta = np.asarray(delta, dtype=float)
    beta0 = np.asarray(beta0, dtype=float)

    # feasibility mask
    ok = (
        np.isfinite(A0) & (A0 > 0) &
        np.isfinite(L) & (L > 0) &
        np.isfinite(alpha) & (alpha > 0) &
        np.isfinite(delta) & (delta > 0) &
        np.isfinite(beta1) & (np.abs(beta1) < alpha) &
        np.isfinite(beta0)
    )

    out = np.full(A0.shape, np.nan, dtype=float)
    if not np.any(ok):
        return out

    x = np.log(L[ok] / A0[ok])

    # SciPy parameterization: (a, b, loc, scale)
    # where a = alpha*scale, b = beta*scale
    scale = delta[ok] * T
    loc = beta0[ok] * T
    a = alpha[ok] * scale
    b = beta1[ok] * scale

    out[ok] = norminvgauss.cdf(x, a=a, b=b, loc=loc, scale=scale)
    return out
