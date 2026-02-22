# nig_pd.py
import numpy as np
import pandas as pd
from scipy.stats import norminvgauss
from nig_apath import NIGParams


def pd_terminal_nig_weekly(
    A_t: float,
    L_t: float,
    p: NIGParams,
    *,
    horizon_weeks: float = 52.0,
) -> float:
    """
    1Y terminal PD under NIG with weekly-calibrated params.

    Weekly increments: X_1 ~ NIG(alpha, beta, delta, mu)
    Over T weeks (iid increments):
        delta_T = delta * T
        mu_T    = mu    * T
        alpha, beta unchanged

    SciPy norminvgauss(a,b,loc,scale) mapping:
        scale = delta
        a     = alpha * delta
        b     = beta  * delta
        loc   = mu
    """
    if (A_t <= 0) or (L_t <= 0):
        return np.nan

    p.validate()

    x = float(np.log(L_t / A_t))
    T = float(horizon_weeks)

    delta_T = p.delta * T
    mu_T = p.mu * T

    a = p.alpha * delta_T
    b = p.beta * delta_T

    # basic safety: scipy requires a>0 and |b|<a
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or abs(b) >= a:
        return np.nan

    pd_val = norminvgauss.cdf(x, a=a, b=b, loc=mu_T, scale=delta_T)
    return float(np.clip(pd_val, 0.0, 1.0))


def pd_weekly_one_firm(
    assets_weekly: pd.DataFrame,
    *,
    gvkey: str,
    param_updates: pd.DataFrame,
    p0: NIGParams,
    horizon_weeks: float = 52.0,
) -> pd.DataFrame:
    """
    Compute weekly 1Y PDs using:
      - weekly asset path (A_hat)
      - weekly liabilities (L)
      - quarterly parameter updates (param_updates)
    No re-inversion.

    assets_weekly must include: date, A_hat, L
    param_updates must include: date, alpha, beta, delta, mu
    """
    df = assets_weekly.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])

    need_cols = {"date", "A_hat", "L"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"assets_weekly missing columns: {sorted(missing)}")

    upd = param_updates.sort_values("date").copy()
    upd["date"] = pd.to_datetime(upd["date"])

    need_u = {"date", "alpha", "beta", "delta", "mu"}
    missing_u = need_u - set(upd.columns)
    if missing_u:
        raise ValueError(f"param_updates missing columns: {sorted(missing_u)}")

    # Build a forward-filled parameter schedule aligned to weekly dates:
    # for each weekly date, pick the most recent update <= date; if none, fall back to p0.
    sched = pd.merge_asof(
        df[["date"]].sort_values("date"),
        upd[["date", "alpha", "beta", "delta", "mu"]].sort_values("date"),
        on="date",
        direction="backward",
    )

    # Fill pre-first-update weeks with p0
    sched["alpha"] = sched["alpha"].fillna(p0.alpha)
    sched["beta"] = sched["beta"].fillna(p0.beta)
    sched["delta"] = sched["delta"].fillna(p0.delta)
    sched["mu"] = sched["mu"].fillna(p0.mu)

    out = df.merge(sched, on="date", how="left")

    # Vectorized PD computation (same formula as pd_terminal_nig_weekly)
    A = out["A_hat"].to_numpy(float)
    L = out["L"].to_numpy(float)
    alpha = out["alpha"].to_numpy(float)
    beta = out["beta"].to_numpy(float)
    delta = out["delta"].to_numpy(float)
    mu = out["mu"].to_numpy(float)

    ok = np.isfinite(A) & np.isfinite(L) & (A > 0) & (L > 0)

    x = np.full(len(out), np.nan, float)
    x[ok] = np.log(L[ok] / A[ok])

    T = float(horizon_weeks)
    delta_T = delta * T
    mu_T = mu * T

    a = alpha * delta_T
    b = beta * delta_T

    # SciPy constraints: a>0, |b|<a
    ok2 = ok & np.isfinite(a) & np.isfinite(b) & (a > 0) & (np.abs(b) < a)

    pd_1y = np.full(len(out), np.nan, float)
    if np.any(ok2):
        pd_1y[ok2] = norminvgauss.cdf(
            x[ok2],
            a=a[ok2],
            b=b[ok2],
            loc=mu_T[ok2],
            scale=delta_T[ok2],
        )

    out["PD_1y"] = np.clip(pd_1y, 0.0, 1.0)

    out.insert(0, "gvkey", str(gvkey))
    return out[["gvkey", "date", "A_hat", "L", "alpha", "beta", "delta", "mu", "PD_1y"]].sort_values("date").reset_index(drop=True)