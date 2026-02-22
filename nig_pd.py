import numpy as np
import pandas as pd
from scipy.stats import norminvgauss
from nig_apath import NIGParams


# -------------------------------------------------------------
# 1️⃣ Correct NIG scaling over horizon T
# -------------------------------------------------------------
def pd_terminal_nig_weekly(
    A_t: float,
    L_t: float,
    p: NIGParams,
    *,
    horizon_weeks: float = 52.0,
) -> float:
    """
    1Y terminal PD under NIG with weekly-calibrated params.

    Assumes:
        Weekly increments follow NIG(alpha, beta, delta, mu)

    Over T weeks:
        delta_T = delta * T
        mu_T    = mu * T
        alpha, beta unchanged
    """
    if (A_t <= 0) or (L_t <= 0):
        return np.nan

    p.validate()

    x = float(np.log(L_t / A_t))

    T = float(horizon_weeks)

    delta_T = p.delta * T
    mu_T    = p.mu    * T

    # SciPy parametrization:
    # norminvgauss(a, b, loc, scale)
    # corresponds to:
    # a = alpha * delta
    # b = beta  * delta
    # scale = delta

    a = p.alpha * delta_T
    b = p.beta  * delta_T

    pd_val = norminvgauss.cdf(x, a=a, b=b, loc=mu_T, scale=delta_T)

    return float(np.clip(pd_val, 0.0, 1.0))


# -------------------------------------------------------------
# 2️⃣ Build parameter schedule (forward-fill)
# -------------------------------------------------------------
def build_param_schedule(
    weekly_dates: pd.Series,
    param_updates: pd.DataFrame,
) -> dict:
    """
    Converts quarterly EM updates into a forward-filled param schedule.

    Returns:
        dict {date -> NIGParams}
    """
    param_updates = param_updates.sort_values("date")

    p_cache = {}

    for _, row in param_updates.iterrows():
        p_cache[pd.to_datetime(row["date"])] = NIGParams(
            alpha=row["alpha"],
            beta=row["beta"],
            delta=row["delta"],
            mu=row["mu"],
        )

    return p_cache


def pick_params_for_date(
    p_cache: dict,
    d: pd.Timestamp,
    *,
    fallback: NIGParams,
) -> NIGParams:
    if not p_cache:
        return fallback

    d = pd.to_datetime(d)

    keys = sorted(p_cache.keys())
    eligible = [k for k in keys if k <= d]

    if not eligible:
        return fallback

    return p_cache[eligible[-1]]


# -------------------------------------------------------------
# 3️⃣ Weekly PD computation (NO inversion)
# -------------------------------------------------------------
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
        - weekly liabilities L
        - quarterly parameter updates

    No re-inversion.
    """

    df = assets_weekly.sort_values("date").copy()

    if "A_hat" not in df.columns:
        raise ValueError("assets_weekly must contain 'A_hat' column.")

    if "L" not in df.columns:
        raise ValueError("assets_weekly must contain 'L' column.")

    p_cache = build_param_schedule(df["date"], param_updates)

    out = []

    for _, row in df.iterrows():
        d = pd.to_datetime(row["date"])
        A_t = float(row["A_hat"])
        L_t = float(row["L"])

        p_t = pick_params_for_date(p_cache, d, fallback=p0)

        pd_1y = pd_terminal_nig_weekly(
            A_t,
            L_t,
            p_t,
            horizon_weeks=horizon_weeks,
        )

        out.append({
            "gvkey": str(gvkey),
            "date": d,
            "A_hat": A_t,
            "L": L_t,
            "alpha": p_t.alpha,
            "beta": p_t.beta,
            "delta": p_t.delta,
            "mu": p_t.mu,
            "PD_1y": pd_1y,
        })

    return pd.DataFrame(out).sort_values("date").reset_index(drop=True)