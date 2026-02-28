import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.optimize import brentq


# Helpers
def norm_cdf(x):
    return ndtr(np.asarray(x, dtype=float))


def build_weekly_calendar_from_panel(g: pd.DataFrame, *, week_ending: str = "W-FRI") -> pd.DatetimeIndex:
    """
    Build a weekly inversion calendar from the firm's (or window's) DAILY panel.
    Uses 'W-FRI' periods by default and selects the LAST available trading date in each week.
    """
    g = g.sort_values("date").copy()
    g["week"] = g["date"].dt.to_period(week_ending)
    week_ends = g.groupby("week")["date"].max().sort_values()
    return pd.DatetimeIndex(week_ends.values)


# Merton pricing + inversion
def merton_equity_from_assets(V, B, r, T, sigmaV):
    """
    Equity value as a European call on firm assets (no payouts):
      E = V N(d1) - B e^{-rT} N(d2)
    """
    eps = 1e-14
    V = float(max(V, eps))
    B = float(max(B, eps))
    T = float(max(T, eps))
    sigmaV = float(max(sigmaV, eps))

    sig_sqrtT = max(sigmaV * np.sqrt(T), eps)
    d1 = (np.log(V / B) + (r + 0.5 * sigmaV**2) * T) / sig_sqrtT
    d2 = d1 - sig_sqrtT
    Nd1 = norm_cdf(d1)
    Nd2 = norm_cdf(d2)
    E_model = V * Nd1 - B * np.exp(-r * T) * Nd2
    return float(E_model), float(d1), float(d2), float(Nd1)


def _bracket_root_for_V(E, B, r, T, sigmaV, V_prev=None, max_expand=60):
    """
    Find bracket [lo, hi] such that f(lo)<=0 and f(hi)>=0 for f(V)=E_model(V)-E.
    Faster/more robust: choose lo ~ 0 so f(lo) ~ -E < 0; only expand hi.
    """
    eps = 1e-14
    E = float(max(E, eps))

    lo = 1e-12  # ensures E_model(lo) ~ 0 => f(lo) ~ -E < 0

    if V_prev is not None and np.isfinite(V_prev) and V_prev > 0:
        hi = max(1.5 * V_prev, E + B, 2.0 * E, 1.0)
    else:
        hi = max(E + B, 2.0 * E, 1.0)

    def f(V):
        Emod, *_ = merton_equity_from_assets(V, B, r, T, sigmaV)
        return Emod - E

    flo = f(lo)
    fhi = f(hi)

    # expand hi until fhi >= 0
    if fhi < 0:
        for _ in range(max_expand):
            hi *= 2.0
            fhi = f(hi)
            if fhi >= 0:
                break

    return lo, hi, flo, fhi


# Get weekly asset path
def invert_asset_one_week_merton(E_obs, B, r, T, sigmaV, *, V_prev=None, tol=1e-6, maxiter=200):
    """
    Invert V (assets) from Merton equity equation given sigmaV (constant in window).
    Returns: V_hat, d1, d2
    """
    if not (np.isfinite(E_obs) and np.isfinite(B) and np.isfinite(r) and np.isfinite(T) and np.isfinite(sigmaV)):
        raise RuntimeError("nonfinite input to inversion")
    if E_obs <= 0 or B <= 0 or T <= 0 or sigmaV <= 0:
        raise RuntimeError("invalid sign input to inversion")

    lo, hi, flo, fhi = _bracket_root_for_V(E_obs, B, r, T, sigmaV, V_prev=V_prev)

    if not (np.isfinite(flo) and np.isfinite(fhi)) or (flo > 0) or (fhi < 0):
        raise RuntimeError("could not bracket root for V")

    def f(V):
        Emod, *_ = merton_equity_from_assets(V, B, r, T, sigmaV)
        return Emod - E_obs

    V_hat = float(brentq(f, lo, hi, xtol=tol, rtol=tol, maxiter=maxiter))
    Emod, d1, d2, _ = merton_equity_from_assets(V_hat, B, r, T, sigmaV)

    # light sanity: call value cannot exceed underlying
    if Emod > V_hat + 1e-8:
        raise RuntimeError("sanity check failed: E_model > V")

    return V_hat, d1, d2


def invert_assets_weekly_for_firm_merton(
    g: pd.DataFrame,
    *,
    sigmaV: float,
    week_ending: str = "W-FRI",
    E_col: str = "E",
    B_col: str = "B_used",
    r_col: str = "r",
    T_col: str = "T",
    dates: pd.DatetimeIndex | None = None,      # NEW
    g_indexed: pd.DataFrame | None = None,      # NEW (date-indexed, deduped)
) -> pd.DataFrame:
    """
    DAILY input -> WEEKLY output (last trading day each week):
    returns weekly V_hat path + d1,d2 + logV + dlogV.
    """
    #NEW: allow cached preprocessing (saves a lot of time inside sigma-iterations)
    if g_indexed is None:
        g = g.sort_values("date").copy()
        g["date"] = pd.to_datetime(g["date"])

        for c in [E_col, B_col, r_col, T_col]:
            g[c] = pd.to_numeric(g[c], errors="coerce")

        if dates is None:
            dates = build_weekly_calendar_from_panel(g, week_ending=week_ending)

        g = g.dropna(subset=["date"]).sort_values("date")
        g = g.groupby("date", as_index=False).last().set_index("date")
    else:
        g = g_indexed
        if dates is None:
            # fallback: if user didn't provide dates, compute from indexed dates
            dates = pd.DatetimeIndex(g.index)

    results = []
    V_prev = None

    for d in dates:
        if d not in g.index:
            continue

        row = g.loc[d]
        E_obs = float(row[E_col])
        B     = float(row[B_col])
        r     = float(row[r_col])
        T     = float(row[T_col])

        V_hat, d1, d2 = invert_asset_one_week_merton(
            E_obs, B, r, T, sigmaV,
            V_prev=V_prev
        )

        results.append((d, E_obs, B, r, T, V_hat, d1, d2))
        V_prev = V_hat

    out = pd.DataFrame(
        results,
        columns=["date", "E", "B", "r", "T", "V_hat", "d1", "d2"]
    ).sort_values("date")

    out["logV"] = np.log(out["V_hat"])
    out["dlogV"] = out["logV"].diff()
    return out


# 1 sigma per 2-year window (weekly data)
def _sigmaV_init_guess_from_equity(window_daily: pd.DataFrame, *, E_col="E", B_col="B_used", sigmaE_col=None):
    """
    Optional: if sigma_E exists in daily data, we use a KMV-ish mapping to seed sigmaV.
    Otherwise fallback to 0.20.
    """
    E = pd.to_numeric(window_daily[E_col], errors="coerce").values.astype(float)
    B = pd.to_numeric(window_daily[B_col], errors="coerce").values.astype(float)
    V0 = np.maximum(E + B, 1e-8)

    if sigmaE_col is not None and sigmaE_col in window_daily.columns:
        sE = pd.to_numeric(window_daily[sigmaE_col], errors="coerce").values.astype(float)
        x = sE * (E / V0)
        x = x[np.isfinite(x) & (x > 0)]
        if x.size:
            return float(np.clip(np.median(x), 1e-4, 3.0))

    return 0.20


def calibrate_sigmaV_window_weekly_merton(
    window_daily: pd.DataFrame,
    *,
    week_ending: str = "W-FRI",
    ann_factor: float = 52.0,      # weekly annualization
    max_iter: int = 30,
    tol_sigma: float = 1e-4,
    sigmaV_init: float | None = None,
    E_col: str = "E",
    B_col: str = "B_used",
    r_col: str = "r",
    T_col: str = "T",
    sigmaE_col: str = "sigma_E",   # optional seed only
):
    """
    Estimate ONE constant sigmaV (annualized) for this 2-year window,
    using WEEKLY implied asset returns.
    Returns: sigmaV_hat, weekly_df, ok, msg
    """
    w = window_daily.copy()
    w["date"] = pd.to_datetime(w["date"])

    # basic numeric cleaning
    for c in [E_col, B_col, r_col, T_col]:
        w[c] = pd.to_numeric(w[c], errors="coerce")

    # NEW: cache weekly calendar + date-indexed daily panel ONCE
    dates_cached = build_weekly_calendar_from_panel(w, week_ending=week_ending)
    w_indexed = (
        w.dropna(subset=["date"])
         .sort_values("date")
         .groupby("date", as_index=False).last()
         .set_index("date")
    )

    # NEW: use weekly endpoints for the sigmaE-based seed (no change to seed function)
    w_weekly_end = w_indexed.reindex(dates_cached).reset_index()

    # initialize sigmaV
    if sigmaV_init is not None and np.isfinite(sigmaV_init) and sigmaV_init > 0:
        sigmaV = float(np.clip(sigmaV_init, 1e-4, 3.0))
    else:
        sigmaV = _sigmaV_init_guess_from_equity(
            w_weekly_end, E_col=E_col, B_col=B_col, sigmaE_col=sigmaE_col
        )

    ok = True
    msg = "ok"
    weekly = None

    for it in range(max_iter):
        try:
            # NEW (A): pass cached dates + indexed panel to avoid recomputation
            weekly = invert_assets_weekly_for_firm_merton(
                w,
                sigmaV=sigmaV,
                week_ending=week_ending,
                E_col=E_col,
                B_col=B_col,
                r_col=r_col,
                T_col=T_col,
                dates=dates_cached,
                g_indexed=w_indexed,
            )
        except Exception as e:
            ok = False
            msg = f"inversion_fail(it={it}):{type(e).__name__}:{str(e)[:120]}"
            break

        dlogV = weekly["dlogV"].values
        dlogV = dlogV[np.isfinite(dlogV)]
        if dlogV.size < 2:
            ok = False
            msg = "too_few_weekly_returns"
            break

        sigma_new = float(np.std(dlogV, ddof=1) * np.sqrt(ann_factor))
        sigma_new = float(np.clip(sigma_new, 1e-4, 3.0))

        if abs(sigma_new - sigmaV) < tol_sigma:
            sigmaV = sigma_new
            msg = f"converged(it={it})"
            break

        sigmaV = sigma_new

        if it == max_iter - 1:
            ok = False
            msg = "max_iter_reached"

    if weekly is None:
        weekly = pd.DataFrame(columns=["date", "E", "B", "r", "T", "V_hat", "d1", "d2", "logV", "dlogV"])

    weekly = weekly.copy()
    weekly["sigma_V_win"] = sigmaV
    weekly["window_ok"] = bool(ok)
    weekly["window_msg"] = msg

    return sigmaV, weekly, ok, msg


# Quarterly-rolling 2-year windows (daily input -> weekly output)
def calibrate_merton_quarterly_rolling_weekly_output(
    df_daily: pd.DataFrame,
    *,
    date_col: str = "date",
    week_ending: str = "W-FRI",
    window_years: int = 2,
    ann_factor: float = 52.0,
    min_weekly_obs: int = 60,     # require enough weekly points inside window
    warm_start_sigma: bool = True,
    # columns
    E_col: str = "E",
    B_col: str = "B",
    r_col: str = "r",
    T_col: str = "T",
    T_default: float = 1.0,
    B_scale: float = 1.0,
    sigmaE_col: str = "sigma_E",  # optional seed only
):
    """
    For each firm (gvkey):
      For each quarter end:
        take DAILY window (end_date-2y, end_date]
        estimate ONE sigma_V_win using WEEKLY implied V returns
        output weekly series for that window

    Returns:
      panel_weekly_out: weekly rows per window (has gvkey, window_start, window_end)
      windows_out: one row per window with sigma_V_win and status
    """
    df = df_daily.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # ensure T exists
    if T_col not in df.columns:
        df[T_col] = float(T_default)

    # numeric coercion
    for c in [E_col, B_col, r_col, T_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # scaled debt column like your previous code
    df["B_used"] = df[B_col].astype(float) * float(B_scale)
    df["B_scale_used"] = float(B_scale)

    df = df.sort_values(["gvkey", date_col]).reset_index(drop=True)

    panel_chunks = []
    win_rows = []

    for gvkey, gdf in df.groupby("gvkey", sort=False):
        gdf = gdf.sort_values(date_col).reset_index(drop=True)
        dates = gdf[date_col]

        if dates.isna().all():
            continue

        q_ends = pd.date_range(start=dates.min(), end=dates.max(), freq="Q")

        last_sigma = None

        for q_end in q_ends:
            # pick actual end_date as last available trading date <= quarter end
            mask_end = dates <= q_end
            if not mask_end.any():
                continue
            end_date = dates[mask_end].max()
            start_date = end_date - pd.DateOffset(years=window_years)

            window_daily = gdf[(dates > start_date) & (dates <= end_date)].copy()
            if window_daily.empty:
                continue

            sigma_seed = last_sigma if (warm_start_sigma and last_sigma is not None) else None

            sigmaV_hat, weekly_df, ok, msg = calibrate_sigmaV_window_weekly_merton(
                window_daily,
                week_ending=week_ending,
                ann_factor=ann_factor,
                sigmaV_init=sigma_seed,
                E_col=E_col,
                B_col="B_used",
                r_col=r_col,
                T_col=T_col,
                sigmaE_col=sigmaE_col if sigmaE_col in window_daily.columns else None,
            )

            # enforce min weekly obs
            if len(weekly_df) < min_weekly_obs:
                continue

            if ok:
                last_sigma = sigmaV_hat

            weekly_df = weekly_df.copy()
            weekly_df["gvkey"] = gvkey
            weekly_df["window_start"] = weekly_df["date"].iloc[0]
            weekly_df["window_end"] = weekly_df["date"].iloc[-1]

            panel_chunks.append(weekly_df)

            win_rows.append(
                {
                    "gvkey": gvkey,
                    "window_start": weekly_df["window_start"].iloc[0],
                    "window_end": weekly_df["window_end"].iloc[-1],
                    "n_weekly": len(weekly_df),
                    "sigma_V_win": sigmaV_hat,
                    "ok": ok,
                    "msg": msg,
                }
            )

    panel_weekly_out = pd.concat(panel_chunks, ignore_index=True) if panel_chunks else pd.DataFrame()
    windows_out = pd.DataFrame(win_rows)

    return panel_weekly_out, windows_out
