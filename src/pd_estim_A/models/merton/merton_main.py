import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.optimize import brentq
from pd_estim_A.models.merton.merton_pd import estimate_mu_from_weekly_implied_assets, merton_pd_rn_1y, merton_pd_physical_1y


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

    # call value cannot exceed underlying
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
    dates: pd.DatetimeIndex | None = None,
    g_indexed: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    DAILY input -> WEEKLY output (last trading day each week):
    returns weekly V_hat path + d1,d2 + logV + dlogV.
    """
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
        B = float(row[B_col])
        r = float(row[r_col])
        T = float(row[T_col])

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
    out["week"] = out["date"].dt.to_period(week_ending)

    week_ord = out["week"].astype("int64")
    is_consecutive = week_ord.diff().eq(1)

    out["dlogV"] = out["logV"].diff()
    out.loc[~is_consecutive, "dlogV"] = np.nan
    out = out.drop(columns="week")
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
    sigmaE_col: str = None,
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

    # cache weekly calendar + date-indexed daily panel once
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
            # pass cached dates + indexed panel to avoid recomputation
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

    # ensure final weekly path matches the final returned sigmaV
    if weekly is not None and (msg.startswith("converged") or msg == "max_iter_reached"):
        try:
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
            msg = f"final_inversion_fail:{type(e).__name__}:{str(e)[:120]}"
            weekly = None

    if weekly is None:
        weekly = pd.DataFrame(columns=["date", "E", "B", "r", "T", "V_hat", "d1", "d2", "logV", "dlogV"])

    weekly = weekly.copy()
    weekly["sigma_V_win"] = sigmaV
    weekly["window_ok"] = bool(ok)
    weekly["window_msg"] = msg

    return sigmaV, weekly, ok, msg


def run_merton_window_for_firm(
    g_firm_daily: pd.DataFrame,
    *,
    train_start,
    train_end,
    sigmaV_init: float | None = None,
    date_col: str = "date",
    week_ending: str = "W-FRI",
    ann_factor: float = 52.0,
    min_daily_rows: int = 10,
    min_weekly_obs: int | None = None,
    min_weekly_returns: int = 2,
    E_col: str = "E",
    B_col: str = "B",
    r_col: str = "r",
    T_col: str = "T",
    T_default: float = 1.0,
    B_scale: float = 1.0,
    sigmaE_col: str | None = None,
):
    """
    Run the classical Merton training step for ONE firm and ONE training window.
    """
    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)

    # Standardize input to a daily dataframe with an explicit date col
    g = g_firm_daily.copy()

    if date_col not in g.columns:
        if isinstance(g.index, pd.DatetimeIndex):
            g = g.reset_index()
            # rename the reset index column to date_col if needed
            if date_col not in g.columns:
                g = g.rename(columns={g.columns[0]: date_col})
        else:
            raise ValueError(
                f"Input must either contain a '{date_col}' column or have a DatetimeIndex."
            )

    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")

    # Optional firm identifier for diagnostics
    gvkey_val = None
    if "gvkey" in g.columns:
        vals = g["gvkey"].dropna().astype(str).unique()
        if len(vals) == 1:
            gvkey_val = vals[0]

    # Ensure T exists
    if T_col not in g.columns:
        g[T_col] = float(T_default)

    # Numeric coercion
    for c in [E_col, B_col, r_col, T_col]:
        g[c] = pd.to_numeric(g[c], errors="coerce")

    # Optional scaled debt column used by the training calibration
    g["B_used"] = pd.to_numeric(g[B_col], errors="coerce").astype(float) * float(B_scale)
    g["B_scale_used"] = float(B_scale)

    # Slice requested training window, then clean
    g_train = g.loc[(g[date_col] >= train_start) & (g[date_col] <= train_end)].copy()

    if g_train.empty:
        summary = {
            "gvkey": gvkey_val,
            "train_start_req": train_start,
            "train_end_req": train_end,
            "ok": False,
            "msg": "empty_training_slice",
            "sigma_hat": np.nan,
            "mu_hat_train": np.nan,
            "n_daily_train": 0,
            "n_weekly_train": 0,
            "n_weekly_returns": 0,
        }
        return summary, pd.DataFrame()

    g_train = (
        g_train.dropna(subset=[date_col, E_col, B_col, r_col, T_col, "B_used"])
               .query(f"{E_col} > 0 and {B_col} > 0 and B_used > 0")
               .sort_values(date_col)
               .groupby(date_col, as_index=False)
               .last()
               .reset_index(drop=True)
    )

    n_daily_train = int(len(g_train))
    if n_daily_train < int(min_daily_rows):
        summary = {
            "gvkey": gvkey_val,
            "train_start_req": train_start,
            "train_end_req": train_end,
            "train_start_used_daily": g_train[date_col].min() if n_daily_train else pd.NaT,
            "train_end_used_daily": g_train[date_col].max() if n_daily_train else pd.NaT,
            "ok": False,
            "msg": "too_few_daily_rows_train_after_clean",
            "sigma_hat": np.nan,
            "mu_hat_train": np.nan,
            "n_daily_train": n_daily_train,
            "n_weekly_train": 0,
            "n_weekly_returns": 0,
        }
        return summary, pd.DataFrame()

    # Liability used at the end of the training window
    B_end_raw = float(g_train[B_col].iloc[-1])
    B_end_used = float(g_train["B_used"].iloc[-1])

    # Estimate annual sigma_V and recover in-sample weekly asset path
    sigmaV_hat, weekly_df, ok, msg = calibrate_sigmaV_window_weekly_merton(
        g_train,
        week_ending=week_ending,
        ann_factor=ann_factor,
        sigmaV_init=sigmaV_init,
        E_col=E_col,
        B_col="B_used",
        r_col=r_col,
        T_col=T_col,
        sigmaE_col=sigmaE_col if (sigmaE_col is not None and sigmaE_col in g_train.columns) else None,
    )

    if weekly_df is None:
        weekly_df = pd.DataFrame()

    weekly_df = weekly_df.copy()
    if not weekly_df.empty:
        weekly_df["date"] = pd.to_datetime(weekly_df["date"])
        weekly_df = weekly_df.sort_values("date").reset_index(drop=True)

    n_weekly_train = int(len(weekly_df))
    n_weekly_returns = int(
        weekly_df["dlogV"].notna().sum()
    ) if ("dlogV" in weekly_df.columns and not weekly_df.empty) else 0

    # Optional extra acceptance filters at the window-function level
    if min_weekly_obs is not None and n_weekly_train < int(min_weekly_obs):
        ok = False
        msg = f"too_few_weekly_obs<{int(min_weekly_obs)}"

    if n_weekly_returns < int(min_weekly_returns):
        ok = False
        msg = f"too_few_weekly_returns<{int(min_weekly_returns)}"

    # Estimate in-sample mu from weekly implied asset returns
    mu_hat_train = np.nan
    if n_weekly_returns >= int(min_weekly_returns):
        try:
            mu_hat_train = estimate_mu_from_weekly_implied_assets(
                weekly_df,
                float(sigmaV_hat),
                ann_factor=float(ann_factor),
            )
            mu_hat_train = float(mu_hat_train) if np.isfinite(mu_hat_train) else np.nan
        except Exception:
            mu_hat_train = np.nan

    # Add useful metadata to the weekly output
    if not weekly_df.empty:
        weekly_df["sigma_hat"] = float(sigmaV_hat) if np.isfinite(sigmaV_hat) else np.nan
        weekly_df["mu_hat_train"] = float(mu_hat_train) if np.isfinite(mu_hat_train) else np.nan
        weekly_df["train_start_req"] = train_start
        weekly_df["train_end_req"] = train_end
        weekly_df["B_end_raw"] = B_end_raw
        weekly_df["B_end_used"] = B_end_used
        if gvkey_val is not None:
            weekly_df["gvkey"] = gvkey_val

    # Training-end values useful later for PD computation
    if not weekly_df.empty:
        pd_date_is = pd.Timestamp(weekly_df["date"].iloc[-1])
        V_last_is = float(weekly_df["V_hat"].iloc[-1])
        r_last_is = float(weekly_df["r"].iloc[-1])
        weekly_start_used = pd.Timestamp(weekly_df["date"].iloc[0])
        weekly_end_used = pd.Timestamp(weekly_df["date"].iloc[-1])
    else:
        pd_date_is = pd.NaT
        V_last_is = np.nan
        r_last_is = np.nan
        weekly_start_used = pd.NaT
        weekly_end_used = pd.NaT

    # Summary output
    summary = {
        "gvkey": gvkey_val,
        "train_start_req": train_start,
        "train_end_req": train_end,
        "train_start_used_daily": pd.Timestamp(g_train[date_col].iloc[0]),
        "train_end_used_daily": pd.Timestamp(g_train[date_col].iloc[-1]),
        "train_start_used_weekly": weekly_start_used,
        "train_end_used_weekly": weekly_end_used,
        "ok": bool(ok),
        "msg": msg,
        "sigma_hat": float(sigmaV_hat) if np.isfinite(sigmaV_hat) else np.nan,
        "mu_hat_train": float(mu_hat_train) if np.isfinite(mu_hat_train) else np.nan,
        "B_end_raw": B_end_raw,
        "B_end_used": B_end_used,
        "pd_date_is": pd_date_is,
        "V_last_is": V_last_is,
        "r_last_is": r_last_is,
        "n_daily_train": n_daily_train,
        "n_weekly_train": n_weekly_train,
        "n_weekly_returns": n_weekly_returns,
    }

    return summary, weekly_df


def process_one_firm_merton(
    g_firm_daily: pd.DataFrame,
    *,
    windows,
    gvkey: str | None = None,
    date_col: str = "date",
    week_ending: str = "W-FRI",
    ann_factor: float = 52.0,
    T_horizon: float = 1.0,
    min_daily_rows: int = 10,
    min_weekly_obs: int | None = None,
    min_weekly_returns: int = 2,
    E_col: str = "E",
    B_col: str = "B",
    r_col: str = "r",
    T_col: str = "T",
    B_scale: float = 1.0,
    sigmaE_col: str | None = None,
):
    """
    Run the full rolling Merton workflow for ONE firm across all windows.
    For each window:
      1) call run_merton_window_for_firm(...) for the training step
      2) invert the weekly OOS asset path for the following quarter
      3) compute weekly OOS PD_Q and PD_P
    """
    def _last_trading_day_each_week(df_daily_indexed: pd.DataFrame, week_ending: str = "W-FRI"):
        if df_daily_indexed.empty:
            return []
        tmp = df_daily_indexed.reset_index()
        if "date" not in tmp.columns:
            tmp = tmp.rename(columns={tmp.columns[0]: "date"})
        tmp["date"] = pd.to_datetime(tmp["date"])
        tmp["week"] = tmp["date"].dt.to_period(week_ending)
        wk = tmp.groupby("week")["date"].max().sort_values()
        return wk.tolist()

    # standardize input daily panel
    g = g_firm_daily.copy()

    if date_col not in g.columns:
        if isinstance(g.index, pd.DatetimeIndex):
            g = g.reset_index()
            if date_col not in g.columns:
                g = g.rename(columns={g.columns[0]: date_col})
        else:
            raise ValueError(
                f"Input must either contain a '{date_col}' column or have a DatetimeIndex."
            )

    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")

    # infer gvkey if not provided
    if gvkey is None:
        if "gvkey" in g.columns:
            vals = g["gvkey"].dropna().astype(str).unique()
            if len(vals) == 1:
                gvkey = vals[0]

    # numeric coercion once
    keep_num = [c for c in [E_col, B_col, r_col] if c in g.columns]
    if sigmaE_col is not None and sigmaE_col in g.columns:
        keep_num.append(sigmaE_col)

    for c in keep_num:
        g[c] = pd.to_numeric(g[c], errors="coerce")

    if T_col not in g.columns:
        g[T_col] = float(T_horizon)
    else:
        g[T_col] = pd.to_numeric(g[T_col], errors="coerce")
        g[T_col] = g[T_col].fillna(float(T_horizon))

    g = (
        g.dropna(subset=[date_col, E_col, B_col, r_col, T_col])
         .query(f"{E_col} > 0 and {B_col} > 0")
         .sort_values(date_col)
         .groupby(date_col, as_index=False)
         .last()
    )

    if g.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    g = g.set_index(date_col).sort_index()

    # rolling outputs
    summary_rows = []
    weekly_is_parts = []
    weekly_oos_parts = []

    # warm start only within this firm
    prev_sigma_init = None

    for w in windows:
        train_start = pd.Timestamp(w["train_start"])
        train_end   = pd.Timestamp(w["train_end"])
        oos_start   = pd.Timestamp(w["oos_start"])
        oos_end     = pd.Timestamp(w["oos_end"])

        summary, weekly_df = run_merton_window_for_firm(
            g,
            train_start=train_start,
            train_end=train_end,
            sigmaV_init=prev_sigma_init,
            date_col=date_col,
            week_ending=week_ending,
            ann_factor=ann_factor,
            min_daily_rows=min_daily_rows,
            min_weekly_obs=min_weekly_obs,
            min_weekly_returns=min_weekly_returns,
            E_col=E_col,
            B_col=B_col,
            r_col=r_col,
            T_col=T_col,
            T_default=float(T_horizon),
            B_scale=float(B_scale),
            sigmaE_col=sigmaE_col,
        )

        ok = bool(summary.get("ok", False))
        msg = summary.get("msg", "")

        sigmaV_hat = float(summary["sigma_hat"]) if np.isfinite(summary.get("sigma_hat", np.nan)) else np.nan
        mu_hat = float(summary["mu_hat_train"]) if np.isfinite(summary.get("mu_hat_train", np.nan)) else np.nan
        B_end = float(summary["B_end_used"]) if np.isfinite(summary.get("B_end_used", np.nan)) else np.nan
        pd_date = pd.Timestamp(summary["pd_date_is"]) if pd.notna(summary.get("pd_date_is", pd.NaT)) else pd.NaT
        V_pd = float(summary["V_last_is"]) if np.isfinite(summary.get("V_last_is", np.nan)) else np.nan
        r_pd = float(summary["r_last_is"]) if np.isfinite(summary.get("r_last_is", np.nan)) else np.nan

        PD_Q_1y_is = np.nan
        PD_P_1y_is = np.nan
        if np.isfinite(V_pd) and np.isfinite(B_end) and np.isfinite(r_pd) and np.isfinite(sigmaV_hat) and sigmaV_hat > 0:
            PD_Q_1y_is = float(merton_pd_rn_1y(V_pd, B_end, r_pd, sigmaV_hat))
            if np.isfinite(mu_hat):
                PD_P_1y_is = float(merton_pd_physical_1y(V_pd, B_end, mu_hat, sigmaV_hat))

        summary_rows.append({
            "gvkey": gvkey,
            "train_start": train_start,
            "train_end": train_end,
            "oos_start": oos_start,
            "oos_end": oos_end,
            "ok": ok,
            "msg": msg,
            "sigma_hat": sigmaV_hat,
            "mu_hat_train": mu_hat,
            "B_end": B_end,
            "pd_date_is": pd_date,
            "PD_Q_1y_is": PD_Q_1y_is,
            "PD_P_1y_is": PD_P_1y_is,
            "n_daily_train": int(summary.get("n_daily_train", 0)),
            "n_weekly_train": int(summary.get("n_weekly_train", 0)),
            "n_weekly_returns": int(summary.get("n_weekly_returns", 0)),
        })

        # if training failed, skip OOS for this window
        if (not ok) or (weekly_df is None) or weekly_df.empty or (not np.isfinite(sigmaV_hat)) or (sigmaV_hat <= 0):
            continue

        # warm start next window
        prev_sigma_init = float(sigmaV_hat)

        # in-sample weekly rows
        w_is = weekly_df.copy()
        w_is["date"] = pd.to_datetime(w_is["date"])
        w_is = w_is.sort_values("date").reset_index(drop=True)

        w_store = w_is[["date", "V_hat", "dlogV", "r"]].copy()
        w_store["gvkey"] = gvkey
        w_store["train_end"] = train_end
        w_store["sigma_hat"] = sigmaV_hat
        w_store["mu_hat_train"] = mu_hat
        w_store["B_end"] = B_end
        weekly_is_parts.append(w_store)

        # OOS weekly inversion + PDs
        g_oos = g.loc[(g.index >= oos_start) & (g.index <= oos_end)].copy()
        if g_oos.empty:
            continue

        g_oos = g_oos.dropna(subset=[E_col, B_col, r_col]).query(f"{E_col} > 0 and {B_col} > 0").copy()
        if g_oos.empty:
            continue

        weekly_dates = _last_trading_day_each_week(g_oos, week_ending=week_ending)
        if len(weekly_dates) == 0:
            continue

        train_dlogV = w_is["dlogV"].to_numpy(dtype=float)
        train_dlogV = train_dlogV[np.isfinite(train_dlogV)]
        expanding_dlogV = train_dlogV.tolist()

        V_prev = V_pd if (np.isfinite(V_pd) and V_pd > 0) else None
        V_prev_for_dlog = V_prev

        g_oos_idx = g_oos.groupby(g_oos.index).last()
        oos_rows_this = []

        for d in weekly_dates:
            if d not in g_oos_idx.index:
                continue

            row = g_oos_idx.loc[d]

            E_obs = float(row[E_col])
            B_obs = float(row[B_col]) * float(B_scale)
            r_obs = float(row[r_col])

            V_hat, d1, d2 = invert_asset_one_week_merton(
                E_obs,
                B_obs,
                r_obs,
                float(T_horizon),
                float(sigmaV_hat),
                V_prev=V_prev,
                tol=1e-6,
                maxiter=200,
            )

            dlogV = np.nan
            if (
                V_prev_for_dlog is not None
                and np.isfinite(V_prev_for_dlog) and V_prev_for_dlog > 0
                and np.isfinite(V_hat) and V_hat > 0
            ):
                dlogV = float(np.log(V_hat / V_prev_for_dlog))

            V_prev_for_dlog = V_hat
            V_prev = V_hat

            if np.isfinite(dlogV):
                expanding_dlogV.append(float(dlogV))

            mu_hat_expanding = estimate_mu_from_weekly_implied_assets(
                pd.DataFrame({"dlogV": np.asarray(expanding_dlogV, dtype=float)}),
                float(sigmaV_hat),
                ann_factor=float(ann_factor),
            )
            mu_hat_expanding = float(mu_hat_expanding) if np.isfinite(mu_hat_expanding) else np.nan

            PD_Q_week = float(merton_pd_rn_1y(V_hat, B_obs, r_obs, float(sigmaV_hat)))
            PD_P_week = (
                float(merton_pd_physical_1y(V_hat, B_obs, mu_hat_expanding, float(sigmaV_hat)))
                if np.isfinite(mu_hat_expanding) else np.nan
            )

            oos_rows_this.append({
                "gvkey": gvkey,
                "train_end": train_end,
                "date": pd.Timestamp(d),
                "sigma_hat": float(sigmaV_hat),
                "mu_hat_train": mu_hat,
                "mu_hat_oos_expanding": mu_hat_expanding,
                "E": E_obs,
                "B_used": B_obs,
                "r": r_obs,
                "V_hat_oos": float(V_hat),
                "dlogV_oos": dlogV,
                "PD_Q_1y_oos": PD_Q_week,
                "PD_P_1y_oos": PD_P_week,
            })

        if len(oos_rows_this):
            weekly_oos_parts.append(
                pd.DataFrame(oos_rows_this).sort_values("date").reset_index(drop=True)
            )

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["train_end", "gvkey"])
        .reset_index(drop=True)
    ) if len(summary_rows) else pd.DataFrame()

    weekly_is_df = (
        pd.concat(weekly_is_parts, ignore_index=True)
        .sort_values(["train_end", "gvkey", "date"])
        .reset_index(drop=True)
    ) if len(weekly_is_parts) else pd.DataFrame()

    weekly_oos_df = (
        pd.concat(weekly_oos_parts, ignore_index=True)
        .sort_values(["train_end", "gvkey", "date"])
        .reset_index(drop=True)
    ) if len(weekly_oos_parts) else pd.DataFrame()

    return summary_df, weekly_is_df, weekly_oos_df   