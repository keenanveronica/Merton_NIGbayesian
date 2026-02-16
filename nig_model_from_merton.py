import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norminvgauss

from model_dfs import equity_volatility
from merton_calibration import calibrate_merton_panel
from nig_pd import compute_pd_physical_vec

TRADING_DAYS = 252


def nig_prepare_inputs(df_panel, *, ret_col="logret_mcap", vol_window=252, vol_min_obs=126):
    """
    Hybrid-NIG pipeline step 1:
    compute sigma_E (annualized) from equity log returns.
    """
    df = df_panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["gvkey"] = df["gvkey"].astype(str)

    # Ensure numeric
    for c in ["E", "B", "r", ret_col]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute rolling sigma_E using your canonical helper
    df = equity_volatility(
        df,
        ret_col=ret_col,
        window=int(vol_window),
        min_obs=int(vol_min_obs),
        trading_days=TRADING_DAYS,
    )
    return df


def nig_infer_assets_via_merton(df_inputs, *, T=1.0, B_scale=1.0):
    """
    Hybrid-NIG pipeline step 2:
    infer latent assets V_t, sigma_V,t by Merton inversion (warm-started).
    """
    df = df_inputs.copy()

    # Keep only usable rows
    df = df.dropna(subset=["E", "B", "r", "sigma_E"])
    df = df[(df["E"] > 0) & (df["B"] > 0) & (df["sigma_E"] > 0)].copy()

    # Merton inversion adds: V, sigma_V, d1,d2, solver_success, DD_rn, PD_rn
    df["T"] = float(T)
    out = calibrate_merton_panel(
        df,
        B_col="B",
        E_col="E",
        sigmaE_col="sigma_E",
        r_col="r",
        T_col="T",
        warm_start=True,
        B_scale=float(B_scale),
    )
    return out


def nig_build_implied_asset_returns(df_assets):
    """
    Hybrid-NIG pipeline step 3:
    build dlogV to fit NIG later.
    """
    df = df_assets.copy()
    df = df[df["solver_success"]].copy()
    df = df.sort_values(["gvkey", "date"]).reset_index(drop=True)

    df["logV"] = np.log(df["V"])
    df["dlogV"] = df.groupby("gvkey")["logV"].diff()
    return df.drop(columns=["logV"])


def nig_winsorize_returns_by_gvkey(df, col="dlogV", p=0.001):
    df = df.copy()

    def _clip(s):
        lo, hi = s.quantile(p), s.quantile(1 - p)
        return s.clip(lo, hi)
    df[col] = df.groupby("gvkey")[col].transform(_clip)
    return df


def nig_compute_pd_1y_hybrid(
    df_assets,
    df_params,
    *,
    horizon=1.0,
):
    """
    Compute daily 1Y physical PD under hybrid NIG model.
    """
    df_params = df_params.rename(columns={"beta": "beta1", "mu": "beta0"})
    df = df_assets.merge(df_params, on=["gvkey", "date"], how="left")
    df = df.sort_values(["gvkey", "date"])

    # forward-fill parameters between refits
    param_cols = ["alpha", "beta1", "delta", "beta0"]
    df[param_cols] = df.groupby("gvkey")[param_cols].ffill()

    valid = (
        df["V"].notna() & (df["V"] > 0) &
        df["B"].notna() & (df["B"] > 0) &
        df["alpha"].notna() & (df["alpha"] > 0) &
        df["beta1"].notna() & (df["beta1"].abs() < df["alpha"]) &
        df["delta"].notna() & (df["delta"] > 0) &
        df["beta0"].notna()
    )


    df["PD_nig_1y"] = np.nan

    df.loc[valid, "PD_nig_1y"] = compute_pd_physical_vec(
        A0=df.loc[valid, "V"].to_numpy(),
        L=df.loc[valid, "B"].to_numpy(),
        T=horizon,
        alpha=df.loc[valid, "alpha"].to_numpy(),
        beta1=df.loc[valid, "beta1"].to_numpy(),
        beta0=df.loc[valid, "beta0"].to_numpy(),
        delta=df.loc[valid, "delta"].to_numpy(),
        warn=False,
    )

    return df


def _nig_mle_fit_window(x, *, maxiter=120):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 50:
        return None

    # objective
    def nll(u):
        a_raw, b_raw, log_delta, mu = u

        # hard bounds via clipping as extra safety
        a_raw = np.clip(a_raw, -20.0, 12.0)
        log_delta = np.clip(log_delta, -20.0, 0.0)
        b_raw = np.clip(b_raw, -10.0, 10.0)
        mu = np.clip(mu, -0.2, 0.2)

        alpha = np.exp(a_raw) + 1e-6
        delta = np.exp(log_delta) + 1e-8
        beta1 = (alpha - 1e-8) * np.tanh(b_raw)

        a = alpha * delta
        b = beta1 * delta

        # early penalty if numerically dangerous
        if (not np.isfinite(a)) or (not np.isfinite(b)) or (a <= 0.0) or (abs(b) >= a) or (delta <= 0.0):
            return 1e50
        if a > 1e6:
            return 1e50

        ll = norminvgauss.logpdf(x, a=a, b=b, loc=mu, scale=delta)
        if not np.all(np.isfinite(ll)):
            return 1e50
        return -float(np.sum(ll))

    # initial guess (stable)
    mu0 = float(np.mean(x))
    s0 = float(np.std(x, ddof=1))
    delta0 = max(s0, 1e-3)
    alpha0 = 10.0

    u0 = np.array([np.log(alpha0), 0.0, np.log(delta0), np.clip(mu0, -0.2, 0.2)], dtype=float)

    # IMPORTANT: optimizer bounds (prevents wandering + speeds up)
    bounds = [
        (-20.0, 12.0),   # a_raw
        (-10.0, 10.0),   # b_raw
        (-20.0, 0.0),    # log_delta
        (-0.2, 0.2),     # mu
    ]

    res = minimize(
        nll,
        u0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-9},
    )
    if not res.success:
        return None

    # Extract params safely (clip again to avoid overflow in exp)
    a_raw, b_raw, log_delta, mu = res.x
    a_raw = float(np.clip(a_raw, -20.0, 12.0))
    b_raw = float(np.clip(b_raw, -10.0, 10.0))
    log_delta = float(np.clip(log_delta, -20.0, 0.0))
    mu = float(np.clip(mu, -0.2, 0.2))

    alpha = float(np.exp(a_raw) + 1e-6)
    delta = float(np.exp(log_delta) + 1e-8)
    beta1 = float((alpha - 1e-8) * np.tanh(b_raw))
    beta0 = float(mu)

    return {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0}


def nig_fit_rolling_params_hybrid(
    df,
    *,
    ret_col="dlogV",
    window=252,
    refit_every=20,
    min_obs=200,
    winsor_p=None,
    maxiter=120,
):
    """
    Rolling NIG fit per firm on implied asset returns.

    Output: one row per (gvkey, refit_date) with alpha,beta1,delta,beta0.
    Keep it sparse (only refit dates) and forward-fill later.
    """
    xdf = df[["gvkey", "date", ret_col]].copy()
    xdf["date"] = pd.to_datetime(xdf["date"])
    xdf = xdf.dropna(subset=[ret_col]).sort_values(["gvkey", "date"])

    rows = []

    for gvkey, g in xdf.groupby("gvkey", sort=False):
        g = g.reset_index(drop=True)
        x = g[ret_col].to_numpy()

        for i in range(window, len(g), refit_every):
            w = x[i - window: i]
            w = w[np.isfinite(w)]
            if w.size < min_obs:
                continue

            if winsor_p is not None:
                lo = np.quantile(w, winsor_p)
                hi = np.quantile(w, 1.0 - winsor_p)
                w = np.clip(w, lo, hi)

            fit = _nig_mle_fit_window(w, maxiter=maxiter)
            if fit is None:
                continue

            rows.append(
                {
                    "gvkey": gvkey,
                    "date": g.loc[i, "date"],
                    **fit,
                    "nobs": int(w.size),
                }
            )

    return pd.DataFrame(rows)


def nig_attach_params_and_pd_hybrid(
    df_assets,
    df_params,
    *,
    horizon_years=1.0,
):
    """
    Merge rolling NIG params onto daily asset panel, forward-fill, compute PD_1Y (physical).
    """
    df = df_assets.copy().sort_values(["gvkey", "date"])
    p = df_params.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["gvkey", "date"])

    df = df.merge(p[["gvkey", "date", "alpha", "beta1", "delta", "beta0"]], on=["gvkey", "date"], how="left")

    # forward-fill between refits
    df[["alpha", "beta1", "delta", "beta0"]] = df.groupby("gvkey")[["alpha", "beta1", "delta", "beta0"]].ffill()

    # compute PD where feasible (A0=V, L=B)
    ok = (
        df["V"].notna() & (df["V"] > 0) &
        df["B"].notna() & (df["B"] > 0) &
        df["alpha"].notna() & df["beta1"].notna() & df["delta"].notna() & df["beta0"].notna()
    )
    ok = ok & (df["alpha"] > 0) & (df["delta"] > 0) & (df["beta1"].abs() < df["alpha"])

    df["PD_nig_1y"] = np.nan
    if ok.any():
        df.loc[ok, "PD_nig_1y"] = compute_pd_physical_vec(
            A0=df.loc[ok, "V"].to_numpy(),
            L=df.loc[ok, "B"].to_numpy(),
            T=horizon_years,
            alpha=df.loc[ok, "alpha"].to_numpy(),
            beta1=df.loc[ok, "beta1"].to_numpy(),
            delta=df.loc[ok, "delta"].to_numpy(),
            beta0=df.loc[ok, "beta0"].to_numpy(),
            warn=False,
        )

        eps = 1e-12
        df.loc[ok, "PD_nig_1y"] = df.loc[ok, "PD_nig_1y"].clip(eps, 1 - eps)

    return df
