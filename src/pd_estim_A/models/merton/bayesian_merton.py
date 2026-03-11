import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from pytensor.scan import scan

from pd_estim_A.models.merton.merton_main import (
    invert_asset_one_week_merton,
)
from pd_estim_A.models.merton.merton_pd import (
    merton_pd_rn_1y,
    merton_pd_physical_1y,
)


def _norm_cdf_pt(z):
    return 0.5 * (1.0 + pt.erf(z / pt.sqrt(2.0)))


def _merton_equity_value_pt(V, F, r, tau, sigma):
    eps = 1e-12
    V = pt.maximum(V, eps)
    F = pt.maximum(F, eps)
    tau = pt.maximum(tau, eps)
    sigma = pt.maximum(sigma, eps)

    sqrt_tau = pt.sqrt(tau)
    d1 = (pt.log(V / F) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return V * _norm_cdf_pt(d1) - F * pt.exp(-r * tau) * _norm_cdf_pt(d2)


def _as_scalar_float(x, default=np.nan):
    try:
        x = float(x)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _safe_quantile_triplet(x, lo=0.025, hi=0.975):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    q_lo, q_med, q_hi = np.quantile(x, [lo, 0.5, hi])
    return float(q_lo), float(q_med), float(q_hi)


def _summarize_trace(trace, *, hdi_prob=0.95, max_treedepth=15, delta_fixed=0.01):
    var_names = ["mu", "sigma", "V0_latent", "V_last_latent"]
    summ = az.summary(trace, var_names=var_names, hdi_prob=hdi_prob)

    hdi_cols = [c for c in summ.columns if c.startswith("hdi_")]
    hdi_lo = hdi_cols[0] if len(hdi_cols) >= 1 else None
    hdi_hi = hdi_cols[1] if len(hdi_cols) >= 2 else None

    def pick(row, col):
        if col is None or col not in summ.columns or row not in summ.index:
            return np.nan
        val = summ.loc[row, col]
        return float(val) if np.isfinite(val) else np.nan

    div = np.asarray(trace.sample_stats["diverging"].values, dtype=int)
    td = np.asarray(trace.sample_stats["tree_depth"].values, dtype=int)
    n_total = int(div.size) if div.size else 0
    n_div = int(div.sum()) if div.size else 0
    n_max_td = int((td >= int(max_treedepth)).sum()) if td.size else 0

    return {
        "n_posterior_draws_total": n_total,
        "n_divergences": n_div,
        "divergences_pct": (100.0 * n_div / n_total) if n_total else np.nan,
        "n_max_treedepth": n_max_td,
        "max_treedepth_pct": (100.0 * n_max_td / n_total) if n_total else np.nan,
        "max_treedepth_setting": int(max_treedepth),

        "mu_mean": pick("mu", "mean"),
        "mu_sd": pick("mu", "sd"),
        "mu_hdi_lo": pick("mu", hdi_lo),
        "mu_hdi_hi": pick("mu", hdi_hi),
        "mu_ess_bulk": pick("mu", "ess_bulk"),
        "mu_ess_tail": pick("mu", "ess_tail"),
        "mu_rhat": pick("mu", "r_hat"),

        "sigma_mean": pick("sigma", "mean"),
        "sigma_sd": pick("sigma", "sd"),
        "sigma_hdi_lo": pick("sigma", hdi_lo),
        "sigma_hdi_hi": pick("sigma", hdi_hi),
        "sigma_ess_bulk": pick("sigma", "ess_bulk"),
        "sigma_ess_tail": pick("sigma", "ess_tail"),
        "sigma_rhat": pick("sigma", "r_hat"),

        # fixed delta: keep the fields for compatibility
        "delta_fixed": float(delta_fixed),
        "delta_mean": float(delta_fixed),
        "delta_sd": 0.0,
        "delta_hdi_lo": float(delta_fixed),
        "delta_hdi_hi": float(delta_fixed),
        "delta_ess_bulk": np.nan,
        "delta_ess_tail": np.nan,
        "delta_rhat": np.nan,

        "V0_latent_mean": pick("V0_latent", "mean"),
        "V0_latent_hdi_lo": pick("V0_latent", hdi_lo),
        "V0_latent_hdi_hi": pick("V0_latent", hdi_hi),
        "V_last_latent_mean": pick("V_last_latent", "mean"),
        "V_last_latent_hdi_lo": pick("V_last_latent", hdi_lo),
        "V_last_latent_hdi_hi": pick("V_last_latent", hdi_hi),
    }


def fit_bayesian_merton_window(
    window_df: pd.DataFrame,
    *,
    mu0: float,
    sigma0: float,
    V0: float,
    delta0: float = 0.01,
    draws: int = 200,
    tune: int = 200,
    chains: int = 1,
    cores: int = 1,
    target_accept: float = 0.95,
    max_treedepth: int = 15,
    seed: int = 123,
):
    """
    Bayesian Merton benchmark fit for ONE weekly training window,
    with fixed observation-noise scale delta.
    """
    window_df = window_df.sort_values("date").copy()

    x = np.log(window_df["E"].to_numpy(dtype=float))
    r = window_df["r"].to_numpy(dtype=float)
    F = window_df["B"].to_numpy(dtype=float)

    tau = np.ones_like(x, dtype=float) * 1.0
    dt = 1.0 / 52.0
    T = int(len(x))

    mu0 = float(mu0)
    sigma0 = float(max(sigma0, 1e-6))
    V0 = float(max(V0, 1e-12))
    delta0 = float(max(delta0, 1e-8))   # now interpreted as fixed delta

    h0_init = float(np.log(V0))
    eps0 = np.zeros(max(T - 1, 0), dtype=float)

    # Keep your current sigma prior
    alpha_p, beta_p = 3.0, 1e-4
    var_sigma_paper = (beta_p**2) / (((alpha_p - 1.0) ** 2) * (alpha_p - 2.0))
    alpha_sigma = 2.0 + (sigma0**2) / var_sigma_paper
    beta_sigma = sigma0 * (alpha_sigma - 1.0)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=mu0, sigma=2.0)
        sigma = pm.InverseGamma("sigma", alpha=float(alpha_sigma), beta=float(beta_sigma))

        # FIXED delta
        delta = float(delta0)

        h0 = pt.as_tensor_variable(h0_init)

        if T > 1:
            eps = pm.Normal("eps", mu=0.0, sigma=1.0, shape=T - 1)

            def step(eps_t, h_prev, mu, sigma):
                drift = (mu - 0.5 * sigma**2) * dt
                return h_prev + drift + sigma * pt.sqrt(dt) * eps_t

            h_path, _ = scan(
                fn=step,
                sequences=[eps],
                outputs_info=[h0],
                non_sequences=[mu, sigma],
            )
            h = pt.concatenate([h0[None], h_path])
        else:
            eps = None
            h = h0[None]

        V_latent = pt.exp(h)
        pm.Deterministic("V0_latent", V_latent[0])
        pm.Deterministic("V_last_latent", V_latent[-1])

        E_model = _merton_equity_value_pt(V_latent, F, r, tau, sigma)
        logE_model = pt.log(pt.maximum(E_model, 1e-12))
        pm.Normal("x_obs", mu=logE_model, sigma=delta, observed=x)

        initvals = {
            "mu": mu0,
            "sigma": sigma0,
        }
        if eps is not None:
            initvals["eps"] = eps0

        trace = pm.sample(
            draws=int(draws),
            tune=int(tune),
            chains=int(chains),
            cores=int(cores),
            random_seed=int(seed),
            target_accept=float(target_accept),
            max_treedepth=int(max_treedepth),
            init="adapt_diag",
            initvals=initvals,
            progressbar=True,
            return_inferencedata=True,
        )

    return trace

def _posterior_draws_to_dataframe(
    trace,
    *,
    gvkey,
    train_start,
    train_end,
    n_obs_used,
    delta_fixed=0.01,
):
    post = trace.posterior[["mu", "sigma", "V0_latent", "V_last_latent"]].stack(sample=("chain", "draw"))
    sample_index = post["mu"].coords["sample"].to_index()

    out = pd.DataFrame({
        "gvkey": gvkey,
        "train_start": pd.Timestamp(train_start),
        "train_end": pd.Timestamp(train_end),
        "n_obs_used": int(n_obs_used),
        "chain": sample_index.get_level_values("chain").to_numpy().astype(int),
        "draw": sample_index.get_level_values("draw").to_numpy().astype(int),
        "sample_id": np.arange(len(sample_index), dtype=int),
        "mu": np.asarray(post["mu"].values, dtype=float),
        "sigma": np.asarray(post["sigma"].values, dtype=float),
        "delta": float(delta_fixed),   # keep constant column for compatibility
        "V0_latent": np.asarray(post["V0_latent"].values, dtype=float),
        "V_last_latent": np.asarray(post["V_last_latent"].values, dtype=float),
    })
    return out.reset_index(drop=True)

def get_precomputed_merton_init(
    classical_df: pd.DataFrame,
    *,
    gvkey,
    train_end,
    gvkey_col: str = "gvkey",
    train_end_col: str = "train_end_date",
    training_end_col: str = "training_end",
    mu_col: str = "mu_hat",
    sigma_col: str = "sigma_hat",
    V0_col: str = "V_0",
):
    """
    Fetch precomputed classical Merton initialization for one firm and one training window.

    Expected logic:
    - one saved row per firm-window endpoint is flagged with training_end == 1
    - that row contains the classical mu_hat, sigma_hat, and V_0 to initialize the Bayesian fit

    Returns
    -------
    dict
        {
            "ok": bool,
            "msg": str,
            "gvkey": str,
            "train_end": pd.Timestamp,
            "mu0": float,
            "sigma0": float,
            "V0": float,
        }
    """
    train_end = pd.Timestamp(train_end)

    if classical_df is None or classical_df.empty:
        return {
            "ok": False,
            "msg": "classical_df_empty",
            "gvkey": str(gvkey) if gvkey is not None else None,
            "train_end": train_end,
            "mu0": np.nan,
            "sigma0": np.nan,
            "V0": np.nan,
        }

    df = classical_df.copy()

    if gvkey_col not in df.columns:
        return {
            "ok": False,
            "msg": f"missing_column:{gvkey_col}",
            "gvkey": str(gvkey) if gvkey is not None else None,
            "train_end": train_end,
            "mu0": np.nan,
            "sigma0": np.nan,
            "V0": np.nan,
        }

    if train_end_col not in df.columns:
        return {
            "ok": False,
            "msg": f"missing_column:{train_end_col}",
            "gvkey": str(gvkey) if gvkey is not None else None,
            "train_end": train_end,
            "mu0": np.nan,
            "sigma0": np.nan,
            "V0": np.nan,
        }

    for c in [mu_col, sigma_col, V0_col]:
        if c not in df.columns:
            return {
                "ok": False,
                "msg": f"missing_column:{c}",
                "gvkey": str(gvkey) if gvkey is not None else None,
                "train_end": train_end,
                "mu0": np.nan,
                "sigma0": np.nan,
                "V0": np.nan,
            }

    df[gvkey_col] = df[gvkey_col].astype(str)
    df[train_end_col] = pd.to_datetime(df[train_end_col], errors="coerce")

    if training_end_col in df.columns:
        df[training_end_col] = (
            pd.to_numeric(df[training_end_col], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    df[mu_col] = pd.to_numeric(df[mu_col], errors="coerce")
    df[sigma_col] = pd.to_numeric(df[sigma_col], errors="coerce")
    df[V0_col] = pd.to_numeric(df[V0_col], errors="coerce")

    sel = df.loc[
        (df[gvkey_col] == str(gvkey)) &
        (df[train_end_col] == train_end)
    ].copy()

    if training_end_col in sel.columns:
        sel = sel.loc[sel[training_end_col] == 1].copy()

    if sel.empty:
        return {
            "ok": False,
            "msg": "no_matching_precomputed_init",
            "gvkey": str(gvkey),
            "train_end": train_end,
            "mu0": np.nan,
            "sigma0": np.nan,
            "V0": np.nan,
        }

    # In case duplicates exist, keep the last valid one
    sel = sel.dropna(subset=[mu_col, sigma_col, V0_col])

    if sel.empty:
        return {
            "ok": False,
            "msg": "matching_row_but_missing_values",
            "gvkey": str(gvkey),
            "train_end": train_end,
            "mu0": np.nan,
            "sigma0": np.nan,
            "V0": np.nan,
        }

    row = sel.iloc[-1]

    mu0 = _as_scalar_float(row[mu_col])
    sigma0 = _as_scalar_float(row[sigma_col])
    V0 = _as_scalar_float(row[V0_col])

    if not (np.isfinite(mu0) and np.isfinite(sigma0) and sigma0 > 0 and np.isfinite(V0) and V0 > 0):
        return {
            "ok": False,
            "msg": "invalid_precomputed_init_values",
            "gvkey": str(gvkey),
            "train_end": train_end,
            "mu0": mu0,
            "sigma0": sigma0,
            "V0": V0,
        }

    return {
        "ok": True,
        "msg": "ok",
        "gvkey": str(gvkey),
        "train_end": train_end,
        "mu0": mu0,
        "sigma0": sigma0,
        "V0": V0,
    }

def run_bayesian_merton_window_for_firm(
    g_firm_daily: pd.DataFrame,
    *,
    train_start,
    train_end,
    classical_init_df: pd.DataFrame,
    gvkey: str | None = None,
    date_col: str = "date",
    week_ending: str = "W-FRI",
    ann_factor: float = 52.0,
    T_horizon: float = 1.0,
    min_daily_rows: int = 10,
    min_weekly_obs: int = 104,
    min_weekly_returns: int = 2,
    E_col: str = "E",
    B_col: str = "B",
    r_col: str = "r",
    T_col: str = "T",
    B_scale: float = 1.0,
    sigmaE_col: str | None = None,
    delta0: float = 0.01,
    draws: int = 200,
    tune: int = 200,
    chains: int = 1,
    cores: int = 1,
    target_accept: float = 0.95,
    max_treedepth: int = 15,
    seed: int = 123,
    hdi_prob: float = 0.95,
    classical_gvkey_col: str = "gvkey",
    classical_train_end_col: str = "train_end_date",
    classical_training_end_col: str = "training_end",
    classical_mu_col: str = "mu_hat",
    classical_sigma_col: str = "sigma_hat",
    classical_V0_col: str = "V_0",
):
    """
    Run Bayesian Merton training for ONE firm and ONE training window,
    using precomputed classical Merton initial values from `classical_init_df`.

    Required precomputed inputs per firm-window:
    - mu_hat
    - sigma_hat
    - V_0
    typically stored on the row where training_end == 1.
    """
    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)

    empty_draws = pd.DataFrame()

    g = g_firm_daily.copy()

    if date_col not in g.columns:
        if isinstance(g.index, pd.DatetimeIndex):
            g = g.reset_index()
            if date_col not in g.columns:
                g = g.rename(columns={g.columns[0]: date_col})
        else:
            raise ValueError(f"Input must either contain '{date_col}' or have a DatetimeIndex.")

    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")

    if gvkey is None and "gvkey" in g.columns:
        vals = g["gvkey"].dropna().astype(str).unique()
        if len(vals) == 1:
            gvkey = vals[0]

    keep_num = [c for c in [E_col, B_col, r_col] if c in g.columns]
    if sigmaE_col is not None and sigmaE_col in g.columns:
        keep_num.append(sigmaE_col)
    if T_col in g.columns:
        keep_num.append(T_col)

    for c in keep_num:
        g[c] = pd.to_numeric(g[c], errors="coerce")

    needed_cols = [date_col, E_col, B_col, r_col]
    missing = [c for c in needed_cols if c not in g.columns]
    if missing:
        summary = {
            "gvkey": gvkey,
            "train_start": train_start,
            "train_end": train_end,
            "ok": False,
            "msg": f"missing_input_columns:{','.join(missing)}",
            "n_obs_used": 0,
            "mu_init": np.nan,
            "sigma_init": np.nan,
            "V0_init": np.nan,
            "B_end_used": np.nan,
            "r_last_is": np.nan,
        }
        return summary, pd.DataFrame(), empty_draws

    g = (
        g.dropna(subset=[date_col, E_col, B_col, r_col])
         .query(f"{E_col} > 0 and {B_col} > 0")
         .sort_values(date_col)
         .groupby(date_col, as_index=False)
         .last()
    )

    g_train = g.loc[(g[date_col] >= train_start) & (g[date_col] <= train_end)].copy()

    if len(g_train) < int(min_daily_rows):
        summary = {
            "gvkey": gvkey,
            "train_start": train_start,
            "train_end": train_end,
            "ok": False,
            "msg": f"too_few_daily_rows<{int(min_daily_rows)}",
            "n_obs_used": 0,
            "mu_init": np.nan,
            "sigma_init": np.nan,
            "V0_init": np.nan,
            "B_end_used": np.nan,
            "r_last_is": np.nan,
        }
        return summary, pd.DataFrame(), empty_draws

    g_train["week"] = g_train[date_col].dt.to_period(week_ending)

    window_df = (
        g_train.sort_values(date_col)
        .groupby("week", as_index=False)
        .last()[[date_col, E_col, B_col, r_col]]
        .rename(columns={
            date_col: "date",
            E_col: "E",
            B_col: "B",
            r_col: "r",
        })
        .sort_values("date")
        .reset_index(drop=True)
    )

    n_obs_used = int(len(window_df))
    if n_obs_used < int(min_weekly_obs):
        summary = {
            "gvkey": gvkey,
            "train_start": train_start,
            "train_end": train_end,
            "ok": False,
            "msg": f"too_few_weekly_obs<{int(min_weekly_obs)}",
            "n_obs_used": n_obs_used,
            "mu_init": np.nan,
            "sigma_init": np.nan,
            "V0_init": np.nan,
            "B_end_used": np.nan,
            "r_last_is": np.nan,
        }
        return summary, window_df, empty_draws

    if n_obs_used >= 2:
        dlogE = np.diff(np.log(window_df["E"].to_numpy(dtype=float)))
        n_ret = int(np.isfinite(dlogE).sum())
    else:
        n_ret = 0

    if n_ret < int(min_weekly_returns):
        summary = {
            "gvkey": gvkey,
            "train_start": train_start,
            "train_end": train_end,
            "ok": False,
            "msg": f"too_few_weekly_returns<{int(min_weekly_returns)}",
            "n_obs_used": n_obs_used,
            "mu_init": np.nan,
            "sigma_init": np.nan,
            "V0_init": np.nan,
            "B_end_used": np.nan,
            "r_last_is": np.nan,
        }
        return summary, window_df, empty_draws

    init_info = get_precomputed_merton_init(
        classical_init_df,
        gvkey=gvkey,
        train_end=train_end,
        gvkey_col=classical_gvkey_col,
        train_end_col=classical_train_end_col,
        training_end_col=classical_training_end_col,
        mu_col=classical_mu_col,
        sigma_col=classical_sigma_col,
        V0_col=classical_V0_col,
    )

    if not bool(init_info.get("ok", False)):
        summary = {
            "gvkey": gvkey,
            "train_start": train_start,
            "train_end": train_end,
            "ok": False,
            "msg": f"precomputed_init_failed:{init_info.get('msg', 'unknown')}",
            "n_obs_used": n_obs_used,
            "mu_init": init_info.get("mu0", np.nan),
            "sigma_init": init_info.get("sigma0", np.nan),
            "V0_init": init_info.get("V0", np.nan),
            "B_end_used": np.nan,
            "r_last_is": np.nan,
        }
        return summary, window_df, empty_draws

    mu0 = _as_scalar_float(init_info.get("mu0", np.nan))
    sigma0 = _as_scalar_float(init_info.get("sigma0", np.nan))
    V0 = _as_scalar_float(init_info.get("V0", np.nan))

    B_end_used = _as_scalar_float(window_df["B"].iloc[-1] * float(B_scale))
    r_last_is = _as_scalar_float(window_df["r"].iloc[-1])

    if not (np.isfinite(mu0) and np.isfinite(sigma0) and sigma0 > 0 and np.isfinite(V0) and V0 > 0):
        summary = {
            "gvkey": gvkey,
            "train_start": train_start,
            "train_end": train_end,
            "ok": False,
            "msg": "invalid_precomputed_initial_values",
            "n_obs_used": n_obs_used,
            "mu_init": mu0,
            "sigma_init": sigma0,
            "V0_init": V0,
            "B_end_used": B_end_used,
            "r_last_is": r_last_is,
        }
        return summary, window_df, empty_draws

    try:
        trace = fit_bayesian_merton_window(
            window_df,
            mu0=mu0,
            sigma0=sigma0,
            V0=V0,
            delta0=delta0,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            seed=seed,
        )

        param_draws_df = _posterior_draws_to_dataframe(
            trace,
            gvkey=gvkey,
            train_start=train_start,
            train_end=train_end,
            n_obs_used=n_obs_used,
            delta_fixed=delta0,
        )

        trace_stats = _summarize_trace(
            trace,
            hdi_prob=hdi_prob,
            max_treedepth=max_treedepth,
            delta_fixed=delta0,
        )

        if np.isfinite(B_end_used) and B_end_used > 0 and np.isfinite(r_last_is):
            pd_q_is = [
                merton_pd_rn_1y(v, B_end_used, r_last_is, s)
                for v, s in zip(param_draws_df["V_last_latent"], param_draws_df["sigma"])
            ]
            pd_p_is = [
                merton_pd_physical_1y(v, B_end_used, mu, s)
                for v, mu, s in zip(
                    param_draws_df["V_last_latent"],
                    param_draws_df["mu"],
                    param_draws_df["sigma"],
                )
            ]
        else:
            pd_q_is = []
            pd_p_is = []

        lo = (1.0 - float(hdi_prob)) / 2.0
        hi = 1.0 - lo

        q_lo, q_med, q_hi = _safe_quantile_triplet(pd_q_is, lo=lo, hi=hi)
        p_lo, p_med, p_hi = _safe_quantile_triplet(pd_p_is, lo=lo, hi=hi)

        summary = {
            "gvkey": gvkey,
            "train_start": train_start,
            "train_end": train_end,
            "ok": True,
            "msg": "ok",
            "n_obs_used": n_obs_used,
            "mu_init": mu0,
            "sigma_init": sigma0,
            "V0_init": V0,
            "B_end_used": B_end_used,
            "r_last_is": r_last_is,
            "delta_fixed": float(delta0),
            "PD_Q_1y_is_med": q_med,
            "PD_Q_1y_is_ci_lo": q_lo,
            "PD_Q_1y_is_ci_hi": q_hi,
            "PD_P_1y_is_med": p_med,
            "PD_P_1y_is_ci_lo": p_lo,
            "PD_P_1y_is_ci_hi": p_hi,
            "precomputed_init_msg": init_info.get("msg", "ok"),
        }
        summary.update(trace_stats)

        return summary, window_df, param_draws_df

    except Exception as e:
        summary = {
            "gvkey": gvkey,
            "train_start": train_start,
            "train_end": train_end,
            "ok": False,
            "msg": f"bayesian_fit_fail:{type(e).__name__}:{str(e)[:200]}",
            "n_obs_used": n_obs_used,
            "mu_init": mu0,
            "sigma_init": sigma0,
            "V0_init": V0,
            "B_end_used": B_end_used,
            "r_last_is": r_last_is,
        }
        return summary, window_df, empty_draws
    
    
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


def compute_oos_pd_bayesian_merton(
    g_firm_daily: pd.DataFrame,
    *,
    param_draws_df: pd.DataFrame,
    train_end,
    oos_start,
    oos_end,
    gvkey: str | None = None,
    date_col: str = "date",
    week_ending: str = "W-FRI",
    T_horizon: float = 1.0,
    E_col: str = "E",
    B_col: str = "B",
    r_col: str = "r",
    B_scale: float = 1.0,
    hdi_prob: float = 0.95,
    store_draw_level: bool = False,
):
    """
    For the next quarter, use each posterior draw (mu_k, sigma_k, V_last_k)
    to invert the weekly OOS asset path draw-by-draw and compute weekly PDs.

    Returns
    -------
    oos_summary_df : pd.DataFrame
        One row per week with posterior median / interval summaries.
    oos_draws_df : pd.DataFrame
        One row per (week, posterior draw) if store_draw_level=True, else empty.
    """
    train_end = pd.Timestamp(train_end)
    oos_start = pd.Timestamp(oos_start)
    oos_end = pd.Timestamp(oos_end)

    if param_draws_df is None or param_draws_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    g = g_firm_daily.copy()
    if date_col not in g.columns:
        if isinstance(g.index, pd.DatetimeIndex):
            g = g.reset_index()
            if date_col not in g.columns:
                g = g.rename(columns={g.columns[0]: date_col})
        else:
            raise ValueError(f"Input must either contain '{date_col}' or have a DatetimeIndex.")

    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")
    for c in [E_col, B_col, r_col]:
        g[c] = pd.to_numeric(g[c], errors="coerce")

    g = (
        g.dropna(subset=[date_col, E_col, B_col, r_col])
         .query(f"{E_col} > 0 and {B_col} > 0")
         .sort_values(date_col)
         .groupby(date_col, as_index=False)
         .last()
         .set_index(date_col)
    )

    g_oos = g.loc[(g.index >= oos_start) & (g.index <= oos_end)].copy()
    if g_oos.empty:
        return pd.DataFrame(), pd.DataFrame()

    weekly_dates = _last_trading_day_each_week(g_oos, week_ending=week_ending)
    if len(weekly_dates) == 0:
        return pd.DataFrame(), pd.DataFrame()

    lo = (1.0 - float(hdi_prob)) / 2.0
    hi = 1.0 - lo

    draw_rows = []
    summary_rows = []

    draw_state = param_draws_df.copy().reset_index(drop=True)
    draw_state = draw_state.loc[
        np.isfinite(pd.to_numeric(draw_state["mu"], errors="coerce")) &
        np.isfinite(pd.to_numeric(draw_state["sigma"], errors="coerce")) &
        (pd.to_numeric(draw_state["sigma"], errors="coerce") > 0) &
        np.isfinite(pd.to_numeric(draw_state["V_last_latent"], errors="coerce")) &
        (pd.to_numeric(draw_state["V_last_latent"], errors="coerce") > 0)
    ].copy()

    if draw_state.empty:
        return pd.DataFrame(), pd.DataFrame()

    draw_state["V_prev"] = pd.to_numeric(draw_state["V_last_latent"], errors="coerce").astype(float)
    draw_state["active"] = True

    for d in weekly_dates:
        row = g_oos.loc[d]
        E_obs = float(row[E_col])
        B_obs = float(row[B_col]) * float(B_scale)
        r_obs = float(row[r_col])

        vals_v = []
        vals_q = []
        vals_p = []

        for i in range(len(draw_state)):
            if not bool(draw_state.at[i, "active"]):
                continue

            mu_i = float(draw_state.at[i, "mu"])
            sigma_i = float(draw_state.at[i, "sigma"])
            V_prev_i = float(draw_state.at[i, "V_prev"]) if np.isfinite(draw_state.at[i, "V_prev"]) else None

            try:
                V_hat_i, _, _ = invert_asset_one_week_merton(
                    E_obs,
                    B_obs,
                    r_obs,
                    float(T_horizon),
                    sigma_i,
                    V_prev=V_prev_i,
                    tol=1e-6,
                    maxiter=200,
                )
            except Exception:
                draw_state.at[i, "active"] = False
                draw_state.at[i, "V_prev"] = np.nan
                continue

            draw_state.at[i, "V_prev"] = float(V_hat_i)

            pd_q_i = float(merton_pd_rn_1y(V_hat_i, B_obs, r_obs, sigma_i))
            pd_p_i = float(merton_pd_physical_1y(V_hat_i, B_obs, mu_i, sigma_i))

            vals_v.append(float(V_hat_i))
            vals_q.append(pd_q_i)
            vals_p.append(pd_p_i)

            if store_draw_level:
                draw_rows.append({
                    "gvkey": gvkey,
                    "train_end": train_end,
                    "date": pd.Timestamp(d),
                    "chain": int(draw_state.at[i, "chain"]),
                    "draw": int(draw_state.at[i, "draw"]),
                    "sample_id": int(draw_state.at[i, "sample_id"]),
                    "mu": mu_i,
                    "sigma": sigma_i,
                    "delta": float(draw_state.at[i, "delta"]),
                    "V_hat_oos": float(V_hat_i),
                    "PD_Q_1y_oos": pd_q_i,
                    "PD_P_1y_oos": pd_p_i,
                    "E_obs": E_obs,
                    "B_used": B_obs,
                    "r_obs": r_obs,
                })

        if len(vals_q) == 0:
            continue

        v_lo, v_med, v_hi = _safe_quantile_triplet(vals_v, lo=lo, hi=hi)
        q_lo, q_med, q_hi = _safe_quantile_triplet(vals_q, lo=lo, hi=hi)
        p_lo, p_med, p_hi = _safe_quantile_triplet(vals_p, lo=lo, hi=hi)

        summary_rows.append({
            "gvkey": gvkey,
            "train_end": train_end,
            "date": pd.Timestamp(d),
            "n_draws_used": int(len(vals_q)),
            "E_obs": E_obs,
            "B_used": B_obs,
            "r_obs": r_obs,
            "V_hat_oos_med": v_med,
            "V_hat_oos_ci_lo": v_lo,
            "V_hat_oos_ci_hi": v_hi,
            "PD_Q_1y_oos_med": q_med,
            "PD_Q_1y_oos_ci_lo": q_lo,
            "PD_Q_1y_oos_ci_hi": q_hi,
            "PD_P_1y_oos_med": p_med,
            "PD_P_1y_oos_ci_lo": p_lo,
            "PD_P_1y_oos_ci_hi": p_hi,
        })

    oos_summary_df = pd.DataFrame(summary_rows).sort_values("date").reset_index(drop=True)
    oos_draws_df = (
        pd.DataFrame(draw_rows)
        .sort_values(["date", "sample_id"])
        .reset_index(drop=True)
        if len(draw_rows)
        else pd.DataFrame()
    )
    return oos_summary_df, oos_draws_df

def process_one_firm_bayesian_merton(
    g_firm_daily: pd.DataFrame,
    *,
    windows,
    classical_init_df: pd.DataFrame,
    gvkey: str | None = None,
    date_col: str = "date",
    week_ending: str = "W-FRI",
    ann_factor: float = 52.0,
    T_horizon: float = 1.0,
    min_daily_rows: int = 10,
    min_weekly_obs: int = 104,
    min_weekly_returns: int = 2,
    E_col: str = "E",
    B_col: str = "B",
    r_col: str = "r",
    T_col: str = "T",
    B_scale: float = 1.0,
    sigmaE_col: str | None = None,
    delta0: float = 0.01,
    draws: int = 200,
    tune: int = 200,
    chains: int = 1,
    cores: int = 1,
    target_accept: float = 0.95,
    max_treedepth: int = 15,
    seed: int = 123,
    hdi_prob: float = 0.95,
    store_oos_draws: bool = False,
    classical_gvkey_col: str = "gvkey",
    classical_train_end_col: str = "train_end_date",
    classical_training_end_col: str = "training_end",
    classical_mu_col: str = "mu_hat",
    classical_sigma_col: str = "sigma_hat",
    classical_V0_col: str = "V_0",
):
    """
    Run the full rolling Bayesian Merton workflow for ONE firm across all windows,
    using precomputed classical Merton initial values from `classical_init_df`.

    Returns
    -------
    window_summary_df
        One row per training window.
    param_draws_all_df
        One row per posterior parameter draw per training window.
    oos_summary_all_df
        One row per OOS week with posterior median / interval summaries.
    oos_draws_all_df
        Optional draw-level OOS PD dataframe (can be large).
    """
    g = g_firm_daily.copy()

    if date_col not in g.columns:
        if isinstance(g.index, pd.DatetimeIndex):
            g = g.reset_index()
            if date_col not in g.columns:
                g = g.rename(columns={g.columns[0]: date_col})
        else:
            raise ValueError(f"Input must either contain '{date_col}' or have a DatetimeIndex.")

    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")

    if gvkey is None and "gvkey" in g.columns:
        vals = g["gvkey"].dropna().astype(str).unique()
        if len(vals) == 1:
            gvkey = vals[0]

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
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    param_draws_parts = []
    oos_summary_parts = []
    oos_draw_parts = []

    for w_idx, w in enumerate(windows):
        train_start = pd.Timestamp(w["train_start"])
        train_end = pd.Timestamp(w["train_end"])
        oos_start = pd.Timestamp(w["oos_start"])
        oos_end = pd.Timestamp(w["oos_end"])

        summary, window_df, param_draws_df = run_bayesian_merton_window_for_firm(
            g,
            train_start=train_start,
            train_end=train_end,
            classical_init_df=classical_init_df,
            gvkey=gvkey,
            date_col=date_col,
            week_ending=week_ending,
            ann_factor=ann_factor,
            T_horizon=T_horizon,
            min_daily_rows=min_daily_rows,
            min_weekly_obs=min_weekly_obs,
            min_weekly_returns=min_weekly_returns,
            E_col=E_col,
            B_col=B_col,
            r_col=r_col,
            T_col=T_col,
            B_scale=B_scale,
            sigmaE_col=sigmaE_col,
            delta0=delta0,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            seed=seed + int(w_idx),
            hdi_prob=hdi_prob,
            classical_gvkey_col=classical_gvkey_col,
            classical_train_end_col=classical_train_end_col,
            classical_training_end_col=classical_training_end_col,
            classical_mu_col=classical_mu_col,
            classical_sigma_col=classical_sigma_col,
            classical_V0_col=classical_V0_col,
        )

        summary_row = dict(summary)
        summary_row["oos_start"] = oos_start
        summary_row["oos_end"] = oos_end
        summary_rows.append(summary_row)

        if param_draws_df is not None and not param_draws_df.empty:
            param_draws_df = param_draws_df.copy()
            param_draws_df["oos_start"] = oos_start
            param_draws_df["oos_end"] = oos_end
            param_draws_parts.append(param_draws_df)

        if not bool(summary.get("ok", False)) or param_draws_df is None or param_draws_df.empty:
            continue

        oos_summary_df, oos_draws_df = compute_oos_pd_bayesian_merton(
            g,
            param_draws_df=param_draws_df,
            train_end=train_end,
            oos_start=oos_start,
            oos_end=oos_end,
            gvkey=gvkey,
            date_col=date_col,
            week_ending=week_ending,
            T_horizon=T_horizon,
            E_col=E_col,
            B_col=B_col,
            r_col=r_col,
            B_scale=B_scale,
            hdi_prob=hdi_prob,
            store_draw_level=store_oos_draws,
        )

        if oos_summary_df is not None and not oos_summary_df.empty:
            oos_summary_parts.append(oos_summary_df)

        if store_oos_draws and oos_draws_df is not None and not oos_draws_df.empty:
            oos_draw_parts.append(oos_draws_df)

    window_summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["train_end", "gvkey"])
        .reset_index(drop=True)
        if len(summary_rows)
        else pd.DataFrame()
    )

    param_draws_all_df = (
        pd.concat(param_draws_parts, ignore_index=True)
        .sort_values(["train_end", "gvkey", "sample_id"])
        .reset_index(drop=True)
        if len(param_draws_parts)
        else pd.DataFrame()
    )

    oos_summary_all_df = (
        pd.concat(oos_summary_parts, ignore_index=True)
        .sort_values(["train_end", "gvkey", "date"])
        .reset_index(drop=True)
        if len(oos_summary_parts)
        else pd.DataFrame()
    )

    oos_draws_all_df = (
        pd.concat(oos_draw_parts, ignore_index=True)
        .sort_values(["train_end", "gvkey", "date", "sample_id"])
        .reset_index(drop=True)
        if len(oos_draw_parts)
        else pd.DataFrame()
    )

    return window_summary_df, param_draws_all_df, oos_summary_all_df, oos_draws_all_df

