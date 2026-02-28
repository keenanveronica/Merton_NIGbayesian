import numpy as np
import pandas as pd
from NIG_weekly.nig_apath import call_nig_with_theta, NIGParams


# Helpers
def _to_dt(s):
    return pd.to_datetime(s, errors="coerce")


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad <= 1e-18:
        return (x - med) / (np.nanstd(x) + 1e-12)
    return 0.6745 * (x - med) / (mad + 1e-12)


def _infer_param_cols(df: pd.DataFrame):
    # supports either alpha,beta,delta,mu or p_alpha,p_beta,p_delta,p_mu
    if {"alpha","beta","delta","mu"}.issubset(df.columns):
        return "alpha","beta","delta","mu"
    if {"p_alpha","p_beta","p_delta","p_mu"}.issubset(df.columns):
        return "p_alpha","p_beta","p_delta","p_mu"
    raise ValueError("Cannot infer parameter columns (alpha/beta/delta/mu or p_alpha/p_beta/p_delta/p_mu).")


# (i) Equity reconstruction errors vs leverage
def check_equity_reconstruction_errors(
    assets_weekly_all: pd.DataFrame,
    *,
    tau_inv: float = 1.0,
    use_theta_col: str = "theta",
    max_abs_rel_err_warn: float = 5e-4,
    plot: bool = False,
):
    """
    Recompute model equity value E_model(Â_t) and compare to observed E_t.

    Requires columns:
      gvkey,date,E,L,r,A_hat and either:
        - theta (preferred), or you can set use_theta_col=None and skip this check.
    """
    df = assets_weekly_all.copy()
    req = {"gvkey","date","E","L","r","A_hat"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"assets_weekly_all missing columns: {missing}")

    if use_theta_col is None or use_theta_col not in df.columns:
        raise ValueError("Equity reconstruction check needs a theta column (set use_theta_col='theta').")

    acol,bcol,dcol,mcol = _infer_param_cols(df)
    df["date"] = _to_dt(df["date"])

    # compute E_model pointwise (not vectorized; still fine for ~26k rows)
    E_model = np.empty(len(df), dtype=float)
    E_model[:] = np.nan

    for i, row in enumerate(df.itertuples(index=False)):
        try:
            p = NIGParams(
                alpha=float(getattr(row, acol)),
                beta=float(getattr(row, bcol)),
                delta=float(getattr(row, dcol)),
                mu=float(getattr(row, mcol)),
            )
            th = float(getattr(row, use_theta_col))
            A = float(row.A_hat); L = float(row.L); r = float(row.r)
            E_model[i] = call_nig_with_theta(A, L, r, tau_inv, p, th)
        except Exception:
            E_model[i] = np.nan

    df["E_model"] = E_model
    df["E_abs_err"] = df["E_model"] - df["E"]
    df["E_rel_err"] = df["E_abs_err"] / (df["E"].abs() + 1e-12)
    df["log_lev"] = np.log((df["A_hat"] + 1e-18) / (df["L"] + 1e-18))

    summ = df["E_rel_err"].abs().describe(percentiles=[0.5,0.9,0.95,0.99,0.999])
    share_bad = (df["E_rel_err"].abs() > max_abs_rel_err_warn).mean()

    # error in high leverage states (close to default): low log(A/L)
    high_lev = df["log_lev"] < np.nanpercentile(df["log_lev"], 10)
    summ_high_lev = df.loc[high_lev, "E_rel_err"].abs().describe(percentiles=[0.5,0.9,0.95,0.99])

    out = {
        "summary_abs_rel_error": summ,
        "share_abs_rel_err_gt_thresh": float(share_bad),
        "summary_abs_rel_error_high_leverage_tail": summ_high_lev,
        "worst_rows": df.loc[df["E_rel_err"].abs().nlargest(10).index,
                             ["gvkey","date","E","E_model","E_rel_err","A_hat","L","log_lev"]]
                     .sort_values("E_rel_err", key=lambda s: s.abs(), ascending=False),
        "df_with_errors": df,
    }

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(df["log_lev"], df["E_rel_err"], s=6)
        plt.axhline(0.0)
        plt.title("Equity reconstruction relative error vs log(A/L)")
        plt.xlabel("log(A_hat / L)")
        plt.ylabel("E_model - E_obs (relative)")
        plt.show()

    return out


# (ii) Tail outliers in implied asset returns (dlogA)
def check_asset_return_outliers(
    assets_weekly_all: pd.DataFrame,
    *,
    ret_col: str = "dlogA",
    z_thresh: float = 10.0,
    topk: int = 10,
):
    """
    Detect whether tails are driven by a few inversion outliers.

    Flags per firm:
      - share(|z|>z_thresh)
      - max |dlogA|
      - how much of total variance is contributed by top 1% |dlogA|
    """
    df = assets_weekly_all.copy()
    req = {"gvkey","date",ret_col}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"assets_weekly_all missing columns: {missing}")

    df["date"] = _to_dt(df["date"])
    df = df.dropna(subset=[ret_col])

    rows = []
    for gv, g in df.groupby("gvkey"):
        x = g[ret_col].to_numpy(float)
        z = _robust_z(x)

        share_out = float(np.mean(np.abs(z) > z_thresh))
        max_abs = float(np.nanmax(np.abs(x))) if x.size else np.nan

        # variance contribution of top 1% absolute moves
        absx = np.abs(x)
        if x.size >= 50:
            cut = np.nanpercentile(absx, 99)
            big = absx >= cut
            var_total = float(np.nanvar(x, ddof=1))
            var_big = float(np.nanvar(x[big], ddof=1)) if np.sum(big) > 2 else 0.0
            # proxy: compare energy (sum of squares)
            e_total = float(np.nansum((x - np.nanmean(x))**2))
            e_big = float(np.nansum((x[big] - np.nanmean(x))**2))
            energy_share = float(e_big / (e_total + 1e-18))
        else:
            energy_share = np.nan

        # topk outliers for inspection
        idx = np.argsort(-np.abs(z))[: min(topk, len(z))]
        ex = g.iloc[idx][["date", ret_col]].copy()
        ex["robust_z"] = z[idx]
        ex["abs_ret"] = np.abs(ex[ret_col].to_numpy(float))

        rows.append({
            "gvkey": gv,
            "n": int(len(g)),
            "share_|z|>thr": share_out,
            "max_|ret|": max_abs,
            "top1pct_energy_share": energy_share,
            "examples_top_outliers": ex.sort_values("abs_ret", ascending=False),
        })

    summary = pd.DataFrame([{k:v for k,v in r.items() if k!="examples_top_outliers"} for r in rows])
    worst = summary.sort_values(["share_|z|>thr","top1pct_energy_share","max_|ret|"], ascending=False).head(10)

    return {"per_firm": rows, "summary": summary, "worst_firms": worst}


# (iii) Artificial jumps at refit dates (params + PD)
def check_refit_date_jumps(
    pds_weekly_all: pd.DataFrame,
    updates_all: pd.DataFrame,
    *,
    pd_col: str = "PD_1y",
    refit_merge_tol_days: int = 7,
):
    """
    Compares week-to-week changes on refit weeks vs non-refit weeks.

    Needs:
      pds_weekly_all: gvkey,date,PD_1y and params (alpha/beta/delta/mu)
      updates_all: gvkey,date and params (alpha/beta/delta/mu)
    """
    dfp = pds_weekly_all.copy()
    dfu = updates_all.copy()

    dfp["date"] = _to_dt(dfp["date"])
    dfu["date"] = _to_dt(dfu["date"])

    if "gvkey" not in dfp.columns or "gvkey" not in dfu.columns:
        raise ValueError("Both pds_weekly_all and updates_all must have gvkey.")

    if pd_col not in dfp.columns:
        raise ValueError(f"pds_weekly_all missing {pd_col}.")

    # identify refit weeks per gvkey by nearest match within tolerance
    refit_dates = dfu[["gvkey","date"]].drop_duplicates().copy()
    refit_dates = refit_dates.rename(columns={"date":"refit_date"})

    dfp = dfp.sort_values(["gvkey","date"]).copy()
    dfp["pd_prev"] = dfp.groupby("gvkey")[pd_col].shift(1)
    dfp["dpd"] = dfp[pd_col] - dfp["pd_prev"]

    # logit change is more meaningful near 0/1
    eps = 1e-9
    p = np.clip(dfp[pd_col].to_numpy(float), eps, 1-eps)
    p_prev = np.clip(dfp["pd_prev"].to_numpy(float), eps, 1-eps)
    dfp["dlogit_pd"] = np.log(p/(1-p)) - np.log(p_prev/(1-p_prev))

    # nearest merge
    # for each PD row, find closest refit date for same gvkey
    dfp = dfp.merge(refit_dates, on="gvkey", how="left")
    dfp["refit_dist_days"] = (dfp["date"] - dfp["refit_date"]).abs().dt.days
    dfp["is_refit_week"] = dfp["refit_dist_days"] <= refit_merge_tol_days

    # distribution comparison
    refit = dfp.loc[dfp["is_refit_week"] & dfp["dlogit_pd"].notna(), "dlogit_pd"].abs()
    nonref = dfp.loc[(~dfp["is_refit_week"]) & dfp["dlogit_pd"].notna(), "dlogit_pd"].abs()

    out = {
        "abs_dlogit_pd_refit_desc": refit.describe(percentiles=[0.5,0.9,0.95,0.99]) if len(refit) else None,
        "abs_dlogit_pd_nonrefit_desc": nonref.describe(percentiles=[0.5,0.9,0.95,0.99]) if len(nonref) else None,
        "ratio_median_refit_to_nonrefit": float(np.nanmedian(refit) / (np.nanmedian(nonref) + 1e-18)) if len(refit) and len(nonref) else np.nan,
        "top_abs_jumps": dfp.loc[dfp["dlogit_pd"].abs().nlargest(20).index, ["gvkey","date",pd_col,"pd_prev","dlogit_pd","is_refit_week","refit_date","refit_dist_days"]],
    }

    return out


# (iv) Stability under small perturbations (rerun vs baseline)
def compare_two_pd_panels(
    pd_base: pd.DataFrame,
    pd_alt: pd.DataFrame,
    *,
    pd_col: str = "PD_1y",
):
    """
    Compares two PD panels (baseline vs perturbed config).
    Expects columns: gvkey,date,PD_1y
    """
    a = pd_base[["gvkey","date",pd_col]].copy()
    b = pd_alt[["gvkey","date",pd_col]].copy()
    a["date"] = _to_dt(a["date"])
    b["date"] = _to_dt(b["date"])

    m = a.merge(b, on=["gvkey","date"], suffixes=("_base","_alt"), how="inner")
    if len(m) == 0:
        raise ValueError("No overlap between baseline and alternative PD panels.")

    x = m[f"{pd_col}_base"].to_numpy(float)
    y = m[f"{pd_col}_alt"].to_numpy(float)

    # correlation on levels + logit scale
    eps = 1e-9
    xb = np.clip(x, eps, 1-eps); yb = np.clip(y, eps, 1-eps)
    logit_x = np.log(xb/(1-xb)); logit_y = np.log(yb/(1-yb))

    def _corr(u,v):
        u = u - np.nanmean(u); v = v - np.nanmean(v)
        return float(np.nansum(u*v) / (np.sqrt(np.nansum(u*u))*np.sqrt(np.nansum(v*v)) + 1e-18))

    out = {
        "n_overlap": int(len(m)),
        "corr_level": _corr(x, y),
        "corr_logit": _corr(logit_x, logit_y),
        "median_abs_diff": float(np.nanmedian(np.abs(x - y))),
        "p95_abs_diff": float(np.nanpercentile(np.abs(x - y), 95)),
        "worst_diffs": m.assign(abs_diff=np.abs(x-y)).sort_values("abs_diff", ascending=False).head(20),
    }
    return out