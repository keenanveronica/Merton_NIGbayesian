import numpy as np
import pandas as pd
from math import erf, sqrt


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _two_sided_p_from_z(z: float) -> float:
    return 2.0 * (1.0 - _norm_cdf(abs(z)))

def _safe_corr(x: pd.Series, y: pd.Series, method="pearson", min_obs=10) -> float:
    ok = x.notna() & y.notna()
    if ok.sum() < min_obs:
        return np.nan
    return float(x[ok].corr(y[ok], method=method))

def _log_ratio_err(x, y, eps=1e-12):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.log(x + eps) - np.log(y + eps)


# Level-fit metrics (PD space + log-PD space)
def level_fit_metrics(df: pd.DataFrame, model: str, bench: str,
                      eps_log=1e-12, mape_floor=1e-6) -> dict:
    x = pd.to_numeric(df[model], errors="coerce")
    y = pd.to_numeric(df[bench], errors="coerce")
    ok = x.notna() & y.notna() & (x > 0) & (y > 0)

    e = (x[ok] - y[ok]).to_numpy(dtype=float)
    loge = _log_ratio_err(x[ok], y[ok], eps=eps_log)

    out = {
        "model": model,
        "benchmark": bench,
        "N": int(ok.sum()),
        # PD-level errors (signed + unsigned)
        "Bias_PD": float(np.mean(e)),
        "MAE_PD": float(np.mean(np.abs(e))),
        "RMSE_PD": float(np.sqrt(np.mean(e**2))),
        # scale-free percentage error (floor on denominator)
        "MAPE_PD_%": float(100.0 * np.mean(np.abs(e) / np.maximum(y[ok].to_numpy(), mape_floor))),
        # log-PD errors (multiplicative)
        "Bias_logPD": float(np.mean(loge)),
        "MAE_logPD": float(np.mean(np.abs(loge))),
        "RMSE_logPD": float(np.sqrt(np.mean(loge**2))),
    }
    return out


# Tracking metrics (firm-level time-series & cross-sectional by date)
def tracking_metrics(df: pd.DataFrame, model: str, bench: str,
                     id_col="gvkey", date_col="date",
                     min_obs_firm=30, min_firms_date=10,
                     use_log_changes=False, eps_log=1e-12) -> dict:
    d = df[[id_col, date_col, model, bench]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d[model] = pd.to_numeric(d[model], errors="coerce")
    d[bench] = pd.to_numeric(d[bench], errors="coerce")

    # keep positive PDs
    d = d.dropna(subset=[model, bench])
    d = d[(d[model] > 0) & (d[bench] > 0)].copy()
    d = d.sort_values([id_col, date_col])

    # within-firm first differences
    if use_log_changes:
        d["_x"] = np.log(d[model].to_numpy() + eps_log)
        d["_y"] = np.log(d[bench].to_numpy() + eps_log)
        d["dx"] = d.groupby(id_col)["_x"].diff()
        d["dy"] = d.groupby(id_col)["_y"].diff()
    else:
        d["dx"] = d.groupby(id_col)[model].diff()
        d["dy"] = d.groupby(id_col)[bench].diff()

    # firm-level time-series correlations
    firm_rows = []
    for gv, g in d.groupby(id_col):
        rho_levels = _safe_corr(g[model], g[bench], method="pearson", min_obs=min_obs_firm)
        rho_changes = _safe_corr(g["dx"], g["dy"], method="pearson", min_obs=max(10, min_obs_firm // 3))
        rho_s_levels = _safe_corr(g[model], g[bench], method="spearman", min_obs=min_obs_firm)

        # magnitude tracking error in changes (PD points)
        ok_ch = g["dx"].notna() & g["dy"].notna()
        mae_delta = float(np.mean(np.abs((g.loc[ok_ch, "dx"] - g.loc[ok_ch, "dy"]).to_numpy()))) if ok_ch.any() else np.nan

        firm_rows.append({
            id_col: gv,
            "rho_levels": rho_levels,
            "rho_changes": rho_changes,
            "rhoS_levels": rho_s_levels,
            "MAE_delta": mae_delta,
            "n_obs": int(g[[model, bench]].dropna().shape[0]),
        })

    firm_df = pd.DataFrame(firm_rows)

    # cross-sectional rank fit by date (one correlation per date)
    date_rows = []
    for t, g in d.groupby(date_col):
        if g[id_col].nunique() < min_firms_date:
            continue
        date_rows.append({
            "date": t,
            "cs_spearman": _safe_corr(g[model], g[bench], method="spearman", min_obs=min_firms_date),
            "cs_pearson": _safe_corr(g[model], g[bench], method="pearson", min_obs=min_firms_date),
        })
    date_df = pd.DataFrame(date_rows).sort_values("date")

    # summarize
    out = {
        "model": model,
        "benchmark": bench,
        "firm_level_mean_rho_levels": float(firm_df["rho_levels"].mean()),
        "firm_level_mean_rho_changes": float(firm_df["rho_changes"].mean()),
        "firm_level_mean_rhoS_levels": float(firm_df["rhoS_levels"].mean()),
        # report MAE of changes in PD-bps (1bp = 1e-4)
        "firm_level_mean_MAE_delta_PD_bps": float(1e4 * firm_df["MAE_delta"].mean()),
        "cs_by_date_mean_spearman": float(date_df["cs_spearman"].mean()),
        "cs_by_date_mean_pearson": float(date_df["cs_pearson"].mean()),
        "n_firms_used": int(firm_df.shape[0]),
        "n_dates_used": int(date_df.shape[0]),
        "firm_metrics_df": firm_df,
        "date_metrics_df": date_df,
    }
    return out


# Newey–West (HAC) test on mean loss differential d_t
def newey_west_mean_test(d_t: pd.Series, L: int | None = None) -> dict:
    d = pd.to_numeric(d_t, errors="coerce").dropna().to_numpy(dtype=float)
    T = d.shape[0]
    if T < 10:
        return {"T": T, "mean": np.nan, "se_hac": np.nan, "z": np.nan, "p": np.nan, "L": L}

    mu = float(d.mean())
    u = d - mu

    # automatic lag choice
    if L is None:
        L = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
        L = max(0, L)

    # gamma_0
    gamma0 = float(np.mean(u * u))
    lr_var = gamma0

    # Bartlett weights
    for ell in range(1, L + 1):
        w = 1.0 - ell / (L + 1.0)
        gamma = float(np.mean(u[ell:] * u[:-ell]))
        lr_var += 2.0 * w * gamma

    var_mu = lr_var / T
    se = float(np.sqrt(max(var_mu, 0.0)))
    z = mu / se if se > 0 else np.nan
    p = _two_sided_p_from_z(z) if np.isfinite(z) else np.nan

    return {"T": T, "mean": mu, "se_hac": se, "z": float(z), "p": float(p), "L": int(L)}

def dm_compare_models(df: pd.DataFrame, modelA: str, modelB: str, bench: str,
                      date_col="date", eps_log=1e-12, loss="sq_log", L=None) -> dict:
    d = df[[date_col, modelA, modelB, bench]].copy()
    d[date_col] = pd.to_datetime(d[date_col])
    for c in [modelA, modelB, bench]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[modelA, modelB, bench])
    d = d[(d[modelA] > 0) & (d[modelB] > 0) & (d[bench] > 0)].copy()

    def loss_fn(x, y):
        if loss == "sq_log":
            e = _log_ratio_err(x, y, eps=eps_log)
            return e**2
        if loss == "abs_log":
            e = _log_ratio_err(x, y, eps=eps_log)
            return np.abs(e)
        if loss == "sq":
            e = (x - y)
            return e**2
        if loss == "abs":
            e = (x - y)
            return np.abs(e)
        raise ValueError("loss must be one of: sq_log, abs_log, sq, abs")

    # firm-date losses
    LA = loss_fn(d[modelA].to_numpy(dtype=float), d[bench].to_numpy(dtype=float))
    LB = loss_fn(d[modelB].to_numpy(dtype=float), d[bench].to_numpy(dtype=float))
    d["d_loss"] = LA - LB

    # collapse by date
    d_t = d.groupby(date_col)["d_loss"].mean().sort_index()
    test = newey_west_mean_test(d_t, L=L)
    test.update({"modelA": modelA, "modelB": modelB, "benchmark": bench, "loss": loss})
    return test


# Paired t-test on firm-level correlation differences
def paired_t_on_firm_metric(firm_df_A: pd.DataFrame, firm_df_B: pd.DataFrame,
                            id_col="gvkey", metric="rho_levels") -> dict:
    a = firm_df_A[[id_col, metric]].rename(columns={metric: "a"})
    b = firm_df_B[[id_col, metric]].rename(columns={metric: "b"})
    m = a.merge(b, on=id_col, how="inner").dropna()
    if m.shape[0] < 5:
        return {"metric": metric, "N_firms": int(m.shape[0]), "t": np.nan, "p": np.nan, "mean_diff": np.nan}

    diff = (m["a"] - m["b"]).to_numpy(dtype=float)
    n = diff.shape[0]
    mean_diff = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / np.sqrt(n) if sd > 0 else np.nan
    t = mean_diff / se if se and np.isfinite(se) else np.nan

    # normal approx p-value
    p = _two_sided_p_from_z(t) if np.isfinite(t) else np.nan
    return {"metric": metric, "N_firms": int(n), "t": float(t), "p": float(p), "mean_diff": mean_diff}


# Master evaluation runner
def evaluate_models_vs_cds(df: pd.DataFrame,
                           models: list[str],
                           bench_col: str,
                           id_col="gvkey", date_col="date",
                           eps_log=1e-12, mape_floor=1e-6,
                           min_obs_firm=30, min_firms_date=10,
                           loss_for_dm="sq_log") -> dict:
    # level fit table
    lvl = pd.DataFrame([level_fit_metrics(df, m, bench_col, eps_log=eps_log, mape_floor=mape_floor) for m in models])

    # tracking metrics + keep firm-level dfs for paired tests
    trk = []
    firm_metrics = {}
    for m in models:
        out = tracking_metrics(df, m, bench_col, id_col=id_col, date_col=date_col,
                               min_obs_firm=min_obs_firm, min_firms_date=min_firms_date,
                               use_log_changes=False, eps_log=eps_log)
        trk.append({k: v for k, v in out.items() if not k.endswith("_df")})
        firm_metrics[m] = out["firm_metrics_df"]

    trk_tbl = pd.DataFrame(trk)

    # DM-style HAC comparisons
    dm = None
    if len(models) >= 2:
        dm = dm_compare_models(df, models[0], models[1], bench_col, date_col=date_col,
                               eps_log=eps_log, loss=loss_for_dm, L=None)

    # paired tests on firm-level correlations (levels and changes)
    paired = []
    if len(models) >= 2:
        A, B = models[0], models[1]
        paired.append(paired_t_on_firm_metric(firm_metrics[A], firm_metrics[B], id_col=id_col, metric="rho_levels"))
        paired.append(paired_t_on_firm_metric(firm_metrics[A], firm_metrics[B], id_col=id_col, metric="rho_changes"))
        paired = pd.DataFrame(paired)

    return {"level_fit": lvl, "tracking": trk_tbl, "dm_test": dm, "paired_corr_tests": paired,
            "firm_metrics": firm_metrics}
