import numpy as np
import pandas as pd
from math import erf, sqrt
from scipy.stats import kstest, norm, norminvgauss


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


def jarque_bera_test(x, min_obs=30) -> dict:
    s = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    n = int(s.shape[0])

    if n < min_obs:
        return {
            "n": n,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "excess_kurtosis": np.nan,
            "JB": np.nan,
            "p_value": np.nan,
            "reject_5pct": np.nan,
        }

    z = s.to_numpy(dtype=float)
    z = z - z.mean()

    m2 = float(np.mean(z ** 2))
    if (not np.isfinite(m2)) or (m2 <= 0):
        return {
            "n": n,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "excess_kurtosis": np.nan,
            "JB": np.nan,
            "p_value": np.nan,
            "reject_5pct": np.nan,
        }

    m3 = float(np.mean(z ** 3))
    m4 = float(np.mean(z ** 4))

    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2)
    excess_kurt = kurt - 3.0

    jb = (n / 6.0) * (skew ** 2 + 0.25 * (excess_kurt ** 2))

    # JB is asymptotically chi-square with 2 df -> survival function = exp(-x/2)
    p = float(np.exp(-0.5 * jb))

    return {
        "n": n,
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "excess_kurtosis": float(excess_kurt),
        "JB": float(jb),
        "p_value": p,
        "reject_5pct": bool(p < 0.05),
    }


def jb_panel_on_returns(df: pd.DataFrame,
                        value_col: str,
                        id_col="gvkey",
                        date_col="date",
                        min_obs=30,
                        value_is_return=False,
                        log_returns=True) -> dict:
    d = df[[id_col, date_col, value_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[id_col, date_col, value_col]).sort_values([id_col, date_col]).copy()

    if value_is_return:
        d["ret"] = d[value_col]
        series_tested = "provided returns"
    else:
        if log_returns:
            d = d[d[value_col] > 0].copy()
            d["ret"] = d.groupby(id_col)[value_col].transform(lambda s: np.log(s).diff())
            series_tested = "log returns from levels"
        else:
            d["ret"] = d.groupby(id_col)[value_col].pct_change()
            series_tested = "simple returns from levels"

    ret_df = d[[id_col, date_col, "ret"]].dropna().copy()

    rows = []
    for gv, g in ret_df.groupby(id_col):
        out = jarque_bera_test(g["ret"], min_obs=min_obs)
        out[id_col] = gv
        rows.append(out)

    by_firm = pd.DataFrame(rows)
    if by_firm.empty:
        summary = pd.DataFrame([{
            "value_col": value_col,
            "series_tested": series_tested,
            "n_firms_total": 0,
            "n_firms_tested": 0,
            "share_reject_5pct": np.nan,
            "mean_skewness": np.nan,
            "mean_excess_kurtosis": np.nan,
            "median_excess_kurtosis": np.nan,
            "pooled_JB": np.nan,
            "pooled_p_value": np.nan,
            "pooled_reject_5pct": np.nan,
            "n_returns_pooled": 0,
        }])
        return {"summary": summary, "by_firm": by_firm, "returns_panel": ret_df}

    pooled = jarque_bera_test(ret_df["ret"], min_obs=min_obs)

    summary = pd.DataFrame([{
        "value_col": value_col,
        "series_tested": series_tested,
        "n_firms_total": int(ret_df[id_col].nunique()),
        "n_firms_tested": int(by_firm["JB"].notna().sum()),
        "share_reject_5pct": float(pd.to_numeric(by_firm["reject_5pct"], errors="coerce").mean()),
        "mean_skewness": float(by_firm["skewness"].mean()),
        "mean_excess_kurtosis": float(by_firm["excess_kurtosis"].mean()),
        "median_excess_kurtosis": float(by_firm["excess_kurtosis"].median()),
        "pooled_JB": float(pooled["JB"]) if np.isfinite(pooled["JB"]) else np.nan,
        "pooled_p_value": float(pooled["p_value"]) if np.isfinite(pooled["p_value"]) else np.nan,
        "pooled_reject_5pct": pooled["reject_5pct"],
        "n_returns_pooled": int(pooled["n"]),
    }])

    by_firm = by_firm[[id_col, "n", "skewness", "kurtosis", "excess_kurtosis", "JB", "p_value", "reject_5pct"]]
    by_firm = by_firm.sort_values(["p_value", "JB"], ascending=[True, False], na_position="last").reset_index(drop=True)

    return {
        "summary": summary,
        "by_firm": by_firm,
        "returns_panel": ret_df,
    }


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


def _ks_last_row_per_firm_quarter(df: pd.DataFrame,
                                  selected_quarters: list[str],
                                  id_col="gvkey",
                                  date_col="date",
                                  ok_col: str | None = None) -> pd.DataFrame:
    d = df.copy()
    d[id_col] = d[id_col].astype(str)
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d["quarter"] = d[date_col].dt.to_period("Q").astype(str)

    if ok_col is not None and ok_col in d.columns:
        d = d[d[ok_col].fillna(False)].copy()

    d = d[d["quarter"].isin(selected_quarters)].copy()
    d = d.sort_values([id_col, "quarter", date_col])

    # representative parameter row per firm-quarter = last available row in quarter
    out = (
        d.groupby([id_col, "quarter"], as_index=False)
         .tail(1)
         .reset_index(drop=True)
    )
    return out


def _window_log_returns(level_df: pd.DataFrame,
                        gvkey: str,
                        level_col: str,
                        start_date,
                        end_date,
                        id_col="gvkey",
                        date_col="date") -> np.ndarray:
    g = level_df.copy()
    g[id_col] = g[id_col].astype(str)
    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")
    g[level_col] = pd.to_numeric(g[level_col], errors="coerce")

    g = g[
        (g[id_col] == str(gvkey)) &
        (g[date_col] >= pd.Timestamp(start_date)) &
        (g[date_col] <= pd.Timestamp(end_date))
    ].copy()

    g = g.dropna(subset=[date_col, level_col]).sort_values(date_col)
    g = g[g[level_col] > 0].copy()

    if g.shape[0] < 3:
        return np.array([], dtype=float)

    r = np.diff(np.log(g[level_col].to_numpy(dtype=float)))
    r = r[np.isfinite(r)]
    return r


def ks_fixed_normal_by_quarter(param_df: pd.DataFrame,
                               level_df: pd.DataFrame,
                               selected_quarters: list[str],
                               *,
                               level_col="V_used",
                               mu_col="mu_hat",
                               sigma_col="sigma_hat",
                               train_end_col="training_end",
                               train_start_col=None,
                               merton_lookback_weeks: int | None = None,
                               id_col="gvkey",
                               date_col="date",
                               min_obs=20) -> dict:
    """
    Quarter-specific KS(norm): test observed log-returns in the training window
    against N(mu_hat, sigma_hat^2), where the parameter row is fixed for that quarter.

    If train_start_col is absent, you must provide merton_lookback_weeks.
    """
    rows_q = _ks_last_row_per_firm_quarter(
        param_df, selected_quarters, id_col=id_col, date_col=date_col, ok_col=None
    )

    out_rows = []
    for _, row in rows_q.iterrows():
        gv = str(row[id_col])
        q = row["quarter"]

        mu = pd.to_numeric(row.get(mu_col), errors="coerce")
        sig = pd.to_numeric(row.get(sigma_col), errors="coerce")

        if not np.isfinite(mu) or not np.isfinite(sig) or sig <= 0:
            continue

        if train_start_col is not None and train_start_col in row.index:
            train_start = pd.to_datetime(row[train_start_col], errors="coerce")
        else:
            if merton_lookback_weeks is None:
                raise ValueError("For Merton KS, provide train_start_col or merton_lookback_weeks.")
            train_end_tmp = pd.to_datetime(row.get(train_end_col), errors="coerce")
            if pd.isna(train_end_tmp):
                train_end_tmp = pd.to_datetime(row[date_col], errors="coerce")
            train_start = train_end_tmp - pd.Timedelta(weeks=merton_lookback_weeks)

        train_end = pd.to_datetime(row.get(train_end_col), errors="coerce")
        if pd.isna(train_end):
            train_end = pd.to_datetime(row[date_col], errors="coerce")

        r = _window_log_returns(
            level_df=level_df,
            gvkey=gv,
            level_col=level_col,
            start_date=train_start,
            end_date=train_end,
            id_col=id_col,
            date_col=date_col,
        )

        if r.size < min_obs:
            continue

        ks = kstest(r, lambda x: norm.cdf(x, loc=float(mu), scale=float(sig)))

        out_rows.append({
            "gvkey": gv,
            "quarter": q,
            "n_obs": int(r.size),
            "train_start": pd.Timestamp(train_start),
            "train_end": pd.Timestamp(train_end),
            "mu_hat": float(mu),
            "sigma_hat": float(sig),
            "KS_D": float(ks.statistic),
            "KS_p_value": float(ks.pvalue),
            "reject_5pct": bool(ks.pvalue < 0.05),
        })

    by_firm_quarter = pd.DataFrame(out_rows).sort_values(["quarter", "gvkey"]).reset_index(drop=True)

    by_quarter = (
        by_firm_quarter.groupby("quarter", as_index=False)
        .agg(
            n_firms=("gvkey", "nunique"),
            mean_n_obs=("n_obs", "mean"),
            mean_KS_D=("KS_D", "mean"),
            median_KS_D=("KS_D", "median"),
            share_reject_5pct=("reject_5pct", "mean"),
        )
        .sort_values("quarter")
    )

    return {"by_firm_quarter": by_firm_quarter, "by_quarter": by_quarter}


def ks_fixed_nig_by_quarter(param_df: pd.DataFrame,
                            level_df: pd.DataFrame,
                            selected_quarters: list[str],
                            *,
                            level_col="A_hat_oos",
                            alpha_col="alpha",
                            beta_col="beta1",
                            delta_col="delta",
                            mu_col="beta0",
                            train_start_col="window_train_start",
                            train_end_col="window_train_end",
                            id_col="gvkey",
                            date_col="date",
                            ok_col="ok",
                            min_obs=20) -> dict:
    """
    Quarter-specific KS(NIG): test observed log-returns in the training window
    against NIG(alpha, beta, delta, mu), assuming:
      beta1 = beta, beta0 = mu
    and SciPy mapping:
      a = alpha * delta, b = beta * delta, loc = mu, scale = delta
    """
    rows_q = _ks_last_row_per_firm_quarter(
        param_df, selected_quarters, id_col=id_col, date_col=date_col, ok_col=ok_col
    )

    out_rows = []
    for _, row in rows_q.iterrows():
        gv = str(row[id_col])
        q = row["quarter"]

        alpha = pd.to_numeric(row.get(alpha_col), errors="coerce")
        beta = pd.to_numeric(row.get(beta_col), errors="coerce")
        delta = pd.to_numeric(row.get(delta_col), errors="coerce")
        mu = pd.to_numeric(row.get(mu_col), errors="coerce")

        if not all(np.isfinite([alpha, beta, delta, mu])):
            continue
        if delta <= 0:
            continue
        if alpha <= abs(beta):
            continue

        train_start = pd.to_datetime(row.get(train_start_col), errors="coerce")
        train_end = pd.to_datetime(row.get(train_end_col), errors="coerce")
        if pd.isna(train_start) or pd.isna(train_end):
            continue

        r = _window_log_returns(
            level_df=level_df,
            gvkey=gv,
            level_col=level_col,
            start_date=train_start,
            end_date=train_end,
            id_col=id_col,
            date_col=date_col,
        )

        if r.size < min_obs:
            continue

        a = float(alpha * delta)
        b = float(beta * delta)

        ks = kstest(
            r,
            lambda x: norminvgauss.cdf(x, a=a, b=b, loc=float(mu), scale=float(delta))
        )

        out_rows.append({
            "gvkey": gv,
            "quarter": q,
            "n_obs": int(r.size),
            "train_start": pd.Timestamp(train_start),
            "train_end": pd.Timestamp(train_end),
            "alpha": float(alpha),
            "beta": float(beta),
            "delta": float(delta),
            "mu": float(mu),
            "KS_D": float(ks.statistic),
            "KS_p_value": float(ks.pvalue),
            "reject_5pct": bool(ks.pvalue < 0.05),
        })

    by_firm_quarter = pd.DataFrame(out_rows).sort_values(["quarter", "gvkey"]).reset_index(drop=True)

    by_quarter = (
        by_firm_quarter.groupby("quarter", as_index=False)
        .agg(
            n_firms=("gvkey", "nunique"),
            mean_n_obs=("n_obs", "mean"),
            mean_KS_D=("KS_D", "mean"),
            median_KS_D=("KS_D", "median"),
            share_reject_5pct=("reject_5pct", "mean"),
        )
        .sort_values("quarter")
    )

    return {"by_firm_quarter": by_firm_quarter, "by_quarter": by_quarter}