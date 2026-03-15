from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from scipy import stats as sps
except Exception:
    sps = None

from pd_estim_A.eval.eval_df_build import (
    KEYS,
    prepare_panel,
    asof_merge_by_gvkey,
    cds_pd_triangle,
    cds_pd_par_1y_flat_hazard,
)
from pd_estim_A.eval.evaluation_frequentist import newey_west_mean_test


# ---------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------

def _load_cds_and_rf(
    *,
    derived_dir: Path,
    rf_path: Path,
    cds_file: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cds = pd.read_csv(derived_dir / cds_file)
    rf = pd.read_csv(rf_path)

    cds2 = cds.copy()
    cds2["gvkey"] = cds2["gvkey"].astype(str)
    cds2["date"] = pd.to_datetime(cds2["date"], errors="coerce")
    cds2["cds"] = pd.to_numeric(cds2["cds"], errors="coerce")
    cds2 = cds2.dropna(subset=["gvkey", "date", "cds"]).sort_values(KEYS).reset_index(drop=True)

    rf2 = rf.rename(columns={"r_1y": "r_f"}).copy()
    rf2["date"] = pd.to_datetime(rf2["date"], errors="coerce")
    rf2["r_f"] = pd.to_numeric(rf2["r_f"], errors="coerce")
    rf2 = rf2.dropna(subset=["date", "r_f"]).sort_values("date").reset_index(drop=True)

    if rf2["r_f"].median(skipna=True) > 0.5:
        rf2["r_f"] = rf2["r_f"] / 100.0

    return cds2, rf2


def _build_single_bayesian_eval_panel(
    raw_df: pd.DataFrame,
    *,
    suffix: str,
    cds: pd.DataFrame,
    rf: pd.DataFrame,
    recovery: float,
    cds_unit: str,
) -> pd.DataFrame:
    keep_cols = [
        "gvkey",
        "date",
        "window_idx",
        "rf_used",
        "B_used",
        "asset_lo",
        "asset_med",
        "asset_hi",
        "asset_mean",
        "PD_Q_lo",
        "PD_Q_med",
        "PD_Q_hi",
        "PD_Q_mean",
        "PD_P_lo",
        "PD_P_med",
        "PD_P_hi",
        "PD_P_mean",
    ]
    keep_cols = [c for c in keep_cols if c in raw_df.columns]

    rename_map = {
        "asset_lo": f"asset_{suffix}_lo",
        "asset_med": f"asset_{suffix}_med",
        "asset_hi": f"asset_{suffix}_hi",
        "asset_mean": f"asset_{suffix}_mean",
        "PD_Q_lo": f"PD_Q_{suffix}_lo",
        "PD_Q_med": f"PD_Q_{suffix}_med",
        "PD_Q_hi": f"PD_Q_{suffix}_hi",
        "PD_Q_mean": f"PD_Q_{suffix}_mean",
        "PD_P_lo": f"PD_P_{suffix}_lo",
        "PD_P_med": f"PD_P_{suffix}_med",
        "PD_P_hi": f"PD_P_{suffix}_hi",
        "PD_P_mean": f"PD_P_{suffix}_mean",
    }

    numeric_cols = [rename_map[c] for c in rename_map if c in keep_cols]
    numeric_cols += [c for c in ["window_idx", "rf_used", "B_used"] if c in keep_cols]

    panel = prepare_panel(
        raw_df,
        keep_cols=keep_cols,
        rename_map=rename_map,
        numeric_cols=numeric_cols,
    )

    panel["gvkey"] = panel["gvkey"].astype(str)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")

    common_gv = sorted(set(panel["gvkey"].unique()) & set(cds["gvkey"].unique()))
    panel = panel[panel["gvkey"].isin(common_gv)].copy()
    cds2 = cds[cds["gvkey"].isin(common_gv)].copy()

    dmin, dmax = panel["date"].min(), panel["date"].max()
    cds2 = cds2[(cds2["date"] >= dmin) & (cds2["date"] <= dmax)].copy()

    merged_cds = asof_merge_by_gvkey(
        panel.sort_values(KEYS).reset_index(drop=True),
        cds2[["gvkey", "date", "cds"]].sort_values(KEYS).reset_index(drop=True),
        on="date",
        by="gvkey",
        direction="backward",
        allow_exact_matches=True,
    )
    out = merged_cds.dropna(subset=["cds"]).reset_index(drop=True)

    out = out.drop(columns=["r_f"], errors="ignore").sort_values("date").reset_index(drop=True)
    out = pd.merge_asof(out, rf[["date", "r_f"]].sort_values("date"), on="date", direction="backward")

    out["PD_1y_CDS_triangle"], out["lambda_CDS_triangle"] = cds_pd_triangle(
        out["cds"],
        R=recovery,
        T=1.0,
        unit=cds_unit,
    )

    out["PD_1y_CDS_par"], out["lambda_CDS_par"] = cds_pd_par_1y_flat_hazard(
        out["cds"],
        r=out["r_f"],
        R=recovery,
        unit=cds_unit,
        pay_freq=4,
        T=1.0,
        max_iter=60,
    )

    ordered_cols = [
        "gvkey",
        "date",
        "window_idx",
        "rf_used",
        "B_used",
        f"asset_{suffix}_lo",
        f"asset_{suffix}_med",
        f"asset_{suffix}_hi",
        f"asset_{suffix}_mean",
        f"PD_Q_{suffix}_lo",
        f"PD_Q_{suffix}_med",
        f"PD_Q_{suffix}_hi",
        f"PD_Q_{suffix}_mean",
        f"PD_P_{suffix}_lo",
        f"PD_P_{suffix}_med",
        f"PD_P_{suffix}_hi",
        f"PD_P_{suffix}_mean",
        "cds",
        "r_f",
        "lambda_CDS_triangle",
        "PD_1y_CDS_triangle",
        "lambda_CDS_par",
        "PD_1y_CDS_par",
    ]
    ordered_cols = [c for c in ordered_cols if c in out.columns]
    extra_cols = [c for c in out.columns if c not in ordered_cols]
    return out[ordered_cols + extra_cols].sort_values(KEYS).reset_index(drop=True)


def build_bayesian_nig_evaluation_dataframe(
    *,
    derived_dir: str | Path = "../data/derived",
    rf_path: str | Path = "../data/raw/ecb_yc_1y_aaa.csv",
    nig_subdir: str = "bayesian_nig",
    nig_file: str = "bayes_nig_oos_weekly_condfixA_iter2000_burn500_thin5_firms10_wins8.csv",
    cds_file: str = "CDS_panel.csv",
    recovery: float = 0.40,
    cds_unit: str = "auto",
) -> pd.DataFrame:
    derived_dir = Path(derived_dir)
    rf_path = Path(rf_path)

    nig = pd.read_csv(derived_dir / nig_subdir / nig_file)
    cds, rf = _load_cds_and_rf(
        derived_dir=derived_dir,
        rf_path=rf_path,
        cds_file=cds_file,
    )
    return _build_single_bayesian_eval_panel(
        nig,
        suffix="NIG",
        cds=cds,
        rf=rf,
        recovery=recovery,
        cds_unit=cds_unit,
    )


def build_bayesian_merton_evaluation_dataframe(
    *,
    derived_dir: str | Path = "../data/derived",
    rf_path: str | Path = "../data/raw/ecb_yc_1y_aaa.csv",
    merton_subdir: str = "bayesian_merton_oos_weekly",
    merton_file: str = "bayesian_merton_oos_weekly.csv",
    cds_file: str = "CDS_panel.csv",
    recovery: float = 0.40,
    cds_unit: str = "auto",
) -> pd.DataFrame:
    derived_dir = Path(derived_dir)
    rf_path = Path(rf_path)

    if merton_subdir:
        raw = pd.read_csv(derived_dir / merton_subdir / merton_file)
    else:
        raw = pd.read_csv(derived_dir / merton_file)

    cds, rf = _load_cds_and_rf(
        derived_dir=derived_dir,
        rf_path=rf_path,
        cds_file=cds_file,
    )
    return _build_single_bayesian_eval_panel(
        raw,
        suffix="Merton",
        cds=cds,
        rf=rf,
        recovery=recovery,
        cds_unit=cds_unit,
    )


# ---------------------------------------------------------------------
# Core numeric helpers
# ---------------------------------------------------------------------

def _to_num(x) -> np.ndarray:
    return np.array(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=float, copy=True)


def _clip_prob(x, eps: float = 1e-6) -> np.ndarray:
    arr = _to_num(x).copy()
    ok = np.isfinite(arr)
    arr[ok] = np.clip(arr[ok], eps, 1.0 - eps)
    return arr


def _logit(x, eps: float = 1e-6) -> np.ndarray:
    z = _clip_prob(x, eps=eps)
    return np.log(z / (1.0 - z))


def _expit(z) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------

def split_panel_by_date(
    df: pd.DataFrame,
    *,
    split_date: str | pd.Timestamp | None = None,
    train_frac: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    out = df.copy()
    out["gvkey"] = out["gvkey"].astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(KEYS).reset_index(drop=True)

    unique_dates = np.array(sorted(out["date"].dropna().unique()))
    if unique_dates.size < 2:
        raise ValueError("Need at least two unique dates to split the panel.")

    if split_date is None:
        idx = int(np.floor(train_frac * unique_dates.size))
        idx = max(1, min(idx, unique_dates.size - 1))
        cutoff = pd.Timestamp(unique_dates[idx - 1])
    else:
        cutoff = pd.Timestamp(split_date)

    train = out.loc[out["date"] <= cutoff].copy()
    test = out.loc[out["date"] > cutoff].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True), cutoff


def split_panel_three_way(
    df: pd.DataFrame,
    *,
    fit_end: str | pd.Timestamp,
    valid_end: str | pd.Timestamp,
    test_end: str | pd.Timestamp | None = None,
    start_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    out = df.copy()
    out["gvkey"] = out["gvkey"].astype(str)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values(KEYS).reset_index(drop=True)

    fit_end = pd.Timestamp(fit_end)
    valid_end = pd.Timestamp(valid_end)
    if valid_end <= fit_end:
        raise ValueError("valid_end must be strictly after fit_end.")

    if test_end is not None:
        test_end = pd.Timestamp(test_end)
        if test_end <= valid_end:
            raise ValueError("test_end must be strictly after valid_end.")

    if start_date is not None:
        start_date = pd.Timestamp(start_date)
        out = out.loc[out["date"] >= start_date].copy()

    fit_df = out.loc[out["date"] <= fit_end].copy()
    valid_df = out.loc[(out["date"] > fit_end) & (out["date"] <= valid_end)].copy()

    if test_end is None:
        test_df = out.loc[out["date"] > valid_end].copy()
    else:
        test_df = out.loc[(out["date"] > valid_end) & (out["date"] <= test_end)].copy()

    if fit_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("One of the three split blocks is empty. Check the chosen dates.")

    meta = {
        "start_date_used": None if start_date is None else pd.Timestamp(start_date),
        "fit_end": fit_end,
        "valid_end": valid_end,
        "test_end": None if test_end is None else test_end,
        "fit_n": int(len(fit_df)),
        "valid_n": int(len(valid_df)),
        "test_n": int(len(test_df)),
        "fit_min_date": fit_df["date"].min(),
        "fit_max_date": fit_df["date"].max(),
        "valid_min_date": valid_df["date"].min(),
        "valid_max_date": valid_df["date"].max(),
        "test_min_date": test_df["date"].min(),
        "test_max_date": test_df["date"].max(),
    }
    return (
        fit_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        meta,
    )


# ---------------------------------------------------------------------
# Calibrator dataclasses
# ---------------------------------------------------------------------

@dataclass
class LinearCalibrator:
    intercept: float
    slope: float
    scale: str = "logit"
    eps: float = 1e-6
    n_train: int = 0


@dataclass
class WidthCalibrator:
    k: float
    objective: str = "interval_score"
    nominal_coverage: float = 0.95
    n_train: int = 0
    train_empirical_coverage: float = np.nan
    train_mean_width: float = np.nan
    train_mean_interval_score: float = np.nan


CALIBRATION_VARIANTS = ("standard", "shift", "k", "firm_shift")
MAIN_SELECTION_VARIANTS = ("standard", "shift", "k")


# ---------------------------------------------------------------------
# Calibrator fitting / mapping
# ---------------------------------------------------------------------

def fit_linear_calibrator(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    scale: str = "logit",
    eps: float = 1e-6,
) -> LinearCalibrator:
    use = df[[x_col, y_col]].copy().dropna()

    x = _to_num(use[x_col])
    y = _to_num(use[y_col])
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if x.size == 0:
        raise ValueError("No valid observations available to fit the calibrator.")

    if scale == "logit":
        xs = _logit(x, eps=eps)
        ys = _logit(y, eps=eps)
    elif scale == "level":
        xs = x
        ys = y
    else:
        raise ValueError("scale must be 'level' or 'logit'.")

    X = np.column_stack([np.ones(xs.size), xs])
    beta, *_ = np.linalg.lstsq(X, ys, rcond=None)

    return LinearCalibrator(
        intercept=float(beta[0]),
        slope=float(beta[1]),
        scale=scale,
        eps=eps,
        n_train=int(xs.size),
    )


def apply_linear_mapping(
    x,
    *,
    calibrator: LinearCalibrator,
) -> np.ndarray:
    x_arr = _to_num(x)
    out = np.full_like(x_arr, np.nan, dtype=float)
    ok = np.isfinite(x_arr)
    xv = x_arr[ok]

    if calibrator.scale == "logit":
        xv = np.clip(xv, calibrator.eps, 1.0 - calibrator.eps)
        mapped = _expit(
            calibrator.intercept + calibrator.slope * _logit(xv, eps=calibrator.eps)
        )
    else:
        mapped = calibrator.intercept + calibrator.slope * xv

    out[ok] = np.clip(mapped, calibrator.eps, 1.0 - calibrator.eps)
    return out


def apply_linear_interval_mapping(
    df: pd.DataFrame,
    *,
    lo_col: str,
    med_col: str,
    hi_col: str,
    calibrator: LinearCalibrator,
    out_prefix: str,
) -> pd.DataFrame:
    out = df.copy()

    lo_map = apply_linear_mapping(out[lo_col], calibrator=calibrator)
    med_map = apply_linear_mapping(out[med_col], calibrator=calibrator)
    hi_map = apply_linear_mapping(out[hi_col], calibrator=calibrator)

    lo_new = np.minimum(lo_map, hi_map)
    hi_new = np.maximum(lo_map, hi_map)
    med_new = np.clip(med_map, lo_new, hi_new)

    bad = np.isnan(lo_map) | np.isnan(med_map) | np.isnan(hi_map)
    lo_new[bad] = np.nan
    med_new[bad] = np.nan
    hi_new[bad] = np.nan

    out[f"{out_prefix}_lo"] = lo_new
    out[f"{out_prefix}_med"] = med_new
    out[f"{out_prefix}_hi"] = hi_new
    return out


def apply_center_mapping_preserve_width(
    df: pd.DataFrame,
    *,
    lo_col: str,
    med_col: str,
    hi_col: str,
    calibrator: LinearCalibrator,
    out_prefix: str,
) -> pd.DataFrame:
    out = df.copy()

    lo_raw = _to_num(out[lo_col])
    med_raw = _to_num(out[med_col])
    hi_raw = _to_num(out[hi_col])
    med_map = apply_linear_mapping(out[med_col], calibrator=calibrator)

    lo_new = np.full_like(med_map, np.nan, dtype=float)
    med_new = np.full_like(med_map, np.nan, dtype=float)
    hi_new = np.full_like(med_map, np.nan, dtype=float)

    ok = np.isfinite(lo_raw) & np.isfinite(med_raw) & np.isfinite(hi_raw) & np.isfinite(med_map)
    if np.any(ok):
        lv = lo_raw[ok]
        mv = med_raw[ok]
        hv = hi_raw[ok]
        mc = med_map[ok]

        lo0 = np.minimum(lv, hv)
        hi0 = np.maximum(lv, hv)
        med0 = np.clip(mv, lo0, hi0)

        left_hw = np.maximum(med0 - lo0, 0.0)
        right_hw = np.maximum(hi0 - med0, 0.0)

        lo1 = np.clip(mc - left_hw, calibrator.eps, 1.0 - calibrator.eps)
        hi1 = np.clip(mc + right_hw, calibrator.eps, 1.0 - calibrator.eps)

        lo2 = np.minimum(lo1, hi1)
        hi2 = np.maximum(lo1, hi1)
        med2 = np.clip(mc, lo2, hi2)

        lo_new[ok] = lo2
        med_new[ok] = med2
        hi_new[ok] = hi2

    out[f"{out_prefix}_lo"] = lo_new
    out[f"{out_prefix}_med"] = med_new
    out[f"{out_prefix}_hi"] = hi_new
    return out


def _rescale_interval_about_median(
    lo,
    med,
    hi,
    *,
    k: float,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo_arr = _to_num(lo)
    med_arr = _to_num(med)
    hi_arr = _to_num(hi)

    lo_new = np.full_like(lo_arr, np.nan, dtype=float)
    med_new = np.full_like(med_arr, np.nan, dtype=float)
    hi_new = np.full_like(hi_arr, np.nan, dtype=float)

    ok = np.isfinite(lo_arr) & np.isfinite(med_arr) & np.isfinite(hi_arr)
    if not np.any(ok):
        return lo_new, med_new, hi_new

    lv = lo_arr[ok]
    mv = med_arr[ok]
    hv = hi_arr[ok]

    lo0 = np.minimum(lv, hv)
    hi0 = np.maximum(lv, hv)
    med0 = np.clip(mv, lo0, hi0)

    left_hw = np.maximum(med0 - lo0, 0.0)
    right_hw = np.maximum(hi0 - med0, 0.0)

    lo1 = np.clip(med0 - float(k) * left_hw, eps, 1.0 - eps)
    hi1 = np.clip(med0 + float(k) * right_hw, eps, 1.0 - eps)

    lo2 = np.minimum(lo1, hi1)
    hi2 = np.maximum(lo1, hi1)
    med2 = np.clip(med0, lo2, hi2)

    lo_new[ok] = lo2
    med_new[ok] = med2
    hi_new[ok] = hi2
    return lo_new, med_new, hi_new


def apply_interval_width_multiplier(
    df: pd.DataFrame,
    *,
    lo_col: str,
    med_col: str,
    hi_col: str,
    k: float,
    out_prefix: str,
    eps: float = 1e-6,
) -> pd.DataFrame:
    out = df.copy()

    lo_new, med_new, hi_new = _rescale_interval_about_median(
        out[lo_col],
        out[med_col],
        out[hi_col],
        k=k,
        eps=eps,
    )

    out[f"{out_prefix}_lo"] = lo_new
    out[f"{out_prefix}_med"] = med_new
    out[f"{out_prefix}_hi"] = hi_new
    return out


def fit_interval_width_multiplier(
    df: pd.DataFrame,
    *,
    target_col: str,
    lo_col: str,
    med_col: str,
    hi_col: str,
    nominal_coverage: float = 0.95,
    objective: str = "interval_score",
    k_grid=None,
    eps: float = 1e-6,
) -> WidthCalibrator:
    use = df[[target_col, lo_col, med_col, hi_col]].copy().dropna()

    if use.empty:
        return WidthCalibrator(
            k=1.0,
            objective=objective,
            nominal_coverage=nominal_coverage,
            n_train=0,
        )

    y = _to_num(use[target_col])
    lo = _to_num(use[lo_col])
    med = _to_num(use[med_col])
    hi = _to_num(use[hi_col])

    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(med) & np.isfinite(hi)
    y = y[ok]
    lo = lo[ok]
    med = med[ok]
    hi = hi[ok]

    if y.size == 0:
        return WidthCalibrator(
            k=1.0,
            objective=objective,
            nominal_coverage=nominal_coverage,
            n_train=0,
        )

    if k_grid is None:
        k_grid = np.geomspace(0.25, 20.0, 161)
    else:
        k_grid = np.asarray(list(k_grid), dtype=float)

    k_grid = k_grid[np.isfinite(k_grid) & (k_grid > 0)]
    if k_grid.size == 0:
        raise ValueError("k_grid must contain at least one positive finite value.")

    alpha = 1.0 - nominal_coverage
    best = None

    for k in k_grid:
        lo_k, med_k, hi_k = _rescale_interval_about_median(
            lo,
            med,
            hi,
            k=float(k),
            eps=eps,
        )

        valid = np.isfinite(y) & np.isfinite(lo_k) & np.isfinite(med_k) & np.isfinite(hi_k)
        if not np.any(valid):
            continue

        yv = y[valid]
        lv = lo_k[valid]
        hv = hi_k[valid]

        cov = float(np.mean((yv >= lv) & (yv <= hv)))
        width = float(np.mean(hv - lv))
        iscore = float(np.nanmean(interval_score(yv, lv, hv, alpha=alpha)))

        if objective == "interval_score":
            rank = (iscore, abs(cov - nominal_coverage), abs(np.log(float(k))))
        elif objective == "coverage":
            rank = (abs(cov - nominal_coverage), iscore, abs(np.log(float(k))))
        else:
            raise ValueError("objective must be 'interval_score' or 'coverage'.")

        if (best is None) or (rank < best["rank"]):
            best = {
                "rank": rank,
                "k": float(k),
                "coverage": cov,
                "width": width,
                "iscore": iscore,
                "n_train": int(valid.sum()),
            }

    if best is None:
        return WidthCalibrator(
            k=1.0,
            objective=objective,
            nominal_coverage=nominal_coverage,
            n_train=0,
        )

    return WidthCalibrator(
        k=best["k"],
        objective=objective,
        nominal_coverage=nominal_coverage,
        n_train=best["n_train"],
        train_empirical_coverage=best["coverage"],
        train_mean_width=best["width"],
        train_mean_interval_score=best["iscore"],
    )


# ---------------------------------------------------------------------
# Firm-specific shift robustness
# ---------------------------------------------------------------------

def fit_firm_specific_calibrators(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    scale: str = "logit",
    eps: float = 1e-6,
    id_col: str = "gvkey",
    min_obs: int = 20,
    fallback_calibrator: LinearCalibrator | None = None,
) -> tuple[dict[str, LinearCalibrator], pd.DataFrame]:
    d = df[[id_col, x_col, y_col]].copy()
    d[id_col] = d[id_col].astype(str)

    calibrators: dict[str, LinearCalibrator] = {}
    rows = []

    for gv, g in d.groupby(id_col, sort=True):
        use = g[[x_col, y_col]].dropna()
        n_obs = int(len(use))

        used_fallback = False
        cal = None
        error_msg = None

        if n_obs >= min_obs:
            try:
                cal = fit_linear_calibrator(
                    use,
                    x_col=x_col,
                    y_col=y_col,
                    scale=scale,
                    eps=eps,
                )
            except Exception as exc:
                error_msg = str(exc)
                cal = None

        if cal is None:
            if fallback_calibrator is None:
                raise ValueError(
                    f"Could not fit firm-specific calibrator for gvkey={gv} and no fallback calibrator was provided."
                )
            cal = fallback_calibrator
            used_fallback = True

        calibrators[str(gv)] = cal
        rows.append(
            {
                "gvkey": str(gv),
                "n_fit_obs": n_obs,
                "used_fallback": bool(used_fallback),
                "intercept": float(cal.intercept),
                "slope": float(cal.slope),
                "scale": cal.scale,
                "eps": float(cal.eps),
                "error_msg": error_msg,
            }
        )

    return calibrators, pd.DataFrame(rows)


def apply_firm_specific_center_mapping_preserve_width(
    df: pd.DataFrame,
    *,
    lo_col: str,
    med_col: str,
    hi_col: str,
    calibrators_by_firm: dict[str, LinearCalibrator],
    fallback_calibrator: LinearCalibrator,
    out_prefix: str,
    id_col: str = "gvkey",
) -> pd.DataFrame:
    out = df.copy()
    out[id_col] = out[id_col].astype(str)

    lo_new = np.full(len(out), np.nan, dtype=float)
    med_new = np.full(len(out), np.nan, dtype=float)
    hi_new = np.full(len(out), np.nan, dtype=float)

    for gv, idx in out.groupby(id_col).groups.items():
        cal = calibrators_by_firm.get(str(gv), fallback_calibrator)
        g = out.loc[list(idx), [lo_col, med_col, hi_col]].copy()

        tmp = apply_center_mapping_preserve_width(
            g,
            lo_col=lo_col,
            med_col=med_col,
            hi_col=hi_col,
            calibrator=cal,
            out_prefix="_tmp",
        )

        lo_new[list(idx)] = tmp["_tmp_lo"].to_numpy(dtype=float)
        med_new[list(idx)] = tmp["_tmp_med"].to_numpy(dtype=float)
        hi_new[list(idx)] = tmp["_tmp_hi"].to_numpy(dtype=float)

    out[f"{out_prefix}_lo"] = lo_new
    out[f"{out_prefix}_med"] = med_new
    out[f"{out_prefix}_hi"] = hi_new
    return out


# ---------------------------------------------------------------------
# Scoring / summaries / selection
# ---------------------------------------------------------------------

def interval_score(
    y,
    lo,
    hi,
    *,
    alpha: float = 0.05,
) -> np.ndarray:
    y_arr = _to_num(y)
    lo_arr = _to_num(lo)
    hi_arr = _to_num(hi)

    out = np.full_like(y_arr, np.nan, dtype=float)
    ok = np.isfinite(y_arr) & np.isfinite(lo_arr) & np.isfinite(hi_arr)

    yv = y_arr[ok]
    lv = lo_arr[ok]
    hv = hi_arr[ok]

    out[ok] = (
        (hv - lv)
        + (2.0 / alpha) * np.maximum(lv - yv, 0.0)
        + (2.0 / alpha) * np.maximum(yv - hv, 0.0)
    )
    return out


def summarize_interval_set(
    df: pd.DataFrame,
    *,
    target_col: str,
    lo_col: str,
    med_col: str,
    hi_col: str,
    model: str,
    sample: str,
    variant: str,
    nominal_coverage: float = 0.95,
) -> dict:
    use = df[[target_col, lo_col, med_col, hi_col]].copy().dropna()

    if use.empty:
        return {
            "model": model,
            "sample": sample,
            "variant": variant,
            "n_obs": 0,
            "target_coverage": nominal_coverage,
            "empirical_coverage": np.nan,
            "coverage_gap": np.nan,
            "mean_width": np.nan,
            "mean_interval_score": np.nan,
            "mean_abs_error_med": np.nan,
        }

    y = use[target_col].to_numpy(dtype=float)
    lo = use[lo_col].to_numpy(dtype=float)
    med = use[med_col].to_numpy(dtype=float)
    hi = use[hi_col].to_numpy(dtype=float)
    alpha = 1.0 - nominal_coverage
    coverage = float(np.mean((y >= lo) & (y <= hi)))

    return {
        "model": model,
        "sample": sample,
        "variant": variant,
        "n_obs": int(len(use)),
        "target_coverage": float(nominal_coverage),
        "empirical_coverage": coverage,
        "coverage_gap": float(abs(coverage - nominal_coverage)),
        "mean_width": float(np.mean(hi - lo)),
        "mean_interval_score": float(np.nanmean(interval_score(y, lo, hi, alpha=alpha))),
        "mean_abs_error_med": float(np.mean(np.abs(y - med))),
    }


def _variant_output_prefix(model_label: str, variant: str) -> str:
    return f"{model_label}_Q_{variant}"


def _interval_score_col(model_label: str, variant: str) -> str:
    if variant == "raw":
        return f"IS_raw_{model_label}"
    return f"IS_{variant}_{model_label}"


def build_variant_selection_table(
    summary_table: pd.DataFrame,
    *,
    model_label: str,
    sample: str = "validation",
    variants: tuple[str, ...] | list[str] = MAIN_SELECTION_VARIANTS,
    primary_metric: str = "mean_interval_score",
) -> pd.DataFrame:
    use = summary_table.copy()
    use = use[
        (use["model"] == model_label)
        & (use["sample"] == sample)
        & (use["variant"].isin(list(variants)))
    ].copy()

    if use.empty:
        return use

    use = use.sort_values(
        [primary_metric, "coverage_gap", "mean_abs_error_med", "mean_width"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    use["rank"] = np.arange(1, len(use) + 1)
    return use


# ---------------------------------------------------------------------
# Variant application
# ---------------------------------------------------------------------

def _apply_variant_with_metadata(
    fit_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    model_label: str,
    lo_col: str,
    med_col: str,
    hi_col: str,
    target_col: str,
    calibrator: LinearCalibrator,
    nominal_coverage: float,
    variant: str,
    width_objective: str = "interval_score",
    width_k: float | None = None,
    width_k_grid=None,
    firm_min_obs: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, WidthCalibrator | None, pd.DataFrame | None]:
    out_prefix = _variant_output_prefix(model_label, variant)
    width_cal = None
    firm_cal_table = None

    if variant == "standard":
        fit_out = apply_linear_interval_mapping(
            fit_df,
            lo_col=lo_col,
            med_col=med_col,
            hi_col=hi_col,
            calibrator=calibrator,
            out_prefix=out_prefix,
        )
        valid_out = apply_linear_interval_mapping(
            valid_df,
            lo_col=lo_col,
            med_col=med_col,
            hi_col=hi_col,
            calibrator=calibrator,
            out_prefix=out_prefix,
        )
        test_out = apply_linear_interval_mapping(
            test_df,
            lo_col=lo_col,
            med_col=med_col,
            hi_col=hi_col,
            calibrator=calibrator,
            out_prefix=out_prefix,
        )

    elif variant == "shift":
        fit_out = apply_center_mapping_preserve_width(
            fit_df,
            lo_col=lo_col,
            med_col=med_col,
            hi_col=hi_col,
            calibrator=calibrator,
            out_prefix=out_prefix,
        )
        valid_out = apply_center_mapping_preserve_width(
            valid_df,
            lo_col=lo_col,
            med_col=med_col,
            hi_col=hi_col,
            calibrator=calibrator,
            out_prefix=out_prefix,
        )
        test_out = apply_center_mapping_preserve_width(
            test_df,
            lo_col=lo_col,
            med_col=med_col,
            hi_col=hi_col,
            calibrator=calibrator,
            out_prefix=out_prefix,
        )

    elif variant == "k":
        base_prefix = f"{out_prefix}_base"

        fit_base = apply_linear_interval_mapping(
            fit_df,
            lo_col=lo_col,
            med_col=med_col,
            hi_col=hi_col,
            calibrator=calibrator,
            out_prefix=base_prefix,
        )
        valid_base = apply_linear_interval_mapping(
            valid_df,
            lo_col=lo_col,
            med_col=med_col,
            hi_col=hi_col,
            calibrator=calibrator,
            out_prefix=base_prefix,
        )
        test_base = apply_linear_interval_mapping(
            test_df,
            lo_col=lo_col,
            med_col=med_col,
            hi_col=hi_col,
            calibrator=calibrator,
            out_prefix=base_prefix,
        )

        if width_k is None:
            width_cal = fit_interval_width_multiplier(
                fit_base,
                target_col=target_col,
                lo_col=f"{base_prefix}_lo",
                med_col=f"{base_prefix}_med",
                hi_col=f"{base_prefix}_hi",
                nominal_coverage=nominal_coverage,
                objective=width_objective,
                k_grid=width_k_grid,
                eps=calibrator.eps,
            )
        else:
            width_cal = WidthCalibrator(
                k=float(width_k),
                objective="fixed",
                nominal_coverage=nominal_coverage,
                n_train=int(
                    fit_base[
                        [target_col, f"{base_prefix}_lo", f"{base_prefix}_med", f"{base_prefix}_hi"]
                    ]
                    .dropna()
                    .shape[0]
                ),
            )

        fit_out = apply_interval_width_multiplier(
            fit_base,
            lo_col=f"{base_prefix}_lo",
            med_col=f"{base_prefix}_med",
            hi_col=f"{base_prefix}_hi",
            k=width_cal.k,
            out_prefix=out_prefix,
            eps=calibrator.eps,
        )
        valid_out = apply_interval_width_multiplier(
            valid_base,
            lo_col=f"{base_prefix}_lo",
            med_col=f"{base_prefix}_med",
            hi_col=f"{base_prefix}_hi",
            k=width_cal.k,
            out_prefix=out_prefix,
            eps=calibrator.eps,
        )
        test_out = apply_interval_width_multiplier(
            test_base,
            lo_col=f"{base_prefix}_lo",
            med_col=f"{base_prefix}_med",
            hi_col=f"{base_prefix}_hi",
            k=width_cal.k,
            out_prefix=out_prefix,
            eps=calibrator.eps,
        )

    elif variant == "firm_shift":
        firm_cals, firm_cal_table = fit_firm_specific_calibrators(
            fit_df,
            x_col=med_col,
            y_col=target_col,
            scale=calibrator.scale,
            eps=calibrator.eps,
            min_obs=firm_min_obs,
            fallback_calibrator=calibrator,
        )

        fit_out = apply_firm_specific_center_mapping_preserve_width(
            fit_df,
            lo_col=lo_col,
            med_col=med_col,
            hi_col=hi_col,
            calibrators_by_firm=firm_cals,
            fallback_calibrator=calibrator,
            out_prefix=out_prefix,
        )
        valid_out = apply_firm_specific_center_mapping_preserve_width(
            valid_df,
            lo_col=lo_col,
            med_col=med_col,
            hi_col=hi_col,
            calibrators_by_firm=firm_cals,
            fallback_calibrator=calibrator,
            out_prefix=out_prefix,
        )
        test_out = apply_firm_specific_center_mapping_preserve_width(
            test_df,
            lo_col=lo_col,
            med_col=med_col,
            hi_col=hi_col,
            calibrators_by_firm=firm_cals,
            fallback_calibrator=calibrator,
            out_prefix=out_prefix,
        )

    else:
        raise ValueError(f"Unknown variant: {variant}")

    alpha = 1.0 - nominal_coverage
    score_col = _interval_score_col(model_label, variant)

    for d in [fit_out, valid_out, test_out]:
        d[score_col] = interval_score(
            d[target_col],
            d[f"{out_prefix}_lo"],
            d[f"{out_prefix}_hi"],
            alpha=alpha,
        )

    return fit_out, valid_out, test_out, width_cal, firm_cal_table


# ---------------------------------------------------------------------
# Main 3-way runner
# ---------------------------------------------------------------------

def run_all_interval_calibrations_three_way(
    eval_df: pd.DataFrame,
    *,
    model_label: str,
    lo_col: str,
    med_col: str,
    hi_col: str,
    fit_end: str | pd.Timestamp,
    valid_end: str | pd.Timestamp,
    test_end: str | pd.Timestamp | None = None,
    start_date: str | pd.Timestamp | None = None,
    target_col: str = "PD_1y_CDS_par",
    nominal_coverage: float = 0.95,
    calibration_scale: str = "logit",
    variants: tuple[str, ...] | list[str] = CALIBRATION_VARIANTS,
    selection_variants: tuple[str, ...] | list[str] = MAIN_SELECTION_VARIANTS,
    selection_metric: str = "mean_interval_score",
    width_objective: str = "interval_score",
    width_k: float | None = None,
    width_k_grid=None,
    firm_min_obs: int = 20,
) -> dict:
    variants = tuple(variants)
    invalid = [v for v in variants if v not in CALIBRATION_VARIANTS]
    if invalid:
        raise ValueError(f"Unknown variants: {invalid}")

    fit_df, valid_df, test_df, split_info = split_panel_three_way(
        eval_df,
        fit_end=fit_end,
        valid_end=valid_end,
        test_end=test_end,
        start_date=start_date,
    )

    pooled_calibrator = fit_linear_calibrator(
        fit_df,
        x_col=med_col,
        y_col=target_col,
        scale=calibration_scale,
    )

    fit_master = fit_df.copy()
    valid_master = valid_df.copy()
    test_master = test_df.copy()

    alpha = 1.0 - nominal_coverage
    raw_score_col = _interval_score_col(model_label, "raw")

    for d in [fit_master, valid_master, test_master]:
        d[raw_score_col] = interval_score(
            d[target_col],
            d[lo_col],
            d[hi_col],
            alpha=alpha,
        )

    summary_rows = []
    for sample_name, sample_df in [
        ("fit", fit_master),
        ("validation", valid_master),
        ("test", test_master),
    ]:
        summary_rows.append(
            summarize_interval_set(
                sample_df,
                target_col=target_col,
                lo_col=lo_col,
                med_col=med_col,
                hi_col=hi_col,
                model=model_label,
                sample=sample_name,
                variant="raw",
                nominal_coverage=nominal_coverage,
            )
        )

    calibrator_rows = []
    firm_calibrator_tables: dict[str, pd.DataFrame] = {}
    results: dict[str, dict] = {}

    for variant in variants:
        fit_var, valid_var, test_var, width_cal, firm_cal_table = _apply_variant_with_metadata(
            fit_df,
            valid_df,
            test_df,
            model_label=model_label,
            lo_col=lo_col,
            med_col=med_col,
            hi_col=hi_col,
            target_col=target_col,
            calibrator=pooled_calibrator,
            nominal_coverage=nominal_coverage,
            variant=variant,
            width_objective=width_objective,
            width_k=width_k,
            width_k_grid=width_k_grid,
            firm_min_obs=firm_min_obs,
        )

        prefix = _variant_output_prefix(model_label, variant)
        score_col = _interval_score_col(model_label, variant)
        keep_cols = [f"{prefix}_lo", f"{prefix}_med", f"{prefix}_hi", score_col]

        fit_master = fit_master.merge(
            fit_var[["gvkey", "date"] + keep_cols],
            on=["gvkey", "date"],
            how="left",
            validate="one_to_one",
        )
        valid_master = valid_master.merge(
            valid_var[["gvkey", "date"] + keep_cols],
            on=["gvkey", "date"],
            how="left",
            validate="one_to_one",
        )
        test_master = test_master.merge(
            test_var[["gvkey", "date"] + keep_cols],
            on=["gvkey", "date"],
            how="left",
            validate="one_to_one",
        )

        for sample_name, sample_df in [
            ("fit", fit_var),
            ("validation", valid_var),
            ("test", test_var),
        ]:
            summary_rows.append(
                summarize_interval_set(
                    sample_df,
                    target_col=target_col,
                    lo_col=f"{prefix}_lo",
                    med_col=f"{prefix}_med",
                    hi_col=f"{prefix}_hi",
                    model=model_label,
                    sample=sample_name,
                    variant=variant,
                    nominal_coverage=nominal_coverage,
                )
            )

        calibrator_rows.append(
            {
                "model": model_label,
                "variant": variant,
                "intercept": pooled_calibrator.intercept,
                "slope": pooled_calibrator.slope,
                "scale": pooled_calibrator.scale,
                "eps": pooled_calibrator.eps,
                "n_fit": pooled_calibrator.n_train,
                "width_k": np.nan if width_cal is None else width_cal.k,
                "width_objective": None if width_cal is None else width_cal.objective,
                "width_train_n": None if width_cal is None else width_cal.n_train,
                "width_train_empirical_coverage": None if width_cal is None else width_cal.train_empirical_coverage,
                "width_train_mean_width": None if width_cal is None else width_cal.train_mean_width,
                "width_train_mean_interval_score": None if width_cal is None else width_cal.train_mean_interval_score,
                "firm_min_obs": None if variant != "firm_shift" else firm_min_obs,
                "n_firm_specific_calibrators": None if firm_cal_table is None else int(len(firm_cal_table)),
                "n_firm_fallbacks": None if firm_cal_table is None else int(firm_cal_table["used_fallback"].sum()),
            }
        )

        if firm_cal_table is not None:
            firm_calibrator_tables[variant] = firm_cal_table.copy()

        results[variant] = {
            "fit_df": fit_var,
            "validation_df": valid_var,
            "test_df": test_var,
            "calibrator": pooled_calibrator,
            "width_calibrator": width_cal,
            "firm_calibrator_table": firm_cal_table,
        }

    summary_table = pd.DataFrame(summary_rows)
    calibrator_table = pd.DataFrame(calibrator_rows)

    selection_table = build_variant_selection_table(
        summary_table,
        model_label=model_label,
        sample="validation",
        variants=selection_variants,
        primary_metric=selection_metric,
    )

    selected_variant = None
    if not selection_table.empty:
        selected_variant = str(selection_table.iloc[0]["variant"])

    return {
        "split_info": split_info,
        "fit_df": fit_master,
        "validation_df": valid_master,
        "test_df": test_master,
        "results": results,
        "summary_table": summary_table,
        "calibrator_table": calibrator_table,
        "selection_table": selection_table,
        "selected_variant": selected_variant,
        "selected_variant_test_summary": (
            summary_table[
                (summary_table["model"] == model_label)
                & (summary_table["sample"] == "test")
                & (summary_table["variant"] == selected_variant)
            ].reset_index(drop=True)
            if selected_variant is not None
            else pd.DataFrame()
        ),
        "pooled_calibrator": pooled_calibrator,
        "firm_calibrator_tables": firm_calibrator_tables,
    }


# ---------------------------------------------------------------------
# Merge / DM helpers
# ---------------------------------------------------------------------

def merge_bayesian_interval_panels(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    keep_common: list[str] | None = None,
) -> pd.DataFrame:
    keep_common = keep_common or [
        "gvkey",
        "date",
        "window_idx",
        "rf_used",
        "B_used",
        "cds",
        "r_f",
        "lambda_CDS_triangle",
        "PD_1y_CDS_triangle",
        "lambda_CDS_par",
        "PD_1y_CDS_par",
    ]

    left = left_df.copy()
    right = right_df.copy()

    common_cols = [c for c in keep_common if c in left.columns]
    right_extra = [c for c in right.columns if c not in keep_common and c not in left.columns]

    return (
        left[common_cols + [c for c in left.columns if c not in common_cols]]
        .merge(
            right[["gvkey", "date"] + right_extra],
            on=["gvkey", "date"],
            how="inner",
            validate="one_to_one",
        )
        .sort_values(KEYS)
        .reset_index(drop=True)
    )


def dm_test_from_interval_scores(
    test_df: pd.DataFrame,
    *,
    score_a: str,
    score_b: str,
    date_col: str = "date",
    L: int | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
    variant: str | None = None,
    sample: str | None = None,
) -> pd.DataFrame:
    d = test_df[[date_col, score_a, score_b]].copy().dropna()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")

    by_date = d.groupby(date_col)[[score_a, score_b]].mean().sort_index()
    d_t = by_date[score_a] - by_date[score_b]

    out = newey_west_mean_test(d_t, L=L)
    out.update(
        {
            "score_a": score_a,
            "score_b": score_b,
            "label_a": score_a if label_a is None else label_a,
            "label_b": score_b if label_b is None else label_b,
            "variant": variant,
            "sample": sample,
        }
    )
    return pd.DataFrame([out])


def run_pairwise_interval_dm(
    sample_df: pd.DataFrame,
    *,
    model_a: str,
    model_b: str,
    variants: tuple[str, ...] | list[str] = ("raw", "standard", "shift", "k", "firm_shift"),
    L: int | None = None,
    sample_name: str | None = None,
) -> pd.DataFrame:
    rows = []
    for variant in variants:
        score_a = _interval_score_col(model_a, variant)
        score_b = _interval_score_col(model_b, variant)
        if score_a not in sample_df.columns or score_b not in sample_df.columns:
            continue

        rows.append(
            dm_test_from_interval_scores(
                sample_df,
                score_a=score_a,
                score_b=score_b,
                L=L,
                label_a=model_a,
                label_b=model_b,
                variant=variant,
                sample=sample_name,
            )
        )

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------
# Raw-width vs CDS-volatility diagnostic
# ---------------------------------------------------------------------

def build_daily_cds_rolling_volatility(
    cds_daily_df: pd.DataFrame,
    *,
    id_col: str = "gvkey",
    date_col: str = "date",
    spread_col: str = "cds",
    window_days: int = 20,
    min_periods: int = 10,
    return_type: str = "logdiff",   # "logdiff" or "diff"
    out_col: str | None = None,
) -> pd.DataFrame:
    """
    Build a daily rolling CDS volatility series by firm.

    Main choice:
        return_type="logdiff"  -> std of daily log spread changes
        return_type="diff"     -> std of daily spread changes
    """
    d = cds_daily_df[[id_col, date_col, spread_col]].copy()
    d[id_col] = d[id_col].astype(str)
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[spread_col] = pd.to_numeric(d[spread_col], errors="coerce")
    d = d.dropna(subset=[id_col, date_col, spread_col]).sort_values([id_col, date_col]).reset_index(drop=True)

    if out_col is None:
        suffix = "logdiff" if return_type == "logdiff" else "diff"
        out_col = f"cds_vol_{window_days}d_{suffix}"

    if return_type == "logdiff":
        d["_log_spread"] = np.where(d[spread_col] > 0, np.log(d[spread_col]), np.nan)
        d["_ret"] = d.groupby(id_col)["_log_spread"].diff()
    elif return_type == "diff":
        d["_ret"] = d.groupby(id_col)[spread_col].diff()
    else:
        raise ValueError("return_type must be 'logdiff' or 'diff'.")

    d[out_col] = (
        d.groupby(id_col)["_ret"]
         .rolling(window_days, min_periods=min_periods)
         .std()
         .reset_index(level=0, drop=True)
    )

    return d[[id_col, date_col, spread_col, out_col]].sort_values([id_col, date_col]).reset_index(drop=True)


def attach_cds_rolling_volatility_to_weekly_panel(
    weekly_df: pd.DataFrame,
    cds_daily_df: pd.DataFrame,
    *,
    id_col: str = "gvkey",
    date_col: str = "date",
    spread_col: str = "cds",
    window_days: int = 20,
    min_periods: int = 10,
    return_type: str = "logdiff",
    out_col: str | None = None,
) -> pd.DataFrame:
    """
    Build daily rolling CDS volatility and backward as-of merge it onto a weekly panel.
    """
    vol_daily = build_daily_cds_rolling_volatility(
        cds_daily_df,
        id_col=id_col,
        date_col=date_col,
        spread_col=spread_col,
        window_days=window_days,
        min_periods=min_periods,
        return_type=return_type,
        out_col=out_col,
    )

    if out_col is None:
        suffix = "logdiff" if return_type == "logdiff" else "diff"
        out_col = f"cds_vol_{window_days}d_{suffix}"

    weekly = weekly_df.copy()
    weekly[id_col] = weekly[id_col].astype(str)
    weekly[date_col] = pd.to_datetime(weekly[date_col], errors="coerce")
    weekly = weekly.sort_values([id_col, date_col]).reset_index(drop=True)

    merged = asof_merge_by_gvkey(
        weekly,
        vol_daily[[id_col, date_col, out_col]].sort_values([id_col, date_col]).reset_index(drop=True),
        on=date_col,
        by=id_col,
        direction="backward",
        allow_exact_matches=True,
    )
    return merged


def add_raw_interval_widths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add raw interval-width columns when the corresponding lo/hi columns exist.

    Output names:
        W_Q_NIG_raw, W_P_NIG_raw, W_Q_Merton_raw, W_P_Merton_raw
    """
    out = df.copy()

    specs = [
        ("PD_Q_NIG_lo", "PD_Q_NIG_hi", "W_Q_NIG_raw"),
        ("PD_P_NIG_lo", "PD_P_NIG_hi", "W_P_NIG_raw"),
        ("PD_Q_Merton_lo", "PD_Q_Merton_hi", "W_Q_Merton_raw"),
        ("PD_P_Merton_lo", "PD_P_Merton_hi", "W_P_Merton_raw"),
    ]

    for lo_col, hi_col, out_col in specs:
        if lo_col in out.columns and hi_col in out.columns:
            lo = pd.to_numeric(out[lo_col], errors="coerce")
            hi = pd.to_numeric(out[hi_col], errors="coerce")
            out[out_col] = hi - lo

    return out


def _check_scipy_available() -> None:
    if sps is None:
        raise ImportError(
            "This diagnostic uses scipy.stats. Please install scipy in your environment."
        )


def _safe_corr(x: pd.Series, y: pd.Series, *, method: str = "pearson") -> tuple[float, float]:
    _check_scipy_available()

    xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(xv) & np.isfinite(yv)

    xv = xv[ok]
    yv = yv[ok]

    if xv.size < 3:
        return np.nan, np.nan
    if np.nanstd(xv) <= 0 or np.nanstd(yv) <= 0:
        return np.nan, np.nan

    if method == "pearson":
        r, p = sps.pearsonr(xv, yv)
    elif method == "spearman":
        r, p = sps.spearmanr(xv, yv, nan_policy="omit")
    else:
        raise ValueError("method must be 'pearson' or 'spearman'.")

    return float(r), float(p)


def compute_firm_level_width_vol_correlations(
    panel_df: pd.DataFrame,
    *,
    width_col: str,
    vol_col: str,
    id_col: str = "gvkey",
    date_col: str = "date",
    method: str = "pearson",
    min_obs: int = 20,
) -> pd.DataFrame:
    """
    For each firm, compute corr(width, CDS volatility) over time.
    """
    out_cols = ["gvkey", "n_obs", "corr", "p_value", "width_mean", "vol_mean", "fisher_z"]

    d = panel_df[[id_col, date_col, width_col, vol_col]].copy()
    d[id_col] = d[id_col].astype(str)
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[width_col] = pd.to_numeric(d[width_col], errors="coerce")
    d[vol_col] = pd.to_numeric(d[vol_col], errors="coerce")
    d = d.sort_values([id_col, date_col]).reset_index(drop=True)

    rows = []
    for gv, g in d.groupby(id_col, sort=True):
        gg = g.dropna(subset=[width_col, vol_col]).copy()
        n_obs = int(len(gg))
        if n_obs < min_obs:
            continue

        r, p = _safe_corr(gg[width_col], gg[vol_col], method=method)

        rows.append(
            {
                "gvkey": str(gv),
                "n_obs": n_obs,
                "corr": r,
                "p_value": p,
                "width_mean": float(np.nanmean(gg[width_col])),
                "vol_mean": float(np.nanmean(gg[vol_col])),
            }
        )

    if not rows:
        return pd.DataFrame(columns=out_cols)

    out = pd.DataFrame(rows)
    clipped = np.clip(out["corr"].to_numpy(dtype=float), -0.999999, 0.999999)
    out["fisher_z"] = np.arctanh(clipped)
    return out[out_cols].sort_values(["gvkey"]).reset_index(drop=True)



def paired_fisher_z_test(
    corr_df_a: pd.DataFrame,
    corr_df_b: pd.DataFrame,
    *,
    label_a: str,
    label_b: str,
    id_col: str = "gvkey",
    corr_col: str = "corr",
    fisher_col: str = "fisher_z",
    n_col: str = "n_obs",
    alternative: str = "two-sided",   # "two-sided", "greater", "less"
    min_pairs: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Paired test across firms on Fisher-z transformed correlations.

    Difference tested:
        d_i = z_i(label_a) - z_i(label_b)
    """
    _check_scipy_available()

    needed = [id_col, corr_col, fisher_col, n_col]

    if corr_df_a.empty or any(c not in corr_df_a.columns for c in needed):
        pairs = pd.DataFrame(columns=[
            id_col, "corr_a", "fisher_z_a", "n_obs_a",
            "corr_b", "fisher_z_b", "n_obs_b"
        ])
        summary = pd.DataFrame([{
            "label_a": label_a,
            "label_b": label_b,
            "alternative": alternative,
            "n_firms": 0,
            "mean_corr_a": np.nan,
            "median_corr_a": np.nan,
            "positive_share_a": np.nan,
            "mean_corr_b": np.nan,
            "median_corr_b": np.nan,
            "positive_share_b": np.nan,
            "mean_diff_fisher_z": np.nan,
            "t_stat": np.nan,
            "p_value_t": np.nan,
            "wilcoxon_stat": np.nan,
            "p_value_wilcoxon": np.nan,
        }])
        return summary, pairs

    if corr_df_b.empty or any(c not in corr_df_b.columns for c in needed):
        pairs = pd.DataFrame(columns=[
            id_col, "corr_a", "fisher_z_a", "n_obs_a",
            "corr_b", "fisher_z_b", "n_obs_b"
        ])
        summary = pd.DataFrame([{
            "label_a": label_a,
            "label_b": label_b,
            "alternative": alternative,
            "n_firms": 0,
            "mean_corr_a": np.nan,
            "median_corr_a": np.nan,
            "positive_share_a": np.nan,
            "mean_corr_b": np.nan,
            "median_corr_b": np.nan,
            "positive_share_b": np.nan,
            "mean_diff_fisher_z": np.nan,
            "t_stat": np.nan,
            "p_value_t": np.nan,
            "wilcoxon_stat": np.nan,
            "p_value_wilcoxon": np.nan,
        }])
        return summary, pairs

    left = corr_df_a[[id_col, corr_col, fisher_col, n_col]].copy().rename(
        columns={
            corr_col: "corr_a",
            fisher_col: "fisher_z_a",
            n_col: "n_obs_a",
        }
    )
    right = corr_df_b[[id_col, corr_col, fisher_col, n_col]].copy().rename(
        columns={
            corr_col: "corr_b",
            fisher_col: "fisher_z_b",
            n_col: "n_obs_b",
        }
    )

    pairs = (
        left.merge(right, on=id_col, how="inner", validate="one_to_one")
            .dropna(subset=["fisher_z_a", "fisher_z_b"])
            .reset_index(drop=True)
    )

    if len(pairs) < min_pairs:
        summary = pd.DataFrame([{
            "label_a": label_a,
            "label_b": label_b,
            "alternative": alternative,
            "n_firms": int(len(pairs)),
            "mean_corr_a": np.nan if pairs.empty else float(np.nanmean(pairs["corr_a"])),
            "median_corr_a": np.nan if pairs.empty else float(np.nanmedian(pairs["corr_a"])),
            "positive_share_a": np.nan if pairs.empty else float(np.mean(pairs["corr_a"] > 0)),
            "mean_corr_b": np.nan if pairs.empty else float(np.nanmean(pairs["corr_b"])),
            "median_corr_b": np.nan if pairs.empty else float(np.nanmedian(pairs["corr_b"])),
            "positive_share_b": np.nan if pairs.empty else float(np.mean(pairs["corr_b"] > 0)),
            "mean_diff_fisher_z": np.nan,
            "t_stat": np.nan,
            "p_value_t": np.nan,
            "wilcoxon_stat": np.nan,
            "p_value_wilcoxon": np.nan,
        }])
        return summary, pairs

    diff = pairs["fisher_z_a"].to_numpy(dtype=float) - pairs["fisher_z_b"].to_numpy(dtype=float)
    n = diff.size
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else np.nan

    if np.isfinite(sd_diff) and sd_diff > 0 and n > 1:
        t_stat = mean_diff / (sd_diff / np.sqrt(n))
        if alternative == "two-sided":
            p_t = float(2.0 * sps.t.sf(np.abs(t_stat), df=n - 1))
        elif alternative == "greater":
            p_t = float(sps.t.sf(t_stat, df=n - 1))
        elif alternative == "less":
            p_t = float(sps.t.cdf(t_stat, df=n - 1))
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
    else:
        t_stat = np.nan
        p_t = np.nan

    try:
        w = sps.wilcoxon(diff, alternative=alternative, zero_method="wilcox", correction=False)
        w_stat = float(w.statistic)
        p_w = float(w.pvalue)
    except Exception:
        w_stat = np.nan
        p_w = np.nan

    summary = pd.DataFrame([{
        "label_a": label_a,
        "label_b": label_b,
        "alternative": alternative,
        "n_firms": int(n),
        "mean_corr_a": float(np.mean(pairs["corr_a"])),
        "median_corr_a": float(np.median(pairs["corr_a"])),
        "positive_share_a": float(np.mean(pairs["corr_a"] > 0)),
        "mean_corr_b": float(np.mean(pairs["corr_b"])),
        "median_corr_b": float(np.median(pairs["corr_b"])),
        "positive_share_b": float(np.mean(pairs["corr_b"] > 0)),
        "mean_diff_fisher_z": mean_diff,
        "t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "p_value_t": p_t,
        "wilcoxon_stat": w_stat,
        "p_value_wilcoxon": p_w,
    }])

    return summary, pairs


def run_width_vol_model_comparison(
    panel_df: pd.DataFrame,
    *,
    width_col_a: str,
    width_col_b: str,
    vol_col: str,
    label_a: str,
    label_b: str,
    method: str = "pearson",
    min_obs: int = 20,
    alternative: str = "two-sided",
) -> dict:
    """
    Main compact diagnostic:
        1) one corr(width, CDSVol) per firm for model A
        2) one corr(width, CDSVol) per firm for model B
        3) paired Fisher-z test across firms
    """
    corr_a = compute_firm_level_width_vol_correlations(
        panel_df,
        width_col=width_col_a,
        vol_col=vol_col,
        method=method,
        min_obs=min_obs,
    )
    corr_b = compute_firm_level_width_vol_correlations(
        panel_df,
        width_col=width_col_b,
        vol_col=vol_col,
        method=method,
        min_obs=min_obs,
    )

    test_table, paired_corrs = paired_fisher_z_test(
        corr_a,
        corr_b,
        label_a=label_a,
        label_b=label_b,
        alternative=alternative,
    )

    test_table["width_col_a"] = width_col_a
    test_table["width_col_b"] = width_col_b
    test_table["vol_col"] = vol_col
    test_table["corr_method"] = method
    test_table["min_obs"] = min_obs

    return {
        "corr_a": corr_a,
        "corr_b": corr_b,
        "paired_corrs": paired_corrs,
        "test_table": test_table,
    }


def run_all_width_vol_model_comparisons(
    panel_df: pd.DataFrame,
    *,
    vol_col: str = "cds_vol_20d_logdiff",
    method: str = "pearson",
    min_obs: int = 20,
    alternative: str = "two-sided",
) -> dict:
    """
    Convenience wrapper for:
        - Q widths: NIG vs Merton
        - P widths: NIG vs Merton
    """
    specs = [
        ("Q", "W_Q_NIG_raw", "W_Q_Merton_raw"),
        ("P", "W_P_NIG_raw", "W_P_Merton_raw"),
    ]

    results = {}
    rows = []

    for width_family, col_a, col_b in specs:
        if col_a not in panel_df.columns or col_b not in panel_df.columns:
            continue

        out = run_width_vol_model_comparison(
            panel_df,
            width_col_a=col_a,
            width_col_b=col_b,
            vol_col=vol_col,
            label_a="NIG",
            label_b="Merton",
            method=method,
            min_obs=min_obs,
            alternative=alternative,
        )
        tbl = out["test_table"].copy()
        tbl["width_family"] = width_family
        rows.append(tbl)
        results[width_family] = out

    summary_table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return {
        "results": results,
        "summary_table": summary_table,
    }


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def plot_interval_panels_by_firm(
    plot_df: pd.DataFrame,
    *,
    target_col: str,
    lo_col: str,
    hi_col: str,
    n_firms: int = 21,
    use_specific_firms: list[str] | None = None,
    title_suffix: str = "",
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (5.0, 3.2),
):
    d = plot_df.copy()
    d["gvkey"] = d["gvkey"].astype(str)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.sort_values(["gvkey", "date"]).reset_index(drop=True)

    needed = ["gvkey", "date", target_col, lo_col, hi_col]
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise KeyError(f"Missing columns for plotting: {missing}")

    if use_specific_firms is None:
        firm_order = (
            d.groupby("gvkey")[target_col]
            .size()
            .sort_values(ascending=False)
            .index.tolist()[:n_firms]
        )
    else:
        firm_order = [str(x) for x in use_specific_firms][:n_firms]

    n = len(firm_order)
    if n == 0:
        raise ValueError("No firms available to plot.")

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    axes = axes.ravel()

    for ax, gv in zip(axes, firm_order):
        g = d[d["gvkey"] == gv].dropna(subset=[target_col, lo_col, hi_col]).copy()
        if g.empty:
            ax.set_visible(False)
            continue

        ax.plot(g["date"], g[target_col], label="CDS-implied PD")
        ax.plot(g["date"], g[lo_col], linestyle="--", label="Lower")
        ax.plot(g["date"], g[hi_col], linestyle="--", label="Upper")
        ax.set_title(f"gvkey {gv}")
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[n:]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(title_suffix)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_average_interval(
    plot_df: pd.DataFrame,
    *,
    target_col: str,
    lo_col: str,
    hi_col: str,
    title: str,
    target_label: str = "Average CDS-implied PD",
    lo_label: str = "Average lower",
    hi_label: str = "Average upper",
):
    avg_df = (
        plot_df.groupby("date", as_index=False)[[target_col, lo_col, hi_col]]
        .mean()
        .sort_values("date")
    )

    fig = plt.figure(figsize=(14, 5))
    plt.plot(avg_df["date"], avg_df[target_col], label=target_label)
    plt.plot(avg_df["date"], avg_df[lo_col], linestyle="--", label=lo_label)
    plt.plot(avg_df["date"], avg_df[hi_col], linestyle="--", label=hi_label)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("PD")
    plt.legend()
    plt.tight_layout()
    return fig