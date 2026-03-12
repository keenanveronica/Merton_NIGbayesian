from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["gvkey", "date"]


def prepare_panel(
    df: pd.DataFrame,
    *,
    keep_cols: list[str],
    rename_map: dict[str, str],
    numeric_cols: list[str],
) -> pd.DataFrame:
    out = (
        df.copy()
        .assign(
            gvkey=lambda x: x["gvkey"].astype(str),
            date=lambda x: pd.to_datetime(x["date"], errors="coerce"),
        )
        .loc[:, keep_cols]
        .rename(columns=rename_map)
        .sort_values(KEYS)
        .drop_duplicates(KEYS, keep="last")
        .reset_index(drop=True)
    )
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return out


def asof_merge_by_gvkey(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: str = "date",
    by: str = "gvkey",
    direction: str = "backward",
    allow_exact_matches: bool = True,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    right_groups = {k: g.sort_values(on).reset_index(drop=True) for k, g in right.groupby(by, sort=False)}

    for gv, g_left in left.groupby(by, sort=False):
        g_right = right_groups.get(gv)
        if g_right is None or g_right.empty:
            tmp = g_left.copy()
            missing_cols = [c for c in right.columns if c not in {by, on}]
            for c in missing_cols:
                tmp[c] = np.nan
            pieces.append(tmp)
            continue

        merged = pd.merge_asof(
            g_left.sort_values(on).reset_index(drop=True),
            g_right,
            on=on,
            direction=direction,
            allow_exact_matches=allow_exact_matches,
        )
        merged[by] = gv
        pieces.append(merged)

    out = pd.concat(pieces, axis=0, ignore_index=True)
    return out.sort_values([by, on]).reset_index(drop=True)


def _to_spread_decimal(cds: pd.Series | np.ndarray, unit: str = "auto") -> np.ndarray:
    x = pd.to_numeric(cds, errors="coerce").astype(float)
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)

    if unit == "auto":
        unit = "bps" if med > 1.0 else "decimal"

    if unit == "bps":
        return x / 1e4
    if unit == "decimal":
        return x
    raise ValueError("unit must be 'bps', 'decimal', or 'auto'")


def cds_pd_triangle(
    cds: pd.Series | np.ndarray,
    *,
    R: float = 0.40,
    T: float = 1.0,
    unit: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    s = _to_spread_decimal(cds, unit=unit)
    lam = s / max(1e-12, 1.0 - R)
    pd_1y = 1.0 - np.exp(-lam * T)
    return pd_1y, lam


def cds_pd_par_1y_flat_hazard(
    cds: pd.Series | np.ndarray,
    *,
    r: pd.Series | np.ndarray | float = 0.0,
    R: float = 0.40,
    unit: str = "auto",
    pay_freq: int = 4,
    T: float = 1.0,
    max_iter: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    s = _to_spread_decimal(cds, unit=unit)
    s = np.asarray(s, dtype=float)
    n = s.shape[0]

    if np.isscalar(r):
        r_arr = np.full(n, float(r), dtype=float)
    else:
        r_arr = np.asarray(pd.to_numeric(pd.Series(r), errors="coerce").fillna(0.0), dtype=float)
        if r_arr.shape[0] != n:
            raise ValueError("r must be scalar or have the same length as cds")

    dt = 1.0 / pay_freq
    t = np.arange(dt, T + 1e-12, dt)
    delta = np.full_like(t, dt)
    P = np.exp(-r_arr[:, None] * t[None, :])

    def objective(lam: np.ndarray) -> np.ndarray:
        lam = np.asarray(lam, dtype=float)
        Q = np.exp(-lam[:, None] * t[None, :])
        Qprev = np.concatenate([np.ones((n, 1)), Q[:, :-1]], axis=1)
        dQ = Qprev - Q

        rpv01 = (delta[None, :] * P * Q).sum(axis=1) + (0.5 * delta[None, :] * P * dQ).sum(axis=1)
        prot = (1.0 - R) * (P * dQ).sum(axis=1)
        return s * rpv01 - prot

    low = np.full(n, 1e-12)
    high = np.full(n, 5.0)

    f_high = objective(high)
    need_expand = f_high > 0
    it = 0
    while np.any(need_expand) and it < 20:
        high[need_expand] *= 2.0
        f_high = objective(high)
        need_expand = f_high > 0
        it += 1

    high = np.minimum(high, 200.0)

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = objective(mid)
        pos = f_mid > 0
        low[pos] = mid[pos]
        high[~pos] = mid[~pos]

    lam_hat = 0.5 * (low + high)
    pd_1y = 1.0 - np.exp(-lam_hat * T)
    return pd_1y, lam_hat


def build_evaluation_dataframe(
    *,
    derived_dir: str | Path = "../data/derived",
    rf_path: str | Path = "../data/raw/ecb_yc_1y_aaa.csv",
    merton_file: str = "merton_weekly.csv",
    nig_file: str = "nig_em_weekly.csv",
    cds_file: str = "CDS_panel.csv",
    recovery: float = 0.40,
    cds_unit: str = "auto",
    require_both_models: bool = True,
) -> pd.DataFrame:
    derived_dir = Path(derived_dir)
    rf_path = Path(rf_path)

    merton = pd.read_csv(derived_dir / merton_file)
    nig = pd.read_csv(derived_dir / nig_file)
    cds = pd.read_csv(derived_dir / cds_file)
    rf = pd.read_csv(rf_path)

    merton_keep = prepare_panel(
        merton,
        keep_cols=["gvkey", "date", "PD_Q", "PD_P"],
        rename_map={"PD_Q": "PD_1y_Merton_Q", "PD_P": "PD_1y_Merton_P"},
        numeric_cols=["PD_1y_Merton_Q", "PD_1y_Merton_P"],
    )

    nig_keep = prepare_panel(
        nig,
        keep_cols=["gvkey", "date", "PD_Q", "PD_P"],
        rename_map={"PD_Q": "PD_1y_NIG_Q", "PD_P": "PD_1y_NIG_P"},
        numeric_cols=["PD_1y_NIG_Q", "PD_1y_NIG_P"],
    )

    merged = (
        merton_keep.merge(nig_keep, on=KEYS, how="outer", validate="one_to_one")
        .sort_values(KEYS)
        .reset_index(drop=True)
    )

    if require_both_models:
        both_cols = ["PD_1y_Merton_Q", "PD_1y_NIG_Q"]
        merged = merged.loc[merged[both_cols].notna().all(axis=1)].copy()

    merged["gvkey"] = merged["gvkey"].astype(str)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")

    cds2 = cds.copy()
    cds2["gvkey"] = cds2["gvkey"].astype(str)
    cds2["date"] = pd.to_datetime(cds2["date"], errors="coerce")
    cds2["cds"] = pd.to_numeric(cds2["cds"], errors="coerce")
    cds2 = cds2.dropna(subset=["gvkey", "date", "cds"]).sort_values(KEYS).reset_index(drop=True)

    common_gv = sorted(set(merged["gvkey"].unique()) & set(cds2["gvkey"].unique()))
    merged = merged[merged["gvkey"].isin(common_gv)].copy()
    cds2 = cds2[cds2["gvkey"].isin(common_gv)].copy()

    dmin, dmax = merged["date"].min(), merged["date"].max()
    cds2 = cds2[(cds2["date"] >= dmin) & (cds2["date"] <= dmax)].copy()

    merged_cds = asof_merge_by_gvkey(
        merged.sort_values(KEYS).reset_index(drop=True),
        cds2[["gvkey", "date", "cds"]].sort_values(KEYS).reset_index(drop=True),
        on="date",
        by="gvkey",
        direction="backward",
        allow_exact_matches=True,
    )

    out = merged_cds.dropna(subset=["cds"]).reset_index(drop=True)

    rf = rf.rename(columns={"r_1y": "r_f"}).copy()
    rf["date"] = pd.to_datetime(rf["date"], errors="coerce")
    rf["r_f"] = pd.to_numeric(rf["r_f"], errors="coerce")
    rf = rf.dropna(subset=["date", "r_f"]).sort_values("date").reset_index(drop=True)

    if rf["r_f"].median(skipna=True) > 0.5:
        rf["r_f"] = rf["r_f"] / 100.0

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
        "PD_1y_Merton_Q",
        "PD_1y_Merton_P",
        "PD_1y_NIG_Q",
        "PD_1y_NIG_P",
        "cds",
        "r_f",
        "lambda_CDS_triangle",
        "PD_1y_CDS_triangle",
        "lambda_CDS_par",
        "PD_1y_CDS_par",
    ]
    extra_cols = [c for c in out.columns if c not in ordered_cols]
    out = out[ordered_cols + extra_cols].sort_values(KEYS).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final evaluation-ready dataframe for Merton vs NIG vs CDS.")
    parser.add_argument("--derived-dir", default="../data/derived")
    parser.add_argument("--rf-path", default="../data/raw/ecb_yc_1y_aaa.csv")
    parser.add_argument("--merton-file", default="merton_weekly.csv")
    parser.add_argument("--nig-file", default="nig_em_weekly.csv")
    parser.add_argument("--cds-file", default="CDS_panel.csv")
    parser.add_argument("--output", default="../data/derived/eval_ready_panel.csv")
    parser.add_argument("--recovery", type=float, default=0.40)
    parser.add_argument("--cds-unit", choices=["auto", "bps", "decimal"], default="auto")
    parser.add_argument("--allow-single-model-rows", action="store_true")
    args = parser.parse_args()

    df = build_evaluation_dataframe(
        derived_dir=args.derived_dir,
        rf_path=args.rf_path,
        merton_file=args.merton_file,
        nig_file=args.nig_file,
        cds_file=args.cds_file,
        recovery=args.recovery,
        cds_unit=args.cds_unit,
        require_both_models=not args.allow_single_model_rows,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Evaluation-ready dataframe built successfully")
    print(f"Saved to: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Firms: {df['gvkey'].nunique()}")
    print(f"Date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"Missing r_f share: {df['r_f'].isna().mean():.6f}")
    print(df.head())


if __name__ == "__main__":
    main()
