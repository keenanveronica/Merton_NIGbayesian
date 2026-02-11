import pandas as pd
import numpy as np


def build_nig_inputs(
    ret_daily: pd.DataFrame,
    bs: pd.DataFrame,
    df_rf: pd.DataFrame,
    equity_col: str = "mcap",
    id_cols=("isin", "company", "country_iso"),
    drop_missing_L: bool = True,
    drop_missing_r: bool = True,
):
    """
    Build a panel suitable for NIG EM initialization (Jovan & Ahčan style):
      - E_t: market equity value (market cap)
      - L_t: liabilities (book debt proxy, as-of merged via final_date <= date)
      - r_t: risk-free rate series aligned by date

    Output columns (minimum):
      gvkey, date, E, L, r, final_date (+ optional id columns)
    """

    # ---------
    # Market data
    # ---------
    keep_mkt = ["gvkey", "date", equity_col]
    for c in id_cols:
        if c in ret_daily.columns:
            keep_mkt.append(c)

    mkt = ret_daily[keep_mkt].copy()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt["gvkey"] = mkt["gvkey"].astype(str)
    mkt[equity_col] = pd.to_numeric(mkt[equity_col], errors="coerce")

    # IMPORTANT for merge_asof(by=...): sort by ON key first, then BY key
    mkt = mkt.sort_values(["date", "gvkey"]).reset_index(drop=True)

    # ---------
    # Balance sheet liabilities (as-of merge within firm)
    # ---------
    bs2 = bs[["gvkey", "final_date", "liabilities_total"]].copy()
    bs2["gvkey"] = bs2["gvkey"].astype(str)
    bs2["final_date"] = pd.to_datetime(bs2["final_date"])
    bs2["liabilities_total"] = pd.to_numeric(bs2["liabilities_total"], errors="coerce")
    bs2 = bs2.dropna(subset=["final_date", "liabilities_total"])

    # IMPORTANT: sort by right_on first, then by
    bs2 = bs2.sort_values(["final_date", "gvkey"]).reset_index(drop=True)

    mkt_bs = pd.merge_asof(
        mkt,
        bs2,
        left_on="date",
        right_on="final_date",
        by="gvkey",
        direction="backward",
        allow_exact_matches=True,
    )

    # Rename to NIG naming
    mkt_bs = mkt_bs.rename(columns={equity_col: "E", "liabilities_total": "L"})

    # ---------
    # Risk-free as-of merge on date
    # ---------
    rf = df_rf[["date", "r_1y"]].copy()
    rf["date"] = pd.to_datetime(rf["date"])
    rf["r_1y"] = pd.to_numeric(rf["r_1y"], errors="coerce")
    rf = rf.dropna(subset=["date", "r_1y"]).sort_values(["date"]).reset_index(drop=True)

    mkt_bs = mkt_bs.sort_values(["date", "gvkey"]).reset_index(drop=True)

    mkt_bs = pd.merge_asof(
        mkt_bs,
        rf,
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    mkt_bs = mkt_bs.rename(columns={"r_1y": "r"})

    # ---------
    # Basic cleaning (EM requires E>0; and you typically want L>0 too)
    # ---------
    mkt_bs["E"] = pd.to_numeric(mkt_bs["E"], errors="coerce")
    mkt_bs["L"] = pd.to_numeric(mkt_bs["L"], errors="coerce")
    mkt_bs["r"] = pd.to_numeric(mkt_bs["r"], errors="coerce")

    mkt_bs = mkt_bs[mkt_bs["E"] > 0].copy()

    if drop_missing_L:
        mkt_bs = mkt_bs.dropna(subset=["L"])
    if drop_missing_r:
        mkt_bs = mkt_bs.dropna(subset=["r"])

    # QA summary
    qa = {
        "rows_out": int(len(mkt_bs)),
        "n_firms_out": int(mkt_bs["gvkey"].nunique()) if len(mkt_bs) else 0,
        "date_min_out": str(mkt_bs["date"].min().date()) if len(mkt_bs) else None,
        "date_max_out": str(mkt_bs["date"].max().date()) if len(mkt_bs) else None,
        "pct_missing_L_after_merge": float(pd.isna(mkt_bs["L"]).mean() * 100) if "L" in mkt_bs else None,
        "pct_missing_r_after_merge": float(pd.isna(mkt_bs["r"]).mean() * 100) if "r" in mkt_bs else None,
        "pct_nonpos_L": float((mkt_bs["L"] <= 0).mean() * 100) if "L" in mkt_bs else None,
    }
    print("NIG input QA:", qa)

    return mkt_bs


def fill_liabilities_L(
    nig_inputs: pd.DataFrame,
    L_col: str = "L",
    final_date_col: str = "final_date",
    method: str = "ffill_then_bfill_initial",
) -> pd.DataFrame:
    """
    Same idea as your fill_liabilities_B, but for L.

    Returns df with:
      - L_filled
      - final_date_filled
      - L_imputed
    """
    df = nig_inputs.copy()
    df = df.sort_values(["gvkey", "date"]).reset_index(drop=True)

    g = df.groupby("gvkey", sort=False)

    if method == "bfill_only":
        df["L_filled"] = g[L_col].transform(lambda s: s.bfill())
        df["final_date_filled"] = g[final_date_col].transform(lambda s: s.bfill())

    elif method == "ffill_then_bfill_initial":
        L_ff = g[L_col].transform(lambda s: s.ffill())
        fd_ff = g[final_date_col].transform(lambda s: s.ffill())

        L_bf = g[L_col].transform(lambda s: s.bfill())
        fd_bf = g[final_date_col].transform(lambda s: s.bfill())

        df["L_filled"] = L_ff.fillna(L_bf)
        df["final_date_filled"] = fd_ff.fillna(fd_bf)

    else:
        raise ValueError("Unknown method. Use 'bfill_only' or 'ffill_then_bfill_initial'.")

    df["L_imputed"] = df[L_col].isna() & df["L_filled"].notna()
    return df


def make_em_inputs(
    nig_panel: pd.DataFrame,
    gvkey: str,
    *,
    end_date: str | pd.Timestamp | None = None,
    window: int = 505,
    use_filled_L: bool = True,
    L_pick: str = "last",   # "last" is the most natural for 'as-of end of window'
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Extract (equity_array, L_scalar, rf_array) in the exact shape expected by
    nig_em_paper.em_init_nig_params(...) :contentReference[oaicite:2]{index=2}

    window=505 matches the empirical setup in Jovan & Ahčan :contentReference[oaicite:3]{index=3}
    """
    df = nig_panel.copy()
    df["gvkey"] = df["gvkey"].astype(str)
    df = df[df["gvkey"] == str(gvkey)].sort_values("date")

    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df = df[df["date"] <= end_date]

    if len(df) < 3:
        raise ValueError("Not enough observations after filtering (need >= 3).")

    df = df.tail(int(window)).copy()

    # arrays
    E = df["E"].to_numpy(dtype=float)
    r = df["r"].to_numpy(dtype=float)

    # liabilities scalar (constant L over the window)
    L_col = "L_filled" if (use_filled_L and "L_filled" in df.columns) else "L"
    if L_pick == "last":
        L = float(df[L_col].dropna().iloc[-1])
    elif L_pick == "median":
        L = float(df[L_col].dropna().median())
    else:
        raise ValueError("L_pick must be 'last' or 'median'.")

    # sanity checks (the EM code will also enforce these)
    if np.any(~np.isfinite(E)) or np.any(E <= 0.0):
        raise ValueError("Equity array contains non-finite or non-positive values.")
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("Liabilities L must be finite and > 0.")
    if np.any(~np.isfinite(r)):
        raise ValueError("Risk-free array contains non-finite values.")
    if E.shape != r.shape:
        raise ValueError("E and r must have identical shapes.")

    return E, L, r
