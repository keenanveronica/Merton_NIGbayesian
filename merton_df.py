import pandas as pd
import numpy as np


def build_merton_inputs(
    ret_daily: pd.DataFrame,
    bs: pd.DataFrame,
    df_rf: pd.DataFrame,
    T: float = 1.0,
    equity_col: str = "mcap",
    returns_col: str = "logret_mcap",
    id_cols=("isin", "company", "country_iso"),
    drop_missing_B: bool = True,
    drop_missing_r: bool = True,
):
    # drop extra columns from market data
    keep_mkt = ["gvkey", "date", equity_col]
    if returns_col in ret_daily.columns:
        keep_mkt.append(returns_col)
    for c in id_cols:
        if c in ret_daily.columns:
            keep_mkt.append(c)

    mkt = ret_daily[keep_mkt].copy()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt["gvkey"] = mkt["gvkey"].astype(str)
    mkt[equity_col] = pd.to_numeric(mkt[equity_col], errors="coerce")
    if returns_col in mkt.columns:
        mkt[returns_col] = pd.to_numeric(mkt[returns_col], errors="coerce")

    # IMPORTANT: for merge_asof with by=..., sort by ON key first, then BY key
    mkt = mkt.sort_values(["date", "gvkey"]).reset_index(drop=True)

    # Balance sheet panel for asof merge (B_t known as of final_date)
    bs2 = bs[["gvkey", "final_date", "liabilities_total"]].copy()
    bs2["gvkey"] = bs2["gvkey"].astype(str)
    bs2["final_date"] = pd.to_datetime(bs2["final_date"])
    bs2["liabilities_total"] = pd.to_numeric(bs2["liabilities_total"], errors="coerce")
    bs2 = bs2.dropna(subset=["final_date", "liabilities_total"])

    # IMPORTANT: sort by right_on first, then by
    bs2 = bs2.sort_values(["final_date", "gvkey"]).reset_index(drop=True)

    # As-of merge within firm: latest final_date <= market date
    mkt_bs = pd.merge_asof(
        mkt,
        bs2,
        left_on="date",
        right_on="final_date",
        by="gvkey",
        direction="backward",
        allow_exact_matches=True,
    )

    mkt_bs = mkt_bs.rename(columns={equity_col: "E", "liabilities_total": "B"})

    # Risk-free: asof merge on date
    rf = df_rf[["date", "r_1y"]].copy()
    rf["date"] = pd.to_datetime(rf["date"])
    rf["r_1y"] = pd.to_numeric(rf["r_1y"], errors="coerce")
    rf = rf.dropna(subset=["date", "r_1y"]).sort_values(["date"]).reset_index(drop=True)

    # mkt_bs must be sorted by on='date' for asof merge
    mkt_bs = mkt_bs.sort_values(["date", "gvkey"]).reset_index(drop=True)

    mkt_bs = pd.merge_asof(
        mkt_bs,
        rf,
        on="date",
        direction="backward",
        allow_exact_matches=True,
    )

    mkt_bs = mkt_bs.rename(columns={"r_1y": "r"})
    mkt_bs["T"] = float(T)

    # QA summary
    qa = {
        "rows_out": int(len(mkt_bs)),
        "n_firms_out": int(mkt_bs["gvkey"].nunique()),
        "date_min_out": str(mkt_bs["date"].min().date()) if len(mkt_bs) else None,
        "date_max_out": str(mkt_bs["date"].max().date()) if len(mkt_bs) else None,
        "pct_missing_B_after_merge": float(pd.isna(mkt_bs["B"]).mean() * 100) if "B" in mkt_bs else None,
        "pct_missing_r_after_merge": float(pd.isna(mkt_bs["r"]).mean() * 100) if "r" in mkt_bs else None,
    }
    print("Merton input QA:", qa)

    return mkt_bs


def equity_volatility(
    merton_inputs: pd.DataFrame,
    ret_col: str = "logret_mcap",
    window: int = 252,
    min_obs: int = 126,
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Adds rolling equity volatility sigma_E(t) (annualized) computed from daily log returns.

    sigma_E_daily(t) = rolling std of log returns over `window` within each firm
    sigma_E_ann(t)   = sigma_E_daily(t) * sqrt(trading_days)

    Parameters
    ----------
    window : int
        Rolling window length in trading days (252 ~ 1 year).
    min_obs : int
        Minimum observations required to compute rolling std (stability rule).
    """

    df = merton_inputs.copy()
    df = df.sort_values(["gvkey", "date"]).reset_index(drop=True)

    # daily rolling std by firm
    df["sigma_E_daily"] = (
        df.groupby("gvkey")[ret_col]
          .transform(lambda s: s.rolling(window=window, min_periods=min_obs).std())
    )

    # annualize
    df["sigma_E"] = df["sigma_E_daily"] * np.sqrt(trading_days)

    return df


def fill_liabilities_B(merton_inputs_vol: pd.DataFrame,
                       B_col: str = "B",
                       final_date_col: str = "final_date",
                       method: str = "ffill_then_bfill_initial") -> pd.DataFrame:
    """
    Create a filled-liabilities version of the Merton input panel.

    method:
      - "bfill_only": pure backfill (uses future info for any missing point)
      - "ffill_then_bfill_initial": forward-fill within firm, then backfill remaining (typically initial block)

    Returns df with:
      - B_filled
      - final_date_filled
      - B_imputed (True if original B was missing and is now filled)
    """

    df = merton_inputs_vol.copy()
    df = df.sort_values(["gvkey", "date"]).reset_index(drop=True)

    g = df.groupby("gvkey", sort=False)

    if method == "bfill_only":
        df["B_filled"] = g[B_col].transform(lambda s: s.bfill())
        df["final_date_filled"] = g[final_date_col].transform(lambda s: s.bfill())

    elif method == "ffill_then_bfill_initial":
        # 1) Use past info first (no look-ahead) wherever possible
        B_ff = g[B_col].transform(lambda s: s.ffill())
        fd_ff = g[final_date_col].transform(lambda s: s.ffill())

        # 2) Only remaining NaNs (usually the initial block) are filled using the first available future statement
        B_bf = g[B_col].transform(lambda s: s.bfill())
        fd_bf = g[final_date_col].transform(lambda s: s.bfill())

        df["B_filled"] = B_ff.fillna(B_bf)
        df["final_date_filled"] = fd_ff.fillna(fd_bf)

    else:
        raise ValueError("Unknown method. Use 'bfill_only' or 'ffill_then_bfill_initial'.")

    df["B_imputed"] = df[B_col].isna() & df["B_filled"].notna()

    return df
