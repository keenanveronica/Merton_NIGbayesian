import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any

def build_market_rf_panel(
    ret_daily: pd.DataFrame,
    df_rf: pd.DataFrame,
    *,
    equity_col: str = "mcap",
    returns_col: str | None = None,
    id_cols=("isin", "company", "country_iso"),
    rf_col: str = "r_1y",
    require_E_pos: bool = True,
    drop_missing_r: bool = True,
) -> pd.DataFrame:
    keep_mkt = ["gvkey", "date", equity_col]
    if returns_col is not None and returns_col in ret_daily.columns:
        keep_mkt.append(returns_col)
    for c in id_cols:
        if c in ret_daily.columns:
            keep_mkt.append(c)

    mkt = ret_daily[keep_mkt].copy()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt["gvkey"] = mkt["gvkey"].astype(str)
    mkt[equity_col] = pd.to_numeric(mkt[equity_col], errors="coerce")
    if returns_col is not None and returns_col in mkt.columns:
        mkt[returns_col] = pd.to_numeric(mkt[returns_col], errors="coerce")
    mkt = mkt.sort_values(["date", "gvkey"]).reset_index(drop=True)

    rf = df_rf[["date", rf_col]].copy()
    rf["date"] = pd.to_datetime(rf["date"])
    rf[rf_col] = pd.to_numeric(rf[rf_col], errors="coerce")
    rf = rf.dropna(subset=["date", rf_col]).sort_values(["date"]).reset_index(drop=True)

    out = pd.merge_asof(mkt, rf, on="date", direction="backward", allow_exact_matches=True)
    out = out.rename(columns={equity_col: "E", rf_col: "r"})

    out["E"] = pd.to_numeric(out["E"], errors="coerce")
    out["r"] = pd.to_numeric(out["r"], errors="coerce")
    if require_E_pos:
        out = out[out["E"] > 0].copy()
    if drop_missing_r:
        out = out.dropna(subset=["r"])

    return out


def build_discounted_liabilities_daily(
    bs: pd.DataFrame,
    df_rf: pd.DataFrame,
    *,
    debt_col: str = "liabilities_total",
    rf_col: str = "r_1y",
    publication_col: str = "final_date",
    gvkey_col: str = "gvkey",
    ref_rule: str = "dec31_prev_year",
    bs_date_col: str = "date",
    daycount: float = 365.0,
    compounding: str = "simple",
    # NEW:
    target_max_year: int | None = None,          # e.g. max year in your market panel dates
    tail_year_policy: str = "ffill_face",              # "drop" or "ffill_face"
) -> pd.DataFrame:
    """
    Builds daily discounted liabilities L_pv(t) within each reference year.

    If ref_rule="dec31_prev_year", then a publication in year Y is mapped to ref_year=Y-1.

    tail_year_policy:
      - "drop": only build years for which a mapped ref_year exists in bs
      - "ffill_face": extend each gvkey to target_max_year by carrying forward last known L_face
    """

    # ---- 1) clean BS ----
    cols = [gvkey_col, publication_col, debt_col]
    if bs_date_col in bs.columns:
        cols.append(bs_date_col)

    bs2 = bs[cols].copy()
    bs2[gvkey_col] = bs2[gvkey_col].astype(str)
    bs2[publication_col] = pd.to_datetime(bs2[publication_col], errors="coerce")
    bs2[debt_col] = pd.to_numeric(bs2[debt_col], errors="coerce")
    bs2 = bs2.dropna(subset=[publication_col, debt_col])

    # reference date
    if ref_rule == "bs_date" and bs_date_col in bs2.columns:
        bs2["ref_date"] = pd.to_datetime(bs2[bs_date_col], errors="coerce")
        bs2 = bs2.dropna(subset=["ref_date"])
    elif ref_rule == "dec31_prev_year":
        ref_year = bs2[publication_col].dt.year - 1
        bs2["ref_date"] = pd.to_datetime(ref_year.astype(str) + "-12-31")
    else:
        raise ValueError("ref_rule must be 'dec31_prev_year' or 'bs_date' (when bs_date_col exists).")

    bs2["ref_year"] = bs2["ref_date"].dt.year
    bs2 = bs2.rename(columns={publication_col: "pub_date", debt_col: "L_face"})

    # keep last publication per gvkey-ref_year
    bs2 = (
        bs2.sort_values([gvkey_col, "ref_year", "pub_date"])
           .groupby([gvkey_col, "ref_year"], as_index=False)
           .tail(1)
           .reset_index(drop=True)
    )

    # ---- 2) optionally extend last year(s) by ffill of L_face ----
    if tail_year_policy not in {"drop", "ffill_face"}:
        raise ValueError("tail_year_policy must be 'drop' or 'ffill_face'.")

    if tail_year_policy == "ffill_face":
        if target_max_year is None:
            # fallback: extend to last rf year (still an assumption)
            target_max_year = int(pd.to_datetime(df_rf["date"]).dt.year.max())

        # reindex each gvkey to a full year grid and ffill the face value
        out_blocks = []
        for gv, g in bs2.groupby(gvkey_col, sort=False):
            g = g.sort_values("ref_year").set_index("ref_year")

            full_years = pd.Index(range(int(g.index.min()), int(target_max_year) + 1), name="ref_year")
            gg = g.reindex(full_years)

            gg[gvkey_col] = gv
            gg["L_face_source_imputed"] = gg["L_face"].isna()

            # forward-fill L_face and pub_date (pub_date indicates staleness)
            gg["L_face"] = gg["L_face"].ffill()
            gg["pub_date"] = gg["pub_date"].ffill()

            # rebuild ref_date consistently
            gg["ref_date"] = pd.to_datetime(gg.index.astype(str) + "-12-31")

            # drop years where we *still* don't have any L_face (e.g. firm starts after)
            gg = gg.dropna(subset=["L_face"])

            out_blocks.append(gg.reset_index())

        bs2 = pd.concat(out_blocks, ignore_index=True)

    # if "drop", do nothing: we keep only years that exist

    # ---- 3) risk-free calendar ----
    rf = df_rf[["date", rf_col]].copy()
    rf["date"] = pd.to_datetime(rf["date"])
    rf[rf_col] = pd.to_numeric(rf[rf_col], errors="coerce")
    rf = rf.dropna(subset=["date", rf_col]).sort_values("date").reset_index(drop=True)

    min_year = int(bs2["ref_year"].min())
    max_year = int(bs2["ref_year"].max())

    cal = pd.DataFrame({"date": pd.date_range(f"{min_year}-01-01", f"{max_year}-12-31", freq="D")})
    cal = pd.merge_asof(cal.sort_values("date"), rf, on="date", direction="backward", allow_exact_matches=True)
    cal["year"] = cal["date"].dt.year

    # ---- 4) DF(t -> Dec31) ----
    df_list = []
    dt = 1.0 / daycount

    for y, g in cal.groupby("year", sort=True):
        r = g[rf_col].to_numpy(dtype=float)
        n = len(r)
        DF = np.empty(n, dtype=float)
        DF[-1] = 1.0

        if compounding == "simple":
            for i in range(n - 2, -1, -1):
                DF[i] = DF[i + 1] / (1.0 + r[i] * dt)
        elif compounding == "continuous":
            acc = 0.0
            for i in range(n - 2, -1, -1):
                acc += r[i] * dt
                DF[i] = np.exp(-acc)
        else:
            raise ValueError("compounding must be 'simple' or 'continuous'.")

        tmp = g[["date"]].copy()
        tmp["ref_year"] = int(y)
        tmp["DF_to_dec31"] = DF
        df_list.append(tmp)

    DF_table = pd.concat(df_list, ignore_index=True)

    # ---- 5) firm-year × daily DF ----
    liab = bs2[[gvkey_col, "ref_year", "ref_date", "pub_date", "L_face"]].copy()
    out = liab.merge(DF_table, on="ref_year", how="left")

    out["L_pv"] = out["L_face"] * out["DF_to_dec31"]

    out = out.rename(columns={gvkey_col: "gvkey"})
    out = out[["gvkey", "date", "L_face", "L_pv", "ref_date", "pub_date", "ref_year"]].sort_values(["gvkey", "date"])
    return out


def attach_debt_daily(
    panel: pd.DataFrame,
    debt_daily: pd.DataFrame,
    *,
    debt_out_col: str = "B",
    pv_col: str = "L_pv",   # whatever name your debt builder uses
) -> pd.DataFrame:
    dd = debt_daily[["gvkey", "date", pv_col]].copy()
    dd["gvkey"] = dd["gvkey"].astype(str)
    dd["date"] = pd.to_datetime(dd["date"])
    out = panel.merge(dd, on=["gvkey", "date"], how="left")
    out = out.rename(columns={pv_col: debt_out_col})
    return out


# Merton specific functions
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


# NIG specific functions
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


def prepare_merton_inputs(
    ret_daily: pd.DataFrame,
    bs: pd.DataFrame,
    df_rf: pd.DataFrame,
    *,
    equity_col: str = "mcap",
    returns_col: str = "logret_mcap",
    id_cols=("isin", "company", "country_iso"),
    add_sigma_E: bool = True,
    sigma_window: int = 252,
    sigma_min_obs: int = 126,
    trading_days: int = 252,
    drop_missing_r: bool = True,
    ref_rule: str = "dec31_prev_year",
    daycount: float = 365.0,
    compounding: str = "simple",
) -> pd.DataFrame:
    """
    Construct a Merton input panel from separate daily returns and
    balance‑sheet data.  This wrapper calls the market/rf builder,
    constructs daily discounted liabilities, attaches them as 'B' and
    optionally computes rolling equity volatility.  It avoids as‑of
    merging of balance‑sheet values and does not fill missing B.  The
    returned DataFrame contains columns:
      gvkey, date, E, (returns_col), optional ID columns, B, r,
      and optionally sigma_E_daily and sigma_E if add_sigma_E=True.
    """

    # 1) build market + risk free panel (includes E and r and returns_col)
    market = build_market_rf_panel(
        ret_daily=ret_daily,
        df_rf=df_rf,
        equity_col=equity_col,
        returns_col=returns_col,
        id_cols=id_cols,
        rf_col="r_1y",
        require_E_pos=True,
        drop_missing_r=drop_missing_r,
    )

    # 2) build discounted liabilities per firm-year
    debt_daily = build_discounted_liabilities_daily(
        bs=bs,
        df_rf=df_rf,
        debt_col="liabilities_total",
        rf_col="r_1y",
        publication_col="final_date",
        gvkey_col="gvkey",
        ref_rule=ref_rule,
        bs_date_col="date",
        daycount=daycount,
        compounding=compounding,
    )

    # 3) attach discounted liabilities to the market panel
    df = attach_debt_daily(
        panel=market,
        debt_daily=debt_daily,
        debt_out_col="B",
        pv_col="L_pv",
    )

    # 4) optionally compute rolling equity volatility
    if add_sigma_E and returns_col in df.columns:
        df = equity_volatility(
            df,
            ret_col=returns_col,
            window=sigma_window,
            min_obs=sigma_min_obs,
            trading_days=trading_days,
        )

    return df


def prepare_nig_inputs(
    ret_daily: pd.DataFrame,
    bs: pd.DataFrame,
    df_rf: pd.DataFrame,
    *,
    equity_col: str = "mcap",
    id_cols=("isin", "company", "country_iso"),
    build_em: bool = False,
    em_window: int = 505,
    em_use_filled_L: bool = True,
    em_L_pick: str = "last",
    drop_missing_r: bool = True,
    ref_rule: str = "dec31_prev_year",
    daycount: float = 365.0,
    compounding: str = "simple",
) -> Tuple[pd.DataFrame, Optional[Dict[Tuple[str, pd.Timestamp], Tuple[Any, Any, Any]]]]:
    """
    Construct a NIG input panel from separate daily returns and balance
    sheet data.  This wrapper builds the market/rf panel (without
    returns), attaches discounted liabilities as 'L' and optionally
    extracts EM windows.  Note that no volatility is computed here.
    """

    # 1) build market + risk free panel (without returns)
    market = build_market_rf_panel(
        ret_daily=ret_daily,
        df_rf=df_rf,
        equity_col=equity_col,
        returns_col=None,
        id_cols=id_cols,
        rf_col="r_1y",
        require_E_pos=True,
        drop_missing_r=drop_missing_r,
    )

    # 2) build discounted liabilities per firm-year
    debt_daily = build_discounted_liabilities_daily(
        bs=bs,
        df_rf=df_rf,
        debt_col="liabilities_total",
        rf_col="r_1y",
        publication_col="final_date",
        gvkey_col="gvkey",
        ref_rule=ref_rule,
        bs_date_col="date",
        daycount=daycount,
        compounding=compounding,
    )

    # 3) attach discounted liabilities to the market panel as L
    df = attach_debt_daily(
        panel=market,
        debt_daily=debt_daily,
        debt_out_col="L",
        pv_col="L_pv",
    )

    # convert to numeric and drop non‑positive L (EM requires positive L)
    df["L"] = pd.to_numeric(df["L"], errors="coerce")
    df = df[df["L"] > 0].copy()

    em_inputs: Optional[Dict[Tuple[str, pd.Timestamp], Tuple[Any, Any, Any]]] = None
    if build_em:
        em_inputs = {}
        # iterate by firm and build windows
        for gv, g in df.groupby("gvkey", sort=False):
            g = g.sort_values("date")
            if len(g) < em_window:
                continue
            for end_date in g["date"].iloc[em_window - 1:]:
                try:
                    E_arr, L_scalar, r_arr = make_em_inputs(
                        nig_panel=df,
                        gvkey=str(gv),
                        end_date=end_date,
                        window=em_window,
                        use_filled_L=em_use_filled_L,
                        L_pick=em_L_pick,
                    )
                    em_inputs[(str(gv), pd.to_datetime(end_date))] = (E_arr, L_scalar, r_arr)
                except Exception:
                    # skip windows with insufficient or invalid data
                    continue
    return df, em_inputs