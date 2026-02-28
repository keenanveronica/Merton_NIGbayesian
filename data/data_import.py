import pandas as pd
import numpy as np
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
from typing import Tuple, Optional, Dict, Any


def load_data(
    xlsx_path: Optional[Path] = None,
    *,
    # Windowing for panel QA / balancing
    start_date: str | None = None,
    end_date: str | None = None,
    # Balanced-panel filter (coverage over the chosen window)
    enforce_coverage: bool = True,
    coverage_tol: float = 0.95,
    # Legacy option: minimum usable days of returns (post-clean)
    min_days_per_firm: int = 0,
    verbose: bool = True,
    liabilities_scale="auto",  # "auto", "none", or numeric factor (e.g., 1e6)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load Accenture Erasmus Case xlsx, clean market data + balance sheet data.

    Key design choices (aligned with notebook):
    - Equity value E_t computed as mcap = close * ffilled(shares_out), using ONLY forward-fill (no bfill).
    - mcap_reported is not used for modeling (it can be 0 when shares are missing).
    - Drops rows where close is missing/non-positive or shares_out still missing after ffill.
    - Computes logret_close and logret_mcap.
    - Optionally enforces a (nearly) balanced panel using coverage over a fixed calendar
      within [start_date, end_date].

    Returns
    -------
    ret_daily : pd.DataFrame
        Clean daily panel with columns including:
        gvkey, date, company, close, shares_out, mcap, logret_close, logret_mcap
    bs : pd.DataFrame
        Clean annual balance sheet panel (latest statement per gvkey-fyear).
    coverage : pd.DataFrame
        Per-firm coverage table for the chosen window (n_dates, coverage_pct).
    """
    if xlsx_path is None:
        xlsx_path = Path.cwd() / "data/raw/Jan2025_Accenture_Dataset_ErasmusCase.xlsx"
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"No xlsx file found at: {xlsx_path}")

    # --- Read excel ---
    mkt_raw = pd.read_excel(xlsx_path, sheet_name="Returns & MktCap")
    bs_raw = pd.read_excel(xlsx_path, sheet_name="Balance Sheet Data")

    # --- Rename market ---
    rename_mkt = {
        "(fic) Current ISO Country Code - Incorporation": "country_iso",
        "(isin) International Security Identification Number": "isin",
        "(datadate) Data Date - Daily Prices": "date",
        "(conm) Company Name": "company",
        "(gvkey) Global Company Key - Company": "gvkey",
        "(cshoc) Shares Outstanding": "shares_out",
        "(prccd) Price - Close - Daily": "close",
        "Market Capitalization (# Shares * Close Price)": "mcap_reported",  # kept only for QA
    }
    mkt = mkt_raw.rename(columns=rename_mkt).copy()

    # Types
    mkt["date"] = pd.to_datetime(mkt["date"], errors="coerce")
    mkt["gvkey"] = mkt["gvkey"].astype(str)
    for c in ["shares_out", "close", "mcap_reported"]:
        if c in mkt.columns:
            mkt[c] = pd.to_numeric(mkt[c], errors="coerce")

    # Drop unusable dates / duplicates early
    mkt = mkt.dropna(subset=["date", "gvkey"])
    mkt = mkt.sort_values(["gvkey", "date"])
    mkt = mkt.drop_duplicates(subset=["gvkey", "date"], keep="last")

    # Optional date windowing for QA + coverage (applied before balancing)
    if start_date is not None:
        mkt = mkt[mkt["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        mkt = mkt[mkt["date"] <= pd.to_datetime(end_date)]

    # --- Core notebook-aligned equity construction ---
    # Forward-fill shares_out only (NO backward-fill)
    mkt["shares_out_ffill"] = mkt.groupby("gvkey")["shares_out"].ffill()

    # Compute model equity value: mcap = close * shares_out_ffill
    mkt["mcap"] = mkt["close"] * mkt["shares_out_ffill"]

    # Drop invalid rows (domain validity for structural models)
    # - close must be > 0 and not NaN
    # - shares_out_ffill must exist
    # - mcap must be > 0
    mkt_clean = mkt[
        mkt["close"].notna()
        & (mkt["close"] > 0)
        & mkt["shares_out_ffill"].notna()
        & (mkt["mcap"].notna())
        & (mkt["mcap"] > 0)
    ].copy()

    # Compute log returns
    mkt_clean["logret_close"] = (
        mkt_clean.groupby("gvkey")["close"].transform(lambda s: np.log(s).diff())
    )
    mkt_clean["logret_mcap"] = (
        mkt_clean.groupby("gvkey")["mcap"].transform(lambda s: np.log(s).diff())
    )

    # Drop first observation per firm (diff -> NaN)
    ret_daily = mkt_clean.dropna(subset=["logret_close", "logret_mcap"]).copy()

    # --- Coverage (balanced panel logic) ---
    # Calendar is the union of dates in ret_daily within the chosen window.
    cal = pd.Series(sorted(ret_daily["date"].unique()))
    n_cal = len(cal)
    if n_cal == 0:
        raise ValueError("No usable market dates after cleaning. Check date window or data quality.")

    cov = ret_daily.groupby("gvkey")["date"].nunique().to_frame("n_dates")
    cov["coverage_pct"] = cov["n_dates"] / n_cal
    cov = cov.sort_values("coverage_pct")

    if enforce_coverage:
        keep = cov.index[cov["coverage_pct"] >= float(coverage_tol)]
        ret_daily = ret_daily[ret_daily["gvkey"].isin(keep)].copy()

        # Recompute coverage after filtering (useful for reporting)
        cal2 = pd.Series(sorted(ret_daily["date"].unique()))
        n_cal2 = len(cal2)
        cov = ret_daily.groupby("gvkey")["date"].nunique().to_frame("n_dates")
        cov["coverage_pct"] = cov["n_dates"] / n_cal2
        cov = cov.sort_values("coverage_pct")

    if min_days_per_firm > 0:
        cnt = ret_daily.groupby("gvkey")["date"].nunique()
        keep = cnt.index[cnt >= int(min_days_per_firm)]
        ret_daily = ret_daily[ret_daily["gvkey"].isin(keep)].copy()

    # --- Balance sheet cleaning (keep as in old version, but compatible) ---
    rename_bs = {
        "(fic) Current ISO Country Code - Incorporation": "country_iso",
        "(costat) Active/Inactive Status Marker": "status",
        "(datafmt) Data Format": "data_format",
        "(indfmt) Industry Format": "industry_format",
        "(consol) Level of Consolidation - Company Annual Descriptor": "consolidation",
        "(isin) International Security Identification Number": "isin",
        "(datadate) Data Date": "date",
        "(conm) Company Name": "company",
        "(gvkey) Global Company Key - Company": "gvkey",
        "(fyear) Data Year - Fiscal": "fyear",
        "(fdate) Final Date": "final_date",
        "(lt) Liabilities - Total": "liabilities_total",
    }
    bs = bs_raw.rename(columns=rename_bs).copy()
    bs["date"] = pd.to_datetime(bs["date"], errors="coerce")
    bs["final_date"] = pd.to_datetime(bs["final_date"], errors="coerce")
    bs["gvkey"] = bs["gvkey"].astype(str)
    bs["liabilities_total"] = pd.to_numeric(bs["liabilities_total"], errors="coerce")

    bs = bs.dropna(subset=["gvkey", "final_date", "liabilities_total"])

    # keep only latest final_date per (gvkey, fyear)
    bs = (
        bs.sort_values(["gvkey", "fyear", "final_date"])
        .groupby(["gvkey", "fyear"], as_index=False)
        .tail(1)
        .sort_values(["gvkey", "final_date"])
        .reset_index(drop=True)
    )

    # AUTO-SCALE liabilities into same units as mcap (as in old version)
    scale_used = 1.0
    if liabilities_scale == "auto":
        # merge_asof requires sorted by merge key then by gvkey
        mkt_tmp = (
            ret_daily[["gvkey", "date", "mcap"]]
            .dropna(subset=["mcap"])
            .sort_values(["date", "gvkey"])
            .reset_index(drop=True)
        )
        bs_tmp = (
            bs[["gvkey", "final_date", "liabilities_total"]]
            .dropna(subset=["final_date", "liabilities_total"])
            .sort_values(["final_date", "gvkey"])
            .reset_index(drop=True)
        )
        merged = pd.merge_asof(
            mkt_tmp,
            bs_tmp,
            left_on="date",
            right_on="final_date",
            by="gvkey",
            direction="backward",
            allow_exact_matches=True,
        )
        mask = (
            np.isfinite(merged["mcap"].values)
            & np.isfinite(merged["liabilities_total"].values)
            & (merged["mcap"].values > 0)
            & (merged["liabilities_total"].values > 0)
        )
        med_ratio = float(np.median(merged.loc[mask, "liabilities_total"] / merged.loc[mask, "mcap"])) if mask.any() else np.nan
        if np.isfinite(med_ratio) and med_ratio < 1e-4:
            scale_used = 1e6
    elif liabilities_scale in ("none", None):
        scale_used = 1.0
    else:
        scale_used = float(liabilities_scale)

    bs["liabilities_total_raw"] = bs["liabilities_total"].astype(float)
    bs["liabilities_total"] = bs["liabilities_total_raw"] * scale_used
    bs["liabilities_scale_used"] = float(scale_used)

    if verbose:
        print(f"[load_data] Firms (ret_daily): {ret_daily['gvkey'].nunique()}")
        print(f"[load_data] Date range (ret_daily): {ret_daily['date'].min().date()} .. {ret_daily['date'].max().date()}")
        print(f"[load_data] Coverage min/median/max: {cov['coverage_pct'].min():.3f} / {cov['coverage_pct'].median():.3f} / {cov['coverage_pct'].max():.3f}")
        print(f"[load_data] liabilities_scale_used: {scale_used:g}")

        # QA: how many reported zeros remain (informational)
        if "mcap_reported" in mkt.columns:
            n_zero_reported = int((mkt["mcap_reported"] <= 0).sum(skipna=True))
            print(f"[load_data] QA mcap_reported<=0 rows (raw windowed mkt): {n_zero_reported}")

    return ret_daily.reset_index(drop=True), bs.reset_index(drop=True), cov.reset_index()


def load_ecb_1y_yield(
    startPeriod="2010-01-01",
    endPeriod="2025-12-31",
    out_file="data/raw/ecb_yc_1y_aaa.xml",
    verify_ssl=False,
    return_response=False
):
    """
    Downloads ECB YC 1Y AAA zero-coupon spot rate as SDMX-ML XML,
    saves it to disk, parses it into a DataFrame, and returns the DataFrame.

    Output DataFrame columns:
      - date: datetime64[ns]
      - r_1y: decimal yield (OBS_VALUE is % p.a., so divided by 100)
    """

    url = "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_1Y"

    params = {"startPeriod": startPeriod, "endPeriod": endPeriod}

    headers = {"Accept": "application/vnd.sdmx.structurespecificdata+xml;version=2.1"}

    response = requests.get(url, headers=headers, params=params, verify=verify_ssl)

    if response.status_code == 200:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "wb") as file:
            file.write(response.content)
        print(f"Data has been written to {out_file}")
    else:
        print(f"Failed to retrieve data: Status code {response.status_code}")
        print("Response text (first 500 chars):")
        print(response.text[:500])
        response.raise_for_status()

    # Parse XML into DataFrame
    root = ET.fromstring(response.content)
    obs_elems = root.findall(".//{*}Obs")  # namespace-agnostic

    data = [(o.attrib["TIME_PERIOD"], float(o.attrib["OBS_VALUE"])) for o in obs_elems]
    df = pd.DataFrame(data, columns=["date", "r_1y_pct"])
    df["date"] = pd.to_datetime(df["date"])

    # OBS_VALUE is "Percent per annum" -> convert to decimal
    df["r_1y"] = df["r_1y_pct"] / 100.0

    df = df[["date", "r_1y"]].sort_values("date").reset_index(drop=True)

    if return_response:
        return df, response
    return df


# -------- PANEL CREATION FOR MODELS --------
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


def fill_liabilities(
    bs: pd.DataFrame,
    df_rf: pd.DataFrame,
    *,
    debt_col: str = "liabilities_total",
    publication_col: str = "final_date",
    gvkey_col: str = "gvkey",
    target_max_year: int | None = None,
    tail_year_policy: str = "ffill_face",   # "ffill_face" or "drop"
) -> pd.DataFrame:
    """
    Build a DAILY liabilities series using publication dates:

    - Forward-fill (no look-ahead): for each (gvkey, date t), use latest liabilities with pub_date <= t.
    - Backfill only at the start: dates before the first pub_date get the first available liabilities (flagged).
    - No discounting is performed here. Output L_pv is a face-value proxy for L (strike).

    Returns columns: gvkey, date, L_pv, is_backfilled_initial
    """

    # safety checks
    if tail_year_policy not in {"drop", "ffill_face"}:
        raise ValueError("tail_year_policy must be 'drop' or 'ffill_face'.")

    if "date" not in df_rf.columns:
        raise ValueError("df_rf must contain a 'date' column for the output calendar.")

    for c in (gvkey_col, publication_col, debt_col):
        if c not in bs.columns:
            raise ValueError(f"bs is missing required column '{c}'.")

    # 1) clean BS: keep only valid (gvkey, pub_date, L_face)
    bs2 = bs[[gvkey_col, publication_col, debt_col]].copy()
    bs2[gvkey_col] = bs2[gvkey_col].astype(str)
    bs2["pub_date"] = pd.to_datetime(bs2[publication_col], errors="coerce")
    bs2["L_face"] = pd.to_numeric(bs2[debt_col], errors="coerce")
    bs2 = bs2.dropna(subset=["pub_date", "L_face"])

    if bs2.empty:
        raise ValueError("After cleaning, bs contains no valid (pub_date, liabilities) rows.")

    # de-duplicate exact same (gvkey, pub_date) keeping the last row
    bs2 = (
        bs2.sort_values([gvkey_col, "pub_date"])
           .drop_duplicates(subset=[gvkey_col, "pub_date"], keep="last")
           .reset_index(drop=True)
    )

    # 2) calendar from df_rf dates (unique, sorted)
    cal_dates = pd.to_datetime(df_rf["date"], errors="coerce").dropna().sort_values().drop_duplicates()

    if cal_dates.empty:
        raise ValueError("df_rf['date'] has no valid dates; cannot build a calendar.")

    if target_max_year is not None:
        cal_dates = cal_dates[cal_dates.dt.year <= int(target_max_year)]

    if cal_dates.empty:
        raise ValueError("Calendar is empty after applying target_max_year filter.")

    # 3) firm-date panel
    gvkeys = bs2[gvkey_col].dropna().unique()
    panel = pd.MultiIndex.from_product([gvkeys, cal_dates], names=[gvkey_col, "date"]).to_frame(index=False)
    panel = panel.sort_values(["date", gvkey_col]).reset_index(drop=True)
    rhs = bs2[[gvkey_col, "pub_date", "L_face"]].sort_values(["pub_date", gvkey_col]).reset_index(drop=True)

    # 4) as-of join (forward-fill with no look-ahead)
    out = pd.merge_asof(
        panel,
        rhs,
        left_on="date",
        right_on="pub_date",
        by=gvkey_col,
        direction="backward",
        allow_exact_matches=True,
    )

    # 5) initial backfill (only before first pub_date)
    first_L = rhs.groupby(gvkey_col)["L_face"].first()
    out["is_backfilled_initial"] = out["L_face"].isna()
    out.loc[out["is_backfilled_initial"], "L_face"] = (
        out.loc[out["is_backfilled_initial"], gvkey_col].map(first_L).to_numpy()
    )

    # If some firms still have NaN, drop them safely
    out = out.dropna(subset=["L_face"])

    # 6) tail policy
    if tail_year_policy == "drop":
        last_pub = rhs.groupby(gvkey_col)["pub_date"].max()
        out = out[out["date"] <= out[gvkey_col].map(last_pub)]

    # 7) output (keep L_pv name for integration)
    out = out.rename(columns={gvkey_col: "gvkey"})
    out["L_pv"] = out["L_face"]
    out = out[["gvkey", "date", "L_pv", "is_backfilled_initial"]].sort_values(["gvkey", "date"]).reset_index(drop=True)
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


def drop_high_leverage_firms(
    ret_daily: pd.DataFrame,
    bs: pd.DataFrame,
    *,
    df_calendar: pd.DataFrame | None = None,
    debt_daily: pd.DataFrame | None = None,
    lev_threshold: float = 8.0,
    lev_agg: str = "median",          # "median" or "mean" or "p95"
    tail_year_policy: str = "ffill_face",  # passed to fill_liabilities if needed
    keep_backfilled_initial: bool = True,
    require_common_dates: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Drops firms whose leverage L/E is persistently high, as measured by an aggregate
    (default: firm-level median) of daily leverage L_t / E_t.

    Parameters
    ----------
    ret_daily : DataFrame
        Must contain columns: 'gvkey', 'date', 'mcap' (equity value).
        If 'company' exists, it is used for diagnostics output.
    bs : DataFrame
        Must contain: 'gvkey', 'final_date', 'liabilities_total' (or whatever fill_liabilities expects).
    df_calendar : DataFrame, optional
        Calendar DataFrame with a 'date' column to define the daily grid for liabilities.
        If None, uses unique sorted ret_daily['date'].
    debt_daily : DataFrame, optional
        Precomputed daily liabilities from fill_liabilities with columns:
        ['gvkey','date','L_pv','is_backfilled_initial'].
        If provided, avoids recomputing.
    lev_threshold : float
        Firms with aggregate leverage > threshold are dropped.
    lev_agg : {"median","mean","p95"}
        Aggregation method across time to decide firm-level leverage regime.
    tail_year_policy : {"ffill_face","drop"}
        Policy for liabilities after last publication date (only used if debt_daily is None).
    keep_backfilled_initial : bool
        If False, rows flagged is_backfilled_initial are excluded from leverage computation.
    require_common_dates : bool
        If True, restrict leverage computation to the intersection of dates across firms
        (usually unnecessary given your near-balanced panel; can reduce bias if calendars differ).
    verbose : bool
        Print a short summary.

    Returns
    -------
    ret_filt : DataFrame
        ret_daily filtered to keep firms not flagged as high leverage.
    bs_filt : DataFrame
        bs filtered to keep the same firms.
    lev_by_firm : DataFrame
        Firm-level leverage aggregate used for selection: columns ['gvkey','lev_stat', 'lev_agg'].
    dropped : DataFrame
        Diagnostics table for excluded firms with company names if available.
    """

    # --- Validate inputs ---
    need_cols = {"gvkey", "date", "mcap"}
    missing = need_cols - set(ret_daily.columns)
    if missing:
        raise ValueError(f"ret_daily missing required columns: {sorted(missing)}")

    ret = ret_daily.copy()
    ret["gvkey"] = ret["gvkey"].astype(str)
    ret["date"] = pd.to_datetime(ret["date"], errors="coerce")
    ret["mcap"] = pd.to_numeric(ret["mcap"], errors="coerce")

    ret = ret.dropna(subset=["gvkey", "date", "mcap"])
    ret = ret[ret["mcap"] > 0].copy()

    # Calendar for liabilities build
    if df_calendar is None:
        df_calendar = ret[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    else:
        if "date" not in df_calendar.columns:
            raise ValueError("df_calendar must contain a 'date' column.")
        df_calendar = df_calendar[["date"]].copy()
        df_calendar["date"] = pd.to_datetime(df_calendar["date"], errors="coerce")
        df_calendar = df_calendar.dropna().drop_duplicates().sort_values("date").reset_index(drop=True)

    # Build liabilities daily if not provided
    if debt_daily is None:
        # fill_liabilities must be in scope (same module) or imported
        debt_daily = fill_liabilities(
            bs=bs,
            df_rf=df_calendar,
            debt_col="liabilities_total",
            publication_col="final_date",
            gvkey_col="gvkey",
            tail_year_policy=tail_year_policy,
        )

    debt = debt_daily.copy()
    debt["gvkey"] = debt["gvkey"].astype(str)
    debt["date"] = pd.to_datetime(debt["date"], errors="coerce")
    debt["L_pv"] = pd.to_numeric(debt["L_pv"], errors="coerce")
    debt = debt.dropna(subset=["gvkey", "date", "L_pv"])
    debt = debt[debt["L_pv"] > 0].copy()

    if not keep_backfilled_initial and "is_backfilled_initial" in debt.columns:
        debt = debt[~debt["is_backfilled_initial"]].copy()

    # Optionally enforce common-date intersection across firms (rarely needed here)
    if require_common_dates:
        dates_by_firm = debt.groupby("gvkey")["date"].apply(set)
        common_dates = set.intersection(*dates_by_firm.tolist()) if len(dates_by_firm) else set()
        if not common_dates:
            raise ValueError("No common dates across firms after applying require_common_dates.")
        common_dates = pd.to_datetime(sorted(common_dates))
        debt = debt[debt["date"].isin(common_dates)].copy()
        ret = ret[ret["date"].isin(common_dates)].copy()

    # Merge to compute leverage
    merged = pd.merge(
        ret[["gvkey", "date", "mcap"] + (["company"] if "company" in ret.columns else [])],
        debt[["gvkey", "date", "L_pv"]],
        on=["gvkey", "date"],
        how="inner",
        validate="many_to_one",  # one L per firm-date expected
    )
    merged["lev"] = merged["L_pv"] / merged["mcap"]
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["lev"])
    merged = merged[merged["lev"] > 0].copy()

    if merged.empty:
        raise ValueError("Merged leverage panel is empty; check date alignment and inputs.")

    # Aggregate leverage by firm
    if lev_agg == "median":
        lev_stat = merged.groupby("gvkey")["lev"].median()
    elif lev_agg == "mean":
        lev_stat = merged.groupby("gvkey")["lev"].mean()
    elif lev_agg == "p95":
        lev_stat = merged.groupby("gvkey")["lev"].quantile(0.95)
    else:
        raise ValueError("lev_agg must be one of {'median','mean','p95'}.")

    lev_by_firm = lev_stat.rename("lev_stat").to_frame()
    lev_by_firm["lev_agg"] = lev_agg
    lev_by_firm = lev_by_firm.sort_values("lev_stat")

    # Determine drop list
    drop_gvkeys = lev_by_firm.index[lev_by_firm["lev_stat"] > float(lev_threshold)].tolist()

    # Diagnostics table
    dropped = lev_by_firm.loc[drop_gvkeys].copy()
    dropped = dropped.reset_index().rename(columns={"index": "gvkey"})
    if "company" in merged.columns:
        names = merged[["gvkey", "company"]].drop_duplicates().set_index("gvkey")
        dropped["company"] = dropped["gvkey"].map(names["company"])

    # Filter outputs
    keep_gvkeys = lev_by_firm.index[~lev_by_firm.index.isin(drop_gvkeys)]
    ret_filt = ret_daily[ret_daily["gvkey"].astype(str).isin(keep_gvkeys)].copy()
    bs_filt = bs[bs["gvkey"].astype(str).isin(keep_gvkeys)].copy()

    if verbose:
        print(f"[drop_high_leverage_firms] agg={lev_agg}, threshold={lev_threshold}")
        print(f"[drop_high_leverage_firms] firms before: {ret_daily['gvkey'].nunique()} | after: {ret_filt['gvkey'].nunique()}")
        if len(drop_gvkeys):
            print(f"[drop_high_leverage_firms] dropped firms: {len(drop_gvkeys)}")

    return ret_filt, bs_filt, lev_by_firm.reset_index().rename(columns={"index": "gvkey"}), dropped


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
   -------
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
    debt_daily: pd.DataFrame | None = None,   # NEW
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
    if debt_daily is None:
        df_cal = market[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
        debt_daily = fill_liabilities(
            bs=bs,
            df_rf=df_cal,
            debt_col="liabilities_total",
            publication_col="final_date",
            gvkey_col="gvkey",
        )
    else:
        # defensive copy + standardize types
        debt_daily = debt_daily.copy()
        debt_daily["gvkey"] = debt_daily["gvkey"].astype(str)
        debt_daily["date"] = pd.to_datetime(debt_daily["date"])

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
    debt_daily: pd.DataFrame | None = None,   # NEW
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

    # market panel (E + rf)
    market = build_market_rf_panel(
        ret_daily=ret_daily,
        df_rf=df_rf,
        equity_col=equity_col,
        id_cols=id_cols,
        drop_missing_r=drop_missing_r,
    )

    # liabilities daily: use provided debt_daily or compute internally
    if debt_daily is None:
        df_cal = market[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
        debt_daily = fill_liabilities(
            bs=bs,
            df_rf=df_cal,
            debt_col="liabilities_total",
            publication_col="final_date",
            gvkey_col="gvkey",
            tail_year_policy="ffill_face",   # or expose as parameter if you want
        )
    else:
        debt_daily = debt_daily.copy()
        debt_daily["gvkey"] = debt_daily["gvkey"].astype(str)
        debt_daily["date"] = pd.to_datetime(debt_daily["date"])

    # attach liabilities (as L)
    df = market.merge(
        debt_daily[["gvkey", "date", "L_pv"]],
        on=["gvkey", "date"],
        how="left",
    )

    df = df.rename(columns={"L_pv": "L"})
    df["L"] = pd.to_numeric(df["L"], errors="coerce")
    df = df.dropna(subset=["L"])
    df = df[df["L"] > 0].copy()

    em_inputs: Optional[Dict[Tuple[str, pd.Timestamp], Tuple[Any, Any, Any]]] = None
    if build_em:
        em_inputs = {}
        # iterate by firm and build windows
        for gv, g in df.groupby("gvkey", sort=False):
            g = g.sort_values("date")
            if len(g) < em_window:
                continue

            # sliding end dates: each window ending at each date from (em_window-1) onward
            for end_date in g["date"].iloc[em_window - 1:]:
                try:
                    E_arr, L_scalar, r_arr = make_em_inputs(
                        nig_panel=df,              # full panel OK; function filters internally
                        gvkey=str(gv),
                        end_date=end_date,
                        window=em_window,
                        use_filled_L=em_use_filled_L,
                        L_pick=em_L_pick,
                    )
                    em_inputs[(str(gv), pd.to_datetime(end_date))] = (E_arr, L_scalar, r_arr)
                except Exception:
                    continue

    return df, em_inputs