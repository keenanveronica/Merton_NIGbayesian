import pandas as pd
import numpy as np
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
from typing import Optional, Tuple


def load_data(
    xlsx_path: Optional[Path] = None,
    min_days_per_firm: int = 0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Accenture Erasmus Case xlsx, clean market data and balance sheet data,
    and return (ret_daily, bs).

    Parameters
    ----------
    xlsx_path : Path or None
        Full path to the Excel file. If None,
        uses Path.cwd() / "data/raw/Jan2025_Accenture_Dataset_ErasmusCase.xlsx".
    min_days_per_firm : int
        If > 0, keeps only firms with at least this many daily return observations.
    verbose : bool
        If True, prints QA summaries.

    Returns
    -------
    ret_daily : pd.DataFrame
        Clean daily market panel with log returns computed from close and market cap.
    bs : pd.DataFrame
        Clean balance sheet panel (latest statement per gvkey-fyear).
    """
    # check file existance in current working directory
    if xlsx_path is None:
        xlsx_path = Path.cwd() / "data/raw/Jan2025_Accenture_Dataset_ErasmusCase.xlsx"
    xlsx_path = Path(xlsx_path)

    if not xlsx_path.exists():
        raise FileNotFoundError(f"No xlsx file found at: {xlsx_path}")

    # read excel sheets into dataframes
    mkt_raw = pd.read_excel(xlsx_path, sheet_name="Returns & MktCap")
    bs_raw = pd.read_excel(xlsx_path, sheet_name="Balance Sheet Data")

    # rename
    rename_mkt = {
        "(fic) Current ISO Country Code - Incorporation": "country_iso",
        "(isin) International Security Identification Number": "isin",
        "(datadate) Data Date - Daily Prices": "date",
        "(conm) Company Name": "company",
        "(gvkey) Global Company Key - Company": "gvkey",
        "(cshoc) Shares Outstanding": "shares_out",
        "(prccd) Price - Close - Daily": "close",
        "Market Capitalization (# Shares * Close Price)": "mcap_reported",
    }

    mkt = mkt_raw.rename(columns=rename_mkt).copy()

    # types
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt["gvkey"] = mkt["gvkey"].astype(str)
    for c in ["shares_out", "close", "mcap_reported"]:
        mkt[c] = pd.to_numeric(mkt[c], errors="coerce")

    # sorting
    mkt = mkt.sort_values(["gvkey", "date"])

    # cleaning steps for market data
    # 1 - fill shares outstanding within each firm (forward-fill then back-fill)
    mkt["shares_out_filled"] = (
        mkt.groupby("gvkey")["shares_out"]
        .ffill()
        .bfill()
    )

    # 2 - recompute mrkt cap
    mkt["mcap"] = mkt["close"] * mkt["shares_out_filled"]

    # 3 - flag “bad” days: insufficient daily pricing data
    mkt["bad_day"] = (
        mkt["close"].isna() | (mkt["close"] <= 0) |
        mkt["mcap"].isna() | (mkt["mcap"] <= 0)
    )

    mkt_clean = mkt.loc[~mkt["bad_day"]].copy()

    # compute log returns from closing prices
    mkt_clean["logret_close"] = (
        mkt_clean.groupby("gvkey")["close"]
                .transform(lambda s: np.log(s).diff())
    )

    # market-cap log returns
    mkt_clean["logret_mcap"] = (
        mkt_clean.groupby("gvkey")["mcap"]
                .transform(lambda s: np.log(s).diff())
    )

    # drop first observation per firm
    ret_daily = mkt_clean.dropna(subset=["logret_close"]).copy()

    # cleaning steps for balance sheet data
    rename_bs = {
        "(fic) Current ISO Country Code - Incorporation": "country_iso",
        "(costat) Active/Inactive Status Marker": "status",
        "(datafmt) Data Format": "data_format",
        "(indfmt) Industry Format": "industry_format",
        "(consol) Level of Consolidation - Company Annual "
        "Descriptor": "consolidation",
        "(isin) International Security Identification Number": "isin",
        "(datadate) Data Date": "date",
        "(conm) Company Name": "company",
        "(gvkey) Global Company Key - Company": "gvkey",
        "(fyear) Data Year - Fiscal": "fyear",
        "(fdate) Final Date": "final_date",
        "(lt) Liabilities - Total": "liabilities_total",
    }

    bs = bs_raw.rename(columns=rename_bs).copy()
    bs["date"] = pd.to_datetime(bs["date"])
    bs["final_date"] = pd.to_datetime(bs["final_date"], errors="coerce")
    bs["gvkey"] = bs["gvkey"].astype(str)
    bs["liabilities_total"] = pd.to_numeric(bs["liabilities_total"],
                                            errors="coerce")

    # drop Na liabilities
    bs = bs.dropna(subset=["liabilities_total"])

    # keep only latest final_date for statement
    bs = (
        bs.sort_values(["gvkey", "fyear", "final_date"])
        .groupby(["gvkey", "fyear"], as_index=False)
        .tail(1)
        .sort_values(["gvkey", "final_date"])
    )

    return ret_daily, bs


def load_ecb_1y_yield(
    startPeriod="2010-01-01",
    endPeriod="2025-12-31",
    out_file="ecb_yc_1y_aaa.xml",
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
        with open(out_file, "wb") as file:
            file.write(response.content)
        print(f"Data has been written to {out_file}")
    else:
        print(f"Failed to retrieve data: Status code {response.status_code}")
        print("Response text (first 500 chars):")
        print(response.text[:500])
        response.raise_for_status()

    # ---- Parse XML into DataFrame ----
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
