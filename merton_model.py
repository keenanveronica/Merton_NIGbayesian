import pandas as pd
import numpy as np
from pathlib import Path
from data_import import (
    load_data, load_ecb_1y_yield,
    fill_liabilities, drop_high_leverage_firms,
    prepare_merton_inputs
)
from merton_calibration import calibrate_merton_panel, add_physical_pd_from_implied_assets

# data loading and initial processing
ret_daily, bs, coverage = load_data(
    Path.cwd() / "data/raw/Jan2025_Accenture_Dataset_ErasmusCase.xlsx",
    start_date="2012-01-01",
    end_date="2025-12-19",
    enforce_coverage=True,
    coverage_tol=0.95,
    liabilities_scale="auto",
    verbose=True,
)

df_rf = load_ecb_1y_yield(
    startPeriod="2010-01-01",
    endPeriod="2025-12-31",
    out_file="ecb_yc_1y_aaa.xml",
    verify_ssl=True,  # recommended if it works
)

df_cal = ret_daily[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)

debt_daily = fill_liabilities(bs, df_cal)

ret_filt, bs_filt, lev_by_firm, dropped = drop_high_leverage_firms(
    ret_daily,
    bs,
    df_calendar=df_cal,
    debt_daily=debt_daily,
    lev_threshold=8.0,
    lev_agg="median",
    verbose=True,
)

# keep debt panel consistent with filtered firms
keep = set(ret_filt["gvkey"].astype(str).unique())
debt_daily_filt = debt_daily[debt_daily["gvkey"].astype(str).isin(keep)].copy()

# Merton
merton_df = prepare_merton_inputs(ret_filt, bs_filt, df_rf, debt_daily=debt_daily_filt)

# BUILDING THE CALIBRATION DATASET DROPPING ROWS WITH MISSING INPUTS
df = merton_df.copy()

# first date where B becomes available for each firm
first_B_date = (
    df.dropna(subset=["B"])
      .groupby("gvkey")["date"]
      .min()
      .rename("first_B_date")
)
# first date where sigma_E becomes available for each firm
first_sigma_date = (
    df.dropna(subset=["sigma_E"])
      .groupby("gvkey")["date"]
      .min()
      .rename("first_sigma_date")
)

starts = pd.concat([first_B_date, first_sigma_date], axis=1)
starts["calib_start"] = starts[["first_B_date","first_sigma_date"]].max(axis=1)

# attach and filter
df2 = df.merge(starts["calib_start"], on="gvkey", how="left")

calib = (
    df2[df2["date"] >= df2["calib_start"]]
      .dropna(subset=["E","B","r","sigma_E"])
      .query("E > 0 and B > 0")
      .copy()
      .rename(columns={"B":"B_drop"})
)

calib_drop = calib.copy()

# CALIBRATE MERTON MODEL
# Dropped missing B calibration
merton_calib = calibrate_merton_panel(
    calib_drop,
    B_col="B_drop",
    warm_start=True,
)

# physical PD via implied-asset drift
merton_calib = add_physical_pd_from_implied_assets(merton_calib)

# diagnostics
for name, df in [("dropped", merton_calib)]:
    print("\n---", name, "---")
    print("Solver success rate:", df["solver_success"].mean())
    print("V/E min:", np.nanmin(df["V"]/df["E"]), "median:", np.nanmedian(df["V"]/df["E"]))
    print("PD_rn summary:")
    print(df["PD_rn"].describe())


# printout dataset
cols = [
    "gvkey", "date", "isin", "company", "country_iso",
    "E", "sigmaE", "B", "r", "T",
    "solver_success", "DD_rn", "PD_rn",
    "logV", "dlogV", "mu_V", "DD_p", "PD_p",
]
df_show = df.loc[:, [c for c in cols if c in df.columns]]
print(df_show.head(5))
print(df_show.tail(5))
