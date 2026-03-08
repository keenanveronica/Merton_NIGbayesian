from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd

from data_import import load_data


def _norm_name(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\b(sa|plc|ag|nv|se|spa|s\.a\.|s\.p\.a\.|ltd|limited|inc|corp|co)\b", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _best_fuzzy_match(sheet_norm: str, candidates_norm: pd.Series) -> Tuple[str, float]:
    best = ""
    best_score = -1.0
    for c in candidates_norm:
        score = SequenceMatcher(None, sheet_norm, c).ratio()
        if score > best_score:
            best_score = score
            best = c
    return best, float(best_score)


def get_cds_panel(
    *,
    project_root: Union[str, Path, None] = None,
    accenture_xlsx: Union[str, Path, None] = None,
    cds_xlsx: Union[str, Path, None] = None,
    start_date: str = "2012-01-01",
    end_date: str = "2025-12-19",
    enforce_coverage: bool = True,
    coverage_tol: float = 0.95,
    liabilities_scale: Union[str, float] = "auto",
    manual_sheet_to_gvkey: Optional[Dict[str, str]] = None,
    min_fuzzy_score: float = 0.88,
    save_csv: bool = False,
    out_csv: Union[str, Path, None] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Build a CDS panel from an Excel file with one sheet per firm.

    Returns
    -------
    cds_panel : DataFrame
        Columns: ['gvkey','date','cds'] (deduplicated, sorted).

    If save_csv=True, writes cds_panel to out_csv (or default location).
    """
    root = Path(project_root) if project_root is not None else Path.cwd()

    if accenture_xlsx is None:
        accenture_xlsx = root / "raw" / "Jan2025_Accenture_Dataset_ErasmusCase.xlsx"
    else:
        accenture_xlsx = Path(accenture_xlsx)

    if cds_xlsx is None:
        cds_xlsx = root / "raw" / "CDS_data_raw.xlsx"
    else:
        cds_xlsx = Path(cds_xlsx)

    if manual_sheet_to_gvkey is None:
        manual_sheet_to_gvkey = {}

    # output path
    if out_csv is None:
        out_csv = root / "data" / "derived" / "cds_panel.csv"
    else:
        out_csv = Path(out_csv)

    # --- Load Accenture market panel (for gvkey<->company mapping)
    ret_daily, _, _ = load_data(
        accenture_xlsx,
        start_date=start_date,
        end_date=end_date,
        enforce_coverage=enforce_coverage,
        coverage_tol=coverage_tol,
        liabilities_scale=liabilities_scale,
        verbose=verbose,
    )

    if "company" not in ret_daily.columns:
        raise ValueError("ret_daily is missing column 'company' (needed for sheet-name mapping).")

    firm_ref = (
        ret_daily[["gvkey", "company"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    firm_ref["gvkey"] = firm_ref["gvkey"].astype(str)
    firm_ref["company_norm"] = firm_ref["company"].map(_norm_name)
    firm_ref = firm_ref.groupby(["gvkey", "company_norm"], as_index=False).agg(company=("company", "first"))

    norm_to_gv = (
        firm_ref.groupby("company_norm")["gvkey"]
        .apply(lambda x: x.unique().tolist())
        .to_dict()
    )

    # --- Read CDS Excel (one sheet per firm)
    xls = pd.ExcelFile(cds_xlsx)
    sheets = xls.sheet_names

    rows = []
    unmapped = []

    for sh in sheets:
        df_sh = pd.read_excel(xls, sheet_name=sh).copy()
        df_sh.columns = [str(c).strip() for c in df_sh.columns]

        if df_sh.shape[1] < 2:
            unmapped.append({"sheet": sh, "reason": "Sheet has <2 columns"})
            continue

        c0, c1 = df_sh.columns[:2]
        df_sh = df_sh.rename(columns={c0: "date_raw", c1: "cds_raw"})

        df_sh["date"] = pd.to_datetime(df_sh["date_raw"], errors="coerce")
        df_sh["cds"] = pd.to_numeric(df_sh["cds_raw"], errors="coerce")
        df_sh = df_sh.dropna(subset=["date", "cds"]).copy()

        if sh in manual_sheet_to_gvkey:
            gv = str(manual_sheet_to_gvkey[sh])
        else:
            sh_norm = _norm_name(sh)
            gv_list = norm_to_gv.get(sh_norm, [])

            if len(gv_list) != 1:
                best_norm, score = _best_fuzzy_match(sh_norm, firm_ref["company_norm"])
                gv_list = norm_to_gv.get(best_norm, [])
                if not (len(gv_list) == 1 and score >= float(min_fuzzy_score)):
                    unmapped.append(
                        {"sheet": sh, "sheet_norm": sh_norm, "best_norm": best_norm, "score": score, "gv_list": gv_list}
                    )
                    continue

            gv = str(gv_list[0])

        df_sh["gvkey"] = gv
        df_sh["sheet"] = sh
        rows.append(df_sh[["gvkey", "date", "cds", "sheet"]])

    if not rows:
        raise ValueError("No CDS sheets were successfully parsed/mapped; check input file and mapping rules.")

    cds_long = pd.concat(rows, ignore_index=True)
    unmapped_df = pd.DataFrame(unmapped)

    if verbose:
        print(f"[get_cds_panel] sheets read: {len(sheets)} | rows parsed: {len(cds_long)} | unmapped sheets: {len(unmapped_df)}")
        if not unmapped_df.empty:
            cols = [c for c in ["sheet", "score", "best_norm", "gv_list", "reason"] if c in unmapped_df.columns]
            print(unmapped_df[cols].head(25))

    cds_panel = (
        cds_long[["gvkey", "date", "cds"]]
        .dropna(subset=["gvkey", "date", "cds"])
        .assign(gvkey=lambda d: d["gvkey"].astype(str), date=lambda d: pd.to_datetime(d["date"]))
        .sort_values(["gvkey", "date"])
        .drop_duplicates(["gvkey", "date"], keep="last")
        .reset_index(drop=True)
    )

    if save_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        cds_panel.to_csv(out_csv, index=False, date_format="%Y-%m-%d")
        if verbose:
            print(f"[get_cds_panel] saved: {out_csv}")

    return cds_panel


if __name__ == "__main__":
    cds_panel = get_cds_panel(save_csv=True, verbose=True)
    print(cds_panel.head())
