import requests
import pandas as pd
import xml.etree.ElementTree as ET


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
