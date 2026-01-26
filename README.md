# Merton_NIGbayesian (team README)

**Important:** the Accenture/Erasmus dataset is **not** stored in this repo (or should not be). Place it locally in `data/raw/`.

---

## Repo structure (current)

- `data_import.py`  
  Loads the Excel dataset and (optionally) downloads/parses ECB risk-free data.
- `merton_calibration.py`  
  Merton calibration utilities (solving for implied asset value/volatility, DD, PD).
- `rf_import.py`  
  Risk-free rate importing/processing (ECB).
- `merton_notebook_attempt1.ipynb`  
  Working notebook for exploration / pipeline checks.

**Local-only (not tracked):**
Please note that the following folders need to be created and placed locally before running the code. 
- `data/raw/` – provided raw dataset(s)
- `data/derived/` – intermediate outputs (e.g., `calib_*.csv`)
---

## 1) Setup (Conda)

### Create the environment
If `environment.yml` exists:
```bash
conda env create -f environment.yml
conda activate <ENV_NAME>
