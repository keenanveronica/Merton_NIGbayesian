# Merton_NIGbayesian (team README)

# Merton_NIGbayesian

Structural credit-risk project for the Accenture / Erasmus School of Economics case study on extending the classical Merton model to address the non-normality of asset returns.

The project implements and compares two structural Probability of Default (PD) models for listed European firms:

1. **Classical Merton model**
2. **Extended structural model with NIG returns**

The aim is to estimate **1-year PDs**, compare the models quantitatively, and evaluate whether the non-Gaussian extension improves statistical fit, predictive usefulness, and economic interpretation.

---

## Important
This project uses:
- the **Accenture / Erasmus case dataset** for listed European firms (EUROSTOXX 50)
- **ECB risk-free / yield-curve data** (ECB Data Portal)
- **CDS data** used as an external benchmark in the evaluation stage

---

## Project objective

The project follows the Accenture case-study assignment:

- implement the **standard Merton model** to estimate **1-year PDs** for a sample of listed European firms
- develop and implement an **extended model** that relaxes the normality assumption
- compare both models in terms of **statistical fit**, **predictive accuracy**, and **economic interpretation**

Since the dataset does **not contain defaults**, model performance must be assessed creatively using indirect benchmarks and validation exercises.

---

## Theoretical background

In the classical structural model of Merton, the firm's equity is interpreted as a **European call option** on firm assets, with liabilities acting as the strike. Default occurs at the horizon if asset value falls below the debt threshold. The classical model relies on a **Gaussian / GBM-type asset-return assumption**, which is often violated in practice.

This repo extends that framework using a **Normal Inverse Gaussian (NIG) Lévy process**, following the direction proposed in the case study and the paper by Jovan and Ahčan. The motivation is that empirical return distributions often display skewness and fat tails that are not captured well by a normal distribution.

---

## Repo structure (current)

```text
.
├── data/
│   ├── raw/
│   │   ├── CDS_data_raw.xlsx
│   │   ├── ecb_yc_1y_aaa.csv
│   │   ├── ecb_yc_1y_aaa.xml
│   │   └── Jan2025_Accenture_Dataset_ErasmusCase.xlsx
│   └── derived/
│       ├── CDS_panel.csv
│       ├── ecb_riskfree_1y_daily.csv
│       ├── ecb_yc_1y_aaa.xml
│       ├── final_df.csv
│       ├── merton_weekly.csv
│       └── NIG_weekly.csv
│
├── notebooks_clean/
│   ├── eval_frequentist.ipynb
│   ├── merton_bayesian.ipynb
│   ├── merton_freq.ipynb
│   └── nig_freq.ipynb
│
├── notebooks_test/
│   └── ... exploratory / testing notebooks
│
├── src/
│   └── pd_estim_A/
│       ├── data/
│       │   ├── cds_df.py
│       │   └── data_import.py
│       ├── eval/
│       │   └── evaluation_frequentist.py
│       └── models/
│           ├── merton/
│           └── nig/
│
├── .gitignore
├── environment.yml
├── pyproject.toml
└── README.md
---

## Setup (Conda)

### Create the environment
If `environment.yml` exists:
```bash
conda env create -f environment.yml
conda activate <ENV_NAME>

