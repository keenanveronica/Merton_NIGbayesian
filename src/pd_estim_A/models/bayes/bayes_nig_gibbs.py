"""
Bayesian weekly NIG structural model utilities via Gibbs sampling.

This module mirrors the high-level workflow and calendar logic of
`nig_em_afonso.py`, but replaces the frequentist EM estimation step with a
Gibbs sampler following the data-augmentation approach of Karlis & Lillestol
(2004), Approach 1 (Gamma-GIG) for the IG mixing law.

Core workflow for one rolling window:
1) infer the in-sample weekly asset path by inverting the NIG call-price
   formula under the current annual parameter draw,
2) compute weekly log-asset returns from that inferred asset path,
3) run Gibbs updates for the NIG mixture representation,
4) summarize posterior draws in-sample and over the requested OOS window.

Conventions
-----------
- Parameters are stored and returned in ANNUAL units:
    alpha, beta1, delta, beta0
  where beta1 is the NIG skewness parameter and beta0 is the drift/location.
- Weekly scaling uses h = 1 / ann_factor, with ann_factor defaulting to 52.
- Daily inputs can be converted to a Friday-aligned weekly panel using the
  same helper used by the frequentist NIG module.
- Training-window inversion follows the same paper-style logic as the
  frequentist module: liabilities are fixed at the end-of-window level and
  time-to-maturity decreases backward through the training sample.
- OOS inversion uses current liabilities and fixed tau = forecast_horizon_years.

Notes on the Gibbs blocks
-------------------------
Let y_t denote weekly log-asset returns inferred from the current asset path.
Using the NIG mixture representation,

    y_t | z_t ~ N(beta0_h + beta1 * z_t, z_t)
    z_t       ~ IG(gamma, delta_h)

with
    gamma   = sqrt(alpha^2 - beta1^2)
    delta_h = delta / ann_factor
    beta0_h = beta0 / ann_factor

Approach 1 of Karlis & Lillestol is implemented on the IG2(mu, phi)
reparametrization:
    mu_ig = delta_h / gamma
    phi   = delta_h * gamma

The default priors are weakly informative and centered on the window-specific
frequentist estimates supplied by the caller.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from scipy.stats import geninvgauss
from scipy.stats import multivariate_normal

from pd_estim_A.models.nig.nig_em import (
    _as_ts,
    _clean_weekly_panel,
    _make_params_window_feasible,
    _require_columns,
    _slice_window,
    daily_to_weekly_nig_panel,
    forecast_nig_oos_window,
    infer_training_asset_path_nig,
    update_theta_series,
    validate_nig_params,
)


EPS = 1e-12


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------


def _coerce_param_dict(params: Dict[str, float] | pd.Series) -> Dict[str, float]:
    """
    Accept either the frequentist naming convention
      {alpha, beta1, delta, beta0}
    or the alternative convention
      {alpha, beta, delta, mu}
    and return a standardized annual-parameter dictionary.
    """
    if isinstance(params, pd.Series):
        params = params.to_dict()

    beta_key = "beta1" if "beta1" in params else "beta"
    mu_key = "beta0" if "beta0" in params else "mu"

    out = {
        "alpha": float(params["alpha"]),
        "beta1": float(params[beta_key]),
        "delta": float(params["delta"]),
        "beta0": float(params[mu_key]),
    }

    if not all(np.isfinite(v) for v in out.values()):
        raise ValueError("All NIG parameters must be finite.")
    validate_nig_params(out["alpha"], out["beta1"], out["delta"])
    return out



def _annual_to_step_params(params_annual: Dict[str, float], ann_factor: float) -> Dict[str, float]:
    """Convert annual parameters to one-step (weekly) quantities."""
    p = _coerce_param_dict(params_annual)
    h = 1.0 / float(ann_factor)

    alpha = float(p["alpha"])
    beta1 = float(p["beta1"])
    delta_h = float(p["delta"]) * h
    beta0_h = float(p["beta0"]) * h

    gamma_val = float(np.sqrt(max(alpha * alpha - beta1 * beta1, 0.0)))
    if delta_h <= 0.0 or gamma_val <= 0.0:
        raise ValueError("Need delta_h > 0 and gamma > 0.")

    mu_ig = float(delta_h / gamma_val)
    phi = float(delta_h * gamma_val)

    return {
        "alpha": alpha,
        "beta1": beta1,
        "delta_h": delta_h,
        "beta0_h": beta0_h,
        "gamma": gamma_val,
        "mu_ig": mu_ig,
        "phi": phi,
        "h": h,
    }



def _step_to_annual_params(
    *,
    mu_ig: float,
    phi: float,
    beta1: float,
    beta0_h: float,
    ann_factor: float,
) -> Dict[str, float]:
    """
    Map the Gibbs-updated step-scale parameters back to annual NIG parameters.

    Using Approach 1 reparametrization:
      mu_ig = delta_h / gamma
      phi   = delta_h * gamma

    Hence
      delta_h = sqrt(mu_ig * phi)
      gamma   = sqrt(phi / mu_ig)
      alpha   = sqrt(beta1^2 + gamma^2)
      delta   = delta_h * ann_factor
      beta0   = beta0_h * ann_factor
    """
    mu_ig = float(mu_ig)
    phi = float(phi)
    beta1 = float(beta1)
    beta0_h = float(beta0_h)

    if mu_ig <= 0.0 or phi <= 0.0:
        raise ValueError("mu_ig and phi must be positive.")

    delta_h = float(np.sqrt(mu_ig * phi))
    gamma_val = float(np.sqrt(phi / mu_ig))
    alpha = float(np.sqrt(beta1 * beta1 + gamma_val * gamma_val))
    delta = float(delta_h) * float(ann_factor)
    beta0 = float(beta0_h) * float(ann_factor)

    out = {
        "alpha": alpha,
        "beta1": beta1,
        "delta": delta,
        "beta0": beta0,
    }
    validate_nig_params(out["alpha"], out["beta1"], out["delta"])
    return out


# ---------------------------------------------------------------------------
# Gibbs building blocks
# ---------------------------------------------------------------------------


def sample_gig(lam: float, chi: float, psi: float, rng: np.random.Generator) -> float:
    """
    Draw from GIG(lambda, chi, psi) using SciPy's standardized geninvgauss.

    If X ~ GIG(lambda, chi, psi), then
      Y = X / sqrt(chi / psi)
    follows SciPy's `geninvgauss(p=lambda, b=sqrt(chi*psi))`.
    """
    chi = float(chi)
    psi = float(psi)
    if chi <= 0.0 or psi <= 0.0:
        raise ValueError("chi and psi must be positive.")

    b = float(np.sqrt(chi * psi))
    y = geninvgauss.rvs(lam, b, random_state=rng)
    return float(y * np.sqrt(chi / psi))



def sample_latent_z(
    returns: np.ndarray,
    *,
    alpha: float,
    delta_h: float,
    beta0_h: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw z_t | y_t for the NIG normal-IG mixture.

    With y_t | z_t ~ N(beta0_h + beta1 z_t, z_t) and z_t ~ IG(gamma, delta_h),
    the conditional posterior is
      z_t | y_t ~ GIG(-1, chi_t, psi)
    where
      chi_t = (y_t - beta0_h)^2 + delta_h^2
      psi   = alpha^2
    """
    x = np.asarray(returns, dtype=float).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("returns must be a non-empty finite vector.")
    if alpha <= 0.0 or delta_h <= 0.0:
        raise ValueError("alpha and delta_h must be positive.")

    out = np.empty_like(x, dtype=float)
    psi = float(alpha * alpha)
    d2 = float(delta_h * delta_h)

    for i, yt in enumerate(x):
        chi = float((yt - beta0_h) ** 2 + d2)
        out[i] = sample_gig(-1.0, chi, psi, rng)
    return out



def sample_regression_block(
    returns: np.ndarray,
    z: np.ndarray,
    b0: np.ndarray,
    B0: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Draw (beta0_h, beta1) from the heteroscedastic normal regression block.

      y_t = beta0_h + beta1 * z_t + eps_t,
      eps_t | z_t ~ N(0, z_t)

    Prior:
      [beta0_h, beta1]' ~ N(b0, B0)
    """
    y = np.asarray(returns, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    if y.shape != z.shape:
        raise ValueError("returns and z must have the same shape.")
    if y.size < 2:
        raise ValueError("Need at least two returns to draw regression parameters.")

    b0 = np.asarray(b0, dtype=float).reshape(2, 1)
    B0 = np.asarray(B0, dtype=float)
    if B0.shape != (2, 2):
        raise ValueError("B0 must be 2x2.")

    w = 1.0 / np.maximum(z, EPS)
    X = np.column_stack([np.ones_like(z), z])
    XtW = X.T * w

    try:
        B0_inv = np.linalg.inv(B0)
    except np.linalg.LinAlgError as exc:
        raise ValueError("B0 must be invertible.") from exc

    post_prec = XtW @ X + B0_inv
    post_cov = np.linalg.inv(post_prec)
    post_mean = post_cov @ (XtW @ y.reshape(-1, 1) + B0_inv @ b0)
    draw = multivariate_normal.rvs(mean=post_mean.reshape(-1), cov=post_cov, random_state=rng)

    return float(draw[0]), float(draw[1])



def sample_ig2_block(
    z: np.ndarray,
    *,
    phi_current: float,
    hyper: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Draw (mu_ig, phi) for the IG2(mu_ig, phi) mixing law, using
    Karlis & Lillestol Approach 1.

    Hyperparameters:
      phi          ~ Gamma(xi, chi)      [shape-rate]
      mu_ig | phi  ~ IG2(eta, phi * omega)
    """
    z = np.asarray(z, dtype=float).reshape(-1)
    if z.size < 2:
        raise ValueError("Need at least two latent z draws.")
    if not np.all(np.isfinite(z)) or np.any(z <= 0.0):
        raise ValueError("z must be positive and finite.")

    xi = float(hyper["xi"])
    chi = float(hyper["chi"])
    eta = float(hyper["eta"])
    omega = float(hyper["omega"])

    if xi <= 0.0 or chi <= 0.0 or eta <= 0.0 or omega <= 0.0:
        raise ValueError("All IG2 hyperparameters must be positive.")
    if phi_current <= 0.0:
        raise ValueError("phi_current must be positive.")

    n = z.size
    zbar = float(np.mean(z))
    zbar_r = float(np.mean(1.0 / z))

    v = float(n + 2.0 * xi)
    u1 = float(n * zbar + omega * eta)
    u2 = float(n + omega - chi)
    u3 = float(n * zbar_r + omega / eta)

    # mu_ig | phi ~ GIG((n-1)/2, phi*u1, phi*u3)
    lam_mu = 0.5 * float(n - 1)
    mu_new = sample_gig(lam_mu, phi_current * u1, phi_current * u3, rng)

    # phi | mu_ig ~ Gamma((v+1)/2, rate)
    shape_phi = 0.5 * (v + 1.0)
    rate_phi = float(u1 / (2.0 * mu_new) - u2 + 0.5 * u3 * mu_new)
    rate_phi = max(rate_phi, EPS)
    phi_new = float(gamma_dist.rvs(shape_phi, scale=1.0 / rate_phi, random_state=rng))

    return float(mu_new), float(phi_new)


# ---------------------------------------------------------------------------
# Prior helpers
# ---------------------------------------------------------------------------


def build_default_priors_from_em(
    em_params: Dict[str, float] | pd.Series,
    *,
    ann_factor: float = 52.0,
    B0_diag: Tuple[float, float] = (50.0, 50.0),
    phi_prior_variance: float = 100.0,
    omega: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Build weakly informative priors centered on the frequentist annual estimate.

    Returns
    -------
    b0 : ndarray, shape (2,)
        Prior mean for [beta0_h, beta1].
    B0 : ndarray, shape (2, 2)
        Prior covariance for [beta0_h, beta1].
    hyper : dict
        Hyperparameters for the IG2(mu_ig, phi) block.
    """
    em = _coerce_param_dict(em_params)
    step = _annual_to_step_params(em, ann_factor=ann_factor)

    b0 = np.array([step["beta0_h"], em["beta1"]], dtype=float)
    B0 = np.diag(np.asarray(B0_diag, dtype=float))

    phi_center = float(step["phi"])
    if phi_prior_variance <= 0.0:
        raise ValueError("phi_prior_variance must be positive.")
    if omega <= 0.0:
        raise ValueError("omega must be positive.")

    xi = float((phi_center * phi_center) / phi_prior_variance)
    chi = float(phi_center / phi_prior_variance)

    hyper = {
        "xi": max(xi, 1e-8),
        "chi": max(chi, 1e-8),
        "eta": float(step["mu_ig"]),
        "omega": float(omega),
    }
    return b0, B0, hyper



def _prepare_prior_inputs(
    em_params: Dict[str, float] | pd.Series,
    *,
    ann_factor: float,
    prior_b0: Optional[np.ndarray] = None,
    prior_B0: Optional[np.ndarray] = None,
    prior_hyper: Optional[Dict[str, float]] = None,
    default_B0_diag: Tuple[float, float] = (50.0, 50.0),
    phi_prior_variance: float = 100.0,
    omega: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Prepare prior inputs for the Gibbs sampler.

    Conventions
    -----------
    - `em_params` are annual.
    - If `prior_b0` is passed, it is interpreted as [beta0_annual, beta1] and
      converted internally to [beta0_h, beta1].
    - If `prior_B0` is passed, it is interpreted directly on the Gibbs-regression
      scale [beta0_h, beta1].
    - If `prior_hyper` is passed, it is assumed already on the IG2 step scale.
    """
    em = _coerce_param_dict(em_params)
    step = _annual_to_step_params(em, ann_factor=ann_factor)

    b0_default, B0_default, hyper_default = build_default_priors_from_em(
        em,
        ann_factor=ann_factor,
        B0_diag=default_B0_diag,
        phi_prior_variance=phi_prior_variance,
        omega=omega,
    )

    if prior_b0 is None:
        b0 = b0_default
    else:
        prior_b0 = np.asarray(prior_b0, dtype=float).reshape(2)
        b0 = np.array([float(prior_b0[0]) / float(ann_factor), float(prior_b0[1])], dtype=float)

    B0 = B0_default if prior_B0 is None else np.asarray(prior_B0, dtype=float)
    if B0.shape != (2, 2):
        raise ValueError("prior_B0 must be 2x2.")
    if not np.all(np.isfinite(B0)):
        raise ValueError("prior_B0 must be finite.")
    try:
        np.linalg.inv(B0)
    except np.linalg.LinAlgError as exc:
        raise ValueError("prior_B0 must be invertible.") from exc

    if prior_hyper is None:
        hyper = hyper_default
    else:
        hyper = {k: float(v) for k, v in prior_hyper.items()}

    required = {"xi", "chi", "eta", "omega"}
    missing = required.difference(hyper)
    if missing:
        raise ValueError(f"prior_hyper missing keys: {sorted(missing)}")
    if any(hyper[k] <= 0.0 or not np.isfinite(hyper[k]) for k in required):
        raise ValueError("All prior_hyper values must be positive and finite.")

    # Keep the step quantities around for diagnostics if needed.
    hyper["_phi_center_from_em"] = float(step["phi"])
    hyper["_mu_ig_center_from_em"] = float(step["mu_ig"])

    return b0.astype(float), B0.astype(float), hyper


# ---------------------------------------------------------------------------
# Posterior summaries
# ---------------------------------------------------------------------------


def _posterior_summary(x: np.ndarray) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float)
    return {
        "mean": np.nanmean(x, axis=0),
        "median": np.nanmedian(x, axis=0),
        "q05": np.nanquantile(x, 0.05, axis=0),
        "q95": np.nanquantile(x, 0.95, axis=0),
    }



def summarize_training_posterior_panel(
    result: Dict[str, Any],
    *,
    gvkey: Optional[str] = None,
    train_start: Any = None,
    train_end: Any = None,
) -> pd.DataFrame:
    """Summarize in-sample posterior draws for one training window."""
    base = result["train_input_df"].copy().sort_values("date").reset_index(drop=True)

    A_sum = _posterior_summary(result["A_train_draws"])
    theta_sum = _posterior_summary(result["theta_train_draws"])
    params_sum = _posterior_summary(result["params_draws_annual"])

    out = base.copy()
    out["A_train_mean"] = A_sum["mean"]
    out["A_train_median"] = A_sum["median"]
    out["A_train_q05"] = A_sum["q05"]
    out["A_train_q95"] = A_sum["q95"]

    out["theta_train_mean"] = theta_sum["mean"]
    out["theta_train_median"] = theta_sum["median"]
    out["theta_train_q05"] = theta_sum["q05"]
    out["theta_train_q95"] = theta_sum["q95"]

    out["logA_train_median"] = np.log(np.maximum(out["A_train_median"].to_numpy(dtype=float), EPS))
    out["dlogA_train_median"] = pd.Series(out["logA_train_median"]).diff().to_numpy(dtype=float)

    out["alpha_post_mean"] = float(params_sum["mean"][0])
    out["beta1_post_mean"] = float(params_sum["mean"][1])
    out["delta_post_mean"] = float(params_sum["mean"][2])
    out["beta0_post_mean"] = float(params_sum["mean"][3])

    out["alpha_post_median"] = float(params_sum["median"][0])
    out["beta1_post_median"] = float(params_sum["median"][1])
    out["delta_post_median"] = float(params_sum["median"][2])
    out["beta0_post_median"] = float(params_sum["median"][3])

    if gvkey is not None:
        out["gvkey"] = str(gvkey)
    if train_start is not None:
        out["window_train_start"] = pd.Timestamp(train_start)
    if train_end is not None:
        out["window_train_end"] = pd.Timestamp(train_end)
    return out



def summarize_oos_posterior_panel(
    result: Dict[str, Any],
    *,
    gvkey: Optional[str] = None,
    train_start: Any = None,
    train_end: Any = None,
    oos_start: Any = None,
    oos_end: Any = None,
    window_idx: Optional[int] = None,
    lo_q: float = 0.025,
    hi_q: float = 0.975,
) -> pd.DataFrame:
    """Weekly OOS posterior summaries in the requested wide format."""
    if "oos_input_df" not in result or result["oos_input_df"].empty:
        return pd.DataFrame()

    base = result["oos_input_df"].copy().sort_values("date").reset_index(drop=True)
    out = pd.DataFrame({"date": pd.to_datetime(base["date"])})

    if gvkey is not None:
        out["gvkey"] = str(gvkey)
    if window_idx is not None:
        out["window_idx"] = int(window_idx)

    out["rf_used"] = pd.to_numeric(base["rf"], errors="coerce")
    out["B_used"] = pd.to_numeric(base["debt_face"], errors="coerce")

    mapping = {
        "A_oos_draws": "asset",
        "PD_Q_draws": "PD_Q",
        "PD_P_draws": "PD_P",
    }

    n_dates = len(base)
    for key, stem in mapping.items():
        arr = np.asarray(result.get(key, np.empty((0, n_dates))), dtype=float)
        if arr.ndim != 2 or arr.shape[1] != n_dates:
            arr = np.empty((0, n_dates), dtype=float)

        if arr.shape[0] == 0:
            out[f"{stem}_lo"] = np.nan
            out[f"{stem}_med"] = np.nan
            out[f"{stem}_hi"] = np.nan
            out[f"{stem}_mean"] = np.nan
            out[f"{stem}_n_valid"] = 0
            continue

        out[f"{stem}_lo"] = np.nanquantile(arr, lo_q, axis=0)
        out[f"{stem}_med"] = np.nanquantile(arr, 0.50, axis=0)
        out[f"{stem}_hi"] = np.nanquantile(arr, hi_q, axis=0)
        out[f"{stem}_mean"] = np.nanmean(arr, axis=0)
        out[f"{stem}_n_valid"] = np.sum(np.isfinite(arr), axis=0).astype(int)

    if train_start is not None:
        out["train_start"] = pd.Timestamp(train_start)
    if train_end is not None:
        out["train_end"] = pd.Timestamp(train_end)
    if oos_start is not None:
        out["oos_start"] = pd.Timestamp(oos_start)
    if oos_end is not None:
        out["oos_end"] = pd.Timestamp(oos_end)

    return out


def summarize_parameter_window_posterior(
    result: Dict[str, Any],
    *,
    gvkey: Optional[str] = None,
    train_start: Any = None,
    train_end: Any = None,
    window_idx: Optional[int] = None,
    lo_q: float = 0.05,
    hi_q: float = 0.95,
) -> pd.DataFrame:
    """
    One-row posterior summary for the window-level NIG parameters.

    `n_obs` is the number of weekly log-returns used by the Gibbs step.
    """
    params_draws = np.asarray(result.get("params_draws_annual", np.empty((0, 4))), dtype=float)
    if params_draws.ndim != 2 or params_draws.shape[0] == 0:
        return pd.DataFrame()

    if not (0.0 < lo_q < 0.5 < hi_q < 1.0):
        raise ValueError("Need 0 < lo_q < 0.5 < hi_q < 1.")

    row: Dict[str, Any] = {}

    if gvkey is not None:
        row["gvkey"] = str(gvkey)
    if train_start is not None:
        row["train_start"] = pd.Timestamp(train_start)
    if train_end is not None:
        row["train_end"] = pd.Timestamp(train_end)
    if window_idx is not None:
        row["window_idx"] = int(window_idx)

    row["n_obs"] = int(np.asarray(result["A_train_fixed"]).size)

    names = ["alpha", "beta1", "delta", "beta0"]
    for j, name in enumerate(names):
        col = params_draws[:, j]
        row[f"{name}_lo"] = float(np.nanquantile(col, lo_q))
        row[f"{name}_med"] = float(np.nanquantile(col, 0.50))
        row[f"{name}_hi"] = float(np.nanquantile(col, hi_q))
        row[f"{name}_mean"] = float(np.nanmean(col))

    meta = result.get("meta", {})
    row["n_keep_requested"] = int(meta.get("n_keep_requested", 0))
    row["n_keep_actual"] = int(meta.get("n_keep_actual", params_draws.shape[0]))
    row["n_rejects"] = int(meta.get("n_reject", 0))
    row["n_fail_z"] = int(meta.get("n_fail_z", 0))
    row["n_fail_candidate"] = int(meta.get("n_fail_candidate", 0))
    row["ok"] = True
    row["msg"] = "ok"

    return pd.DataFrame([row])    


# ---------------------------------------------------------------------------
# Core sampler for one training window
# ---------------------------------------------------------------------------


def _infer_fixed_training_path_from_em(
    train_df: pd.DataFrame,
    *,
    em_params: Dict[str, float] | pd.Series,
    date_col: str = "date",
    equity_col: str = "market_cap",
    debt_col: str = "debt_face",
    rf_col: str = "rf",
    ann_factor: float = 52.0,
    forecast_horizon_years: float = 1.0,
    discounting: str = "continuous",
) -> Dict[str, Any]:
    """
    Infer the training-window asset path ONCE from the frequentist point estimate
    and treat it as fixed in the conditional Gibbs step.
    """
    _require_columns(train_df, [date_col, equity_col, debt_col, rf_col])
    w = train_df.copy().sort_values(date_col).reset_index(drop=True)
    if len(w) < 5:
        raise ValueError("Training window too short for Gibbs estimation.")

    E = w[equity_col].to_numpy(dtype=float)
    L = w[debt_col].to_numpy(dtype=float)
    rf = w[rf_col].to_numpy(dtype=float)

    em = _coerce_param_dict(em_params)
    em_feasible, theta_em = _make_params_window_feasible(em, rf)
    A_fixed, theta_fixed = infer_training_asset_path_nig(
        E_series=E,
        L_face_series=L,
        rf_series=rf,
        params=em_feasible,
        ann_factor=ann_factor,
        forecast_horizon_years=forecast_horizon_years,
        discounting=discounting,
    )

    if not np.all(np.isfinite(A_fixed)) or np.any(A_fixed <= 0.0):
        raise ValueError("Fixed EM inversion failed to produce a valid training asset path.")

    rA_fixed = np.diff(np.log(A_fixed))
    if rA_fixed.size < 2 or not np.all(np.isfinite(rA_fixed)):
        raise ValueError("Fixed EM inversion produced invalid weekly asset returns.")

    train_input_df = w[[date_col, equity_col, debt_col, rf_col]].copy().rename(
        columns={
            date_col: "date",
            equity_col: "market_cap",
            debt_col: "debt_face",
            rf_col: "rf",
        }
    )

    return {
        "train_input_df": train_input_df,
        "rf_train": rf,
        "A_train_fixed": A_fixed,
        "theta_train_fixed": theta_fixed,
        "rA_train_fixed": rA_fixed,
        "em_params_raw": em,
        "em_params_feasible": em_feasible,
        "theta_em_feasible": theta_em,
    }


def gibbs_nig_training_window(
    train_df: pd.DataFrame,
    *,
    em_params: Dict[str, float] | pd.Series,
    date_col: str = "date",
    equity_col: str = "market_cap",
    debt_col: str = "debt_face",
    rf_col: str = "rf",
    ann_factor: float = 52.0,
    forecast_horizon_years: float = 1.0,
    max_iter: int = 2000,
    burn_in: int = 500,
    thin: int = 5,
    prior_b0: Optional[np.ndarray] = None,
    prior_B0: Optional[np.ndarray] = None,
    prior_hyper: Optional[Dict[str, float]] = None,
    default_B0_diag: Tuple[float, float] = (50.0, 50.0),
    phi_prior_variance: float = 100.0,
    omega: float = 1e-3,
    discounting: str = "continuous",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Run the conditional Gibbs sampler on one weekly training window.

    The training panel must already be weekly and cleaned. The implied asset
    path is inverted ONCE from the frequentist window estimate and then treated
    as fixed throughout the Gibbs iterations.
    """
    if rng is None:
        rng = np.random.default_rng()

    if burn_in < 0 or burn_in >= max_iter:
        raise ValueError("burn_in must satisfy 0 <= burn_in < max_iter.")
    if thin <= 0:
        raise ValueError("thin must be >= 1.")

    fixed = _infer_fixed_training_path_from_em(
        train_df,
        em_params=em_params,
        date_col=date_col,
        equity_col=equity_col,
        debt_col=debt_col,
        rf_col=rf_col,
        ann_factor=ann_factor,
        forecast_horizon_years=forecast_horizon_years,
        discounting=discounting,
    )

    rf = np.asarray(fixed["rf_train"], dtype=float)
    rA_fixed = np.asarray(fixed["rA_train_fixed"], dtype=float)
    A_fixed = np.asarray(fixed["A_train_fixed"], dtype=float)
    theta_fixed = np.asarray(fixed["theta_train_fixed"], dtype=float)
    em = fixed["em_params_raw"]
    params_current = dict(fixed["em_params_feasible"])

    b0, B0, hyper = _prepare_prior_inputs(
        em,
        ann_factor=ann_factor,
        prior_b0=prior_b0,
        prior_B0=prior_B0,
        prior_hyper=prior_hyper,
        default_B0_diag=default_B0_diag,
        phi_prior_variance=phi_prior_variance,
        omega=omega,
    )

    step_current = _annual_to_step_params(params_current, ann_factor=ann_factor)
    mu_ig_current = float(step_current["mu_ig"])
    phi_current = float(step_current["phi"])

    keep_idx = [it for it in range(max_iter) if (it >= burn_in and ((it - burn_in) % thin == 0))]
    keep_set = set(keep_idx)
    n_keep_target = len(keep_idx)

    n_obs = A_fixed.size
    n_ret = rA_fixed.size

    A_train_draws = np.full((n_keep_target, n_obs), np.nan, dtype=float)
    theta_train_draws = np.full((n_keep_target, n_obs), np.nan, dtype=float)
    z_draws = np.full((n_keep_target, n_ret), np.nan, dtype=float)
    params_draws_annual = np.full((n_keep_target, 4), np.nan, dtype=float)
    mu_phi_draws = np.full((n_keep_target, 2), np.nan, dtype=float)

    n_reject = 0
    n_fail_inversion = 0
    n_fail_z = 0
    n_fail_candidate = 0
    keep_pos = 0

    for it in range(max_iter):
        store_this_iter = it in keep_set
        step_current = _annual_to_step_params(params_current, ann_factor=ann_factor)

        # 1) z | y, with y fixed from the EM-implied asset path.
        try:
            z = sample_latent_z(
                rA_fixed,
                alpha=step_current["alpha"],
                delta_h=step_current["delta_h"],
                beta0_h=step_current["beta0_h"],
                rng=rng,
            )
        except Exception:
            n_reject += 1
            n_fail_z += 1
            continue

        # 2) Regression block for (beta0_h, beta1).
        try:
            beta0_h_new, beta1_new = sample_regression_block(rA_fixed, z, b0, B0, rng)
        except Exception:
            beta0_h_new = float(step_current["beta0_h"])
            beta1_new = float(params_current["beta1"])

        # 3) IG2(mu_ig, phi) block, Approach 1.
        try:
            mu_ig_new, phi_new = sample_ig2_block(
                z,
                phi_current=phi_current,
                hyper=hyper,
                rng=rng,
            )
        except Exception:
            mu_ig_new, phi_new = mu_ig_current, phi_current

        # 4) Map back to annual parameters and keep only theta-feasible draws.
        is_feasible = False
        try:
            params_candidate = _step_to_annual_params(
                mu_ig=mu_ig_new,
                phi=phi_new,
                beta1=beta1_new,
                beta0_h=beta0_h_new,
                ann_factor=ann_factor,
            )
            theta_candidate = update_theta_series(params_candidate, rf)
            alpha_needed = float(
                np.nanmax(
                    np.maximum(
                        np.abs(params_candidate["beta1"] + theta_candidate),
                        np.abs(params_candidate["beta1"] + theta_candidate + 1.0),
                    )
                )
            )
            is_feasible = (
                np.all(np.isfinite(theta_candidate))
                and np.isfinite(alpha_needed)
                and params_candidate["alpha"] > alpha_needed
            )
        except Exception:
            is_feasible = False

        if is_feasible:
            params_current = params_candidate
            mu_ig_current = float(mu_ig_new)
            phi_current = float(phi_new)
            theta_store = theta_candidate
        else:
            n_reject += 1
            n_fail_candidate += 1
            theta_store = update_theta_series(params_current, rf)

        if store_this_iter:
            A_train_draws[keep_pos, :] = A_fixed
            theta_train_draws[keep_pos, :] = theta_store
            params_draws_annual[keep_pos, :] = np.array(
                [
                    params_current["alpha"],
                    params_current["beta1"],
                    params_current["delta"],
                    params_current["beta0"],
                ],
                dtype=float,
            )
            mu_phi_draws[keep_pos, :] = np.array([mu_ig_current, phi_current], dtype=float)
            z_draws[keep_pos, :] = z
            keep_pos += 1

    A_train_draws = A_train_draws[:keep_pos, :]
    theta_train_draws = theta_train_draws[:keep_pos, :]
    z_draws = z_draws[:keep_pos, :]
    params_draws_annual = params_draws_annual[:keep_pos, :]
    mu_phi_draws = mu_phi_draws[:keep_pos, :]

    return {
        "train_input_df": fixed["train_input_df"],
        "A_train_fixed": A_fixed,
        "theta_train_fixed": theta_fixed,
        "rA_train_fixed": rA_fixed,
        "em_params_raw": fixed["em_params_raw"],
        "em_params_feasible": fixed["em_params_feasible"],
        "A_train_draws": A_train_draws,
        "theta_train_draws": theta_train_draws,
        "z_draws": z_draws,
        "params_draws_annual": params_draws_annual,
        "mu_phi_draws": mu_phi_draws,
        "priors": {"b0": b0, "B0": B0, "hyper": hyper},
        "init_params_annual": dict(fixed["em_params_feasible"]),
        "last_params_annual": dict(params_current),
        "meta": {
            "max_iter": int(max_iter),
            "burn_in": int(burn_in),
            "thin": int(thin),
            "n_keep_requested": int(n_keep_target),
            "n_keep_actual": int(keep_pos),
            "n_reject": int(n_reject),
            "n_fail_inversion": int(n_fail_inversion),
            "n_fail_z": int(n_fail_z),
            "n_fail_candidate": int(n_fail_candidate),
            "ann_factor": float(ann_factor),
            "forecast_horizon_years": float(forecast_horizon_years),
            "conditional_fixed_asset_path": True,
        },
    }


# ---------------------------------------------------------------------------
# OOS forecasting under posterior draws
# ---------------------------------------------------------------------------


def forecast_oos_posterior_draws(
    oos_df: pd.DataFrame,
    *,
    params_draws_annual: np.ndarray,
    date_col: str = "date",
    equity_col: str = "market_cap",
    debt_col: str = "debt_face",
    rf_col: str = "rf",
    pd_horizon_years: float = 1.0,
    inversion_tau_years: float = 1.0,
    discounting: str = "continuous",
) -> Dict[str, Any]:
    """Run the OOS inversion / PD computation under every kept posterior draw."""
    _require_columns(oos_df, [date_col, equity_col, debt_col, rf_col])
    w = oos_df.copy().sort_values(date_col).reset_index(drop=True)

    n_draws = int(np.asarray(params_draws_annual).shape[0])
    n_oos = len(w)

    A_oos_draws = np.full((n_draws, n_oos), np.nan, dtype=float)
    theta_oos_draws = np.full((n_draws, n_oos), np.nan, dtype=float)
    PD_P_draws = np.full((n_draws, n_oos), np.nan, dtype=float)
    PD_Q_draws = np.full((n_draws, n_oos), np.nan, dtype=float)

    for j, draw in enumerate(np.asarray(params_draws_annual, dtype=float)):
        params = {
            "alpha": float(draw[0]),
            "beta1": float(draw[1]),
            "delta": float(draw[2]),
            "beta0": float(draw[3]),
        }
        try:
            oos_one = forecast_nig_oos_window(
                w,
                params,
                date_col=date_col,
                equity_col=equity_col,
                debt_col=debt_col,
                rf_col=rf_col,
                pd_horizon_years=pd_horizon_years,
                inversion_tau_years=inversion_tau_years,
                discounting=discounting,
            )
            A_oos_draws[j, :] = oos_one["A_hat_oos"].to_numpy(dtype=float)
            theta_oos_draws[j, :] = oos_one["theta_oos"].to_numpy(dtype=float)
            PD_P_draws[j, :] = oos_one["PD_P"].to_numpy(dtype=float)
            PD_Q_draws[j, :] = oos_one["PD_Q"].to_numpy(dtype=float)
        except Exception:
            continue

    base = w[[date_col, equity_col, debt_col, rf_col]].copy().rename(
        columns={
            date_col: "date",
            equity_col: "market_cap",
            debt_col: "debt_face",
            rf_col: "rf",
        }
    )

    return {
        "oos_input_df": base,
        "A_oos_draws": A_oos_draws,
        "theta_oos_draws": theta_oos_draws,
        "PD_P_draws": PD_P_draws,
        "PD_Q_draws": PD_Q_draws,
    }


# ---------------------------------------------------------------------------
# EM source extraction
# ---------------------------------------------------------------------------


def _extract_window_em_params(
    em_params_source,
    *,
    train_start: Any,
    train_end: Any,
    gvkey: Optional[str] = None,
) -> Dict[str, float]:
    """
    Accept either:
      - callable(train_start=..., train_end=..., gvkey=...) -> dict-like
      - dict keyed by (train_start, train_end), train_end timestamp, or str(date)
      - DataFrame with alpha,beta1/delta/beta0 or alpha,beta/delta/mu and either
        train_start/train_end columns or a train_end/date column, optionally gvkey
    """
    if callable(em_params_source):
        out = em_params_source(train_start=train_start, train_end=train_end, gvkey=gvkey)
        return _coerce_param_dict(out)

    if isinstance(em_params_source, dict):
        keys = [
            (pd.Timestamp(train_start), pd.Timestamp(train_end)),
            pd.Timestamp(train_end),
            str(pd.Timestamp(train_end).date()),
        ]
        for key in keys:
            if key in em_params_source:
                return _coerce_param_dict(em_params_source[key])
        raise KeyError(f"No EM params found for window ending {pd.Timestamp(train_end)}")

    if isinstance(em_params_source, pd.DataFrame):
        df = em_params_source.copy()
        for c in [c for c in ["train_start", "train_end", "date"] if c in df.columns]:
            df[c] = pd.to_datetime(df[c])

        if gvkey is not None and "gvkey" in df.columns:
            df = df[df["gvkey"].astype(str) == str(gvkey)].copy()

        if {"train_start", "train_end"}.issubset(df.columns):
            df = df[
                (df["train_start"] == pd.Timestamp(train_start))
                & (df["train_end"] == pd.Timestamp(train_end))
            ].copy()
        elif "train_end" in df.columns:
            df = df[df["train_end"] == pd.Timestamp(train_end)].copy()
        elif "date" in df.columns:
            df = df[df["date"] == pd.Timestamp(train_end)].copy()
        else:
            raise ValueError("EM params DataFrame must have train_end or date column.")

        if df.empty:
            raise KeyError(f"No EM params found for window ending {pd.Timestamp(train_end)}")
        return _coerce_param_dict(df.iloc[-1])

    raise TypeError("Unsupported em_params_source type.")


# ---------------------------------------------------------------------------
# High-level runners
# ---------------------------------------------------------------------------


def run_bayesian_nig_window_for_firm(
    df_firm: pd.DataFrame,
    *,
    train_start: Any,
    train_end: Any,
    oos_start: Any,
    oos_end: Any,
    em_params: Dict[str, float] | pd.Series,
    gvkey_col: str = "gvkey",
    input_frequency: str = "daily",
    week_freq: str = "W-FRI",
    date_col: str = "date",
    equity_col: str = "market_cap",
    debt_col: str = "debt_face",
    rf_col: str = "rf",
    ann_factor: float = 52.0,
    forecast_horizon_years: float = 1.0,
    pd_horizon_years: float = 1.0,
    max_iter: int = 2000,
    burn_in: int = 500,
    thin: int = 5,
    prior_b0: Optional[np.ndarray] = None,
    prior_B0: Optional[np.ndarray] = None,
    prior_hyper: Optional[Dict[str, float]] = None,
    default_B0_diag: Tuple[float, float] = (50.0, 50.0),
    phi_prior_variance: float = 100.0,
    omega: float = 1e-3,
    discounting: str = "continuous",
    rng_seed: Optional[int] = None,
    window_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full Bayesian NIG estimation + OOS posterior forecasting for one firm and
    one rolling window.
    """
    if input_frequency not in {"daily", "weekly"}:
        raise ValueError("input_frequency must be 'daily' or 'weekly'.")

    if input_frequency == "daily":
        df_weekly = daily_to_weekly_nig_panel(
            df_firm,
            date_col=date_col,
            equity_col=equity_col,
            debt_col=debt_col,
            rf_col=rf_col,
            gvkey_col=gvkey_col,
            week_freq=week_freq,
        )
    else:
        df_weekly = df_firm.copy()

    df0 = _clean_weekly_panel(
        df_weekly,
        date_col=date_col,
        equity_col=equity_col,
        debt_col=debt_col,
        rf_col=rf_col,
    )

    gvkey_val = None
    if gvkey_col in df_weekly.columns and len(df_weekly) > 0:
        vals = df_weekly[gvkey_col].dropna().astype(str).unique().tolist()
        gvkey_val = vals[0] if vals else None

    train_df = _slice_window(df0, date_col=date_col, start_date=train_start, end_date=train_end)
    oos_df = _slice_window(df0, date_col=date_col, start_date=oos_start, end_date=oos_end)

    meta = {
        "gvkey": gvkey_val,
        "train_start_req": _as_ts(train_start),
        "train_end_req": _as_ts(train_end),
        "oos_start_req": _as_ts(oos_start),
        "oos_end_req": _as_ts(oos_end),
        "train_start_used_weekly": train_df[date_col].min() if len(train_df) else pd.NaT,
        "train_end_used_weekly": train_df[date_col].max() if len(train_df) else pd.NaT,
        "oos_start_used_weekly": oos_df[date_col].min() if len(oos_df) else pd.NaT,
        "oos_end_used_weekly": oos_df[date_col].max() if len(oos_df) else pd.NaT,
    }

    if len(train_df) < 5:
        return {
            **meta,
            "ok": False,
            "msg": "training_window_too_short",
            "params_df": pd.DataFrame(),
            "oos_df": pd.DataFrame(),
            "result": None,
        }

    if len(oos_df) == 0:
        return {
            **meta,
            "ok": False,
            "msg": "empty_oos_window",
            "params_df": pd.DataFrame(),
            "oos_df": pd.DataFrame(),
            "result": None,
        }

    rng = np.random.default_rng(rng_seed)

    try:
        result = gibbs_nig_training_window(
            train_df,
            em_params=em_params,
            date_col=date_col,
            equity_col=equity_col,
            debt_col=debt_col,
            rf_col=rf_col,
            ann_factor=ann_factor,
            forecast_horizon_years=forecast_horizon_years,
            max_iter=max_iter,
            burn_in=burn_in,
            thin=thin,
            prior_b0=prior_b0,
            prior_B0=prior_B0,
            prior_hyper=prior_hyper,
            default_B0_diag=default_B0_diag,
            phi_prior_variance=phi_prior_variance,
            omega=omega,
            discounting=discounting,
            rng=rng,
        )
    except Exception as exc:
        return {
            **meta,
            "ok": False,
            "msg": f"gibbs_fail:{type(exc).__name__}:{str(exc)[:200]}",
            "params_df": pd.DataFrame(),
            "oos_df": pd.DataFrame(),
            "result": None,
        }

    n_keep_actual = int(result["meta"]["n_keep_actual"])
    if n_keep_actual <= 0:
        return {
            **meta,
            "ok": False,
            "msg": "no_kept_draws",
            "params_df": pd.DataFrame(),
            "oos_df": pd.DataFrame(),
            "result": result,
        }

    oos_draws = forecast_oos_posterior_draws(
        oos_df,
        params_draws_annual=result["params_draws_annual"],
        date_col=date_col,
        equity_col=equity_col,
        debt_col=debt_col,
        rf_col=rf_col,
        pd_horizon_years=pd_horizon_years,
        inversion_tau_years=forecast_horizon_years,
        discounting=discounting,
    )
    result.update(oos_draws)

    params_df = summarize_parameter_window_posterior(
        result,
        gvkey=gvkey_val,
        train_start=train_start,
        train_end=train_end,
        window_idx=window_idx,
    )

    oos_panel = summarize_oos_posterior_panel(
        result,
        gvkey=gvkey_val,
        train_start=train_start,
        train_end=train_end,
        oos_start=oos_start,
        oos_end=oos_end,
        window_idx=window_idx,
    )

    params_draws = np.asarray(result["params_draws_annual"], dtype=float)
    em_clean = _coerce_param_dict(em_params)

    summary = {
        **meta,
        "ok": True,
        "msg": "ok",
        "em_alpha": float(em_clean["alpha"]),
        "em_beta1": float(em_clean["beta1"]),
        "em_delta": float(em_clean["delta"]),
        "em_beta0": float(em_clean["beta0"]),
        "alpha_post_mean": float(np.nanmean(params_draws[:, 0])),
        "beta1_post_mean": float(np.nanmean(params_draws[:, 1])),
        "delta_post_mean": float(np.nanmean(params_draws[:, 2])),
        "beta0_post_mean": float(np.nanmean(params_draws[:, 3])),
        "alpha_post_median": float(np.nanmedian(params_draws[:, 0])),
        "beta1_post_median": float(np.nanmedian(params_draws[:, 1])),
        "delta_post_median": float(np.nanmedian(params_draws[:, 2])),
        "beta0_post_median": float(np.nanmedian(params_draws[:, 3])),
        "n_keep_requested": int(result["meta"]["n_keep_requested"]),
        "n_keep_actual": int(result["meta"]["n_keep_actual"]),
        "n_reject": int(result["meta"]["n_reject"]),
        "n_fail_inversion": int(result["meta"]["n_fail_inversion"]),
        "n_fail_z": int(result["meta"]["n_fail_z"]),
        "n_fail_candidate": int(result["meta"]["n_fail_candidate"]),
    }

    return {
        **summary,
        "params_df": params_df,
        "oos_df": oos_panel,
        "result": result,
    }


def process_one_firm_bayesian_nig(
    df_firm: pd.DataFrame,
    window_plan_df: pd.DataFrame,
    *,
    em_params_source,
    train_start_col: str = "train_start",
    train_end_col: str = "train_end",
    oos_start_col: str = "oos_start",
    oos_end_col: str = "oos_end",
    gvkey_col: str = "gvkey",
    input_frequency: str = "daily",
    week_freq: str = "W-FRI",
    date_col: str = "date",
    equity_col: str = "market_cap",
    debt_col: str = "debt_face",
    rf_col: str = "rf",
    ann_factor: float = 52.0,
    forecast_horizon_years: float = 1.0,
    pd_horizon_years: float = 1.0,
    max_iter: int = 2000,
    burn_in: int = 500,
    thin: int = 5,
    prior_b0: Optional[np.ndarray] = None,
    prior_B0: Optional[np.ndarray] = None,
    prior_hyper: Optional[Dict[str, float]] = None,
    default_B0_diag: Tuple[float, float] = (50.0, 50.0),
    phi_prior_variance: float = 100.0,
    omega: float = 1e-3,
    discounting: str = "continuous",
    rng_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """
    Returns
    -------
    params_df_all, oos_df_all, posterior_results
    """
    _require_columns(window_plan_df, [train_start_col, train_end_col, oos_start_col, oos_end_col])

    gvkey_val = None
    if gvkey_col in df_firm.columns and len(df_firm) > 0:
        vals = df_firm[gvkey_col].dropna().astype(str).unique().tolist()
        gvkey_val = vals[0] if vals else None

    base_rng = np.random.default_rng(rng_seed)
    params_parts: List[pd.DataFrame] = []
    oos_parts: List[pd.DataFrame] = []
    posterior_results: List[Dict[str, Any]] = []

    param_cols = [
        "gvkey", "train_start", "train_end", "n_obs", "n_weeks", "window_idx",
        "alpha_lo", "alpha_med", "alpha_hi", "alpha_mean",
        "beta1_lo", "beta1_med", "beta1_hi", "beta1_mean",
        "delta_lo", "delta_med", "delta_hi", "delta_mean",
        "beta0_lo", "beta0_med", "beta0_hi", "beta0_mean",
        "n_keep_requested", "n_keep_actual", "n_rejects",
        "n_fail_z", "n_fail_candidate", "ok", "msg",
    ]

    for local_idx, row in window_plan_df.reset_index(drop=True).iterrows():
        train_start = row[train_start_col]
        train_end = row[train_end_col]
        oos_start = row[oos_start_col]
        oos_end = row[oos_end_col]

        if "window_idx" in row.index and pd.notna(row["window_idx"]):
            window_idx = int(row["window_idx"])
        else:
            window_idx = int(local_idx)

        try:
            em_params = _extract_window_em_params(
                em_params_source,
                train_start=train_start,
                train_end=train_end,
                gvkey=gvkey_val,
            )
        except Exception as exc:
            fail = {c: np.nan for c in param_cols}
            fail.update({
                "gvkey": gvkey_val,
                "train_start": pd.Timestamp(train_start),
                "train_end": pd.Timestamp(train_end),
                "window_idx": window_idx,
                "ok": False,
                "msg": f"missing_em_params:{type(exc).__name__}:{str(exc)[:180]}",
                "n_rejects": 0,
            })
            params_parts.append(pd.DataFrame([fail]))
            continue

        out = run_bayesian_nig_window_for_firm(
            df_firm,
            train_start=train_start,
            train_end=train_end,
            oos_start=oos_start,
            oos_end=oos_end,
            em_params=em_params,
            gvkey_col=gvkey_col,
            input_frequency=input_frequency,
            week_freq=week_freq,
            date_col=date_col,
            equity_col=equity_col,
            debt_col=debt_col,
            rf_col=rf_col,
            ann_factor=ann_factor,
            forecast_horizon_years=forecast_horizon_years,
            pd_horizon_years=pd_horizon_years,
            max_iter=max_iter,
            burn_in=burn_in,
            thin=thin,
            prior_b0=prior_b0,
            prior_B0=prior_B0,
            prior_hyper=prior_hyper,
            default_B0_diag=default_B0_diag,
            phi_prior_variance=phi_prior_variance,
            omega=omega,
            discounting=discounting,
            rng_seed=int(base_rng.integers(0, 2**32 - 1)),
            window_idx=window_idx,
        )

        if out.get("params_df") is not None and not out["params_df"].empty:
            params_parts.append(out["params_df"].copy())
        else:
            fail = {c: np.nan for c in param_cols}
            fail.update({
                "gvkey": gvkey_val,
                "train_start": pd.Timestamp(train_start),
                "train_end": pd.Timestamp(train_end),
                "window_idx": window_idx,
                "ok": False,
                "msg": out.get("msg", "window_fail"),
                "n_rejects": 0,
            })
            params_parts.append(pd.DataFrame([fail]))

        if out.get("oos_df") is not None and not out["oos_df"].empty:
            oos_parts.append(out["oos_df"].copy())

        if out.get("result") is not None and bool(out.get("ok", False)):
            posterior_results.append(
                {
                    "gvkey": gvkey_val,
                    "window_idx": window_idx,
                    "train_start": pd.Timestamp(train_start),
                    "train_end": pd.Timestamp(train_end),
                    "oos_start": pd.Timestamp(oos_start),
                    "oos_end": pd.Timestamp(oos_end),
                    "result": out["result"],
                }
            )

    params_df_all = (
        pd.concat(params_parts, ignore_index=True)
        .sort_values(["window_idx", "train_end"], na_position="last")
        .reset_index(drop=True)
        if params_parts else pd.DataFrame(columns=param_cols)
    )

    oos_df_all = (
        pd.concat(oos_parts, ignore_index=True)
        .sort_values(["window_idx", "date"], na_position="last")
        .reset_index(drop=True)
        if oos_parts else pd.DataFrame()
    )

    return params_df_all, oos_df_all, posterior_results


def _newey_west_long_run_variance(x: np.ndarray, max_lag: int | None = None) -> float:
    """
    Estimate long-run variance S(0) using a Bartlett-kernel / Newey-West estimator.

    Parameters
    ----------
    x : array-like
        1D series.
    max_lag : int or None
        Truncation lag. If None, uses a rule-of-thumb bandwidth.

    Returns
    -------
    float
        Estimated long-run variance (spectral density at zero).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size

    if n < 2:
        return np.nan

    x = x - x.mean()

    # Rule-of-thumb bandwidth often used in HAC settings
    if max_lag is None:
        max_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))

    max_lag = max(1, min(max_lag, n - 1))

    gamma0 = np.dot(x, x) / n
    lrv = gamma0

    for k in range(1, max_lag + 1):
        weight = 1.0 - k / (max_lag + 1.0)  # Bartlett kernel
        gamma_k = np.dot(x[k:], x[:-k]) / n
        lrv += 2.0 * weight * gamma_k

    # Numerical safeguard
    return float(max(lrv, 1e-16))


def geweke_z(
    x: np.ndarray,
    first: float = 0.1,
    last: float = 0.5,
    max_lag_a: int | None = None,
    max_lag_b: int | None = None,
) -> float:
    """
    Autocorrelation-adjusted Geweke z-score.

    Compares the mean of the first `first` fraction of the chain with the
    mean of the last `last` fraction, using HAC/Newey-West long-run variance
    estimates for both segments.

    Parameters
    ----------
    x : np.ndarray
        1D MCMC chain.
    first : float
        Fraction of the chain used from the beginning.
    last : float
        Fraction of the chain used from the end.
    max_lag_a, max_lag_b : int or None
        Optional HAC truncation lags for the first and last segments.

    Returns
    -------
    float
        Geweke z-score. Large absolute values indicate nonstationarity or
        slow drift between early and late parts of the chain.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size

    if n < 30:
        return np.nan

    if not (0 < first < 1) or not (0 < last < 1) or (first + last >= 1):
        raise ValueError("Require 0 < first < 1, 0 < last < 1, and first + last < 1.")

    n_a = max(5, int(np.floor(first * n)))
    n_b = max(5, int(np.floor(last * n)))

    a = x[:n_a]
    b = x[-n_b:]

    s0_a = _newey_west_long_run_variance(a, max_lag=max_lag_a)
    s0_b = _newey_west_long_run_variance(b, max_lag=max_lag_b)

    denom = np.sqrt(s0_a / n_a + s0_b / n_b)
    if not np.isfinite(denom) or denom <= 0:
        return np.nan

    return float((a.mean() - b.mean()) / denom)


def summarize_chain_diagnostics(params_draws_annual: np.ndarray) -> pd.DataFrame:
    cols = ["alpha", "beta1", "delta", "beta0"]
    arr = np.asarray(params_draws_annual, dtype=float)

    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("params_draws_annual must be shape (n_draws, 4).")

    rows = []
    for j, name in enumerate(cols):
        s = arr[:, j]
        s = s[np.isfinite(s)]

        if s.size >= 3:
            lag1 = float(np.corrcoef(s[:-1], s[1:])[0, 1])
        else:
            lag1 = np.nan

        rows.append({
            "param": name,
            "n_draws": int(s.size),
            "mean": float(np.nanmean(s)) if s.size else np.nan,
            "sd": float(np.nanstd(s, ddof=1)) if s.size >= 2 else np.nan,
            "geweke_z": float(geweke_z(s)),
            "lag1_acf": lag1,
        })

    return pd.DataFrame(rows)


def build_mcmc_diagnostics_df(
    posterior_results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Build a stacked diagnostics table across all firm-window posterior objects.

    Parameters
    ----------
    posterior_results : list of dict
        Output list returned by `process_one_firm_bayesian_nig(...)`
        or the stacked `posterior_all` list built in the notebook.

    Returns
    -------
    pd.DataFrame
        Columns:
            gvkey, window_idx, param, ac_1, geweke_z
    """
    rows: List[Dict[str, Any]] = []

    for item in posterior_results:
        if not isinstance(item, dict):
            continue

        gvkey = item.get("gvkey", None)
        window_idx = item.get("window_idx", None)
        result = item.get("result", None)

        if result is None or "params_draws_annual" not in result:
            continue

        arr = np.asarray(result["params_draws_annual"], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 4 or arr.shape[0] == 0:
            continue

        diag = summarize_chain_diagnostics(arr)

        if diag.empty:
            continue

        diag = diag.rename(columns={"lag1_acf": "ac_1"}).copy()
        diag["gvkey"] = str(gvkey) if gvkey is not None else np.nan
        diag["window_idx"] = int(window_idx) if pd.notna(window_idx) else np.nan

        rows.extend(
            diag.loc[:, ["gvkey", "window_idx", "param", "ac_1", "geweke_z"]]
            .to_dict(orient="records")
        )

    if not rows:
        return pd.DataFrame(columns=["gvkey", "window_idx", "param", "ac_1", "geweke_z"])

    out = pd.DataFrame(rows)
    out = out.sort_values(["gvkey", "window_idx", "param"], na_position="last").reset_index(drop=True)
    return out


__all__ = [
    "sample_gig",
    "sample_latent_z",
    "sample_regression_block",
    "sample_ig2_block",
    "build_default_priors_from_em",
    "gibbs_nig_training_window",
    "forecast_oos_posterior_draws",
    "summarize_training_posterior_panel",
    "summarize_oos_posterior_panel",
    "summarize_parameter_window_posterior",
    "run_bayesian_nig_window_for_firm",
    "process_one_firm_bayesian_nig",
    "geweke_z",
    "summarize_chain_diagnostics",
    "build_mcmc_diagnostics_df",
]