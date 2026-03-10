import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from scipy.stats import geninvgauss, gamma, multivariate_normal

from pd_estim_A.models.nig.nig_apath import (
    NIGParams,
    invert_asset_one_date,
    solve_esscher_theta,
    build_weekly_calendar_from_panel,
)


# ============================================================
# Helpers
# ============================================================

def mu_phi_from_params(params: Dict[str, float], h_step: float) -> Tuple[float, float]:
    """
    Map per-unit NIG params -> (mu_h, phi_h) for the IG mixing law at step length h_step.

    Parameters in `params` are assumed to be in a fixed time unit (here: annual).
    The returned (mu_h, phi_h) correspond to one step of length h_step in that same unit.

    Uses:
      gamma = sqrt(alpha^2 - beta^2)
      delta_h = delta * h_step
      mu_h  = delta_h / gamma
      phi_h = delta_h * gamma
    """
    alpha = float(params.get("alpha", 0.0))
    beta = float(params.get("beta", 0.0))
    delta = float(params.get("delta", 0.0))

    if delta <= 0.0 or alpha <= 0.0 or abs(beta) >= alpha:
        raise ValueError("Need delta>0, alpha>0, and |beta|<alpha.")

    gamma_val = float(np.sqrt(max(alpha * alpha - beta * beta, 0.0)))
    delta_h = float(delta) * float(h_step)

    mu_h = delta_h / gamma_val
    phi_h = delta_h * gamma_val
    return float(mu_h), float(phi_h)


def params_from_mu_phi(
    mu_h: float,
    phi_h: float,
    mu0_h: float,
    beta: float,
    h_step: float,
) -> Dict[str, float]:
    """
    Map (mu_h, phi_h, mu0_h, beta) at step length h_step back to per-unit params.

      delta_h = sqrt(mu_h * phi_h)
      gamma   = sqrt(phi_h / mu_h)
      alpha   = sqrt(beta^2 + gamma^2)
      delta   = delta_h / h_step
      mu0     = mu0_h / h_step

    If h_step = 1/52, then the recovered params are annual when mu_h/phi_h/mu0_h
    correspond to one weekly increment.
    """
    mu_h = float(mu_h)
    phi_h = float(phi_h)
    if mu_h <= 0.0 or phi_h <= 0.0:
        raise ValueError("mu_h and phi_h must be positive.")

    delta_h = float(np.sqrt(mu_h * phi_h))
    gamma_val = float(np.sqrt(phi_h / mu_h))
    alpha = float(np.sqrt(beta * beta + gamma_val * gamma_val))

    delta = delta_h / float(h_step)
    mu0 = float(mu0_h) / float(h_step)

    return {"alpha": alpha, "beta": beta, "delta": delta, "mu": mu0}


def sample_gig(lam: float, chi: float, psi: float, rng: np.random.Generator) -> float:
    if chi <= 0.0 or psi <= 0.0:
        raise ValueError("chi and psi must be positive for the GIG distribution")
    b = np.sqrt(chi * psi)
    y = geninvgauss.rvs(lam, b, random_state=rng)
    return float(y * np.sqrt(chi / psi))


def sample_z_posterior(
    returns: np.ndarray,
    params: Dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    z_t | r_t for the NIG normal-IG mixture.

    `params` must be STEP parameters for one return increment:
      alpha, delta_step, mu_step
    """
    alpha = float(params["alpha"])
    delta = float(params["delta"])
    mu0 = float(params["mu"])
    alpha2 = alpha * alpha

    z = np.empty_like(returns, dtype=float)
    for i, rt in enumerate(returns):
        q_rt = 1.0 + ((rt - mu0) / delta) ** 2
        chi = delta * delta * q_rt
        psi = alpha2
        z[i] = sample_gig(lam=-1.0, chi=chi, psi=psi, rng=rng)
    return z


def sample_mu_phi(
    z: np.ndarray,
    phi_current: float,
    hyper: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Sample (mu_h, phi_h) for the IG mixing distribution, given z.
    """
    Tn = len(z)
    if Tn <= 1:
        raise ValueError("At least two observations are required to sample mu and phi")

    z = np.asarray(z, dtype=float)
    zbar = float(np.mean(z))
    zbar_r = float(np.mean(1.0 / np.maximum(z, 1e-12)))

    xi = float(hyper.get("xi", 0.0))
    chi_hyp = float(hyper.get("chi", 0.0))
    eta = float(hyper.get("eta", 1.0))
    omega = float(hyper.get("omega", 1.0))

    u1 = Tn * zbar + omega * eta
    u2 = Tn + omega - chi_hyp
    u3 = Tn * zbar_r + omega / max(eta, 1e-12)
    v = Tn + 2.0 * xi

    # mu_h
    lam_mu = (Tn - 1.0) / 2.0
    a_mu2 = float(phi_current * u1)
    b_mu2 = float(phi_current * u3)
    mu_new = sample_gig(lam_mu, a_mu2, b_mu2, rng)

    # phi_h
    shape_phi = (v + 1.0) / 2.0
    rate_phi = u1 / (2.0 * mu_new) - u2 + (u3 * mu_new) / 2.0
    rate_phi = max(float(rate_phi), 1e-12)
    phi_new = float(gamma.rvs(shape_phi, scale=1.0 / rate_phi, random_state=rng))

    return mu_new, phi_new


def sample_beta_mu0(
    returns: np.ndarray,
    z: np.ndarray,
    b0: np.ndarray,
    B0: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Sample (mu0_step, beta) in:
      r_t = mu0_step + beta * z_t + eps_t,  eps_t ~ N(0, z_t)
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    if r.shape != z.shape:
        raise ValueError("returns and z must have the same length")

    T = r.size
    X = np.column_stack([np.ones(T), z])
    w = 1.0 / np.maximum(z, 1e-12)

    B0 = np.asarray(B0, dtype=float)
    b0 = np.asarray(b0, dtype=float).reshape(2, 1)

    B0_inv = np.linalg.inv(B0)
    XtW = X.T * w

    B_new_inv = XtW @ X + B0_inv
    B_new = np.linalg.inv(B_new_inv)

    b_vec = (XtW @ r.reshape(-1, 1)) + (B0_inv @ b0)
    b_mean = (B_new @ b_vec).reshape(2)

    draw = multivariate_normal.rvs(mean=b_mean, cov=B_new, random_state=rng)
    mu0_step = float(draw[0])
    beta = float(draw[1])
    return mu0_step, beta


# ============================================================
# Rates + theta + asset inversion
# ============================================================

def annual_cc_rate_to_weekly(r_annual: np.ndarray, weeks_per_year: int = 52) -> np.ndarray:
    """
    Convert annual continuously-compounded rate to weekly continuously-compounded rate.
    Only used for diagnostics / convenience columns. The Gibbs inversion below uses
    annual rates together with tau_years.
    """
    return np.asarray(r_annual, dtype=float) / float(weeks_per_year)


def theta_series_weekly(p: NIGParams, r_week: np.ndarray, tau_weeks: float) -> np.ndarray:
    """
    Legacy name kept for compatibility.

    In this project's working weekly-NIG convention:
      - p contains weekly-step NIG parameters
      - r_week is the rate series passed by the panel/inversion workflow as-is
      - tau_weeks should be 1.0 in the weekly inversion workflow
    """
    out = np.full_like(r_week, np.nan, dtype=float)
    for i, rw in enumerate(r_week):
        if not np.isfinite(rw):
            continue
        try:
            out[i] = float(solve_esscher_theta(p, float(rw), float(tau_weeks)))
        except Exception:
            out[i] = np.nan
    return out


def invert_assets_on_dates(
    E: np.ndarray,
    L: np.ndarray,
    r_week: np.ndarray,
    dates: np.ndarray,
    p: NIGParams,
    *,
    tau_weeks: float,
    U: float,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Invert A_t for each observation date (already weekly), returning:
      A_path (len n_obs), theta_path (len n_obs)

    Legacy argument names are kept for compatibility with the rest of the file.
    In the working weekly-NIG convention used here:
      - p contains weekly-step parameters
      - r_week is the panel rate series as passed to invert_asset_one_date
      - tau_weeks should be 1.0
    """
    A = np.full_like(E, np.nan, dtype=float)
    th = np.full_like(E, np.nan, dtype=float)

    A_prev = None
    for i in range(len(E)):
        if not (np.isfinite(E[i]) and np.isfinite(L[i]) and np.isfinite(r_week[i])):
            continue
        A[i], th[i] = invert_asset_one_date(
            float(E[i]),
            float(L[i]),
            float(r_week[i]),
            float(tau_weeks),
            p,
            A_prev=A_prev,
            U=U,
            n=n,
        )
        A_prev = A[i]
    return A, th


# ============================================================
# Prior helpers
# ============================================================

def _coerce_em_params(em_params: Dict[str, float] | pd.Series) -> Dict[str, float]:
    if isinstance(em_params, pd.Series):
        em_params = em_params.to_dict()

    out = {
        "alpha": float(em_params["alpha"]),
        "beta": float(em_params["beta"]),
        "delta": float(em_params["delta"]),
        "mu": float(em_params["mu"]),
    }

    if (
        not np.isfinite(out["alpha"])
        or not np.isfinite(out["beta"])
        or not np.isfinite(out["delta"])
        or not np.isfinite(out["mu"])
    ):
        raise ValueError("EM parameters must be finite.")
    if out["delta"] <= 0.0 or out["alpha"] <= 0.0 or abs(out["beta"]) >= out["alpha"]:
        raise ValueError("EM parameters must satisfy delta>0, alpha>0, and |beta|<alpha.")

    return out


def _build_default_priors_from_em(
    em_params: Dict[str, float],
    *,
    h_step: float,
    B0_diag: Tuple[float, float] = (50.0, 50.0),
    var_phi: float = 100.0,
    omega: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Default weakly-informative priors centered on the EM point estimate.

    IMPORTANT:
    - em_params are assumed to be in ANNUAL units
    - b0 prior is built for (mu0_step, beta), so mu is converted to mu * h_step
    - hyper is built for (mu_h, phi_h) at the same step length h_step
    """
    em = _coerce_em_params(em_params)
    mu_h, phi_h = mu_phi_from_params(em, h_step=h_step)

    b0_prior = np.array([em["mu"] * h_step, em["beta"]], dtype=float)
    B0_prior = np.diag(np.asarray(B0_diag, dtype=float))

    hyper = {
        "xi": float((phi_h * phi_h) / var_phi),
        "chi": float(phi_h / var_phi),
        "eta": float(mu_h),
        "omega": float(omega),
    }
    return b0_prior, B0_prior, hyper


def _prepare_prior_inputs(
    em_params: Dict[str, float],
    *,
    prior_b0: Optional[np.ndarray] = None,
    prior_B0: Optional[np.ndarray] = None,
    prior_hyper: Optional[Dict[str, float]] = None,
    h_step: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Prepare Gibbs priors.

    Convention:
    - em_params are annual
    - if prior_b0 is passed, it is interpreted as [mu_annual, beta]
      and converted internally to [mu_step, beta]
    - prior_hyper, if passed, is assumed already expressed on the Gibbs step scale
    """
    b0_default, B0_default, hyper_default = _build_default_priors_from_em(
        em_params,
        h_step=h_step,
    )

    if prior_b0 is None:
        b0 = b0_default
    else:
        prior_b0 = np.asarray(prior_b0, dtype=float).reshape(2)
        b0 = np.array([float(prior_b0[0]) * h_step, float(prior_b0[1])], dtype=float)

    B0 = B0_default if prior_B0 is None else np.asarray(prior_B0, dtype=float)
    hyper = hyper_default if prior_hyper is None else {k: float(v) for k, v in prior_hyper.items()}

    if B0.shape != (2, 2):
        raise ValueError("prior_B0 must be 2x2.")
    if np.linalg.det(B0) == 0:
        raise ValueError("prior_B0 must be invertible.")

    required_hyper = {"xi", "chi", "eta", "omega"}
    missing = required_hyper - set(hyper.keys())
    if missing:
        raise ValueError(f"prior_hyper missing keys: {sorted(missing)}")

    return b0.astype(float), B0.astype(float), hyper


# ============================================================
# Posterior summarization
# ============================================================

def _posterior_summary(x: np.ndarray) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float)
    return {
        "mean": np.nanmean(x, axis=0),
        "median": np.nanmedian(x, axis=0),
        "q05": np.nanquantile(x, 0.05, axis=0),
        "q95": np.nanquantile(x, 0.95, axis=0),
    }


def summarize_gibbs_window(
    result: Dict[str, Any],
    weekly_input_df: pd.DataFrame,
    *,
    train_start,
    train_end,
    gvkey: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a weekly posterior-summary panel for one training window.
    """
    df = weekly_input_df.copy().sort_values("date").reset_index(drop=True)

    A_draws = np.asarray(result["A_draws"], dtype=float)
    theta_draws = np.asarray(result["theta_draws"], dtype=float)

    A_sum = _posterior_summary(A_draws)
    th_sum = _posterior_summary(theta_draws)

    base_cols = [c for c in ["date", "E", "L", "r", "r_week"] if c in df.columns]
    out = df[base_cols].copy()

    out["A_mean"] = A_sum["mean"]
    out["A_median"] = A_sum["median"]
    out["A_q05"] = A_sum["q05"]
    out["A_q95"] = A_sum["q95"]

    out["theta_mean"] = th_sum["mean"]
    out["theta_median"] = th_sum["median"]
    out["theta_q05"] = th_sum["q05"]
    out["theta_q95"] = th_sum["q95"]

    out["logA_median"] = np.log(np.maximum(out["A_median"].to_numpy(dtype=float), 1e-300))
    out["dlogA_median"] = pd.Series(out["logA_median"]).diff().to_numpy(dtype=float)

    # use ANNUAL params for the stored summaries
    params_annual = np.asarray(result["params_annual"], dtype=float)
    out["alpha_mean"] = float(np.nanmean(params_annual[:, 0]))
    out["beta_mean"] = float(np.nanmean(params_annual[:, 1]))
    out["delta_mean"] = float(np.nanmean(params_annual[:, 2]))
    out["mu_mean"] = float(np.nanmean(params_annual[:, 3]))

    out["alpha_median"] = float(np.nanmedian(params_annual[:, 0]))
    out["beta_median"] = float(np.nanmedian(params_annual[:, 1]))
    out["delta_median"] = float(np.nanmedian(params_annual[:, 2]))
    out["mu_median"] = float(np.nanmedian(params_annual[:, 3]))

    out["train_start_req"] = pd.Timestamp(train_start)
    out["train_end_req"] = pd.Timestamp(train_end)
    if gvkey is not None:
        out["gvkey"] = str(gvkey)

    return out


# ============================================================
# Main Gibbs sampler
# ============================================================

def gibbs_sampler_weekly(
    E_series: np.ndarray,
    L_series: np.ndarray,
    rf_series_annual_cc: np.ndarray,
    dates: np.ndarray,
    start_date=None,
    end_date=None,
    max_iter: int = 2000,
    *,
    em_params: Dict[str, float],
    burn_in: int = 500,
    thin: int = 5,
    weeks_per_year: int = 52,
    tau_years: float = 1.0,
    U: float = 120.0,
    n_int: int = 2000,
    prior_b0: Optional[np.ndarray] = None,
    prior_B0: Optional[np.ndarray] = None,
    prior_hyper: Optional[Dict[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:

    if rng is None:
        rng = np.random.default_rng()

    E_series = np.asarray(E_series, dtype=float)
    L_series = np.asarray(L_series, dtype=float)
    rf_series_annual_cc = np.asarray(rf_series_annual_cc, dtype=float)
    dates = pd.to_datetime(np.asarray(dates))

    if not (E_series.shape == L_series.shape == rf_series_annual_cc.shape == dates.shape):
        raise ValueError("E_series, L_series, rf_series, dates must have same shape")
    if burn_in < 0 or burn_in >= max_iter:
        raise ValueError("burn_in must be in [0, max_iter-1]")
    if thin <= 0:
        raise ValueError("thin must be >= 1")

    mask = np.ones(dates.size, dtype=bool)
    if start_date is not None:
        mask &= (dates >= pd.Timestamp(start_date))
    if end_date is not None:
        mask &= (dates <= pd.Timestamp(end_date))

    idx = np.where(mask)[0]
    if idx.size < 3:
        raise ValueError("Need at least 3 observations in the training window")

    Ew = E_series[mask]
    Lw = L_series[mask]
    rfw_annual = rf_series_annual_cc[mask]
    dates_w = dates[mask]

    # WORKING convention for this project:
    # - EM params are weekly-step
    # - latent Gibbs updates are weekly-step
    # - pricing/inversion uses weeklyized cc rate
    # - tau = 1.0 (not 52.0)
    rw_week = annual_cc_rate_to_weekly(rfw_annual, weeks_per_year=weeks_per_year)
    tau_input = float(tau_years)   # should be 1.0 in your workflow

    n_obs = int(idx.size)

    em = _coerce_em_params(em_params)
    alpha = float(em["alpha"])
    beta  = float(em["beta"])
    delta = float(em["delta"])
    mu0   = float(em["mu"])

    # weekly-step convention
    h_step = 1.0
    mu_h, phi_h = mu_phi_from_params(em, h_step=h_step)

    b0_prior, B0_prior, hyper = _prepare_prior_inputs(
        em,
        prior_b0=prior_b0,
        prior_B0=prior_B0,
        prior_hyper=prior_hyper,
        h_step=h_step,
    )

    keep_mask = np.zeros(max_iter, dtype=bool)
    for it in range(max_iter):
        if it >= burn_in and ((it - burn_in) % thin == 0):
            keep_mask[it] = True
    n_keep = int(keep_mask.sum())

    A_draws = np.full((n_keep, n_obs), np.nan, dtype=float)
    theta_draws = np.full((n_keep, n_obs), np.nan, dtype=float)
    z_draws = np.full((n_keep, n_obs - 1), np.nan, dtype=float)
    params_weekly = np.full((n_keep, 4), np.nan, dtype=float)
    params_annual = np.full((n_keep, 4), np.nan, dtype=float)
    mu_phi_draws = np.full((n_keep, 2), np.nan, dtype=float)

    n_reject = 0
    k = 0

    for it in range(max_iter):
        p_cur = NIGParams(alpha=alpha, beta=beta, delta=delta, mu=mu0)

        try:
            A_path, theta_path = invert_assets_on_dates(
                Ew,
                Lw,
                rw_week,
                dates_w,
                p_cur,
                tau_weeks=tau_input,
                U=U,
                n=n_int,
            )
        except Exception:
            n_reject += 1
            continue

        if not np.all(np.isfinite(A_path)):
            n_reject += 1
            continue

        rA = np.diff(np.log(A_path))
        if not np.all(np.isfinite(rA)):
            n_reject += 1
            continue

        # store current state BEFORE updating
        if keep_mask[it]:
            A_draws[k, :] = A_path
            theta_draws[k, :] = theta_path
            params_weekly[k, :] = np.array([alpha, beta, delta, mu0], dtype=float)
            params_annual[k, :] = np.array(
                [alpha, beta, delta * weeks_per_year, mu0 * weeks_per_year],
                dtype=float,
            )
            mu_phi_draws[k, :] = np.array([mu_h, phi_h], dtype=float)

        # z | r using weekly-step params
        try:
            z = sample_z_posterior(
                rA,
                {"alpha": alpha, "delta": delta, "mu": mu0},
                rng,
            )
        except Exception:
            n_reject += 1
            if keep_mask[it]:
                k += 1
            continue

        try:
            mu_h_new, phi_h_new = sample_mu_phi(
                z,
                phi_current=phi_h,
                hyper=hyper,
                rng=rng,
            )
        except Exception:
            mu_h_new, phi_h_new = mu_h, phi_h

        try:
            mu0_new, beta_new = sample_beta_mu0(
                rA,
                z,
                b0_prior,
                B0_prior,
                rng,
            )
        except Exception:
            mu0_new, beta_new = mu0, beta

        try:
            prop = params_from_mu_phi(
                mu_h_new,
                phi_h_new,
                mu0_h=mu0_new,
                beta=beta_new,
                h_step=h_step,
            )
            alpha_prop = float(prop["alpha"])
            beta_prop = float(prop["beta"])
            delta_prop = float(prop["delta"])
            mu_prop = float(prop["mu"])
        except Exception:
            n_reject += 1
            if keep_mask[it]:
                z_draws[k, :] = z
                k += 1
            continue

        ok = True
        if not (
            np.isfinite(alpha_prop)
            and np.isfinite(beta_prop)
            and np.isfinite(delta_prop)
            and np.isfinite(mu_prop)
        ):
            ok = False
        if delta_prop <= 0.0 or alpha_prop <= 0.0 or abs(beta_prop) >= alpha_prop:
            ok = False

        # theta feasibility on the window using weeklyized rates and tau=1
        if ok:
            try:
                p_prop = NIGParams(
                    alpha=alpha_prop,
                    beta=beta_prop,
                    delta=delta_prop,
                    mu=mu_prop,
                )
                th_prop = theta_series_weekly(
                    p_prop,
                    rw_week,
                    tau_weeks=tau_input,
                )
                alpha_floor = max(
                    float(np.nanmax(np.abs(beta_prop + th_prop))),
                    float(np.nanmax(np.abs(beta_prop + th_prop + 1.0))),
                ) + 1e-6
                if not np.isfinite(alpha_floor) or alpha_prop <= alpha_floor:
                    ok = False
            except Exception:
                ok = False

        if ok:
            alpha, beta, delta, mu0 = alpha_prop, beta_prop, delta_prop, mu_prop
            mu_h, phi_h = float(mu_h_new), float(phi_h_new)
        else:
            n_reject += 1

        if keep_mask[it]:
            z_draws[k, :] = z
            k += 1

    A_draws = A_draws[:k, :]
    theta_draws = theta_draws[:k, :]
    z_draws = z_draws[:k, :]
    params_weekly = params_weekly[:k, :]
    params_annual = params_annual[:k, :]
    mu_phi_draws = mu_phi_draws[:k, :]

    weekly_input_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates_w),
            "E": Ew.astype(float),
            "L": Lw.astype(float),
            "r": rfw_annual.astype(float),
            "r_week": rw_week.astype(float),
        }
    )

    return {
        "weekly_input_df": weekly_input_df,
        "A_draws": A_draws,
        "theta_draws": theta_draws,
        "z_draws": z_draws,
        "params_weekly": params_weekly,
        "params_annual": params_annual,
        "mu_phi_draws": mu_phi_draws,
        "priors": {"hyper": hyper, "b0": b0_prior, "B0": B0_prior},
        "init_params_weekly": em,
        "meta": {
            "burn_in": burn_in,
            "thin": thin,
            "max_iter": max_iter,
            "n_keep_requested": n_keep,
            "n_keep_actual": int(k),
            "n_reject": int(n_reject),
            "weeks_per_year": int(weeks_per_year),
            "tau_weeks": float(tau_input),
            "U": float(U),
            "n_int": int(n_int),
        },
    }

# ============================================================
# EM source extraction
# ============================================================

def _extract_window_em_params(
    em_params_source,
    *,
    train_start,
    train_end,
    gvkey: Optional[str] = None,
) -> Dict[str, float]:
    """
    Accept either:
      - dict keyed by train_end or (train_start, train_end)
      - DataFrame with columns alpha,beta,delta,mu and either train_end/date,
        optionally also gvkey
      - callable(train_start=..., train_end=..., gvkey=...) -> dict-like
    """
    if callable(em_params_source):
        out = em_params_source(train_start=train_start, train_end=train_end, gvkey=gvkey)
        return _coerce_em_params(out)

    if isinstance(em_params_source, dict):
        candidates = [
            (pd.Timestamp(train_start), pd.Timestamp(train_end)),
            pd.Timestamp(train_end),
            str(pd.Timestamp(train_end).date()),
        ]
        for key in candidates:
            if key in em_params_source:
                return _coerce_em_params(em_params_source[key])
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

        return _coerce_em_params(df.iloc[-1])

    raise TypeError("Unsupported em_params_source type.")


# ============================================================
# One-window wrapper
# ============================================================

def run_nig_window_for_firm(
    g_firm_daily: pd.DataFrame,
    *,
    train_start,
    train_end,
    em_params: Dict[str, float] | pd.Series,
    date_col: str = "date",
    week_ending: str = "W-FRI",
    min_daily_rows: int = 10,
    min_weekly_obs: int | None = None,
    min_weekly_returns: int = 2,
    E_col: str = "E",
    L_col: str = "L",
    r_col: str = "r",
    tau_years: float = 1.0,
    weeks_per_year: int = 52,
    max_iter: int = 2000,
    burn_in: int = 500,
    thin: int = 5,
    U: float = 120.0,
    n_int: int = 2000,
    prior_b0: Optional[np.ndarray] = None,
    prior_B0: Optional[np.ndarray] = None,
    prior_hyper: Optional[Dict[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
):
    """
    Run the Bayesian NIG training step for ONE firm and ONE training window.
    """
    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)

    g = g_firm_daily.copy()

    if date_col not in g.columns:
        if isinstance(g.index, pd.DatetimeIndex):
            g = g.reset_index()
            if date_col not in g.columns:
                g = g.rename(columns={g.columns[0]: date_col})
        else:
            raise ValueError(
                f"Input must either contain a '{date_col}' column or have a DatetimeIndex."
            )

    g = g.loc[:, ~g.columns.duplicated()].copy()
    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")

    gvkey_val = None
    if "gvkey" in g.columns:
        vals = g["gvkey"].dropna().astype(str).unique()
        if len(vals) == 1:
            gvkey_val = vals[0]

    for c in [E_col, L_col, r_col]:
        g[c] = pd.to_numeric(g[c], errors="coerce")

    g_train = g.loc[(g[date_col] >= train_start) & (g[date_col] <= train_end)].copy()

    if g_train.empty:
        summary = {
            "gvkey": gvkey_val,
            "train_start_req": train_start,
            "train_end_req": train_end,
            "ok": False,
            "msg": "empty_training_slice",
            "n_daily_train": 0,
            "n_weekly_train": 0,
            "n_weekly_returns": 0,
            "n_keep_actual": 0,
            "n_reject": 0,
        }
        return summary, pd.DataFrame(), None

    g_train = (
        g_train.dropna(subset=[date_col, E_col, L_col, r_col])
        .query(f"{E_col} > 0 and {L_col} > 0")
        .sort_values(date_col)
        .groupby(date_col, as_index=False)
        .last()
        .reset_index(drop=True)
    )

    n_daily_train = int(len(g_train))
    if n_daily_train < int(min_daily_rows):
        summary = {
            "gvkey": gvkey_val,
            "train_start_req": train_start,
            "train_end_req": train_end,
            "train_start_used_daily": g_train[date_col].min() if n_daily_train else pd.NaT,
            "train_end_used_daily": g_train[date_col].max() if n_daily_train else pd.NaT,
            "ok": False,
            "msg": "too_few_daily_rows_train_after_clean",
            "n_daily_train": n_daily_train,
            "n_weekly_train": 0,
            "n_weekly_returns": 0,
            "n_keep_actual": 0,
            "n_reject": 0,
        }
        return summary, pd.DataFrame(), None

    weekly_dates = build_weekly_calendar_from_panel(
        g_train.rename(columns={date_col: "date"}),
        week_ending=week_ending,
    )

    g_train_idx = g_train.set_index(date_col).sort_index()
    g_train_weekly = (
        g_train_idx.reindex(weekly_dates)
        .reset_index()
        .rename(columns={"index": "date"})
    )

    n_weekly_train = int(len(g_train_weekly))
    n_weekly_returns = max(n_weekly_train - 1, 0)

    if min_weekly_obs is not None and n_weekly_train < int(min_weekly_obs):
        summary = {
            "gvkey": gvkey_val,
            "train_start_req": train_start,
            "train_end_req": train_end,
            "train_start_used_daily": pd.Timestamp(g_train[date_col].iloc[0]),
            "train_end_used_daily": pd.Timestamp(g_train[date_col].iloc[-1]),
            "train_start_used_weekly": pd.Timestamp(g_train_weekly["date"].iloc[0]) if n_weekly_train else pd.NaT,
            "train_end_used_weekly": pd.Timestamp(g_train_weekly["date"].iloc[-1]) if n_weekly_train else pd.NaT,
            "ok": False,
            "msg": f"too_few_weekly_obs<{int(min_weekly_obs)}",
            "n_daily_train": n_daily_train,
            "n_weekly_train": n_weekly_train,
            "n_weekly_returns": n_weekly_returns,
            "n_keep_actual": 0,
            "n_reject": 0,
        }
        return summary, pd.DataFrame(), None

    if n_weekly_returns < int(min_weekly_returns):
        summary = {
            "gvkey": gvkey_val,
            "train_start_req": train_start,
            "train_end_req": train_end,
            "train_start_used_daily": pd.Timestamp(g_train[date_col].iloc[0]),
            "train_end_used_daily": pd.Timestamp(g_train[date_col].iloc[-1]),
            "train_start_used_weekly": pd.Timestamp(g_train_weekly["date"].iloc[0]) if n_weekly_train else pd.NaT,
            "train_end_used_weekly": pd.Timestamp(g_train_weekly["date"].iloc[-1]) if n_weekly_train else pd.NaT,
            "ok": False,
            "msg": f"too_few_weekly_returns<{int(min_weekly_returns)}",
            "n_daily_train": n_daily_train,
            "n_weekly_train": n_weekly_train,
            "n_weekly_returns": n_weekly_returns,
            "n_keep_actual": 0,
            "n_reject": 0,
        }
        return summary, pd.DataFrame(), None

    try:
        result = gibbs_sampler_weekly(
            E_series=g_train_weekly[E_col].to_numpy(dtype=float),
            L_series=g_train_weekly[L_col].to_numpy(dtype=float),
            rf_series_annual_cc=g_train_weekly[r_col].to_numpy(dtype=float),
            dates=g_train_weekly["date"].to_numpy(),
            max_iter=max_iter,
            em_params=em_params,
            burn_in=burn_in,
            thin=thin,
            weeks_per_year=weeks_per_year,
            tau_years=tau_years,
            U=U,
            n_int=n_int,
            prior_b0=prior_b0,
            prior_B0=prior_B0,
            prior_hyper=prior_hyper,
            rng=rng,
        )
    except Exception as e:
        summary = {
            "gvkey": gvkey_val,
            "train_start_req": train_start,
            "train_end_req": train_end,
            "train_start_used_daily": pd.Timestamp(g_train[date_col].iloc[0]),
            "train_end_used_daily": pd.Timestamp(g_train[date_col].iloc[-1]),
            "train_start_used_weekly": pd.Timestamp(g_train_weekly["date"].iloc[0]),
            "train_end_used_weekly": pd.Timestamp(g_train_weekly["date"].iloc[-1]),
            "ok": False,
            "msg": f"gibbs_fail:{type(e).__name__}:{str(e)[:180]}",
            "n_daily_train": n_daily_train,
            "n_weekly_train": n_weekly_train,
            "n_weekly_returns": n_weekly_returns,
            "n_keep_actual": 0,
            "n_reject": 0,
        }
        return summary, pd.DataFrame(), None

    n_keep_actual = int(result["meta"]["n_keep_actual"])
    ok = n_keep_actual > 0
    msg = "ok" if ok else "no_kept_draws"

    weekly_post = pd.DataFrame()
    alpha_mean = np.nan
    beta_mean = np.nan
    delta_mean = np.nan
    mu_mean = np.nan
    alpha_median = np.nan
    beta_median = np.nan
    delta_median = np.nan
    mu_median = np.nan
    A_last_mean = np.nan
    A_last_median = np.nan

    if ok:
        weekly_post = summarize_gibbs_window(
            result,
            result["weekly_input_df"],
            train_start=train_start,
            train_end=train_end,
            gvkey=gvkey_val,
        )

        params_annual = np.asarray(result["params_annual"], dtype=float)
        alpha_mean = float(np.nanmean(params_annual[:, 0]))
        beta_mean = float(np.nanmean(params_annual[:, 1]))
        delta_mean = float(np.nanmean(params_annual[:, 2]))
        mu_mean = float(np.nanmean(params_annual[:, 3]))

        alpha_median = float(np.nanmedian(params_annual[:, 0]))
        beta_median = float(np.nanmedian(params_annual[:, 1]))
        delta_median = float(np.nanmedian(params_annual[:, 2]))
        mu_median = float(np.nanmedian(params_annual[:, 3]))

        A_last_mean = float(np.nanmean(result["A_draws"][:, -1]))
        A_last_median = float(np.nanmedian(result["A_draws"][:, -1]))

    em_clean = _coerce_em_params(em_params)

    summary = {
        "gvkey": gvkey_val,
        "train_start_req": train_start,
        "train_end_req": train_end,
        "train_start_used_daily": pd.Timestamp(g_train[date_col].iloc[0]),
        "train_end_used_daily": pd.Timestamp(g_train[date_col].iloc[-1]),
        "train_start_used_weekly": pd.Timestamp(g_train_weekly["date"].iloc[0]),
        "train_end_used_weekly": pd.Timestamp(g_train_weekly["date"].iloc[-1]),
        "ok": bool(ok),
        "msg": msg,
        "em_alpha": float(em_clean["alpha"]),
        "em_beta": float(em_clean["beta"]),
        "em_delta": float(em_clean["delta"]),
        "em_mu": float(em_clean["mu"]),
        "alpha_post_mean": alpha_mean,
        "beta_post_mean": beta_mean,
        "delta_post_mean": delta_mean,
        "mu_post_mean": mu_mean,
        "alpha_post_median": alpha_median,
        "beta_post_median": beta_median,
        "delta_post_median": delta_median,
        "mu_post_median": mu_median,
        "A_last_mean": A_last_mean,
        "A_last_median": A_last_median,
        "n_daily_train": n_daily_train,
        "n_weekly_train": n_weekly_train,
        "n_weekly_returns": n_weekly_returns,
        "n_keep_requested": int(result["meta"]["n_keep_requested"]),
        "n_keep_actual": int(result["meta"]["n_keep_actual"]),
        "n_reject": int(result["meta"]["n_reject"]),
    }

    return summary, weekly_post, result


# ============================================================
# One-firm rolling wrapper
# ============================================================

def process_one_firm_nig(
    g_firm_daily: pd.DataFrame,
    *,
    windows,
    em_params_source,
    gvkey: str | None = None,
    date_col: str = "date",
    week_ending: str = "W-FRI",
    min_daily_rows: int = 10,
    min_weekly_obs: int | None = None,
    min_weekly_returns: int = 2,
    E_col: str = "E",
    L_col: str = "L",
    r_col: str = "r",
    tau_years: float = 1.0,
    weeks_per_year: int = 52,
    max_iter: int = 2000,
    burn_in: int = 500,
    thin: int = 5,
    U: float = 120.0,
    n_int: int = 2000,
    prior_b0: Optional[np.ndarray] = None,
    prior_B0: Optional[np.ndarray] = None,
    prior_hyper: Optional[Dict[str, float]] = None,
    rng: Optional[np.random.Generator] = None,
):
    """
    Run the rolling Bayesian NIG training workflow for ONE firm across all windows.
    """
    g = g_firm_daily.copy()

    if date_col not in g.columns:
        if isinstance(g.index, pd.DatetimeIndex):
            g = g.reset_index()
            if date_col not in g.columns:
                g = g.rename(columns={g.columns[0]: date_col})
        else:
            raise ValueError(
                f"Input must either contain a '{date_col}' column or have a DatetimeIndex."
            )

    g = g.loc[:, ~g.columns.duplicated()].copy()
    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")

    if gvkey is None and "gvkey" in g.columns:
        vals = g["gvkey"].dropna().astype(str).unique()
        if len(vals) == 1:
            gvkey = vals[0]

    for c in [E_col, L_col, r_col]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")

    g = (
        g.dropna(subset=[date_col, E_col, L_col, r_col])
        .query(f"{E_col} > 0 and {L_col} > 0")
        .sort_values(date_col)
        .groupby(date_col, as_index=False)
        .last()
    )

    if g.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    g = g.set_index(date_col).sort_index()

    summary_rows = []
    weekly_is_parts = []
    posterior_results = []

    for w in windows:
        train_start = pd.Timestamp(w["train_start"])
        train_end = pd.Timestamp(w["train_end"])
        oos_start = pd.Timestamp(w["oos_start"]) if "oos_start" in w else pd.NaT
        oos_end = pd.Timestamp(w["oos_end"]) if "oos_end" in w else pd.NaT

        try:
            em_params = _extract_window_em_params(
                em_params_source,
                train_start=train_start,
                train_end=train_end,
                gvkey=gvkey,
            )
        except Exception as e:
            summary_rows.append(
                {
                    "gvkey": gvkey,
                    "train_start": train_start,
                    "train_end": train_end,
                    "oos_start": oos_start,
                    "oos_end": oos_end,
                    "ok": False,
                    "msg": f"missing_em_params:{type(e).__name__}:{str(e)[:160]}",
                    "n_daily_train": 0,
                    "n_weekly_train": 0,
                    "n_weekly_returns": 0,
                    "n_keep_requested": 0,
                    "n_keep_actual": 0,
                    "n_reject": 0,
                }
            )
            continue

        summary, weekly_df, result = run_nig_window_for_firm(
            g,
            train_start=train_start,
            train_end=train_end,
            em_params=em_params,
            date_col=date_col,
            week_ending=week_ending,
            min_daily_rows=min_daily_rows,
            min_weekly_obs=min_weekly_obs,
            min_weekly_returns=min_weekly_returns,
            E_col=E_col,
            L_col=L_col,
            r_col=r_col,
            tau_years=tau_years,
            weeks_per_year=weeks_per_year,
            max_iter=max_iter,
            burn_in=burn_in,
            thin=thin,
            U=U,
            n_int=n_int,
            prior_b0=prior_b0,
            prior_B0=prior_B0,
            prior_hyper=prior_hyper,
            rng=rng,
        )

        summary_rows.append(
            {
                "gvkey": gvkey,
                "train_start": train_start,
                "train_end": train_end,
                "oos_start": oos_start,
                "oos_end": oos_end,
                **summary,
            }
        )

        if weekly_df is not None and not weekly_df.empty:
            w_store = weekly_df.copy()
            w_store["gvkey"] = gvkey
            w_store["train_end"] = train_end
            weekly_is_parts.append(w_store)

        if result is not None and bool(summary.get("ok", False)):
            posterior_results.append(
                {
                    "gvkey": gvkey,
                    "train_start": train_start,
                    "train_end": train_end,
                    "oos_start": oos_start,
                    "oos_end": oos_end,
                    "result": result,
                }
            )

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["train_end", "gvkey"])
        .reset_index(drop=True)
    ) if len(summary_rows) else pd.DataFrame()

    weekly_is_df = (
        pd.concat(weekly_is_parts, ignore_index=True)
        .sort_values(["train_end", "gvkey", "date"])
        .reset_index(drop=True)
    ) if len(weekly_is_parts) else pd.DataFrame()

    return summary_df, weekly_is_df, posterior_results