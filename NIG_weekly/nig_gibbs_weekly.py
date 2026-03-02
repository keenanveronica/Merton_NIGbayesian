import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from scipy.stats import geninvgauss, gamma, multivariate_normal
from NIG_weekly.nig_apath import NIGParams, invert_asset_one_date, solve_esscher_theta


# Helpers: IG-mixture parameter mappings (your originals, but "step" not "daily")
def mu_phi_from_params(params: Dict[str, float], h_step: float) -> Tuple[float, float]:
    """
    Map 'per-unit' NIG params -> (mu_h, phi_h) for the mixing IG at step length h_step
    (in the SAME time unit as your params).

    Uses:
      gamma = sqrt(alpha^2 - beta^2)
      delta_h = delta * h_step
      mu_h  = delta_h / gamma
      phi_h = delta_h * gamma
    """
    alpha = float(params.get("alpha", 0.0))
    beta  = float(params.get("beta", 0.0))
    delta = float(params.get("delta", 0.0))

    if delta <= 0.0 or alpha <= 0.0 or abs(beta) >= alpha:
        raise ValueError("Need delta>0, alpha>0, and |beta|<alpha.")

    gamma_val = float(np.sqrt(max(alpha * alpha - beta * beta, 0.0)))
    delta_h = float(delta) * float(h_step)

    mu_h = delta_h / gamma_val
    phi_h = delta_h * gamma_val
    return float(mu_h), float(phi_h)


def params_from_mu_phi(
    mu_h: float, phi_h: float, mu0_h: float, beta: float, h_step: float
) -> Dict[str, float]:
    """
    Map (mu_h, phi_h, mu0_h, beta) at step length h_step back to per-unit params.

      delta_h = sqrt(mu_h * phi_h)
      gamma   = sqrt(phi_h / mu_h)
      alpha   = sqrt(beta^2 + gamma^2)
      delta   = delta_h / h_step
      mu0     = mu0_h / h_step
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


def sample_z_posterior(returns: np.ndarray, params: Dict[str, float], rng: np.random.Generator) -> np.ndarray:
    """
    z_t | r_t for NIG normal-IG mixture.
    Here params should be STEP params (i.e., for one weekly increment):
      alpha, delta_step, mu_step
    """
    alpha = float(params["alpha"])
    delta = float(params["delta"])
    mu0   = float(params["mu"])
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
    Sample (mu_h, phi_h) for IG mixing distribution, given z.
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

    # mu | z,phi ~ GIG(...)
    lam_mu = (Tn - 1.0) / 2.0
    a_mu2 = float(phi_current * u1)
    b_mu2 = float(phi_current * u3)
    mu_new = sample_gig(lam_mu, a_mu2, b_mu2, rng)

    # phi | z,mu ~ Gamma(...)
    shape_phi = (v + 1.0) / 2.0
    rate_phi = u1/(2.0 * mu_new) - u2 + (u3*mu_new)/2.0
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
    X = np.column_stack([np.ones(T), z])     # (T,2)
    w = 1.0 / np.maximum(z, 1e-12)          # diag weights

    B0 = np.asarray(B0, dtype=float)
    b0 = np.asarray(b0, dtype=float).reshape(2, 1)

    B0_inv = np.linalg.inv(B0)
    XtW = X.T * w                           # (2,T)

    B_new_inv = XtW @ X + B0_inv
    B_new = np.linalg.inv(B_new_inv)

    b_vec = (XtW @ r.reshape(-1, 1)) + (B0_inv @ b0)
    b_mean = (B_new @ b_vec).reshape(2)

    draw = multivariate_normal.rvs(mean=b_mean, cov=B_new, random_state=rng)
    mu0_step = float(draw[0])
    beta = float(draw[1])
    return mu0_step, beta


# -----------------------------
# Weekly theta + asset inversion inside Gibbs
# -----------------------------
def annual_cc_rate_to_weekly(r_annual: np.ndarray, weeks_per_year: int = 52) -> np.ndarray:
    """
    Convert annual continuously-compounded rate to weekly continuously-compounded rate.
    """
    return np.asarray(r_annual, dtype=float) / float(weeks_per_year)


def theta_series_weekly(p: NIGParams, r_week: np.ndarray, tau_weeks: float) -> np.ndarray:
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
    """
    A = np.full_like(E, np.nan, dtype=float)
    th = np.full_like(E, np.nan, dtype=float)

    A_prev = None
    for i in range(len(E)):
        if not (np.isfinite(E[i]) and np.isfinite(L[i]) and np.isfinite(r_week[i])):
            continue
        A[i], th[i] = invert_asset_one_date(
            float(E[i]), float(L[i]), float(r_week[i]), float(tau_weeks), p,
            A_prev=A_prev,
            U=U, n=n
        )
        A_prev = A[i]
    return A, th


# -----------------------------
# Main Gibbs sampler (weekly)
# -----------------------------
def gibbs_sampler_weekly(
    E_series: np.ndarray,
    L_series: np.ndarray,        # should be FACE VALUE strike used in pricing (not already-discounted)
    rf_series_annual_cc: np.ndarray,
    dates: np.ndarray,
    start_date=None,
    end_date=None,
    max_iter: int = 2000,
    *,
    em_params: Dict[str, float],   # expects weekly-step params: alpha, beta, delta, mu
    burn_in: int = 500,
    thin: int = 5,
    weeks_per_year: int = 52,
    tau_years: float = 1.0,        # maturity horizon in years
    U: float = 120.0,
    n_int: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:

    if rng is None:
        rng = np.random.default_rng()

    E_series = np.asarray(E_series, dtype=float)
    L_series = np.asarray(L_series, dtype=float)
    rf_series_annual_cc = np.asarray(rf_series_annual_cc, dtype=float)
    dates = np.asarray(dates)

    if not (E_series.shape == L_series.shape == rf_series_annual_cc.shape == dates.shape):
        raise ValueError("E_series, L_series, rf_series, dates must have same shape")
    if burn_in < 0 or burn_in >= max_iter:
        raise ValueError("burn_in must be in [0, max_iter-1]")
    if thin <= 0:
        raise ValueError("thin must be >= 1")

    # --- window mask ---
    mask = np.ones(dates.size, dtype=bool)
    if start_date is not None:
        mask &= (dates >= start_date)
    if end_date is not None:
        mask &= (dates <= end_date)

    idx = np.where(mask)[0]
    if idx.size < 3:
        raise ValueError("Need at least 3 observations in the training window")

    # windowed series
    Ew = E_series[mask]
    Lw = L_series[mask]
    rw_week = annual_cc_rate_to_weekly(rf_series_annual_cc[mask], weeks_per_year=weeks_per_year)

    # 1 year expressed in "weeks" (because we treat params as weekly increments)
    tau_weeks = float(tau_years) * float(weeks_per_year)

    n_obs = int(idx.size)

    # --- EM init (WEEKLY params) ---
    alpha = float(em_params["alpha"])
    beta  = float(em_params["beta"])
    delta = float(em_params["delta"])
    mu0   = float(em_params["mu"])

    # step-length in the SAME units as params: weekly step => h_step = 1
    h_step = 1.0

    # convert weekly NIG -> IG-mix params (mu_h, phi_h) at step
    mu_h, phi_h = mu_phi_from_params({"alpha": alpha, "beta": beta, "delta": delta}, h_step=h_step)

    # priors centered at EM (weak)
    b0_prior = np.array([mu0, beta], dtype=float)
    B0_prior = np.diag([50.0, 50.0]).astype(float)

    # hyper centered at EM
    var_phi = 100.0
    hyper = {
        "xi": (phi_h * phi_h) / var_phi,
        "chi": phi_h / var_phi,
        "eta": mu_h,
        "omega": 0.001,
    }

    # --- storage plan ---
    keep_mask = np.zeros(max_iter, dtype=bool)
    for it in range(max_iter):
        if it >= burn_in and ((it - burn_in) % thin == 0):
            keep_mask[it] = True
    n_keep = int(keep_mask.sum())

    A_draws      = np.full((n_keep, n_obs), np.nan, dtype=float)
    theta_draws  = np.full((n_keep, n_obs), np.nan, dtype=float)
    z_draws      = np.full((n_keep, n_obs - 1), np.nan, dtype=float)

    # store BOTH weekly-step params and annualized (for reporting)
    params_weekly = np.full((n_keep, 4), np.nan, dtype=float)  # [alpha,beta,delta,mu]
    params_annual = np.full((n_keep, 4), np.nan, dtype=float)  # [alpha,beta,delta*52,mu*52]
    mu_phi_draws  = np.full((n_keep, 2), np.nan, dtype=float)  # [mu_h,phi_h]

    n_reject = 0
    k = 0

    for it in range(max_iter):

        # current NIG params object (weekly-step)
        p_cur = NIGParams(alpha=alpha, beta=beta, delta=delta, mu=mu0)

        # invert weekly assets under current params
        try:
            A_path, theta_path = invert_assets_on_dates(
                Ew, Lw, rw_week, dates[mask], p_cur,
                tau_weeks=tau_weeks, U=U, n=n_int
            )
        except Exception:
            n_reject += 1
            continue

        if not np.all(np.isfinite(A_path)):
            n_reject += 1
            continue

        rA = np.diff(np.log(A_path))  # weekly asset log returns
        if not np.all(np.isfinite(rA)):
            n_reject += 1
            continue

        # store draw BEFORE updating
        if keep_mask[it]:
            A_draws[k, :] = A_path
            theta_draws[k, :] = theta_path
            params_weekly[k, :] = np.array([alpha, beta, delta, mu0], dtype=float)
            params_annual[k, :] = np.array([alpha, beta, delta * weeks_per_year, mu0 * weeks_per_year], dtype=float)
            mu_phi_draws[k, :] = np.array([mu_h, phi_h], dtype=float)

        # ---- Gibbs updates (weekly-step) ----
        # z | r  (weekly-step delta and mu)
        z = sample_z_posterior(rA, {"alpha": alpha, "delta": delta, "mu": mu0}, rng)

        # mu_h, phi_h | z
        try:
            mu_h_new, phi_h_new = sample_mu_phi(z, phi_current=phi_h, hyper=hyper, rng=rng)
        except Exception:
            mu_h_new, phi_h_new = mu_h, phi_h

        # (mu0, beta) | r,z
        try:
            mu0_new, beta_new = sample_beta_mu0(rA, z, b0_prior, B0_prior, rng)
        except Exception:
            mu0_new, beta_new = mu0, beta

        # map back to weekly-step NIG params
        try:
            prop = params_from_mu_phi(mu_h_new, phi_h_new, mu0_h=mu0_new, beta=beta_new, h_step=h_step)
            alpha_prop = float(prop["alpha"])
            beta_prop  = float(prop["beta"])
            delta_prop = float(prop["delta"])
            mu_prop    = float(prop["mu"])
        except Exception:
            n_reject += 1
            if keep_mask[it]:
                z_draws[k, :] = z
                k += 1
            continue

        # feasibility checks
        ok = True
        if not (np.isfinite(alpha_prop) and np.isfinite(beta_prop) and np.isfinite(delta_prop) and np.isfinite(mu_prop)):
            ok = False
        if delta_prop <= 0.0 or alpha_prop <= 0.0 or abs(beta_prop) >= alpha_prop:
            ok = False

        # theta feasibility on window (mgf exists for theta and theta+1)
        if ok:
            try:
                p_prop = NIGParams(alpha=alpha_prop, beta=beta_prop, delta=delta_prop, mu=mu_prop)
                th_prop = theta_series_weekly(p_prop, rw_week, tau_weeks=tau_weeks)
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

    # truncate to actual stored draws
    A_draws     = A_draws[:k, :]
    theta_draws = theta_draws[:k, :]
    z_draws     = z_draws[:k, :]
    params_weekly = params_weekly[:k, :]
    params_annual = params_annual[:k, :]
    mu_phi_draws  = mu_phi_draws[:k, :]

    return {
        "A_draws": A_draws,
        "theta_draws": theta_draws,
        "z_draws": z_draws,
        "params_weekly": params_weekly,   # [alpha,beta,delta,mu] per week
        "params_annual": params_annual,   # [alpha,beta,delta*52,mu*52] annualized for reporting
        "mu_phi_draws": mu_phi_draws,     # IG mix params at step
        "priors": {"hyper": hyper, "b0": b0_prior, "B0": B0_prior},
        "meta": {
            "burn_in": burn_in,
            "thin": thin,
            "max_iter": max_iter,
            "n_keep_requested": n_keep,
            "n_keep_actual": int(k),
            "n_reject": int(n_reject),
            "weeks_per_year": int(weeks_per_year),
            "tau_weeks": float(tau_weeks),
            "U": float(U),
            "n_int": int(n_int),
        },
    }