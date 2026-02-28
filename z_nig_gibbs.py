import numpy as np
from typing import Dict, Tuple, Optional
from scipy.stats import geninvgauss, gamma, multivariate_normal
from nig_em_paper import update_theta_series, get_asset_path
from typing import Any


def mu_phi_from_params(params: Dict[str, float], h: float) -> Tuple[float, float]:
    """
    Map annual NIG params -> (mu_h, phi_h) for *daily* mixing distribution (step h).
    Uses: delta_h = delta * h, gamma = sqrt(alpha^2 - beta^2)
          mu_h  = delta_h / gamma
          phi_h = delta_h * gamma
    """
    alpha = float(params.get("alpha", 0.0))
    beta1 = float(params.get("beta1", 0.0))
    delta = float(params.get("delta", 0.0))

    if delta <= 0.0 or alpha <= 0.0 or abs(beta1) >= alpha:
        raise ValueError("Need delta>0, alpha>0, and |beta1|<alpha.")

    gamma_val = float(np.sqrt(max(alpha * alpha - beta1 * beta1, 0.0)))
    delta_h = float(delta) * float(h)

    mu_h = delta_h / gamma_val
    phi_h = delta_h * gamma_val
    return float(mu_h), float(phi_h)


def params_from_mu_phi(
    mu_h: float, phi_h: float, beta0_h: float, beta1: float, h: float
) -> Dict[str, float]:
    """
    Map (mu_h, phi_h, beta0_h, beta1) for daily step h back to annual params.

      delta_h = sqrt(mu_h * phi_h)
      gamma   = sqrt(phi_h / mu_h)
      alpha   = sqrt(beta1^2 + gamma^2)
      delta   = delta_h / h
      beta0   = beta0_h / h
    """
    mu_h = float(mu_h)
    phi_h = float(phi_h)
    if mu_h <= 0.0 or phi_h <= 0.0:
        raise ValueError("mu_h and phi_h must be positive.")

    delta_h = float(np.sqrt(mu_h * phi_h))
    gamma_val = float(np.sqrt(phi_h / mu_h))
    alpha = float(np.sqrt(beta1 * beta1 + gamma_val * gamma_val))

    delta = delta_h / float(h)
    beta0 = float(beta0_h) / float(h)

    return {
        "alpha": float(alpha),
        "beta1": float(beta1),
        "delta": float(delta),
        "beta0": float(beta0),
    }


def sample_gig(
    lam: float, chi: float, psi: float, rng: np.random.Generator
) -> float:

    if chi <= 0.0 or psi <= 0.0:
        raise ValueError("chi and psi must be positive for the GIG distribution")
    b = np.sqrt(chi * psi)
    y = geninvgauss.rvs(lam, b, random_state=rng)
    return float(y * np.sqrt(chi / psi))


def sample_z_posterior(
    returns: np.ndarray, params: Dict[str, float], rng: np.random.Generator
) -> np.ndarray:

    alpha = params.get("alpha", 0.0)
    delta = params.get("delta", 0.0)
    beta0 = params.get("beta0", 0.0)
    alpha2 = alpha * alpha
    z = np.empty_like(returns, dtype=float)
    for i, rt in enumerate(returns):
        q_rt = 1.0 + ((rt - beta0) / delta) ** 2
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
    Tn = len(z)

    if Tn <= 1:
        raise ValueError("At least two observations are required to sample mu and phi")

    z = np.asarray(z, dtype=float)
    zbar = float(np.mean(z))
    inv_z = 1.0 / z
    zbar_r = float(np.mean(inv_z))
    xi = hyper.get("xi", 0.0)
    chi_hyp = hyper.get("chi", 0.0)
    eta = hyper.get("eta", 1.0)
    omega = hyper.get("omega", 1.0)

    # Statistics u1, u2, u3 and v
    u1 = Tn * zbar + omega * eta
    u2 = Tn + omega - chi_hyp
    u3 = Tn * zbar_r + omega / max(eta, 1e-12)
    v = Tn + 2.0 * xi

    # Sample mu via GIG
    lam_mu = (Tn - 1.0) / 2.0
    a_mu = np.sqrt(phi_current * u1)
    b_mu = np.sqrt(phi_current * u3)

    mu_new = sample_gig(lam_mu, a_mu * a_mu, b_mu * b_mu, rng) #CHECK SQUARED PARAMS
    # Sample phi via Gamma
    shape_phi = (v + 1.0) / 2.0
    rate_phi = u1/(2.0 * mu_new) - u2 + (u3*mu_new)/2.0

    if rate_phi <= 0.0:
        rate_phi = 1e-12
    scale_phi = 1.0 / rate_phi
    phi_new = gamma.rvs(shape_phi, scale=scale_phi, random_state=rng)

    return mu_new, phi_new


def sample_beta(
    returns: np.ndarray,
    z: np.ndarray,
    b0: np.ndarray,
    B0: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:

    r = np.asarray(returns, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    if r.shape != z.shape:
        raise ValueError("returns and z must have the same length")

    T = r.size
    X = np.column_stack([np.ones(T), z])  # (T,2)

    # weights: Σ_z^{-1} = diag(1/z_t)
    w = 1.0 / np.maximum(z,1e-12)  # (T,)

    B0_inv = np.linalg.inv(np.asarray(B0, dtype=float))
    XtW = X.T * w  # (2,T)

    B_new_inv = XtW @ X + B0_inv
    B_new = np.linalg.inv(B_new_inv)

    b0_vec = np.asarray(b0, dtype=float).reshape(2, 1)
    b_vec = (XtW @ r.reshape(-1, 1)) + (B0_inv @ b0_vec)
    b_mean = (B_new @ b_vec).reshape(2)

    beta_draw = multivariate_normal.rvs(mean=b_mean, cov=B_new, random_state=rng)
    return float(beta_draw[0]), float(beta_draw[1])

def gibbs_sampler(
    E_series: np.ndarray,
    L_series: np.ndarray,        # IMPORTANT: same convention as EM/get_asset_path (your current discounted proxy)
    rf_series: np.ndarray,
    dates: np.ndarray,
    start_date=None,
    end_date=None,
    max_iter: int = 2000,
    *,
    em_params: Dict[str, float],
    burn_in: int = 500,
    thin: int = 5,
    trading_days: int = 250,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:

    if rng is None:
        rng = np.random.default_rng()

    E_series = np.asarray(E_series, dtype=float)
    L_series = np.asarray(L_series, dtype=float)
    rf_series = np.asarray(rf_series, dtype=float)
    dates = np.asarray(dates)

    if not (E_series.shape == L_series.shape == rf_series.shape == dates.shape):
        raise ValueError("E_series, L_series, rf_series, dates must have same shape")
    if burn_in < 0 or burn_in >= max_iter:
        raise ValueError("burn_in must be in [0, max_iter-1]")
    if thin <= 0:
        raise ValueError("thin must be >= 1")

    h = 1.0 / float(trading_days)

    # --- training window mask (same as your EM/get_asset_path) ---
    mask = np.ones(dates.size, dtype=bool)
    if start_date is not None:
        mask &= (dates >= start_date)
    if end_date is not None:
        mask &= (dates <= end_date)

    idx = np.where(mask)[0]
    if idx.size < 3:
        raise ValueError("Need at least 3 observations in the training window")

    n = int(idx.size)

    # --- EM init (annual params) ---
    alpha = float(em_params["alpha"])
    beta1 = float(em_params["beta1"])
    delta = float(em_params["delta"])
    beta0 = float(em_params["beta0"])

    # convert EM -> daily (mu_h, phi_h, beta0_h)
    mu_h, phi_h = mu_phi_from_params({"alpha": alpha, "beta1": beta1, "delta": delta}, h=h)
    beta0_h = beta0 * h

    # priors centered at EM (weakly informative) — keep same spirit, but intercept is daily
    b0_prior = np.array([beta0_h, beta1], dtype=float)
    B0_prior = np.diag([50.0 * h * h, 50.0]).astype(float)  # scale beta0 variance by h^2

    # hyper centered at EM daily (phi variance fixed at 100, per your previous convention)
    var_phi = 100.0
    hyper = {
        "xi": (phi_h * phi_h) / var_phi,
        "chi": phi_h / var_phi,
        "eta": mu_h,
        "omega": 0.001,
    }

    # --- storage ---
    keep_mask = np.zeros(max_iter, dtype=bool)
    for it in range(max_iter):
        if it >= burn_in and ((it - burn_in) % thin == 0):
            keep_mask[it] = True
    n_keep = int(keep_mask.sum())

    A_draws = np.full((n_keep, n), np.nan, dtype=float)
    theta_draws = np.full((n_keep, n), np.nan, dtype=float)
    z_draws = np.full((n_keep, n - 1), np.nan, dtype=float)

    # store ANNUAL params (consistent with EM)
    params_draws = np.full((n_keep, 4), np.nan, dtype=float)  # [alpha,beta1,delta,beta0]
    mu_phi_draws = np.full((n_keep, 2), np.nan, dtype=float)  # [mu_h,phi_h]

    n_reject = 0
    k = 0

    for it in range(max_iter):

        # ---- CURRENT STATE (annual params used for pricing/theta) ----
        params_cur = {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0}

        # theta only needs to be valid in-window; fill full vector for get_asset_path indexing
        theta_full = np.full_like(rf_series, np.nan, dtype=float)
        try:
            theta_full[mask] = update_theta_series(params_cur, rf_series[mask])
        except Exception:
            # current state should almost never fail if your constraints are correct
            n_reject += 1
            continue

        # E-step-like deterministic latent A_t under current params
        A_path = get_asset_path(
            params=params_cur,
            theta_series=theta_full,
            rf_series=rf_series,
            dates=dates,
            E_series=E_series,
            L_face_series=L_series,      # same convention as EM (your discounted proxy)
            start_date=start_date,
            end_date=end_date,
        )

        rA = np.diff(np.log(A_path))  # daily returns
        if not np.all(np.isfinite(rA)):
            n_reject += 1
            continue

        # ---- store a consistent draw BEFORE updating params ----
        if keep_mask[it]:
            A_draws[k, :] = A_path
            theta_draws[k, :] = theta_full[mask]
            params_draws[k, :] = np.array([alpha, beta1, delta, beta0], dtype=float)
            mu_phi_draws[k, :] = np.array([mu_h, phi_h], dtype=float)

        # ---- Gibbs updates on DAILY scale ----
        # z | r  (use daily delta_h and daily beta0_h)
        delta_h = delta * h
        beta0_h = beta0 * h
        z = sample_z_posterior(
            rA,
            {"alpha": alpha, "delta": delta_h, "beta0": beta0_h},
            rng
        )

        # mu_h, phi_h | z  (already daily objects)
        try:
            mu_h_new, phi_h_new = sample_mu_phi(z, phi_current=phi_h, hyper=hyper, rng=rng)
        except Exception:
            mu_h_new, phi_h_new = mu_h, phi_h

        # (beta0_h, beta1) | r, z
        try:
            beta0_h_new, beta1_new = sample_beta(rA, z, b0_prior, B0_prior, rng)
        except Exception:
            beta0_h_new, beta1_new = beta0_h, beta1

        # map back to ANNUAL params
        try:
            prop = params_from_mu_phi(mu_h_new, phi_h_new, beta0_h=beta0_h_new, beta1=beta1_new, h=h)
            alpha_prop = float(prop["alpha"])
            delta_prop = float(prop["delta"])
            beta0_prop = float(prop["beta0"])
        except Exception:
            n_reject += 1
            if keep_mask[it]:
                z_draws[k, :] = z
                k += 1
            continue

        # feasibility checks: NIG + theta existence + pricing validity
        ok = True
        if not (np.isfinite(alpha_prop) and np.isfinite(delta_prop) and np.isfinite(beta0_prop) and np.isfinite(beta1_new)):
            ok = False
        if delta_prop <= 0.0 or alpha_prop <= 0.0 or abs(beta1_new) >= alpha_prop:
            ok = False
        if alpha_prop < 0.5:
            ok = False
        bound = delta_prop * np.sqrt(max(2.0 * alpha_prop - 1.0, 0.0))
        if abs(beta0_prop) > bound:
            ok = False

        # also require theta series computable in-window AND cdf-parameter feasibility
        if ok:
            try:
                theta_prop = update_theta_series(
                    {"alpha": alpha_prop, "beta1": beta1_new, "delta": delta_prop, "beta0": beta0_prop},
                    rf_series[mask],
                )
                # ensure NIG cdf parameters are valid for both beta+theta and beta+theta+1
                alpha_floor = max(np.max(np.abs(beta1_new + theta_prop)),
                                  np.max(np.abs(beta1_new + theta_prop + 1.0))) + 1e-6
                if alpha_prop <= alpha_floor:
                    ok = False
            except Exception:
                ok = False

        if ok:
            # accept new state
            mu_h, phi_h = float(mu_h_new), float(phi_h_new)
            beta1 = float(beta1_new)
            alpha = float(alpha_prop)
            delta = float(delta_prop)
            beta0 = float(beta0_prop)
        else:
            n_reject += 1

        # store z aligned with stored draw
        if keep_mask[it]:
            z_draws[k, :] = z
            k += 1

    # if some iterations were skipped, truncate to actual stored draws
    A_draws = A_draws[:k, :]
    theta_draws = theta_draws[:k, :]
    z_draws = z_draws[:k, :]
    params_draws = params_draws[:k, :]
    mu_phi_draws = mu_phi_draws[:k, :]

    return {
        "A_draws": A_draws,
        "theta_draws": theta_draws,            # theta_t per stored draw (window)
        "z_draws": z_draws,
        "params_draws": params_draws,          # [alpha, beta1, delta, beta0] annual
        "mu_phi_draws": mu_phi_draws,          # [mu_h, phi_h] daily
        "priors": {"hyper": hyper, "b0": b0_prior, "B0": B0_prior},
        "meta": {
            "burn_in": burn_in,
            "thin": thin,
            "max_iter": max_iter,
            "n_keep_requested": n_keep,
            "n_keep_actual": int(k),
            "n_reject": int(n_reject),
            "trading_days": trading_days,
        },
    }
