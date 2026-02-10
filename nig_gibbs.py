from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.stats import geninvgauss, gamma, multivariate_normal
from scipy.stats import norminvgauss
from __future__ import annotations

from nig_initialization import (
    invert_nig_call_price,
    update_theta,
    mu_phi_from_params,
    compute_pd_physical,
    compute_pd_risk_neutral,
)


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
) -> Tuple[float, float]:

    r = np.asarray(returns, dtype=float)
    z = np.asarray(z, dtype=float)
    if r.shape != z.shape:
        raise ValueError("returns and z must have the same length")

    # Prior precision
    B0_inv = np.linalg.inv(B0)

    # Design matrix X with t-th row [1, z_t]
    X = np.column_stack([np.ones(Tn), z])  # (T,2)

    # Σ_z = diag(z_t)  =>  Σ_z^{-1} = diag(1/z_t)
    Sig_inv = np.diag(1.0 / z)  # (T,T)

    # Posterior precision
    B_new_inv = X.T @ Sig_inv @ X + B0_inv  # (2,2)

    # Posterior covariance
    B_new = np.linalg.inv(B_new_inv)  # (2,2)

    # Posterior mean
    b0 = np.asarray(b0, dtype=float).reshape(2, 1)  # (2,1)
    b_vec = X.T @ Sig_inv @ r + B0_inv @ b0  # (2,1)
    b_new = (B_new @ b_vec).reshape(2)  # (2,)

    beta_draw = multivariate_normal.rvs(mean=b_new, cov=B_new, random_state=rng)
    return float(beta_draw[0]), float(beta_draw[1])


def gibbs_sampler(
    equity: np.ndarray,
    liabilities_L: float,
    r_series: Optional[np.ndarray],
    maturity_T: float,
    n_iter: int,
    burn_in: int,
    em_params: Dict[str, float],
    thin: int = 100,                      # like Karlis (store every 100)
    r_f: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    physical_measure: bool = True,
) -> Dict[str, np.ndarray]:

    E = np.asarray(equity, dtype=float)
    N = E.size

    #safety checks
    if N < 2:
        raise ValueError("Equity series must have at least two observations")
    if burn_in >= n_iter:
        raise ValueError("burn_in must be strictly less than n_iter")
    if thin < 1:
        raise ValueError("thin must be >= 1")
    if np.any(E <= 0.0):
        raise ValueError("Equity must be strictly positive")
    if r_series is None:
        r_series = np.zeros(N, dtype=float)
    else:
        r_series = np.asarray(r_series, dtype=float)
        if r_series.shape != E.shape:
            raise ValueError("r_series must have the same shape as equity")
    if rng is None:
        rng = np.random.default_rng()

    dt = maturity_T / (N - 1)  # step size

    # initialize from EM
    alpha = float(em_params["alpha"])
    beta1 = float(em_params["beta1"])
    delta = float(em_params["delta"])
    beta0 = float(em_params["beta0"])
    theta = float(em_params.get("theta", 0.0))

    # convert to (mu,phi) then to step-units
    mu_annual, phi_annual = mu_phi_from_params(
        {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0, "theta": theta}
    )
    mu_step = float(mu_annual * dt)
    phi_step = float(phi_annual * dt)

    # 2) EM-centred and Karlis-diffuse priors (built inside the sampler)
    #    Karlis: xi=0.01, chi=0.01 (phi mean 1 var 100), eta=5, omega=0.001
    #    EM-centred: keep xi,omega but set chi,eta so means match EM.

    xi = 0.01
    omega = 0.001
    chi_rate = xi / max(phi_step, 1e-12)   # so E[phi]=phi_step, var = phi_step^2/xi (very diffuse)
    eta = mu_step                          # centre mu|phi at mu_step

    hyper = {"xi": float(xi), "chi": float(chi_rate), "eta": float(eta), "omega": float(omega)}

    # beta prior for regression in step-units: r_t = beta0_step + beta1 z_t + sqrt(z_t) eps
    b0 = np.array([beta0 * dt, beta1], dtype=float).reshape(2, 1)
    B0 = np.diag([50.0, 50.0]).astype(float)  # "variances up to 50", cov=0
    B0_inv = np.linalg.inv(B0)

    # helper: pricing feasibility for your call-price formula (beta_plus = beta1+theta+1)
    def pricing_ok(alpha_: float, beta1_: float, theta_: float, eps: float = 1e-10) -> bool:
        return (alpha_ > abs(beta1_ + theta_) + eps) and (alpha_ > abs(beta1_ + theta_ + 1.0) + eps)


    # Storage (with thinning)
    n_store = (n_iter - burn_in) // thin
    draws = {
        "beta0": np.zeros(n_store),
        "beta1": np.zeros(n_store),
        "mu": np.zeros(n_store),
        "phi": np.zeros(n_store),
        "delta": np.zeros(n_store),
        "gamma": np.zeros(n_store),
        "alpha": np.zeros(n_store),
        "theta": np.zeros(n_store),
        "pd": np.zeros(n_store),
    }
    store_idx = 0

    #gibbs sampler
    for it in range(n_iter):

        # deterministic asset inversion
        params_pricing = {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0, "theta": theta}
        A = np.zeros(N, dtype=float)

        A[0] = invert_nig_call_price(
            E_obs=E[0], L=liabilities_L, T=maturity_T, r=float(r_series[0]),
            params=params_pricing, physical_measure=physical_measure
        )
        for t in range(1, N):
            T_rem = maturity_T - t * dt
            if T_rem <= 0.0:
                A[t] = E[t] + liabilities_L
            else:
                A[t] = invert_nig_call_price(
                    E_obs=E[t], L=liabilities_L, T=T_rem, r=float(r_series[t]),
                    params=params_pricing, physical_measure=physical_measure
                )

        rA = np.diff(np.log(A))  # step returns, length N-1
        Tn = rA.size

        # sample z | r,Theta  (step-units here)
        beta0_step = beta0 * dt
        delta_step = delta * dt
        alpha2 = alpha * alpha

        z = np.empty(Tn, dtype=float)
        ds = max(delta_step, 1e-12)
        for i, rt in enumerate(rA):
            q_rt = 1.0 + ((rt - beta0_step) / ds) ** 2
            chi_z = (ds * ds) * q_rt
            z[i] = sample_gig(lam=-1.0, chi=chi_z, psi=alpha2, rng=rng)

        # sample (mu_step, phi_step) | z
        mu_step, phi_step = sample_mu_phi(z, phi_step, hyper, rng)

        # map back to annual params (keeping gamma in step-units, same numerically)
        delta_step = float(np.sqrt(max(mu_step * phi_step, 1e-24)))
        gamma_val = float(np.sqrt(max(phi_step / max(mu_step, 1e-12), 0.0)))

        mu_annual = mu_step / dt
        phi_annual = phi_step / dt
        delta = delta_step / dt

        # alpha depends on beta1 and gamma
        alpha = float(np.sqrt(beta1 * beta1 + gamma_val * gamma_val))

        # sample beta0_step, beta1 | r,z  (few redraws; else keep previous)
        X = np.column_stack([np.ones(Tn, dtype=float), z])
        Sig_inv = np.diag(1.0 / np.maximum(z, 1e-12))

        B_new_inv = X.T @ Sig_inv @ X + B0_inv
        B_new = np.linalg.inv(B_new_inv)
        b_vec = X.T @ Sig_inv @ rA.reshape(-1, 1) + B0_inv @ b0
        b_mean = (B_new @ b_vec).reshape(2)

        beta0_step_old = beta0_step
        beta1_old = beta1

        ok = False
        for _ in range(10):
            draw = multivariate_normal.rvs(mean=b_mean, cov=B_new, random_state=rng)
            beta0_step_c = float(draw[0])
            beta1_c = float(draw[1])
            alpha_c = float(np.sqrt(beta1_c * beta1_c + gamma_val * gamma_val))
            if pricing_ok(alpha_c, beta1_c, theta):
                beta0_step, beta1, alpha = beta0_step_c, beta1_c, alpha_c
                ok = True
                break

        if not ok:
            beta0_step, beta1 = beta0_step_old, beta1_old
            alpha = float(np.sqrt(beta1 * beta1 + gamma_val * gamma_val))

        beta0 = beta0_step / dt
        b0 = np.array([beta0_step, beta1], dtype=float).reshape(2, 1)  # keep prior mean centred on current

        # Esscher theta update (optional)
        if (not physical_measure) and (r_f is not None):
            try:
                theta_new = update_theta({"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0}, float(r_f[-1]))
                if pricing_ok(alpha, beta1, theta_new):
                    theta = theta_new
            except ValueError:
                pass

        # store (burn-in + thinning)
        if it >= burn_in and ((it - burn_in) % thin == 0):
            if store_idx >= n_store:
                break

            draws["beta0"][store_idx] = beta0
            draws["beta1"][store_idx] = beta1
            draws["mu"][store_idx] = mu_annual
            draws["phi"][store_idx] = phi_annual
            draws["delta"][store_idx] = delta
            draws["gamma"][store_idx] = gamma_val
            draws["alpha"][store_idx] = alpha
            draws["theta"][store_idx] = theta

            params_now = {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0, "theta": theta}

            # 1-year PD from last asset value (use maturity_T, not dt)
            if physical_measure:
                draws["pd"][store_idx] = compute_pd_physical(A0=float(A[-1]), L=liabilities_L, T=maturity_T, params=params_now)
            else:
                draws["pd"][store_idx] = compute_pd_risk_neutral(A0=float(A[-1]), L=liabilities_L, T=maturity_T, params=params_now)

            store_idx += 1

    # trim if early stop
    if store_idx < n_store:
        for k in draws:
            draws[k] = draws[k][:store_idx]

    return draws
