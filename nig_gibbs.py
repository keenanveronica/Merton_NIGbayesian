import numpy as np
from typing import Dict, Tuple, Optional
from scipy.stats import geninvgauss, gamma, multivariate_normal
from scipy.stats import norminvgauss
from scipy.optimize import brentq
from typing import Any


def mu_phi_from_params(params: Dict[str, float]) -> Tuple[float, float]:
    alpha = float(params.get("alpha", 0.0))
    beta1 = float(params.get("beta1", 0.0))
    delta = float(params.get("delta", 0.0))

    if delta <= 0.0 or alpha <= 0.0 or abs(beta1) >= alpha:
        raise ValueError("Need delta>0, alpha>0, and |beta1|<alpha to compute (mu,phi).")

    gamma = np.sqrt(alpha * alpha - beta1 * beta1)
    mu = delta / gamma
    phi = delta * gamma
    return float(mu), float(phi)


def params_from_mu_phi(
    mu: float, phi: float, beta0: float, beta1: float, theta: float = 0.0
) -> Dict[str, float]:
    delta = np.sqrt(max(mu * phi, 0.0))
    gamma_val = np.sqrt(max(phi / mu, 0.0))
    alpha = np.sqrt(beta1 * beta1 + gamma_val * gamma_val)
    return {
        "alpha": float(alpha),
        "beta1": float(beta1),
        "delta": float(delta),
        "beta0": float(beta0),
        "theta": float(theta),
    }


def update_theta(params: Dict[str, float], r_f: float) -> float:

    alpha = float(params.get("alpha", 0.0))
    beta = float(params.get("beta1", 0.0))
    delta = float(params.get("delta", 0.0))
    mu = float(params.get("beta0", 0.0))

    if delta <= 0.0 or alpha <= 0.0 or abs(beta) >= alpha:
        raise ValueError("Invalid NIG params: require delta>0, alpha>0, and |beta|<alpha.")
    if alpha < 0.5:
        raise ValueError("Theta existence requires alpha >= 0.5.")
    bound = delta * np.sqrt(max(2.0 * alpha - 1.0, 0.0))
    if abs(mu) > bound:
        raise ValueError("Theta existence requires |mu| <= delta*sqrt(2*alpha-1).")

    num = mu - float(r_f)
    inside = (4*(alpha ** 2)*(delta ** 2))/((num ** 2) + (delta ** 2))
    theta = -beta - 0.5 - (num/(2*delta))*np.sqrt(inside - 1.0)
    return float(theta)


def update_theta_series(params: Dict[str, float], r_f_series: np.ndarray) -> np.ndarray:
    """Vector theta_t computed day-by-day from update_theta(..., r_f[t])."""
    r_f_series = np.asarray(r_f_series, dtype=float)
    out = np.empty_like(r_f_series, dtype=float)
    for t in range(r_f_series.size):
        out[t] = update_theta(params, float(r_f_series[t]))
    return out


def nig_call_price(A: float, L_face: float, L_disc: float, T: float, params: Dict[str, float]) -> float:
    if A <= 0.0 or L_face <= 0.0 or L_disc <= 0.0 or T <= 0.0:
        return np.nan

    alpha = float(params.get("alpha", 0.0))
    beta  = float(params.get("beta1", 0.0))
    delta = float(params.get("delta", 0.0))
    mu    = float(params.get("beta0", 0.0))
    theta = float(params.get("theta", 0.0))

    if delta <= 0.0 or alpha <= 0.0 or abs(beta) >= alpha:
        return np.nan

    # threshold must use face value at maturity (paper)
    x0 = np.log(L_face / A)

    loc = mu * T
    scale = delta * T

    beta_plus  = beta + theta + 1.0
    beta_minus = beta + theta

    cdf_plus  = norminvgauss.cdf(x0, a=alpha, b=beta_plus,  loc=loc, scale=scale)
    cdf_minus = norminvgauss.cdf(x0, a=alpha, b=beta_minus, loc=loc, scale=scale)

    tail_plus  = 1.0 - cdf_plus
    tail_minus = 1.0 - cdf_minus

    return float(A * tail_plus - L_disc * tail_minus)


# CAN BE CONVERTED TO A "ROOT" FINDER ALSO - check if we should do this in series
def invert_nig_call_price(
    E_obs: float,
    L: float,
    L_face: float,
    T: float,
    params: Dict[str, float],
    A_min_factor: float = 1e-6,
    A_max_factor: float = 50,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """
    Solve for A such that nig_call_price(A,...) == E_obs (bracketing + Brent).
    Falls back to A = E_obs + L (discounted or not?) if bracketing fails.
    """
    if E_obs <= 0.0 or L <= 0.0 or T <= 0.0 or L_face <= 0.0:
        raise ValueError("E_obs, L, T, L_face must be positive")

    A_min = max(E_obs, A_min_factor * L_face)
    A_max = A_max_factor * (E_obs + L_face)

    def f(A: float) -> float:
        return nig_call_price(A, L_face=L_face, L_disc=L, T=T, params=params) - E_obs

    f_min, f_max = f(A_min), f(A_max)
    if not (np.isfinite(f_min) and np.isfinite(f_max)):
        return float(E_obs + L_face)

    if f_min * f_max > 0.0:
        # Expand bracket
        success = False
        for i in range(1, 18):
            factor = 2.0 ** i
            A_max_ext = A_max * factor
            A_min_ext = max(A_min / factor, 1e-12)
            f_min_ext, f_max_ext = f(A_min_ext), f(A_max_ext)
            if np.isfinite(f_min_ext) and np.isfinite(f_max_ext) and (f_min_ext * f_max_ext <= 0.0):
                A_min, A_max = A_min_ext, A_max_ext
                success = True
                break
        if not success:
            return float(E_obs + L_face)

    try:
        return float(brentq(f, A_min, A_max, xtol=tol, maxiter=max_iter))
    except Exception:
        return float(E_obs + L_face)
#NEED TO MAKE PRINTS TO CHECK IF THIS IS BEING DONE CORRECTLY

#we dont use this function in the gibbs
def get_asset_path(
        params: Dict[str, float],
        theta_series: np.ndarray,
        dates: np.ndarray,
        E_series: np.ndarray,
        L_series: np.ndarray, #NEED TO BE DISCOUNTED
        start_date: None,
        end_date: None,
) -> np.ndarray:

    dates = np.asarray(dates)
    E_series = np.asarray(E_series, dtype=float)
    L_series = np.asarray(L_series, dtype=float)
    theta_series = np.asarray(theta_series, dtype=float)

    if not (len(dates) == len(E_series) == len(L_series) == len(theta_series)):
        raise ValueError("dates, E_series, L_series, theta_series must have the same length")

    # date slice (inclusive)
    mask = np.ones(len(dates), dtype=bool)
    if start_date is not None:
        mask &= (dates >= start_date)
    if end_date is not None:
        mask &= (dates <= end_date)

    E = E_series[mask]
    L = L_series[mask]
    th = theta_series[mask]

    n = len(E)
    h = 1/250
    T = 250 + n-1

    end_idx_full = np.where(mask)[0][-1]     # last index in full arrays that is inside training window
    face_idx_full = end_idx_full + 250       # 1 trading year ahead

    if face_idx_full >= len(L_series):
        raise ValueError("Need liabilities 1y after end_date (end training window earlier).")

    L_face = float(L_series[face_idx_full])
    A_path = np.empty(n, dtype=float)

    for t in range(n):
        T_rem = (T - t) * h  # convert remaining days to years

        params_t = dict(params)
        params_t["theta"] = float(th[t])

        A_path[t] = invert_nig_call_price(
            E_obs=float(E[t]),
            L=float(L[t]),
            L_face=L_face, # WE MIGHT NEED TO CHANGE THIS
            T=float(T_rem),
            params=params_t,
        )

    return A_path


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
    w = 1.0 / z  # (T,)

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
    L_series: np.ndarray,
    rf_series: np.ndarray,
    dates: np.ndarray,
    start_date=None,
    end_date=None,
    max_iter: int = 10,
    *,
    em_params: Dict[str, float],
    burn_in: int = 0,
    thin: int = 1,
    # model/time conventions (paper/proposal)
    trading_days: int = 250,
    pd_horizon_years: float = 1.0,
    # RNG
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:

    if rng is None:
        rng = np.random.default_rng()

    # basic checks
    E_series = np.asarray(E_series, dtype=float)
    L_series = np.asarray(L_series, dtype=float)
    rf_series = np.asarray(rf_series, dtype=float)
    dates = np.asarray(dates)

    if not (E_series.shape == L_series.shape == rf_series.shape == dates.shape):
        raise ValueError("E_series, L_series, rf_series, dates must have same shape")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if burn_in < 0 or burn_in >= max_iter:
        raise ValueError("burn_in must be in [0, max_iter-1]")
    if thin <= 0:
        raise ValueError("thin must be >= 1")


    # slice training window
    mask = np.ones(dates.size, dtype=bool)
    if start_date is not None:
        mask &= (dates >= start_date)
    if end_date is not None:
        mask &= (dates <= end_date)

    idx = np.where(mask)[0]
    if idx.size < 3:
        raise ValueError("Need at least 3 observations in the training window")

    E = E_series[idx]
    rf = rf_series[idx]

    n = idx.size
    h = 1.0 / float(trading_days)

    # Paper/proposal maturity convention: last observation has pd_horizon_years maturity
    T0 = (n - 1) * h + float(pd_horizon_years)

    # Need the liability (face) at maturity date = end_of_window + 1 trading year
    end_idx_full = idx[-1]
    face_idx_full = end_idx_full + trading_days
    if face_idx_full >= L_series.size:
        raise ValueError("Need liabilities at end_date + 1 trading year (extend L_series or shorten window).")
    L_face = float(L_series[face_idx_full])  # strike at maturity


    # get EM estimates
    alpha_em = float(em_params["alpha"])
    beta1_em = float(em_params["beta1"])
    delta_em = float(em_params["delta"])
    beta0_em = float(em_params["beta0"])

    #compute gamma (EM)
    gamma_em = float(np.sqrt(max(alpha_em * alpha_em - beta1_em * beta1_em, 0.0)))

    #convert delta, gammma to mu, phi (also EM)
    mu_em = delta_em / gamma_em
    phi_em = delta_em * gamma_em

    #computing hyperparameter in order to center the priors
    var_phi = 100.0 # fixed (karlis)
    hyper = {
        "xi": (phi_em * phi_em) / var_phi,
        "chi": phi_em / var_phi,
        "eta": mu_em,
        "omega": 0.001,  # fixed (karlis)
    }

    b0_prior = np.array([beta0_em, beta1_em], dtype=float)
    B0_prior = np.diag([50.0, 50.0]).astype(float)

    #initialize chain at EM
    mu = float(mu_em)
    phi = float(phi_em)
    beta0 = float(beta0_em)
    beta1 = float(beta1_em)

    # Map back to (alpha,delta) each time using your helper
    # (assumes you already have params_from_mu_phi in scope)
    params_cur = params_from_mu_phi(mu, phi, beta0=beta0, beta1=beta1, theta=0.0)
    alpha = float(params_cur["alpha"])
    delta = float(params_cur["delta"])

    # storage sizes
    keep_mask = np.zeros(max_iter, dtype=bool)
    for it in range(max_iter):
        if it >= burn_in and ((it - burn_in) % thin == 0):
            keep_mask[it] = True
    n_keep = int(keep_mask.sum())

    # storage vectors
    A_draws = np.full((n_keep, n), np.nan, dtype=float)
    theta_draws = np.full((n_keep, n), np.nan, dtype=float)
    z_draws = np.full((n_keep, n - 1), np.nan, dtype=float)
    params_draws = np.full((n_keep, 4), np.nan, dtype=float)  # [alpha,beta1,delta,beta0]
    mu_phi_mat = np.full((n_keep, 2), np.nan, dtype=float)  # [mu,phi]

    n_reject = 0
    k_count = 0

   # GIBBS LOOP
    for it in range(max_iter):

        #compute theta vector (esscher)
        params_for_theta = {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0}
        try:
            theta_series = update_theta_series(params_for_theta, rf)
        except ValueError:
            n_reject += 1
            continue

        # Asset extraction
        A_path = np.empty(n, dtype=float)
        for t in range(n):
            T_rem = T0 - t * h  # in years
            params_t = {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0, "theta": float(theta_series[t])}

            L_disc = L_face * float(np.exp(-rf[t] * T_rem))
            A_path[t] = invert_nig_call_price(
                E_obs=float(E[t]),
                L=float(L_disc),
                L_face=float(L_face),
                T=float(T_rem),
                params=params_t,
            )

        #compute asset log returns
        rA = np.diff(np.log(A_path))  # length n-1

        # sample z | r
        z = sample_z_posterior(rA, {"alpha": alpha, "delta": delta, "beta0": beta0}, rng)

        # sample (mu,phi) | z
        try:
            mu_new, phi_new = sample_mu_phi(z, phi_current=phi, hyper=hyper, rng=rng)
        except Exception:
            mu_new, phi_new = mu, phi  #fallback: keep

        # sample (beta0,beta1) | r,z
        try:
            beta0_new, beta1_new = sample_beta(rA, z, b0_prior, B0_prior, rng)
        except Exception:
            beta0_new, beta1_new = beta0, beta1  # fallback: keep

        # map back to (alpha,delta)
        params_prop = params_from_mu_phi(mu_new, phi_new, beta0=beta0_new, beta1=beta1_new, theta=0.0)

        #validate restrictions (NIG ans esscher-theta)
        ok = True
        alpha_prop = float(params_prop["alpha"])
        delta_prop = float(params_prop["delta"])

        if not (np.isfinite(alpha_prop) and np.isfinite(delta_prop) and np.isfinite(beta0_new) and np.isfinite(beta1_new)):
            ok = False
        if delta_prop <= 0.0 or alpha_prop <= 0.0 or abs(beta1_new) >= alpha_prop:
            ok = False
        if alpha_prop < 0.5:
            ok = False
        bound = delta_prop * np.sqrt(max(2.0 * alpha_prop - 1.0, 0.0))
        if abs(beta0_new) > bound:
            ok = False

        # also require that θ_t is computable for all rf in-window
        if ok:
            try:
                _ = update_theta_series(
                    {"alpha": alpha_prop, "beta1": beta1_new, "delta": delta_prop, "beta0": beta0_new},
                    rf,
                )
            except Exception:
                ok = False

        if ok:
            mu, phi = float(mu_new), float(phi_new)
            beta0, beta1 = float(beta0_new), float(beta1_new)
            alpha, delta = float(alpha_prop), float(delta_prop)
        else:
            n_reject += 1

        # Store the draws and move to the next it
        if keep_mask[it]:

            A_draws[k_count, :] = A_path
            theta_draws[k_count, :] = theta_series
            z_draws[k_count, :] = z

            params_draws[k_count, :] = np.array([alpha, beta1, delta, beta0], dtype=float)
            mu_phi_mat[k_count, :] = np.array([mu, phi], dtype=float)

            k_count += 1

    return {
        "A_draws": A_draws,                 # (n_keep, n_days_in_window)
        "Esscher-theta_series_draws": theta_draws,  # (n_keep, n_days_in_window)
        "z_draws": z_draws,                 # (n_keep, n_days_in_window-1)
        "Theta_draws": params_draws,           # (n_keep, 4) -> [alpha,beta1,delta,beta0]
        "mu_phi_draws": mu_phi_mat,         # (n_keep, 2) -> [mu,phi]
        "priors": {"hyper": hyper, "b0": b0_prior, "B0": B0_prior},
        "meta": {
            "burn_in": burn_in,
            "thin": thin,
            "max_iter": max_iter,
            "n_keep": n_keep,
            "n_reject": int(n_reject),
            "trading_days": trading_days,
            "pd_horizon_years": pd_horizon_years,
            "L_face_index": int(face_idx_full),
        },
    }














