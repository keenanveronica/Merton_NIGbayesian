# ------------------------------------------------------------
# nig_initialization.py (commented)
#
# What this script does
# - Implements helper utilities for the NIG structural credit-risk model
#   used throughout your pipeline (EM init + Gibbs sampler).
# - Key components:
#     1) Parameter transforms for the NIG distribution
#     2) Equity-as-call pricing under NIG log-returns (Jovan & Ahčan)
#     3) Numerical inversion: solve for asset value A given observed equity E
#     4) Risk-neutral tilt (theta) in closed form (Esscher / exponential tilting)
#     5) Physical and risk-neutral PD computations via NIG CDF
#
# Important design choice in this file
# - The call-price formula uses two NIG CDFs with different beta shifts:
#     beta_plus  = beta1 + theta + 1
#     beta_minus = beta1 + theta
#   (this corresponds to the pricing identity in the paper/proposal).
# ------------------------------------------------------------

import numpy as np
from typing import Tuple, Dict

from scipy.stats import norminvgauss
from scipy.optimize import brentq



# Compute the NIG 'gamma' parameter: gamma = sqrt(alpha^2 - beta^2).
# In your notation beta is stored as beta1; alpha must exceed |beta1|.
def gamma_param(params: Dict[str, float]) -> float:
    alpha = params.get("alpha", 0.0)
    beta1 = params.get("beta1", 0.0)
    return np.sqrt(max(alpha * alpha - beta1 * beta1, 0.0))



# Convert (alpha, beta1, delta, beta0, theta) to the alternative (mu, phi) parameterization
# often used in the variance-mean mixture representation of NIG:
#   mu  = delta / gamma
#   phi = delta * gamma
def mu_phi_from_params(params: Dict[str, float]) -> Tuple[float, float]:
    delta = params.get("delta", 0.0)
    g = gamma_param(params)
    if g > 0.0:
        return delta / g, delta * g
    return float("inf"), float("inf")



# Convert back from (mu, phi, beta0, beta1, theta) to the parameter dict used in the rest of the code:
#   delta = sqrt(mu * phi)
#   gamma = sqrt(phi / mu)
#   alpha = sqrt(beta1^2 + gamma^2)
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



# NIG-based equity pricing (equity as a call option on assets).
# Inputs:
#   A: asset value (unknown in practice)
#   L: liabilities / default barrier (strike)
#   T: time to maturity (years)
#   r: risk-free rate (annualized, continuous comp.)
#   params: dict with alpha, beta1, delta, beta0, theta
# Output:
#   Model equity value E = C_NIG(A, L, T, r, params)
# Implementation detail:
#   Uses two NIG CDF evaluations at x0 = log(L/A) with different beta shifts (beta_plus, beta_minus).
def nig_call_price(
    A: float,
    L: float,
    T: float,
    r: float,
    params: Dict[str, float],
    #physical_measure: bool = True,
) -> float:
    if A <= 0.0 or L <= 0.0 or T <= 0.0:
        raise ValueError("A, L and T must be positive")
    # theta is the risk-neutral tilt; theta=0 corresponds to physical-measure beta shifts
    beta_shift = params.get("theta", 0.0)
    beta1 = params.get("beta1", 0.0)
    #if physical_measure:
    # beta_plus corresponds to the '+1' shift in the call-pricing identity (see proposal/paper)
    beta_plus = beta1 + beta_shift + 1.0
    # beta_minus is the baseline beta under the tilted measure
    beta_minus = beta1 + beta_shift
    """else:
        beta_plus = beta1 + beta_shift
        beta_minus = beta1 + beta_shift"""
    delta_T = params.get("delta", 0.0) * T
    loc_T = params.get("beta0", 0.0) * T
    # Pricing is written in terms of x0 = log(strike / asset)
    x0 = np.log(L / A)
    cdf_plus = norminvgauss.cdf(x0, params.get("alpha", 0.0), beta_plus, loc=loc_T, scale=delta_T)
    cdf_minus = norminvgauss.cdf(x0, params.get("alpha", 0.0), beta_minus, loc=loc_T, scale=delta_T)
    # Tail probabilities enter the call payoff expectation
    tail_plus = 1.0 - cdf_plus
    tail_minus = 1.0 - cdf_minus
    # Call price: E[A*1{...}] - PV(L)*P(...), matching the Black–Scholes-like structure
    call = A * tail_plus - L * np.exp(-r * T) * tail_minus
    return float(call)



# Numerical inversion: given observed equity E_obs, solve for asset value A such that
#   nig_call_price(A, ...) = E_obs.
# Uses a robust bracket-expansion + Brent root finder (brentq).
# If a valid bracket cannot be found, returns a safe fallback (E_obs + L).
def invert_nig_call_price(
    E_obs: float,
    L: float,
    T: float,
    r: float,
    params: Dict[str, float],
    physical_measure: bool = True,
    A_min_factor: float = 1e-3,
    A_max_factor: float = 10.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    if E_obs <= 0.0:
        raise ValueError("Observed equity must be positive")
    # Lower bracket: assets must be at least as large as equity (call value <= asset value)
    A_min = max(E_obs, A_min_factor * L)
    # Upper bracket: generous multiple of (equity + liabilities)
    A_max = A_max_factor * (E_obs + L)

    def f(A_val: float) -> float:
        return nig_call_price(A_val, L, T, r, params) - E_obs

    f_min = f(A_min)
    f_max = f(A_max)
    if f_min * f_max > 0.0:
        success = False
        for i in range(1, 15):
            factor = 2 ** i
            A_max_ext = A_max * factor
            A_min_ext = max(A_min / factor, 1e-12)
            f_min_ext = f(A_min_ext)
            f_max_ext = f(A_max_ext)
            if f_min_ext * f_max_ext <= 0.0:
                A_min, A_max = A_min_ext, A_max_ext
                f_min, f_max = f_min_ext, f_max_ext
                success = True
                break
        if not success:
            return float(E_obs + L)
    try:
        # Brent's method finds the root reliably as long as f(A_min)*f(A_max) <= 0
        sol = brentq(f, A_min, A_max, xtol=tol, maxiter=max_iter)
        return float(sol)
    except Exception:
        return float(E_obs + L)



# Closed-form update for the risk-neutral tilting parameter theta.
# This adjusts the physical NIG distribution to a risk-neutral one (consistent with the pricing equation).
# Includes feasibility checks:
#   - delta > 0, alpha > 0, |beta1| < alpha
#   - alpha >= 0.5 and |beta0| within the bound delta*sqrt(2*alpha - 1)
# Then applies the closed-form expression used in your proposal / Jovan & Ahčan.
def update_theta(params: Dict[str, float], r_f: float) -> float:
    alpha = float(params.get("alpha", 0.0))
    beta1 = float(params.get("beta1", 0.0))
    delta = float(params.get("delta", 0.0))
    beta0 = float(params.get("beta0", 0.0))

    # Basic feasibility checks (you said invalid draws should be rejected)
    if delta <= 0.0:
        raise ValueError("delta must be > 0 for theta closed-form")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0 for theta closed-form")
    if abs(beta1) >= alpha:
        raise ValueError("|beta1| must be < alpha (NIG requires gamma real)")

    # Jovan & Ahčan existence condition
    # theta exists provided alpha >= 0.5 and |mu_A| <= delta * sqrt(2*alpha - 1)
    if alpha < 0.5:
        raise ValueError("theta existence condition violated: alpha < 0.5")
    bound = delta * np.sqrt(2.0 * alpha - 1.0)
    if abs(beta0) > bound:
        raise ValueError("theta existence condition violated: |beta0| too large")

    # Closed-form theta (Proposal eq. 3; equivalent to Jovan eq. 24). 
    inside = 4.0 * (alpha ** 2) * (delta ** 2) * ((beta0 - r_f) ** 2) + (delta ** 2)
    if inside < 0.0:
        raise ValueError("theta closed-form invalid: negative value under sqrt")

    theta = -beta1 - 0.5 - ((beta0 - r_f) / (2.0 * delta)) * (np.sqrt(inside) - 1.0)
    return float(theta)


# Convenience wrapper: compute theta for each element of a risk-free rate time series.
def update_theta_series(params: Dict[str, float], r_f_series: np.ndarray) -> np.ndarray:
    r_f_series = np.asarray(r_f_series, dtype=float)
    theta = np.empty_like(r_f_series)
    for i, r in enumerate(r_f_series):
        # Apply the closed-form theta update pointwise across the rate series
        theta[i] = update_theta(params, float(r))
    return theta



# Internal helper for PD calculation under a specified NIG beta parameter.
# If log(A_T/A0) ~ NIG(alpha, beta, loc=beta0*T, scale=delta*T), then
#   PD = P(A_T <= L) = P(log(A_T/A0) <= log(L/A0))
#      = CDF_NIG( log(L/A0) ; alpha, beta, loc=beta0*T, scale=delta*T ).
def _compute_pd_with_beta(
    A0: float,
    L: float,
    T: float,
    alpha: float,
    beta: float,
    beta0: float,
    delta: float,
) -> float:
    """
    Internal helper: PD = P(log(A_T/A0) < log(L/A0)) when log-return ~ NIG(alpha, beta, loc=beta0*T, scale=delta*T).
    """
    if A0 <= 0.0 or L <= 0.0 or T <= 0.0:
        raise ValueError("A0, L and T must be positive")
    if delta <= 0.0:
        raise ValueError("delta must be > 0")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")
    if abs(beta) >= alpha:
        raise ValueError("|beta| must be < alpha for a valid NIG distribution")

    delta_T = delta * T
    loc_T = beta0 * T
    x_thresh = np.log(L / A0)
    pd = norminvgauss.cdf(x_thresh, alpha, beta, loc=loc_T, scale=delta_T)
    return float(pd)



# Physical-measure PD:
# uses beta = beta1 (no risk-neutral tilt).
def compute_pd_physical(A0: float, L: float, T: float, params: Dict[str, float]) -> float:
    """
    Physical-measure PD: uses beta = beta1.
    """
    alpha = float(params.get("alpha", 0.0))
    beta1 = float(params.get("beta1", 0.0))
    beta0 = float(params.get("beta0", 0.0))
    delta = float(params.get("delta", 0.0))

    return _compute_pd_with_beta(
        A0=A0, L=L, T=T,
        alpha=alpha, beta=beta1,
        beta0=beta0, delta=delta
    )



# Risk-neutral PD:
# uses beta = beta1 + theta (theta must already be present in params).
def compute_pd_risk_neutral(A0: float, L: float, T: float, params: Dict[str, float]) -> float:
    """
    Risk-neutral PD: uses beta = beta1 + theta.
    Assumes params['theta'] has already been computed (e.g., via update_theta(params, r_f)).
    """
    alpha = float(params.get("alpha", 0.0))
    beta1 = float(params.get("beta1", 0.0))
    theta = float(params.get("theta", 0.0))
    beta0 = float(params.get("beta0", 0.0))
    delta = float(params.get("delta", 0.0))

    return _compute_pd_with_beta(
        A0=A0, L=L, T=T,
        alpha=alpha, beta=(beta1 + theta),
        beta0=beta0, delta=delta
    )