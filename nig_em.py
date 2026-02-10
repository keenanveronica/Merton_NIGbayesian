# ------------------------------------------------------------
# nig_em.py (commented)
#
# What this script does
# - Provides an EM-style (iterative) initialization routine for the NIG structural model.
# - The routine alternates between:
#     E-step: recover an implied asset path by inverting the NIG call pricing equation
#     M-step: fit NIG parameters to the implied asset log-returns via MLE
# - It also updates theta (risk-neutral tilt) consistently with the current parameter estimates.
#
# Output is meant to be used as a stable starting point / centering point for the Bayesian Gibbs sampler.
# ------------------------------------------------------------

import numpy as np
from typing import Callable, Dict, Tuple

from scipy.optimize import minimize
from scipy.stats import norminvgauss

from nig_initialization import (
    invert_nig_call_price,
    update_theta,
)

# Negative log-likelihood for NIG returns.
# The paper/proposal uses annual parameters but daily increments:
#   delta_h = delta_annual * h
#   mu_h    = mu_annual * h
# and then evaluates the NIG logpdf on daily returns x.


def _nig_negloglik_returns(
    x: np.ndarray, alpha: float, beta: float, delta_annual: float, mu_annual: float, h: float
) -> float:
    # Feasibility
    if alpha <= 0.0 or delta_annual <= 0.0 or abs(beta) >= alpha or h <= 0.0:
        return np.inf

    # Convert annual to daily increment parameters (Jovan: delta*h, mu*h)
    delta_h = delta_annual * h
    mu_h = mu_annual * h

    ll = np.sum(norminvgauss.logpdf(x, alpha, beta, loc=mu_h, scale=delta_h))
    return float(-ll)

# Fit NIG parameters by MLE (L-BFGS-B) given implied asset log-returns.
# Returns updated (alpha, beta, delta_annual, mu_annual) and enforces |beta| < alpha.


def _fit_nig_mle(
    returns: np.ndarray,
    x0: Tuple[float, float, float, float],
    bounds: Tuple[Tuple[float, float], ...],
    h: float,
) -> Tuple[float, float, float, float]:

    def obj(p: np.ndarray) -> float:
        a, b, d, m = p
        return _nig_negloglik_returns(returns, a, b, d, m, h)

    res = minimize(obj, x0=np.array(x0, dtype=float), method="L-BFGS-B", bounds=bounds)
    a, b, d, m = res.x
    if abs(b) >= a:
        b = np.sign(b) * (a - 1e-6)
    return float(a), float(b), float(d), float(m)

# EM-style initialization for NIG structural parameters.
# Inputs:
#   equity: observed equity series E_t
#   liabilities_L: strike / liabilities (constant L)
#   maturity_T: initial maturity used in pricing (years)
#   rf: risk-free rate series aligned with equity (annualized)
#   inverter: function to invert the NIG call price (usually invert_nig_call_price)
#   start_params: initial guess for (alpha, beta1, delta, beta0)
# Loop:
#   - Build an implied asset path A_t by inverting the call price each day
#   - Compute returns r_t = log(A_t/A_{t-1})
#   - Re-fit NIG parameters to {r_t} via MLE
#   - Update theta and iterate until convergence (after min_iter) or max_iter reached


def em_init_nig_params(
    equity: np.ndarray,
    liabilities_L: float,
    maturity_T: float,
    rf: np.ndarray,
    inverter: Callable[[float, float, float, float, Dict[str, float], bool], float],
    start_params: Dict[str, float],
    min_iter: int = 3,
    max_iter: int = 10,
    tol: float = 1e-3,
) -> Dict[str, object]:

    E = np.asarray(equity, dtype=float)
    if np.any(E <= 0.0):
        raise ValueError("All equity observations must be positive.")

    rf = np.asarray(rf, dtype=float)
    if rf.shape != E.shape:
        raise ValueError("rf must have same length as equity")

    # Jovan & Ahčan daily step (years)
    # Daily time step in years (trading-day convention)
    h = 1.0 / 250.0

    # Read starting parameters (annual)
    alpha = float(start_params["alpha"])
    beta1 = float(start_params["beta1"])
    delta = float(start_params["delta"])
    beta0 = float(start_params["beta0"])

    # Initialize theta using current parameters and the (possibly time-varying) risk-free rate series
    theta = update_theta({"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0}, rf)

    bounds = (
        (0.51, 1000.0),     # alpha
        (-1000.0, 1000.0),  # beta
        (1e-9, 1000.0),     # delta_annual
        (-1000.0, 1000.0),  # mu_annual (beta0)
    )

    n = len(E)
    asset_path = np.full(n, np.nan)
    converged = False

    # Outer loop: iterate E-step / M-step updates
    for it in range(max_iter):
        current_params = {
            "alpha": alpha,
            "beta1": beta1,
            "delta": delta,
            "beta0": beta0,
            "theta": theta,
        }

        # E-step: infer asset value each day by inverting the NIG call price
        for t in range(n):
            # Remaining maturity decreases over the window (rolling maturity)
            T_rem = maturity_T - t*h
            # If r_f varies through time, compute a time-specific theta_t for that day
            theta_t = update_theta({"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0}, rf[t])
            params_t = {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0, "theta": theta_t}
            # Solve for A_t such that model equity price equals observed equity E_t
            asset_path[t] = inverter(E[t], liabilities_L, T_rem, rf[t], params_t)

        # Asset log-returns implied by the current parameter set
        r = np.diff(np.log(asset_path))

        x0 = (alpha, beta1, delta, beta0)
        # M-step: update NIG parameters by fitting the return distribution
        a_new, b_new, d_new, beta0_new = _fit_nig_mle(r, x0, bounds, h=h)

        theta_new = update_theta(
            {"alpha": a_new, "beta1": b_new, "delta": d_new, "beta0": beta0_new},
            rf,
        )

        diff = np.array([
            abs(a_new - alpha),
            abs(b_new - beta1),
            abs(d_new - delta),
            abs(beta0_new - beta0),
        ])
        # Convergence check (only after min_iter iterations)
        if (it + 1) >= min_iter and np.all(diff < tol):
            converged = True
            break

        # Accept the new parameters and continue the EM iterations
        alpha, beta1, delta, beta0, theta = a_new, b_new, d_new, beta0_new, theta_new

    return {
        "alpha": alpha,
        "beta1": beta1,
        "delta": delta,
        "beta0": beta0,
        "theta": theta,
        "asset_path": asset_path.copy(),
        "n_iter": it + 1,
        "converged": converged,
        "h": h,
    }
