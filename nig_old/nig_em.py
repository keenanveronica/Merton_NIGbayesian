from typing import Dict
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norminvgauss
import warnings
from nig_old.nig_base import get_asset_path, update_theta


def _nig_negloglik_daily(
    r,
    alpha,
    beta,
    delta_annual,
    mu_annual,
    *,
    daycount: int = 250,
    penalty: float = 1e50,
    normalize: bool = False,   # if True: return -mean(ll) instead of -sum(ll)
):
    """
    Negative log-likelihood for daily log-returns under NIG with annual params scaled by h=1/daycount.
    This implements the paper’s idea that h converts annual parameter values into daily ones. :contentReference[oaicite:1]{index=1}
    """

    h = 1.0 / float(daycount)

    # parameter feasibility (SciPy norminvgauss requires a>0 and |b|<a;
    # with the mapping a=alpha*scale and b=beta*scale this reduces to alpha>0, delta>0, |beta|<alpha)
    if (alpha <= 0.0) or (delta_annual <= 0.0) or (abs(beta) >= alpha):
        return float(penalty)

    r = np.asarray(r, dtype=float).reshape(-1)
    if r.size == 0 or (not np.all(np.isfinite(r))):
        return float(penalty)

    # daily scaling consistent with the rest of your codebase:
    # T = h, so delta_T = delta*h and mu_T = mu*h (paper uses h to convert annual->daily) :contentReference[oaicite:2]{index=2}
    loc = float(mu_annual) * h
    scale = float(delta_annual) * h
    if not np.isfinite(scale) or scale <= 0.0:
        return float(penalty)

    # guard against underflow to exactly 0
    scale = max(scale, 1e-12)

    a = float(alpha) * scale
    b = float(beta) * scale

    ll = norminvgauss.logpdf(r, a=a, b=b, loc=loc, scale=scale)
    if not np.all(np.isfinite(ll)):
        return float(penalty)

    nll = -float(np.sum(ll))
    if normalize:
        nll /= float(r.size)

    return nll


def _fit_nig_mle_daily(
    r,
    x0,
    *,
    daycount: int = 250,
    min_obs: int = 50,
    dropna: bool = True,
    normalize_obj: bool = False,     # if True: objective uses mean NLL for comparability across window lengths
    maxiter: int = 400,
    warn: bool = True,
):
    """
    M-step: MLE update of (alpha, beta, delta, mu) given daily asset log-returns r.

    Robustifications for A+C(+triggers):
      - handles NaNs in r (drop or fail)
      - uses paper-style feasibility bounds and constrained optimisation behaviour :contentReference[oaicite:2]{index=2}
      - returns old params on failure (EM-friendly)
    """
    alpha0, beta0, delta0, mu0 = map(float, x0)

    # clean returns (important now that E-step can yield NaNs)
    r = np.asarray(r, dtype=float).reshape(-1)
    if dropna:
        finite = np.isfinite(r)
        if warn and (finite.size > 0) and (np.count_nonzero(~finite) > 0):
            warnings.warn(
                f"_fit_nig_mle_daily: dropping {np.count_nonzero(~finite)} non-finite return(s).",
                RuntimeWarning,
                stacklevel=2,
            )
        r = r[finite]
    else:
        if not np.all(np.isfinite(r)):
            return alpha0, beta0, delta0, mu0

    if r.size < min_obs:
        if warn:
            warnings.warn(
                f"_fit_nig_mle_daily: too few observations after cleaning (n={r.size} < {min_obs}). "
                "Returning previous params.",
                RuntimeWarning,
                stacklevel=2,
            )
        return alpha0, beta0, delta0, mu0

    # lower/upper bounds (Ahcan and Jovan)
    alpha_lb = 0.51
    alpha_ub = 1000.0
    delta_lb = 1e-9
    delta_ub = 1000.0
    mu_bound = 1000.0            # keeps mu in [-1000,1000] as in paper :contentReference[oaicite:4]{index=4}
    eps = 1e-8

    # Reparam:
    # alpha = alpha_lb + exp(a_raw)  with a_raw bounded so alpha <= alpha_ub
    # delta = delta_lb + exp(log_delta) with log_delta bounded so delta <= delta_ub
    # beta = (alpha - eps) * tanh(b_raw)  -> guarantees |beta| < alpha
    # mu = mu_bound * tanh(mu_raw / mu_bound) -> keeps mu in [-mu_bound, mu_bound]
    def unpack(u):
        a_raw, b_raw, log_delta, mu_raw = map(float, u)
        alpha = alpha_lb + np.exp(a_raw)
        delta = delta_lb + np.exp(log_delta)
        beta = (alpha - eps) * np.tanh(b_raw)
        mu = mu_bound * np.tanh(mu_raw / mu_bound)
        return float(alpha), float(beta), float(delta), float(mu)

    def obj(u):
        alpha, beta, delta, mu = unpack(u)
        return _nig_negloglik_daily(
            r, alpha, beta, delta, mu,
            daycount=daycount,
            normalize=normalize_obj,
        )

    # initialise u0 from x0
    u0 = np.array([
        np.log(max(alpha0 - alpha_lb, 1e-6)),
        np.arctanh(np.clip(beta0 / max(alpha0, 1e-6), -0.999, 0.999)),
        np.log(max(delta0 - delta_lb, 1e-12)),
        mu0,
    ], dtype=float)

    # L-BFGS-B bounds on raw variables (prevents exp overflow / extreme tanh saturation)
    a_ub_raw = np.log(max(alpha_ub - alpha_lb, 1e-12))
    d_ub_raw = np.log(max(delta_ub - delta_lb, 1e-12))
    bounds = [
        (-30.0, a_ub_raw),   # a_raw
        (-20.0, 20.0),       # b_raw
        (-30.0, d_ub_raw),   # log_delta
        (-2e4, 2e4),         # mu_raw (avoids numeric weirdness)
    ]

    res = minimize(
        obj,
        u0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter)},
    )

    if not res.success:
        # fallback: return old params
        if warn:
            warnings.warn(
                f"_fit_nig_mle_daily: optimizer failed ({res.message}). Returning previous params.",
                RuntimeWarning,
                stacklevel=2,
            )
        return alpha0, beta0, delta0, mu0

    return unpack(res.x)


# EM initialization (paper)
def EM_algo(
    E_series: np.ndarray,
    L_face_series: np.ndarray,   # liabilities proxy (strike), time-varying
    rf_series: np.ndarray,
    dates: np.ndarray,
    start_params: Dict[str, float],
    start_date=None,
    end_date=None,
    max_iter: int = 10,
    min_iter: int = 3,
    tol: float = 1e-3,
    *,
    # calibrate on 1Y-consistent state
    tau_mode: str = "one_year",            # "one_year" vs "paper"
    liability_mode: str = "timevarying",   # "timevarying" vs "paper_const_end"
    daycount: int = 250,
    min_obs: int = 50,                     # minimum finite return obs for M-step
) -> Dict[str, object]:
    """
    EM calibration of NIG parameters on a date slice.

    Practical behaviour in this refactor:
      - E-step: infer implied asset path via get_asset_path(...).
      - M-step: fit NIG params on finite daily log-returns of inferred assets.
      - Uses update_theta(...) directly (vectorized) for feasibility/pricing checks.
      - Rejects parameter updates that yield infeasible theta on the window.
    """

    E_series = np.asarray(E_series, dtype=float)
    L_face_series = np.asarray(L_face_series, dtype=float)
    rf_series = np.asarray(rf_series, dtype=float)
    dates = np.asarray(dates)

    if not (len(E_series) == len(L_face_series) == len(rf_series) == len(dates)):
        raise ValueError("E_series, L_face_series, rf_series, dates must have the same length")

    # training-window mask
    mask = np.ones(len(dates), dtype=bool)
    if start_date is not None:
        mask &= (dates >= start_date)
    if end_date is not None:
        mask &= (dates <= end_date)

    idx = np.where(mask)[0]
    if idx.size < 5:
        raise ValueError("Training window too short after applying start_date/end_date")

    # current params
    alpha = float(start_params["alpha"])
    beta1 = float(start_params["beta1"])
    delta = float(start_params["delta"])
    beta0 = float(start_params["beta0"])

    converged = False
    diff_last = None
    it = -1

    for it in range(max_iter):
        # ---------- E-step ----------
        A_path = get_asset_path(
            params={"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0},
            rf_series=rf_series,
            dates=dates,
            E_series=E_series,
            L_face_series=L_face_series,
            start_date=start_date,
            end_date=end_date,
            daycount=daycount,
            tau_mode=tau_mode,
            liability_mode=liability_mode,
            warm_start=True,
            warn=False,
        )

        # implied daily log-returns (drop non-finite)
        logA = np.log(A_path)
        rA = np.diff(logA)
        rA = rA[np.isfinite(rA)]

        if rA.size < min_obs:
            raise ValueError(
                f"E-step produced too few finite asset returns for M-step (n={rA.size} < {min_obs})."
            )

        # ---------- M-step ----------
        a_new, b_new, d_new, mu_new = _fit_nig_mle_daily(
            rA, (alpha, beta1, delta, beta0), daycount=daycount
        )

        # ---------- feasibility checks for proposed params ----------
        update_accepted = True

        # Compute theta on full firm series (vectorized), then check only the window slice.
        # Use enforce_pricing=False here because we handle the Eq(26) constraints explicitly below.
        theta_new_full = update_theta(
            {"alpha": a_new, "beta1": b_new, "delta": d_new, "beta0": mu_new},
            rf_series,
            enforce_pricing=False,
            warn=False,
        )
        theta_new_win = theta_new_full[mask]

        # If theta is not finite across the window, reject the update
        if not np.all(np.isfinite(theta_new_win)):
            update_accepted = False

        # Ensure Eq(26) pricing feasibility across the window:
        # need |beta+theta|<alpha and |beta+theta+1|<alpha for all t in the window.
        # If alpha is too low, bump it and recompute theta.
        # --- candidate E-step sanity check (prevents "n=0" in next iter) ---
        if update_accepted:
            alpha_floor = max(
                float(np.max(np.abs(b_new + theta_new_win))),
                float(np.max(np.abs(b_new + theta_new_win + 1.0))),
            ) + 1e-6

            if a_new < alpha_floor:
                a_new = float(alpha_floor)
                theta_new_full = update_theta(
                    {"alpha": a_new, "beta1": b_new, "delta": d_new, "beta0": mu_new},
                    rf_series,
                    enforce_pricing=False,
                    warn=False,
                )
                theta_new_win = theta_new_full[mask]
                if not np.all(np.isfinite(theta_new_win)):
                    update_accepted = False

        # --- candidate E-step sanity check (do this LAST, on the FINAL candidate params) ---
        if update_accepted:
            dates_win = dates[mask]
            test_len = max(min_obs + 5, 100)            # e.g. 100 days
            test_start = dates_win[-test_len] if len(dates_win) > test_len else dates_win[0]

            A_test = get_asset_path(
                params={"alpha": a_new, "beta1": b_new, "delta": d_new, "beta0": mu_new},
                rf_series=rf_series,
                dates=dates,
                E_series=E_series,
                L_face_series=L_face_series,
                start_date=test_start,                  # << only last chunk
                end_date=end_date,
                tau_mode=tau_mode,
                liability_mode=liability_mode,
                daycount=daycount,
                warm_start=True,
                warn=False,
            )

            # adjacency-preserving return construction (better than diff(log(A)) then filter)
            with np.errstate(divide="ignore", invalid="ignore"):
                logA = np.log(A_test)
            rA_test = logA[1:] - logA[:-1]
            ok_ret = np.isfinite(logA[1:]) & np.isfinite(logA[:-1])
            rA_test = rA_test[ok_ret]

            if rA_test.size < min_obs:
                update_accepted = False

        # If rejected, keep old params and force diff_last large so we don't "false converge"
        if not update_accepted:
            a_new, b_new, d_new, mu_new = alpha, beta1, delta, beta0
            diff_last = np.array([np.inf, np.inf, np.inf, np.inf], dtype=float)
        else:
            diff_last = np.array(
                [abs(a_new - alpha), abs(b_new - beta1), abs(d_new - delta), abs(mu_new - beta0)],
                dtype=float
            )
            alpha, beta1, delta, beta0 = float(a_new), float(b_new), float(d_new), float(mu_new)

        if (it + 1) >= min_iter and np.all(diff_last < tol):
            converged = True
            break

    # ---------- final outputs on the window ----------
    dates_win = dates[mask]

    theta_full = update_theta(
        {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0},
        rf_series,
        enforce_pricing=False,
        warn=False,
    )
    theta_win = theta_full[mask]

    A_win = get_asset_path(
        params={"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0},
        rf_series=rf_series,
        dates=dates,
        E_series=E_series,
        L_face_series=L_face_series,
        start_date=start_date,
        end_date=end_date,
        daycount=daycount,
        tau_mode=tau_mode,
        liability_mode=liability_mode,
        warm_start=True,
        warn=False,
    )

    return {
        "params": {"alpha": alpha, "beta1": beta1, "delta": delta, "beta0": beta0},
        "converged": converged,
        "n_iter": it + 1,
        "dates_win": dates_win,
        "idx_win": idx,
        "A_win": A_win,
        "theta_win": theta_win,
        "diff_last": diff_last,
        "nA_finite": int(np.count_nonzero(np.isfinite(A_win))),
    }
