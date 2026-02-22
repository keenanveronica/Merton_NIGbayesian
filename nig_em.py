from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.special import kve
from nig_apath import NIGParams, solve_esscher_theta



# Helpers: model moments + projections

def nig_var_one_step(p: NIGParams) -> float:
    """
    One-step variance of Z ~ NIG(alpha,beta,delta,mu) under the "common" parameterization:
      Var(Z) = delta * alpha^2 / (alpha^2 - beta^2)^(3/2)
    """
    p.validate()
    denom = max(p.alpha * p.alpha - p.beta * p.beta, 1e-18)
    return p.delta * (p.alpha * p.alpha) / (denom ** 1.5)


def nig_mean_one_step(p: NIGParams) -> float:
    """
    One-step mean:
      E[Z] = mu + delta * beta / sqrt(alpha^2 - beta^2)
    """
    p.validate()
    denom = max(p.alpha * p.alpha - p.beta * p.beta, 1e-18)
    return float(p.mu + p.delta * p.beta / np.sqrt(denom))


def project_delta_to_sample_var(p: NIGParams, sample_var: float) -> NIGParams:
    """
    Hold alpha,beta,mu fixed; choose delta so that model one-step variance matches sample_var.
    """
    p.validate()
    denom = max(p.alpha * p.alpha - p.beta * p.beta, 1e-18)
    delta_star = float(sample_var) * (denom ** 1.5) / (p.alpha * p.alpha)
    delta_star = max(delta_star, 1e-18)
    return NIGParams(alpha=p.alpha, beta=p.beta, delta=delta_star, mu=p.mu)


def project_mu_to_sample_mean(p: NIGParams, sample_mean: float) -> NIGParams:
    """
    After changing alpha/beta/delta, re-set mu to match the sample mean.
    Using: sample_mean = mu + delta*beta/sqrt(alpha^2-beta^2)
    """
    p.validate()
    denom = max(p.alpha * p.alpha - p.beta * p.beta, 1e-18)
    mu_star = float(sample_mean) - float(p.delta * p.beta / np.sqrt(denom))
    return NIGParams(alpha=p.alpha, beta=p.beta, delta=p.delta, mu=mu_star)



# Theta-aware feasibility (computed inside the loop)

def _compute_theta_window(
    p: NIGParams,
    r_win: np.ndarray,
    *,
    tau_theta: float = 1.0,
) -> np.ndarray:
    """
    Compute Esscher theta for each r in the window with current params p.
    If solve fails for a date, returns NaN for that entry (we ignore NaNs downstream).
    """
    r_win = np.asarray(r_win, float)
    out = np.full(r_win.shape, np.nan, dtype=float)
    for i, r in enumerate(r_win):
        if not np.isfinite(r):
            continue
        try:
            out[i] = float(solve_esscher_theta(p, float(r), float(tau_theta)))
        except Exception:
            out[i] = np.nan
    return out


def enforce_theta_feasible_on_window(p: NIGParams, theta_win: np.ndarray, *, eps: float = 1e-6) -> NIGParams:
    """
    Paper-style feasibility on a theta window:
      need |beta + theta_t| < alpha and |beta + theta_t + 1| < alpha  for all t.
    If theta_win is empty/all-NaN, fallback to alpha > |beta| + 1.

    Note: This changes alpha only (keeps beta,delta,mu).
    """
    p.validate()
    theta_win = np.asarray(theta_win, float)
    theta_win = theta_win[np.isfinite(theta_win)]

    if theta_win.size == 0:
        a_min = abs(p.beta) + 1.0 + eps
        if p.alpha < a_min:
            return NIGParams(alpha=a_min, beta=p.beta, delta=p.delta, mu=p.mu)
        return p

    alpha_floor = max(
        float(np.max(np.abs(p.beta + theta_win))),
        float(np.max(np.abs(p.beta + theta_win + 1.0))),
    ) + eps

    if p.alpha < alpha_floor:
        return NIGParams(alpha=alpha_floor, beta=p.beta, delta=p.delta, mu=p.mu)

    return p


# EM core (NIG via GIG latent variable moments)
def _safe_ratio_kve(nu_num, nu_den, x):
    """
    ratio K_{nu_num}(x)/K_{nu_den}(x) using exponentially-scaled kve.
    Includes guards to prevent NaNs/inf from propagating.
    """
    x = np.asarray(x, float)
    num = kve(nu_num, x)
    den = kve(nu_den, x)
    ok = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out = np.ones_like(x, dtype=float)
    out[ok] = (num[ok] / den[ok]).astype(float)
    return out


def em_fit_nig(
    x: np.ndarray,
    p_start: NIGParams,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
    min_alpha_gap: float = 1e-6,
    psi_floor: float = 1e-6,
    verbose: bool = False,
) -> NIGParams:
    """
    EM for NIG on 1D array x (weekly log-asset returns).
    Returns fitted NIGParams (alpha,beta,delta,mu) in the same time unit as x.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError("Too few observations for EM.")

    p = NIGParams(**vars(p_start))
    p.validate()
    n = x.size

    for it in range(max_iter):
        # derived quantity
        psi = max(p.alpha * p.alpha - p.beta * p.beta, psi_floor)
        gamma = np.sqrt(psi)

        # E-step: Y|X ~ GIG(lambda=-1/2, chi=delta^2+(x-mu)^2, psi=gamma^2)
        chi = p.delta * p.delta + (x - p.mu) ** 2
        chi = np.maximum(chi, 1e-18)
        s = np.sqrt(chi * psi)
        s = np.maximum(s, 1e-12)

        lam = -0.5
        ratio_p1 = _safe_ratio_kve(lam + 1.0, lam, s)   # K_{+1/2}/K_{-1/2}
        ratio_m1 = _safe_ratio_kve(lam - 1.0, lam, s)   # K_{-3/2}/K_{-1/2}

        Ey = ratio_p1 * np.sqrt(chi / psi)
        E1y = ratio_m1 * np.sqrt(psi / chi)

        # M-step for mu, beta
        W = np.sum(E1y)
        A = np.sum(E1y * x)
        X = np.sum(x)
        Y = np.sum(Ey)

        denom = (n * n / max(Y, 1e-18)) - W
        if abs(denom) < 1e-18:
            denom = 1e-18

        mu_new = (n * X / max(Y, 1e-18) - A) / denom
        beta_new = (X - n * mu_new) / max(Y, 1e-18)

        # M-step for delta, gamma
        S1 = np.sum(E1y)
        den_delta = S1 - (n * n / max(Y, 1e-18))
        if den_delta <= 1e-18:
            den_delta = 1e-18

        delta_new = np.sqrt(n / den_delta)
        gamma_new = (n * delta_new) / max(Y, 1e-18)

        alpha_new = np.sqrt(beta_new * beta_new + gamma_new * gamma_new)

        # enforce alpha > |beta|
        if alpha_new <= abs(beta_new) + min_alpha_gap:
            alpha_new = abs(beta_new) + min_alpha_gap + 1e-6

        # clamp beta slightly inside alpha (numerical)
        rho = 0.98
        if abs(beta_new) > rho * alpha_new:
            beta_new = np.sign(beta_new) * rho * alpha_new

        vec_old = np.array([p.alpha, p.beta, p.delta, p.mu], float)
        vec_new = np.array([alpha_new, beta_new, delta_new, mu_new], float)
        rel = float(np.max(np.abs(vec_new - vec_old) / (np.abs(vec_old) + 1e-8)))

        p = NIGParams(alpha=float(alpha_new), beta=float(beta_new), delta=float(delta_new), mu=float(mu_new))

        if verbose:
            print(f"[EM] iter={it:03d} rel={rel:.2e} a={p.alpha:.4f} b={p.beta:.6f} d={p.delta:.6e} m={p.mu:.6f}")

        if rel < tol:
            break

    p.validate()
    return p


# Main API: rolling refits from weekly assets
def _alpha_required_from_theta(p: NIGParams, theta_win: np.ndarray, *, eps: float = 1e-6) -> float:
    """
    Minimum alpha required so that mgf exists for u=theta and u=theta+1 across the window:
        |beta + theta_t| < alpha
        |beta + theta_t + 1| < alpha
    """
    theta_win = np.asarray(theta_win, float)
    theta_win = theta_win[np.isfinite(theta_win)]
    if theta_win.size == 0:
        return abs(p.beta) + 1.0 + eps
    return max(
        float(np.max(np.abs(p.beta + theta_win))),
        float(np.max(np.abs(p.beta + theta_win + 1.0))),
    ) + eps


def fit_nig_params_from_weekly_assets(
    assets_weekly: pd.DataFrame,
    *,
    p0: NIGParams,
    window_weeks: int = 104,
    refit_every: int = 13,         # 13=quarterly, 4=monthly
    em_max_iter: int = 80,
    em_tol: float = 1e-6,
    r_col: str = "r",
    tau_theta: float = 1.0,
    alpha_min: float = 1.25,
    use_theta: bool = True,
    # IMPORTANT: if theta column exists, we use it; only fallback recompute if missing/empty
    prefer_precomputed_theta: bool = True,
    # fixed-point iterations only apply when recomputing theta from r
    theta_iters: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Rolling refit of NIG params on weekly asset log returns (dlogA).

    Input assets_weekly should (ideally) come from nig_apath weekly inversion and contain:
      - date
      - dlogA (weekly log-asset returns)
      - theta (Esscher theta used during inversion)  <-- preferred for feasibility
      - r (optional, only needed if we must recompute theta)

    Output: one row per refit date (parameter updates + diagnostics).
    """
    df = assets_weekly.sort_values("date").copy()

    if "dlogA" not in df.columns:
        raise ValueError("assets_weekly must contain column 'dlogA'.")
    df = df.dropna(subset=["dlogA"]).reset_index(drop=True)

    has_theta_col = "theta" in df.columns
    if use_theta and (not has_theta_col) and (r_col not in df.columns):
        raise ValueError(
            f"use_theta=True but assets_weekly has neither 'theta' nor '{r_col}'. "
            "Provide theta from inversion or provide r to recompute theta."
        )

    updates = []
    p_prev = p0

    for end_ix in range(window_weeks, len(df) + 1, refit_every):
        w = df.iloc[end_ix - window_weeks:end_ix]
        end_date = pd.to_datetime(w["date"].iloc[-1])

        x = w["dlogA"].to_numpy(float)
        x = x[np.isfinite(x)]
        if x.size < 10:
            continue

        emp_mean = float(np.mean(x))
        emp_var = float(np.var(x, ddof=1))
        emp_std = float(np.sqrt(emp_var))

        # 1) EM fit (warm start)
        p_hat = em_fit_nig(x, p_prev, max_iter=em_max_iter, tol=em_tol, verbose=False)

        # 2) Project to empirical moments (stabilizes scale/drift)
        #    (do this early so theta feasibility checks use sensible scale)
        if p_hat.alpha < alpha_min:
            p_hat = NIGParams(alpha=alpha_min, beta=p_hat.beta, delta=p_hat.delta, mu=p_hat.mu)
        p_hat = project_delta_to_sample_var(p_hat, emp_var)
        p_hat = project_mu_to_sample_mean(p_hat, emp_mean)

        alpha_required = np.nan

        # 3) Theta-aware feasibility
        if use_theta:
            # Prefer the *precomputed theta* used during inversion (same theta as your asset path)
            theta_win = None
            if prefer_precomputed_theta and has_theta_col:
                theta_win = w["theta"].to_numpy(float)
                # if theta is almost all NaN, fallback to recompute
                finite_share = np.isfinite(theta_win).mean() if theta_win.size else 0.0
                if finite_share < 0.5:
                    theta_win = None

            if theta_win is not None:
                alpha_required = _alpha_required_from_theta(p_hat, theta_win)
                alpha_target = max(alpha_min, float(alpha_required))
                if p_hat.alpha < alpha_target:
                    p_hat = NIGParams(alpha=alpha_target, beta=p_hat.beta, delta=p_hat.delta, mu=p_hat.mu)
                    # alpha changed -> reproject moments
                    p_hat = project_delta_to_sample_var(p_hat, emp_var)
                    p_hat = project_mu_to_sample_mean(p_hat, emp_mean)
            else:
                # Fallback: recompute theta from r (paper-style fixed point)
                r_win = w[r_col].to_numpy(float)
                for _ in range(max(1, int(theta_iters))):
                    theta_win2 = _compute_theta_window(p_hat, r_win, tau_theta=tau_theta)
                    alpha_required = _alpha_required_from_theta(p_hat, theta_win2)

                    alpha_target = max(alpha_min, float(alpha_required))
                    if p_hat.alpha < alpha_target:
                        p_hat = NIGParams(alpha=alpha_target, beta=p_hat.beta, delta=p_hat.delta, mu=p_hat.mu)
                        p_hat = project_delta_to_sample_var(p_hat, emp_var)
                        p_hat = project_mu_to_sample_mean(p_hat, emp_mean)

        mod_std = float(np.sqrt(nig_var_one_step(p_hat)))
        mod_mean = float(nig_mean_one_step(p_hat))

        updates.append({
            "date": end_date,
            "alpha": p_hat.alpha,
            "beta": p_hat.beta,
            "delta": p_hat.delta,
            "mu": p_hat.mu,
            "emp_mean": emp_mean,
            "mod_mean": mod_mean,
            "emp_std": emp_std,
            "mod_std": mod_std,
            "alpha_required_from_theta": alpha_required,
            "alpha_hits_floor": bool(np.isfinite(alpha_required) and (p_hat.alpha <= max(alpha_min, float(alpha_required)) + 1e-12)),
            "window_len": int(len(w)),
        })

        if verbose:
            msg = (
                f"{end_date.date()} | emp_std={emp_std:.6f} mod_std={mod_std:.6f} | "
                f"emp_mean={emp_mean:+.6f} mod_mean={mod_mean:+.6f} | "
                f"a={p_hat.alpha:.3f} b={p_hat.beta:+.5f} d={p_hat.delta:.6e} m={p_hat.mu:+.6f}"
            )
            if use_theta:
                msg += f" | a_req(theta)={alpha_required if np.isfinite(alpha_required) else np.nan:.3f}"
            print(msg)

        p_prev = p_hat

    return pd.DataFrame(updates)
