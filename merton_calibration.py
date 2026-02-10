import numpy as np
from scipy.special import ndtr


def norm_cdf(x):
    return ndtr(np.asarray(x, dtype=float))


def merton_equity_from_assets(V, B, r, T, sigmaV):
    """
    Equity value as a European call on firm assets (no dividends).
    E = V N(d1) - B e^{-rT} N(d2)
    """
    eps = 1e-12
    sig_sqrtT = np.maximum(sigmaV * np.sqrt(T), eps)
    log_V_B = np.log(np.maximum(V, eps) / np.maximum(B, eps))
    d1 = (log_V_B + (r + 0.5 * sigmaV**2) * T) / sig_sqrtT
    d2 = d1 - sig_sqrtT
    Nd1 = norm_cdf(d1)
    Nd2 = norm_cdf(d2)
    E_model = V * Nd1 - B * np.exp(-r*T) * Nd2
    return E_model, d1, d2, Nd1


def _default_initial_guess(E, B, sigmaE):   # helper not exposed
    V0 = max(E + B, 1e-6)
    sigmaV0 = max(sigmaE * (E / max(E + B, 1e-12)), 1e-4)
    sigmaV0 = min(sigmaV0, 5.0)  # guardrail
    return V0, sigmaV0


def solve_merton_row(E, sigmaE, B, r, T, x0=None, tol=1e-10, maxfev=200):
    """
    Solve the 2 equation system (Merton) for a single row:
      (1) E_obs = V N(d1) - B e^{-rT} N(d2)
      (2) sigmaE_obs = N(d1) (V/E_obs) sigmaV

    Unknowns: V, sigmaV

    Uses SciPy root-finding if available. Falls back to a simple,
    stable fixed-point iteration otherwise.
    """
    # Basic validity
    if not (
        np.isfinite(E)
        and np.isfinite(sigmaE)
        and np.isfinite(B)
        and np.isfinite(r)
        and np.isfinite(T)
    ):
        return np.nan, np.nan, np.nan, np.nan, False, "nonfinite_input"
    if (E <= 0) or (sigmaE <= 0) or (B <= 0) or (T <= 0):
        return np.nan, np.nan, np.nan, np.nan, False, "invalid_sign"

    # initial guess
    if x0 is None:
        V0, sV0 = _default_initial_guess(E, B, sigmaE)
    else:
        V0, sV0 = x0
        V0 = max(V0, 1e-6)
        sV0 = min(max(sV0, 1e-4), 5.0)

    # try SciPy root-finding
    try:
        from scipy.optimize import root

        def fun(z):
            # Solve in log-space (positivity)
            V = np.exp(z[0])
            sV = np.exp(z[1])

            Emod, d1, d2, Nd1 = merton_equity_from_assets(V, B, r, T, sV)
            sigE_mod = Nd1 * (V / E) * sV

            # scale residuals
            f1 = (Emod - E) / max(E, 1e-12)
            f2 = (sigE_mod - sigmaE) / max(sigmaE, 1e-12)
            return np.array([f1, f2], dtype=float)

        z0 = np.array([np.log(V0), np.log(sV0)], dtype=float)
        sol = root(fun, z0, method="hybr", tol=tol, options={"maxfev": maxfev})

        if not sol.success:
            return np.nan, np.nan, np.nan, np.nan, False, sol.message

        V_hat = float(np.exp(sol.x[0]))
        sV_hat = float(np.exp(sol.x[1]))
        Emod, d1, d2, Nd1 = merton_equity_from_assets(V_hat, B, r, T, sV_hat)

        # check: equity call value must be positive
        if (
            (not np.isfinite(V_hat))
            or (not np.isfinite(sV_hat))
            or (V_hat <= 0)
            or (sV_hat <= 0)
        ):
            return np.nan, np.nan, np.nan, np.nan, False, "bad_solution"

        return V_hat, sV_hat, float(d1), float(d2), True, "ok"

    except Exception:
        # Fallback - fixed-point iteration
        V, sV = V0, sV0
        ok = False
        for _it in range(60):
            Emod, d1, d2, Nd1 = merton_equity_from_assets(V, B, r, T, sV)

            # update sigmaV from leverage equation: sigmaE = Nd1*(V/E)*sigmaV
            denom = max(Nd1 * (V / E), 1e-8)
            sV_new = sigmaE / denom
            sV_new = min(max(sV_new, 1e-4), 5.0)

            # update V by nudging to match equity value (damped)
            # damped correction using ratio E/Emod
            ratio = E / max(Emod, 1e-8)
            V_new = V * (0.7 + 0.3 * ratio)
            V_new = max(V_new, 1e-6)

            if (abs(sV_new - sV) < 1e-6 and
                    abs(V_new - V) / max(V, 1e-12) < 1e-6):
                ok = True
                V, sV = V_new, sV_new
                break

            V, sV = V_new, sV_new

        if not ok:
            return (
                np.nan, np.nan, np.nan, np.nan, False, "fallback_no_converge"
            )

        Emod, d1, d2, Nd1 = merton_equity_from_assets(V, B, r, T, sV)
        return float(V), float(sV), float(d1), float(d2), True, "ok_fallback"


def calibrate_merton_panel(
    df,
    B_col="B",
    E_col="E",
    sigmaE_col="sigma_E",
    r_col="r",
    T_col="T",
    warm_start=True,
    B_scale="auto",
    progress_every=20000,
):
    """
    Calibrate Merton row-by-row, warm-starting within
    each gvkey for speed/stability.
    Adds: V, sigma_V, d1, d2, success, msg, DD_rn, PD_rn

    B_scale: (since not explicit in dataset)
    if median(B/E) is tiny (<1e-4), multiplies B by 1e6
    (Compustat-style millions); else leaves B as is.
    """
    out = df.copy()
    out = out.sort_values(["gvkey", "date"]).reset_index(drop=True)

    # scale B into same units as E (checking if needed)
    B_raw = out[B_col].astype(float).values
    E_raw = out[E_col].astype(float).values
    mask = np.isfinite(B_raw) & np.isfinite(E_raw) & (B_raw > 0) & (E_raw > 0)
    if B_scale == "auto":
        med_ratio = (
            np.median(B_raw[mask] / E_raw[mask])
            if mask.any()
            else np.nan
        )
        # intuition: if B/E is ~1e-7, B is likely in millions
        scale = (
            1e6
            if (np.isfinite(med_ratio) and med_ratio < 1e-4)
            else 1.0
        )
    else:
        scale = float(B_scale)

    out["B_used"] = out[B_col].astype(float) * scale

    # containers
    V_hat = np.full(len(out), np.nan)
    sV_hat = np.full(len(out), np.nan)
    d1_hat = np.full(len(out), np.nan)
    d2_hat = np.full(len(out), np.nan)
    success = np.zeros(len(out), dtype=bool)
    msg = np.empty(len(out), dtype=object)

    # iterate firm-by-firm
    idx0 = 0
    for g, gdf in out.groupby("gvkey", sort=False):
        idxs = gdf.index.values
        x0 = None

        for k, i in enumerate(idxs):
            E = float(out.at[i, E_col])
            sE = float(out.at[i, sigmaE_col])
            B = float(out.at[i, "B_used"])
            r = float(out.at[i, r_col])
            T = float(out.at[i, T_col])

            if warm_start and (x0 is not None):
                # previous day solution as initial guess
                guess = x0
            else:
                guess = None

            V, sV, d1, d2, ok, m = solve_merton_row(E, sE, B, r, T, x0=guess)

            V_hat[i] = V
            sV_hat[i] = sV
            d1_hat[i] = d1
            d2_hat[i] = d2
            success[i] = ok
            msg[i] = m

            if ok:
                x0 = (V, sV)
            else:
                x0 = None  # reset if solver failed

        if progress_every and (idxs[-1] >= idx0 + progress_every):
            idx0 = idxs[-1]

    out["V"] = V_hat
    out["sigma_V"] = sV_hat
    out["d1"] = d1_hat
    out["d2"] = d2_hat
    out["solver_success"] = success
    out["solver_msg"] = msg
    out["B_scale_used"] = scale

    # Q measure DD and PD (mu = r) - note: approximately Q measure
    sqrtT = np.sqrt(out[T_col].astype(float).values)
    Vv = out["V"].astype(float).values
    Bb = out["B_used"].astype(float).values
    rv = out[r_col].astype(float).values
    sVv = out["sigma_V"].astype(float).values

    # DD under GBM drift mu=r (risk-neutral DD)
    T_vals = out[T_col].astype(float).values
    numerator = (
        np.log(np.maximum(Vv, 1e-12) / np.maximum(Bb, 1e-12))
        + (rv - 0.5 * sVv**2) * T_vals
    )
    denominator = np.maximum(sVv * sqrtT, 1e-12)
    DD_rn = numerator / denominator
    out["DD_rn"] = DD_rn
    out["PD_rn"] = norm_cdf(-DD_rn)

    return out


def add_physical_pd_from_implied_assets(
    calibrated, window=252, min_periods=126
):
    """
    Optional: estimate physical drift mu_V from implied asset series,
    then compute DD_P and PD_P.
    For GBM: E[Δlog V] ≈ (mu - 0.5 sigma^2) / 252
    =>  mu ≈ mean(ΔlogV)*252 + 0.5 sigma^2
    """
    df = calibrated.sort_values(["gvkey", "date"]).copy()
    df["logV"] = np.log(df["V"])
    df["dlogV"] = df.groupby("gvkey")["logV"].diff()

    # rolling mean of dlogV (daily), then convert to annual mu_V
    mu_minus_half_sig2 = (
        df.groupby("gvkey")["dlogV"]
        .transform(
            lambda s: s.rolling(
                window, min_periods=min_periods
            ).mean() * 252.0
        )
    )
    df["mu_V"] = mu_minus_half_sig2 + 0.5 * df["sigma_V"]**2

    sqrtT = np.sqrt(df["T"])
    numerator = (
        np.log(df["V"] / df["B_used"])
        + (df["mu_V"] - 0.5 * df["sigma_V"] ** 2) * df["T"]
    )
    denominator = df["sigma_V"] * sqrtT
    DD_p = numerator / denominator
    df["DD_p"] = DD_p
    df["PD_p"] = norm_cdf(-DD_p)
    return df
