import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy.optimize import brentq


@dataclass(frozen=True)
class NIGParams:
    # NIG for log-returns Z: (alpha, beta, delta, mu) under P, per unit time
    alpha: float
    beta: float
    delta: float
    mu: float

    def validate(self):
        if not (self.alpha > abs(self.beta)):
            raise ValueError("Need alpha > |beta| for NIG.")
        if not (self.delta > 0):
            raise ValueError("Need delta > 0 for NIG.")


def nig_kappa(u, p: NIGParams, tau: float):
    """
    Vectorized log-mgf kappa(u; tau) for NIG increments.
    u can be scalar or numpy array (real/complex).
    """
    p.validate()
    a, b, d, m = p.alpha, p.beta, p.delta, p.mu

    u = np.asarray(u, dtype=np.complex128)

    term0 = np.sqrt(a*a - b*b)
    term1 = np.sqrt(a*a - (b + u)*(b + u))
    return (m * u + d * (term0 - term1)) * tau


def solve_esscher_theta(p: NIGParams, r: float, tau: float) -> float:
    """
    Solve for theta in:
      kappa(theta+1) - kappa(theta) = r*tau
    with domain-safe bracketing.

    NIG log-MGF (kappa) exists only when |beta + u| < alpha.
    We need this for u=theta and u=theta+1.
    """
    p.validate()
    a, b = float(p.alpha), float(p.beta)

    # Admissible theta interval from:
    # 1) |b + theta|   < a  ->  -a - b < theta < a - b
    # 2) |b + theta+1| < a  ->  -a - b - 1 < theta < a - b - 1
    lo = max(-a - b, -a - b - 1.0)
    hi = min( a - b,  a - b - 1.0)

    # stay away from boundary to avoid numerical blowups
    eps = 1e-10
    lo += eps
    hi -= eps
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
        raise RuntimeError("No admissible theta interval (Esscher transform infeasible for these NIG params).")

    def g(th: float) -> float:
        return (nig_kappa(th + 1.0, p, tau) - nig_kappa(th, p, tau)).real - (r * tau)

    # Evaluate g on a small grid to find a sign change
    # (more robust than hoping endpoints bracket)
    grid = np.linspace(lo, hi, 41)
    vals = np.array([g(t) for t in grid], dtype=float)

    # Find adjacent points with opposite signs
    for i in range(len(grid) - 1):
        f0, f1 = vals[i], vals[i + 1]
        if np.isfinite(f0) and np.isfinite(f1) and (f0 * f1 < 0):
            return float(brentq(g, grid[i], grid[i + 1], maxiter=200))

    # If no sign change, it's either genuinely no-root, or numerically flat.
    # You can choose to fail, or return the minimizer of |g| as a fallback.
    j = int(np.nanargmin(np.abs(vals)))
    th_best = float(grid[j])
    raise RuntimeError(
        "Could not bracket theta root inside admissible domain. "
        f"Best |g|={abs(vals[j]):.3e} at theta≈{th_best:.6f}. "
        "Try different starting params or check nig_kappa parameterization."
    )


def cf_logA_T_vec(u, lnA: float, p: NIGParams, theta: float, tau: float):
    """
    Vectorized characteristic function of ln A_T under Esscher tilt theta.
    u can be scalar or numpy array (real/complex).
    """
    u = np.asarray(u, dtype=np.complex128)
    iu = 1j * u
    psi = nig_kappa(theta + iu, p, tau) - nig_kappa(theta, p, tau)
    return np.exp(1j * u * lnA + psi)


def _Pj_prob(j, A, K, p, theta, tau, *, U=120.0, n=8000) -> float:
    if j not in (1, 2):
        raise ValueError("j must be 1 or 2")
    if A <= 0 or K <= 0 or tau <= 0:
        return np.nan

    lnA = float(np.log(A))
    lnK = float(np.log(K))

    u = np.linspace(1e-8, U, n)

    shift = -1j * (j - 1)
    phi = cf_logA_T_vec(u + shift, lnA, p, theta, tau)
    phi_shift = cf_logA_T_vec(shift, lnA, p, theta, tau)

    expo = np.exp(-1j * u * lnK)
    q = expo * phi / (1j * u * phi_shift)

    integral = np.trapezoid(np.real(q), u)
    Pj = 0.5 + (1.0 / np.pi) * integral
    return float(np.clip(Pj, 0.0, 1.0))


def call_nig_with_theta(A, K, r, tau, p, theta, *, U=120.0, n=8000) -> float:
    P1 = _Pj_prob(1, A, K, p, theta, tau, U=U, n=n)
    P2 = _Pj_prob(2, A, K, p, theta, tau, U=U, n=n)

    C = A * P1 - K * np.exp(-r * tau) * P2

    lower = max(A - K * np.exp(-r * tau), 0.0)
    upper = A

    if not np.isfinite(C):
        return np.nan
    return float(min(upper, max(lower, C)))


def invert_asset_one_date(E_obs, L, r, tau, p, *, A_prev=None, U=120.0, n=8000, bracket_mult=5.0):
    theta = solve_esscher_theta(p, r, tau)

    A0 = max(E_obs + 1e-12, E_obs + L, (A_prev if A_prev is not None else 0.0))
    A_lo = max(1e-12, 0.1 * L)
    A_hi = max(A0, E_obs + bracket_mult * L)

    def f(A):
        return call_nig_with_theta(A, L, r, tau, p, theta, U=U, n=n) - E_obs

    f_lo, f_hi = f(A_lo), f(A_hi)
    tries = 0
    while f_lo * f_hi > 0 and tries < 20:
        A_hi *= 2.0
        f_hi = f(A_hi)
        tries += 1

    if f_lo * f_hi > 0:
        raise RuntimeError("Could not bracket root for A.")

    A_hat = float(brentq(f, A_lo, A_hi, maxiter=200))
    return A_hat, theta


def invert_assets_monthly_for_firm(
    g: pd.DataFrame,
    p: NIGParams,
    tau: float = 1.0,
    U: float = 150.0,
    n: int = 2000,
    verbose: bool = False
):
    g = g.sort_values("date").copy()
    g["month"] = g["date"].dt.to_period("M")
    month_ends = g.groupby("month")["date"].max().sort_values()

    results = []
    A_prev = None

    for d in month_ends:
        row = g[g["date"] == d].iloc[0]
        E_obs = float(row["E"])
        L     = float(row["L"])
        r     = float(row["r"])

        # unpack two outputs
        A_hat, theta = invert_asset_one_date(
            E_obs, L, r, tau, p,
            A_prev=A_prev,
            U=U, n=n
        )

        results.append((d, E_obs, L, r, A_hat, theta))
        A_prev = A_hat

    out = pd.DataFrame(results, columns=["date", "E", "L", "r", "A_hat", "theta"])
    out["logA"] = np.log(out["A_hat"])
    out["logret_A"] = out["logA"].diff()
    return out


def build_weekly_calendar_from_panel(g: pd.DataFrame, *, week_ending: str = "W-FRI") -> pd.DatetimeIndex:
    """
    Build a weekly inversion calendar from the firm's panel dates.
    By default, uses 'W-FRI' periods and selects the LAST available trading date in each week.
    """
    g = g.sort_values("date").copy()
    g["week"] = g["date"].dt.to_period(week_ending)
    week_ends = g.groupby("week")["date"].max().sort_values()
    return pd.DatetimeIndex(week_ends.values)


def invert_assets_weekly_for_firm(
    g: pd.DataFrame,
    p: NIGParams,
    *,
    tau: float = 1.0,
    U: float = 150.0,
    n: int = 2000,
    week_ending: str = "W-FRI",
) -> pd.DataFrame:
    """
    Invert asset values A_hat on weekly calendar (last available trading date each week).
    Returns weekly A_hat path + logA + weekly log returns.
    """
    g = g.sort_values("date").copy()
    dates = build_weekly_calendar_from_panel(g, week_ending=week_ending)

    # index for fast row lookup
    g = g.set_index("date")

    results = []
    A_prev = None

    for d in dates:
        if d not in g.index:
            continue

        row = g.loc[d]
        E_obs = float(row["E"])
        L     = float(row["L"])
        r     = float(row["r"])

        A_hat, theta = invert_asset_one_date(
            E_obs, L, r, tau, p,
            A_prev=A_prev,
            U=U, n=n
        )

        results.append((d, E_obs, L, r, A_hat, theta))
        A_prev = A_hat

    out = pd.DataFrame(results, columns=["date", "E", "L", "r", "A_hat", "theta"]).sort_values("date")
    out["logA"] = np.log(out["A_hat"])
    out["dlogA"] = out["logA"].diff()     # weekly log-return of assets
    return out