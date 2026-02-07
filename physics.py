"""Retarded time solver and Lienard-Wiechert field computation."""

import numpy as np
from scipy import constants, optimize
from trajectories import Trajectory

c = constants.c
eps0 = constants.epsilon_0
MU0 = constants.mu_0


def solve_retarded_time(
    t_obs: float,
    r_obs: np.ndarray,
    trajectory: Trajectory,
    r_max: float = 1e3,
) -> float:
    """Solve for retarded time t_r such that c*(t_obs - t_r) = |r_obs - r_s(t_r)|.

    Uses Brent's method on f(t_r) = c*(t_obs - t_r) - |r_obs - r_s(t_r)|.
    At t_r = t_obs, f = -|r_obs - r_s(t_obs)| <= 0.
    At t_r = t_low (sufficiently early), the light-travel term c*(t_obs - t_r)
    dominates and f > 0, so there is a root in between.
    """
    def f(t_r):
        R_vec = r_obs - trajectory.position(t_r)
        R_mag = np.linalg.norm(R_vec)
        return c * (t_obs - t_r) - R_mag

    # Upper bracket: t_r = t_obs gives f <= 0 (equality only if r_obs == r_s(t_obs))
    t_high = t_obs

    # Lower bracket: go far enough back that the light-travel term dominates
    t_low = t_obs - r_max / c

    # Verify bracket: f(t_low) should be > 0
    f_low = f(t_low)
    if f_low <= 0:
        # Expand bracket
        t_low = t_obs - 10 * r_max / c
        f_low = f(t_low)
        if f_low <= 0:
            raise RuntimeError(
                f"Cannot bracket retarded time: f(t_low={t_low}) = {f_low} <= 0. "
                f"Try increasing r_max."
            )

    f_high = f(t_high)
    if f_high > 0:
        # Observation point is at the charge position; degenerate case
        raise RuntimeError(
            "Observation point coincides with charge position at t_obs."
        )

    t_r = optimize.brentq(f, t_low, t_high, xtol=1e-14, rtol=1e-12)
    return t_r


def compute_fields(
    t_obs: float,
    r_obs: np.ndarray,
    trajectory: Trajectory,
    charge: float,
    r_max: float = 1e3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute E and B fields at (r_obs, t_obs) using the Lienard-Wiechert formulae.

    Returns (E, B) as 3D numpy arrays in SI units.
    """
    t_r = solve_retarded_time(t_obs, r_obs, trajectory, r_max)

    # Vectors at retarded time
    r_s = trajectory.position(t_r)
    v = trajectory.velocity(t_r)
    a = trajectory.acceleration(t_r)

    R_vec = r_obs - r_s
    R_mag = np.linalg.norm(R_vec)
    n = R_vec / R_mag  # unit vector from source to observer

    beta = v / c  # v/c
    beta_dot = a / c  # (dv/dt)/c  = β̇

    kappa = 1.0 - np.dot(n, beta)  # 1 - n·β
    beta_sq = np.dot(beta, beta)

    # Velocity (Coulomb-like) term: (n - β)(1 - β²) / (κ³ R²)
    velocity_term = (n - beta) * (1.0 - beta_sq) / (kappa**3 * R_mag**2)

    # Acceleration (radiation) term: n × ((n - β) × β̇) / (κ³ R c)
    n_minus_beta = n - beta
    cross_inner = np.cross(n_minus_beta, beta_dot)
    radiation_term = np.cross(n, cross_inner) / (kappa**3 * R_mag * c)

    prefactor = charge / (4.0 * np.pi * eps0)
    E = prefactor * (velocity_term + radiation_term)
    B = np.cross(n, E) / c

    return E, B
