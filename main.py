"""Entry point for the electromagnetic field simulator."""

import numpy as np
from scipy import constants

from trajectories import CircularTrajectory, OscillatingTrajectory, LinearUniformTrajectory
from visualization import run_animation

c = constants.c
e = constants.e  # elementary charge


def circular_demo():
    """Charge in circular motion at 0.3c — shows synchrotron-like radiation fields."""
    radius = 1.0  # meters
    speed = 0.01 * c
    trajectory = CircularTrajectory(radius, speed)

    # Observation point: 5 meters away along x-axis
    #r_obs = np.array([5.0, 0.0, 0.0])
    r_obs = np.array([0.0, 0.0, 0.0])

    period = 2 * np.pi * radius / speed
    run_animation(
        trajectory=trajectory,
        charge=e,
        r_obs=r_obs,
        dt=period / 200,
        t_max=5 * period,
        time_window=3 * period,
        r_max=50.0,
    )


def oscillating_demo():
    """Charge oscillating along x-axis — classic dipole radiation pattern."""
    amplitude = 0.5  # meters
    omega = 0.2 * c / amplitude  # max speed = 0.2c
    trajectory = OscillatingTrajectory(amplitude, omega)

    # Observation point: along y-axis (perpendicular to oscillation)
    #r_obs = np.array([0.0, 10.0, 0.0])
    r_obs = np.array([0.0, 1., 0.0])

    period = 2 * np.pi / omega
    run_animation(
        trajectory=trajectory,
        charge=e,
        r_obs=r_obs,
        dt=period / 200,
        t_max=5 * period,
        time_window=3 * period,
        r_max=50.0,
    )


def oscillating_longitudinal_demo():
    """Charge oscillating along x-axis — observer on the axis of oscillation."""
    amplitude = 0.5  # meters
    omega = 0.2 * c / amplitude  # max speed = 0.2c
    trajectory = OscillatingTrajectory(amplitude, omega)

    # Observation point: along x-axis (on the oscillation axis)
    r_obs = np.array([10.0, 0.0, 0.0])

    period = 2 * np.pi / omega
    run_animation(
        trajectory=trajectory,
        charge=e,
        r_obs=r_obs,
        dt=period / 200,
        t_max=5 * period,
        time_window=3 * period,
        r_max=50.0,
    )


def linear_static_demo():
    """Static charge — should produce pure Coulomb field, zero B."""
    trajectory = LinearUniformTrajectory(velocity_x=0.0, x0=0.0)

    r_obs = np.array([3.0, 0.0, 0.0])
    run_animation(
        trajectory=trajectory,
        charge=e,
        r_obs=r_obs,
        dt=1e-9,
        t_max=1e-6,
        r_max=50.0,
    )


def linear_relativistic_demo():
    """Charge moving at 0.9c — shows relativistic field compression."""
    speed = 0.9 * c
    trajectory = LinearUniformTrajectory(velocity_x=speed, x0=-50.0)

    # Observer off to the side to see field compression
    r_obs = np.array([0.0, 5.0, 0.0])
    run_animation(
        trajectory=trajectory,
        charge=e,
        r_obs=r_obs,
        dt=1e-9,
        t_max=500e-9,
        time_window=200e-9,
        r_max=200.0,
    )


DEMOS = {
    "circular": circular_demo,
    "oscillating": oscillating_demo,
    "oscillating_longitudinal": oscillating_longitudinal_demo,
    "static": linear_static_demo,
    "relativistic": linear_relativistic_demo,
}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in DEMOS:
        choice = sys.argv[1]
    else:
        print("Available demos:")
        for name in DEMOS:
            print(f"  {name}")
        choice = input("Select demo (default: circular): ").strip() or "circular"

    if choice in DEMOS:
        DEMOS[choice]()
    else:
        print(f"Unknown demo: {choice}")
        print(f"Available: {', '.join(DEMOS)}")
