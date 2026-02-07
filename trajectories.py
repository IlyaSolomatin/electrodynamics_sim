"""Charge trajectory definitions for the electromagnetic field simulator."""

import numpy as np
from abc import ABC, abstractmethod


class Trajectory(ABC):
    """Base class for charge trajectories."""

    @abstractmethod
    def position(self, t: float) -> np.ndarray:
        """Return position r_s(t) as a 3D vector."""

    @abstractmethod
    def velocity(self, t: float) -> np.ndarray:
        """Return velocity v(t) as a 3D vector."""

    @abstractmethod
    def acceleration(self, t: float) -> np.ndarray:
        """Return acceleration a(t) as a 3D vector."""


class CircularTrajectory(Trajectory):
    """Uniform circular motion in the x-y plane.

    The charge orbits the origin at a given radius and speed.
    position(t) = (R cos(ωt), R sin(ωt), 0)
    where ω = v / R.
    """

    def __init__(self, radius: float, speed: float):
        self.radius = radius
        self.speed = speed
        self.omega = speed / radius

    def position(self, t: float) -> np.ndarray:
        wt = self.omega * t
        return np.array([
            self.radius * np.cos(wt),
            self.radius * np.sin(wt),
            0.0,
        ])

    def velocity(self, t: float) -> np.ndarray:
        wt = self.omega * t
        return np.array([
            -self.speed * np.sin(wt),
            self.speed * np.cos(wt),
            0.0,
        ])

    def acceleration(self, t: float) -> np.ndarray:
        wt = self.omega * t
        a_mag = self.speed * self.omega
        return np.array([
            -a_mag * np.cos(wt),
            -a_mag * np.sin(wt),
            0.0,
        ])


class OscillatingTrajectory(Trajectory):
    """Sinusoidal oscillation along the x-axis.

    position(t) = (A sin(ωt), 0, 0)
    The maximum speed is Aω, which must be < c.
    """

    def __init__(self, amplitude: float, angular_frequency: float):
        self.amplitude = amplitude
        self.omega = angular_frequency

    def position(self, t: float) -> np.ndarray:
        return np.array([
            self.amplitude * np.sin(self.omega * t),
            0.0,
            0.0,
        ])

    def velocity(self, t: float) -> np.ndarray:
        return np.array([
            self.amplitude * self.omega * np.cos(self.omega * t),
            0.0,
            0.0,
        ])

    def acceleration(self, t: float) -> np.ndarray:
        return np.array([
            -self.amplitude * self.omega**2 * np.sin(self.omega * t),
            0.0,
            0.0,
        ])


class LinearUniformTrajectory(Trajectory):
    """Constant velocity along the x-axis.

    position(t) = (v*t + x0, 0, 0)
    Use v=0 for a static charge (Coulomb test).
    """

    def __init__(self, velocity_x: float, x0: float = 0.0):
        self.vx = velocity_x
        self.x0 = x0

    def position(self, t: float) -> np.ndarray:
        return np.array([self.vx * t + self.x0, 0.0, 0.0])

    def velocity(self, t: float) -> np.ndarray:
        return np.array([self.vx, 0.0, 0.0])

    def acceleration(self, t: float) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])
