"""Real-time matplotlib animation of electromagnetic fields."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from trajectories import Trajectory
from physics import compute_fields


def run_animation(
    trajectory: Trajectory,
    charge: float,
    r_obs: np.ndarray,
    dt: float,
    t_max: float,
    time_window: float = None,
    r_max: float = 1e3,
    trail_length: int = 200,
):
    """Launch real-time animation showing charge motion and field evolution.

    Parameters
    ----------
    trajectory : Trajectory instance
    charge : charge in Coulombs
    r_obs : 3D observation point
    dt : time step per frame (seconds)
    t_max : total simulation time (seconds)
    time_window : width of scrolling time axis (seconds); defaults to t_max
    r_max : upper bound on source-observer distance for retarded time solver
    trail_length : number of past positions to show as trajectory trail
    """
    if time_window is None:
        time_window = t_max

    # Pre-compute spatial bounds from trajectory at a few sample times
    sample_times = np.linspace(-t_max, t_max, 500)
    positions = np.array([trajectory.position(t) for t in sample_times])
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    pad_x = max(0.3 * (x_max - x_min), 0.5)
    pad_y = max(0.3 * (y_max - y_min), 0.5)
    # Include observation point in bounds
    x_lo = min(x_min - pad_x, r_obs[0] - pad_x)
    x_hi = max(x_max + pad_x, r_obs[0] + pad_x)
    y_lo = min(y_min - pad_y, r_obs[1] - pad_y)
    y_hi = max(y_max + pad_y, r_obs[1] + pad_y)

    # --- Figure layout ---
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], hspace=0.35, wspace=0.3)
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_E = fig.add_subplot(gs[0, 1])
    ax_B = fig.add_subplot(gs[1, 1])

    # Trajectory panel
    ax_traj.set_xlim(x_lo, x_hi)
    ax_traj.set_ylim(y_lo, y_hi)
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.set_title("Charge trajectory (x-y plane)")

    (trail_line,) = ax_traj.plot([], [], "b-", alpha=0.4, lw=1)
    (charge_dot,) = ax_traj.plot([], [], "ro", ms=8, zorder=5)
    ax_traj.plot(r_obs[0], r_obs[1], "kx", ms=10, mew=2, label="observer")

    # E-field arrow at observer (x-y projection, scaled to plot coordinates)
    arrow_scale = 0.15 * max(x_hi - x_lo, y_hi - y_lo)  # max arrow length in data units
    E_arrow = ax_traj.quiver(
        r_obs[0], r_obs[1], 0, 0,
        angles="xy", scale_units="xy", scale=1,
        color="orange", width=0.015, zorder=6, label="E field",
    )
    ax_traj.legend(loc="upper right", fontsize=8)

    # E-field panel
    ax_E.set_ylabel("E (V/m)")
    ax_E.set_title("Electric field at observer")
    (line_Ex,) = ax_E.plot([], [], "r-", label="Ex", lw=1)
    (line_Ey,) = ax_E.plot([], [], "g-", label="Ey", lw=1)
    (line_Ez,) = ax_E.plot([], [], "b-", label="Ez", lw=1)
    ax_E.legend(loc="upper right", fontsize=7)

    # B-field panel
    ax_B.set_xlabel("t (s)")
    ax_B.set_ylabel("B (T)")
    ax_B.set_title("Magnetic field at observer")
    (line_Bx,) = ax_B.plot([], [], "r-", label="Bx", lw=1)
    (line_By,) = ax_B.plot([], [], "g-", label="By", lw=1)
    (line_Bz,) = ax_B.plot([], [], "b-", label="Bz", lw=1)
    ax_B.legend(loc="upper right", fontsize=7)

    # History storage
    times = []
    Ex_hist, Ey_hist, Ez_hist = [], [], []
    Bx_hist, By_hist, Bz_hist = [], [], []
    trail_x, trail_y = [], []

    n_frames = int(t_max / dt)

    def init():
        trail_line.set_data([], [])
        charge_dot.set_data([], [])
        E_arrow.set_UVC(0, 0)
        for ln in (line_Ex, line_Ey, line_Ez, line_Bx, line_By, line_Bz):
            ln.set_data([], [])
        return (
            trail_line, charge_dot, E_arrow,
            line_Ex, line_Ey, line_Ez,
            line_Bx, line_By, line_Bz,
        )

    def update(frame):
        t = frame * dt

        # Current charge position
        pos = trajectory.position(t)
        trail_x.append(pos[0])
        trail_y.append(pos[1])
        # Keep trail limited
        if len(trail_x) > trail_length:
            del trail_x[0]
            del trail_y[0]

        trail_line.set_data(trail_x, trail_y)
        charge_dot.set_data([pos[0]], [pos[1]])

        # Compute fields
        try:
            E, B = compute_fields(t, r_obs, trajectory, charge, r_max)
        except RuntimeError:
            E = np.zeros(3)
            B = np.zeros(3)

        # Update E-field arrow (x-y projection, normalized to arrow_scale)
        E_xy = np.hypot(E[0], E[1])
        if E_xy > 0:
            E_arrow.set_UVC(E[0] / E_xy * arrow_scale, E[1] / E_xy * arrow_scale)
        else:
            E_arrow.set_UVC(0, 0)

        times.append(t)
        Ex_hist.append(E[0])
        Ey_hist.append(E[1])
        Ez_hist.append(E[2])
        Bx_hist.append(B[0])
        By_hist.append(B[1])
        Bz_hist.append(B[2])

        t_arr = np.array(times)

        # Update E plot
        line_Ex.set_data(t_arr, Ex_hist)
        line_Ey.set_data(t_arr, Ey_hist)
        line_Ez.set_data(t_arr, Ez_hist)

        # Update B plot
        line_Bx.set_data(t_arr, Bx_hist)
        line_By.set_data(t_arr, By_hist)
        line_Bz.set_data(t_arr, Bz_hist)

        # Scrolling time window
        t_lo = max(0, t - time_window)
        t_hi = max(t, time_window)
        ax_E.set_xlim(t_lo, t_hi)
        ax_B.set_xlim(t_lo, t_hi)

        # Auto-scale y axes
        # Find visible data range
        mask = t_arr >= t_lo
        if mask.any():
            E_vals = np.array([Ex_hist, Ey_hist, Ez_hist])[:, mask]
            B_vals = np.array([Bx_hist, By_hist, Bz_hist])[:, mask]

            e_min, e_max = E_vals.min(), E_vals.max()
            b_min, b_max = B_vals.min(), B_vals.max()

            e_pad = max(0.1 * (e_max - e_min), 1e-20)
            b_pad = max(0.1 * (b_max - b_min), 1e-30)

            ax_E.set_ylim(e_min - e_pad, e_max + e_pad)
            ax_B.set_ylim(b_min - b_pad, b_max + b_pad)

        return (
            trail_line, charge_dot, E_arrow,
            line_Ex, line_Ey, line_Ez,
            line_Bx, line_By, line_Bz,
        )

    anim = FuncAnimation(
        fig, update, frames=n_frames, init_func=init,
        interval=max(1, int(dt * 1000)),  # interval in ms (aim for real-time)
        blit=False, repeat=False,
    )

    plt.tight_layout()
    plt.show()
    return anim  # prevent garbage collection
