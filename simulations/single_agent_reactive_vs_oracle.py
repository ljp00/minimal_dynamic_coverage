"""Single-agent Reactive vs Oracle experiment.

Run with default settings::

    python simulations/single_agent_reactive_vs_oracle.py

With custom parameters::

    python simulations/single_agent_reactive_vs_oracle.py \\
        --time 200 --dt 0.1 --horizon 1.0 --seed 0 --show

With animation disabled (faster, headless-friendly)::

    python simulations/single_agent_reactive_vs_oracle.py --no-animation

"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; overridden when --show is set

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ---------------------------------------------------------------------------
# Make the repo root importable regardless of where the script is invoked from
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.environment.dynamic_field import DynamicSensitivityField
from src.agents.single_uav import SingleUAV
from src.control.reactive_controller import ReactiveController, OracleController
from src.metrics.coverage_metrics import CoverageMetrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-agent Reactive vs Oracle coverage experiment."
    )
    parser.add_argument("--time", type=float, default=100.0,
                        help="Total simulation time in seconds (default: 100)")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Time step in seconds (default: 0.1)")
    parser.add_argument("--horizon", type=float, default=0.5,
                        help="Oracle look-ahead horizon in seconds (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no-animation", dest="animation", action="store_false",
                        help="Skip GIF animation generation (faster)")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for the animation (default: 10)")
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively (requires a display)")
    parser.set_defaults(animation=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(field, controller, init_pos, total_time, dt, metrics):
    """Run a single agent through the field and record metrics.

    Parameters
    ----------
    field      : DynamicSensitivityField
    controller : ReactiveController | OracleController
    init_pos   : array-like of shape (2,)
    total_time : float
    dt         : float
    metrics    : CoverageMetrics

    Returns
    -------
    dict with keys:
        times, positions, weighted_coverage, coverage_cost,
        hotspot_distance, speed, sensed_value
    """
    agent = SingleUAV(init_pos=init_pos, max_velocity=10.0)

    times = []
    wc_list = []
    cc_list = []
    hd_list = []
    sp_list = []
    sv_list = []

    t = 0.0
    step = 0
    while t <= total_time + 1e-9:
        u = controller.compute(agent.pos, field, t)
        # Record before stepping (captures state at time t)
        times.append(t)
        wc_list.append(metrics.weighted_coverage(agent.pos, t))
        cc_list.append(metrics.coverage_cost(agent.pos, t))
        hd_list.append(metrics.hotspot_distance(agent.pos, t))
        sp_list.append(float(np.linalg.norm(u)))
        sv_list.append(metrics.sensed_value(agent.pos, t))
        agent.record(field, t, u)

        agent.step(u, dt)
        t += dt
        step += 1

    return {
        "times": np.array(times),
        "positions": agent.trajectory[: len(times)],  # align lengths
        "weighted_coverage": np.array(wc_list),
        "coverage_cost": np.array(cc_list),
        "hotspot_distance": np.array(hd_list),
        "speed": np.array(sp_list),
        "sensed_value": np.array(sv_list),
    }


# ---------------------------------------------------------------------------
# Printing statistics
# ---------------------------------------------------------------------------

def print_stats(r_data, o_data):
    """Print a comparison table to the terminal."""
    metrics_cfg = [
        ("Weighted Coverage", "weighted_coverage", True,  "%"),
        ("Coverage Cost",     "coverage_cost",     False, ""),
        ("Hotspot Distance",  "hotspot_distance",  False, "m"),
        ("Speed",             "speed",             None,  "m/s"),
        ("Sensed Value",      "sensed_value",      True,  ""),
    ]

    header = f"{'Metric':<22}  {'Reactive':>22}  {'Oracle':>22}  {'Improvement':>12}"
    sep = "-" * len(header)
    print()
    print(sep)
    print("  REACTIVE vs ORACLE — SUMMARY STATISTICS")
    print(sep)
    print(header)
    print(sep)

    for label, key, higher_is_better, unit in metrics_cfg:
        rv = r_data[key]
        ov = o_data[key]
        r_str = f"μ={rv.mean():.3f} σ={rv.std():.3f}"
        o_str = f"μ={ov.mean():.3f} σ={ov.std():.3f}"
        if higher_is_better is True:
            pct = (ov.mean() - rv.mean()) / (abs(rv.mean()) + 1e-12) * 100
            imp_str = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
        elif higher_is_better is False:
            pct = (rv.mean() - ov.mean()) / (abs(rv.mean()) + 1e-12) * 100
            imp_str = f"-{abs(pct):.1f}%" if pct >= 0 else f"+{abs(pct):.1f}%"
        else:
            imp_str = "—"
        row = f"{label:<22}  {r_str:>22}  {o_str:>22}  {imp_str:>12}"
        print(row)
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Static result figure
# ---------------------------------------------------------------------------

def save_result_figure(field, r_data, o_data, times_hotspot, hotspot_traj,
                       total_time, oracle_horizon, out_path):
    """Generate and save the 3x2 result figure."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        f"Single-agent Reactive vs Oracle  (horizon={oracle_horizon:.2f} s, T={total_time:.0f} s)",
        fontsize=13, fontweight="bold"
    )

    # ---- Panel 1: final field + trajectories ---------------------------
    ax = axes[0, 0]
    field_img = field.get_values_on_grid(total_time)
    extent = [field.domain[0], field.domain[1], field.domain[2], field.domain[3]]
    im = ax.imshow(field_img, origin="lower", extent=extent, cmap="hot", aspect="equal", alpha=0.7)
    plt.colorbar(im, ax=ax, label="Sensitivity")

    r_traj = r_data["positions"]
    o_traj = o_data["positions"]
    ax.plot(r_traj[:, 0], r_traj[:, 1], "C0-", lw=1.0, alpha=0.7, label="Reactive trajectory")
    ax.plot(o_traj[:, 0], o_traj[:, 1], "C1-", lw=1.0, alpha=0.7, label="Oracle trajectory")
    ax.plot(hotspot_traj[:, 0], hotspot_traj[:, 1], "w--", lw=1.5, alpha=0.8, label="Hotspot trajectory")
    ax.plot(*r_traj[-1], "C0o", ms=8, label="Reactive (end)")
    ax.plot(*o_traj[-1], "C1o", ms=8, label="Oracle (end)")
    ax.plot(*hotspot_traj[-1], "w*", ms=12, label="Hotspot (end)")
    ax.set_xlim(field.domain[0], field.domain[1])
    ax.set_ylim(field.domain[2], field.domain[3])
    ax.set_title("Sensitivity Field + Trajectories (t=T)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=7, loc="upper right")

    # ---- Panel 2: Weighted coverage ------------------------------------
    ax = axes[0, 1]
    r_wc = r_data["weighted_coverage"]
    o_wc = o_data["weighted_coverage"]
    times = r_data["times"]
    imp_wc = (o_wc.mean() - r_wc.mean()) / (abs(r_wc.mean()) + 1e-12) * 100
    ax.plot(times, r_wc, "C0", lw=1.2, label=f"Reactive (μ={r_wc.mean():.3f})")
    ax.plot(times, o_wc, "C1", lw=1.2, label=f"Oracle (μ={o_wc.mean():.3f})")
    ax.set_title(f"Weighted Coverage  (Oracle +{imp_wc:.1f}%)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Coverage rate")
    ax.legend(fontsize=8)
    ax.set_ylim(0, None)

    # ---- Panel 3: Coverage cost ----------------------------------------
    ax = axes[1, 0]
    r_cc = r_data["coverage_cost"]
    o_cc = o_data["coverage_cost"]
    imp_cc = (r_cc.mean() - o_cc.mean()) / (abs(r_cc.mean()) + 1e-12) * 100
    ax.plot(times, r_cc, "C0", lw=1.2, label=f"Reactive (μ={r_cc.mean():.3f})")
    ax.plot(times, o_cc, "C1", lw=1.2, label=f"Oracle (μ={o_cc.mean():.3f})")
    ax.set_title(f"Coverage Cost (↓ better)  (Oracle -{imp_cc:.1f}%)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cost (uncovered fraction)")
    ax.legend(fontsize=8)

    # ---- Panel 4: Hotspot distance ------------------------------------
    ax = axes[1, 1]
    r_hd = r_data["hotspot_distance"]
    o_hd = o_data["hotspot_distance"]
    imp_hd = (r_hd.mean() - o_hd.mean()) / (abs(r_hd.mean()) + 1e-12) * 100
    ax.plot(times, r_hd, "C0", lw=1.2, label=f"Reactive (μ={r_hd.mean():.2f} m)")
    ax.plot(times, o_hd, "C1", lw=1.2, label=f"Oracle (μ={o_hd.mean():.2f} m)")
    ax.set_title(f"Hotspot Distance (↓ better)  (Oracle -{imp_hd:.1f}%)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.legend(fontsize=8)

    # ---- Panel 5: Speed -----------------------------------------------
    ax = axes[2, 0]
    r_sp = r_data["speed"]
    o_sp = o_data["speed"]
    ax.plot(times, r_sp, "C0", lw=1.2, label=f"Reactive (μ={r_sp.mean():.2f} m/s)")
    ax.plot(times, o_sp, "C1", lw=1.2, label=f"Oracle (μ={o_sp.mean():.2f} m/s)")
    ax.set_title("Agent Speed")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.legend(fontsize=8)

    # ---- Panel 6: Sensed value -----------------------------------------
    ax = axes[2, 1]
    r_sv = r_data["sensed_value"]
    o_sv = o_data["sensed_value"]
    imp_sv = (o_sv.mean() - r_sv.mean()) / (abs(r_sv.mean()) + 1e-12) * 100
    ax.plot(times, r_sv, "C0", lw=1.2, label=f"Reactive (μ={r_sv.mean():.3f})")
    ax.plot(times, o_sv, "C1", lw=1.2, label=f"Oracle (μ={o_sv.mean():.3f})")
    ax.set_title(f"Sensed Value (↑ better)  (Oracle +{imp_sv:.1f}%)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sensitivity at agent")
    ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[INFO] Result figure saved to: {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def make_animation(field, r_data, o_data, times_hotspot, hotspot_traj,
                   total_time, fps, out_path):
    """Build and save a GIF animation of the simulation."""
    # Sub-sample frames to keep file size manageable
    n_frames = min(200, len(r_data["times"]))
    frame_idx = np.linspace(0, len(r_data["times"]) - 1, n_frames, dtype=int)

    fig_a, ax_a = plt.subplots(figsize=(7, 7))
    ax_a.set_xlim(field.domain[0], field.domain[1])
    ax_a.set_ylim(field.domain[2], field.domain[3])
    ax_a.set_aspect("equal")

    extent = [field.domain[0], field.domain[1], field.domain[2], field.domain[3]]
    # Initialise with the field at t=0
    field_img0 = field.get_values_on_grid(0.0)
    im_anim = ax_a.imshow(field_img0, origin="lower", extent=extent, cmap="hot",
                          aspect="equal", alpha=0.7, vmin=0, vmax=field.amplitude)
    plt.colorbar(im_anim, ax=ax_a, label="Sensitivity")

    r_line, = ax_a.plot([], [], "C0-", lw=1.0, alpha=0.7, label="Reactive")
    o_line, = ax_a.plot([], [], "C1-", lw=1.0, alpha=0.7, label="Oracle")
    h_line, = ax_a.plot([], [], "w--", lw=1.5, alpha=0.8, label="Hotspot")
    r_dot,  = ax_a.plot([], [], "C0o", ms=9)
    o_dot,  = ax_a.plot([], [], "C1o", ms=9)
    h_dot,  = ax_a.plot([], [], "w*",  ms=12)
    time_txt = ax_a.text(0.02, 0.96, "", transform=ax_a.transAxes,
                         color="white", fontsize=10, va="top")
    ax_a.legend(loc="upper right", fontsize=8)

    r_traj_full = r_data["positions"]
    o_traj_full = o_data["positions"]

    def _update(fi):
        idx = frame_idx[fi]
        t_now = r_data["times"][idx]
        field_now = field.get_values_on_grid(t_now)
        im_anim.set_data(field_now)

        r_line.set_data(r_traj_full[: idx + 1, 0], r_traj_full[: idx + 1, 1])
        o_line.set_data(o_traj_full[: idx + 1, 0], o_traj_full[: idx + 1, 1])
        h_idx = min(idx, len(hotspot_traj) - 1)
        h_line.set_data(hotspot_traj[: h_idx + 1, 0], hotspot_traj[: h_idx + 1, 1])

        r_dot.set_data([r_traj_full[idx, 0]], [r_traj_full[idx, 1]])
        o_dot.set_data([o_traj_full[idx, 0]], [o_traj_full[idx, 1]])
        h_dot.set_data([hotspot_traj[h_idx, 0]], [hotspot_traj[h_idx, 1]])
        time_txt.set_text(f"t = {t_now:.1f} s")
        return im_anim, r_line, o_line, h_line, r_dot, o_dot, h_dot, time_txt

    anim = animation.FuncAnimation(fig_a, _update, frames=n_frames, blit=True, interval=1000 // fps)

    try:
        writer = animation.PillowWriter(fps=fps)
        anim.save(out_path, writer=writer)
        print(f"[INFO] Animation saved to: {out_path}")
    except Exception as exc:
        print(f"[WARN] Could not save animation ({exc}). Skipping.")
    plt.close(fig_a)
    return anim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.show:
        matplotlib.use("TkAgg")  # switch to interactive backend when requested

    rng = np.random.default_rng(args.seed)
    # rng is available here to randomise agent initial positions in the future
    # (e.g. rng.uniform(domain[0], domain[1], size=2)).  Currently both agents
    # start at the domain centroid for a fair baseline comparison.

    # ----- Environment -------------------------------------------------------
    domain = (0.0, 100.0, 0.0, 100.0)
    field = DynamicSensitivityField(
        domain=domain,
        radius=30.0,
        angular_speed=0.2,
        phase=0.0,
        sigma=10.0,
        amplitude=1.0,
        grid_resolution=50,
    )

    # ----- Initial position (same for both agents) ---------------------------
    x_mid = 0.5 * (domain[0] + domain[1])
    y_mid = 0.5 * (domain[2] + domain[3])
    init_pos = np.array([x_mid, y_mid])

    # ----- Controllers -------------------------------------------------------
    gain = 2.0
    max_vel = 10.0
    reactive_ctrl = ReactiveController(gain=gain, max_velocity=max_vel)
    oracle_ctrl = OracleController(horizon=args.horizon, gain=gain, max_velocity=max_vel)

    # ----- Metrics -----------------------------------------------------------
    metrics = CoverageMetrics(field=field, coverage_radius=10.0)

    # ----- Run simulations ---------------------------------------------------
    print(f"[INFO] Running Reactive simulation  (T={args.time}s, dt={args.dt}s) …")
    r_data = run_simulation(field, reactive_ctrl, init_pos.copy(), args.time, args.dt, metrics)

    print(f"[INFO] Running Oracle simulation    (horizon={args.horizon}s) …")
    o_data = run_simulation(field, oracle_ctrl, init_pos.copy(), args.time, args.dt, metrics)

    # Hotspot ground-truth trajectory (aligned with recorded times)
    times_hotspot = r_data["times"]
    hotspot_traj = np.array([field.get_hotspot_position(t) for t in times_hotspot])

    # ----- Terminal statistics -----------------------------------------------
    print_stats(r_data, o_data)

    # ----- Static result figure ----------------------------------------------
    out_dir = os.path.join(_REPO_ROOT, "output")
    fig_path = os.path.join(out_dir, "reactive_vs_oracle_results.png")
    result_fig = save_result_figure(
        field, r_data, o_data, times_hotspot, hotspot_traj,
        args.time, args.horizon, fig_path
    )

    # ----- Animation ---------------------------------------------------------
    if args.animation:
        gif_path = os.path.join(out_dir, "reactive_vs_oracle_animation.gif")
        make_animation(field, r_data, o_data, times_hotspot, hotspot_traj,
                       args.time, args.fps, gif_path)
    else:
        print("[INFO] Animation skipped (--no-animation).")

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
