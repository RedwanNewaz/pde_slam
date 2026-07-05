"""
run_experiment.py
-----------------
PDE SLAM with PINN quadtree map (Method 2), evaluated on the elliptical
trajectory from pde_localization/trajectory/SimTrajectory.py.

Setup
  - Ground truth: unicycle robot tracking get_ellipse_points() waypoints for
    N laps, with actuation noise; senses (u, v, h) of the true SWE field
    (data/swe/solutions.npy) at its position + measurement noise.
  - PDE SLAM (ours): RBPF pose filter whose measurement model queries the
    online-learned quadtree PINN map; only leaves inside the 3-sigma pose
    covariance ellipse are trained each step (uncertainty gating); leaves
    split adaptively on persistent data misfit.
  - Baselines: dead reckoning (commanded controls only) and an oracle RBPF
    that knows the true field (the IROS-2026 localization setting).

All parameters live under conf/ and are composed with Hydra. Examples:
    uv run python run_experiment.py                       # full comparison
    uv run python run_experiment.py method=slam_only      # ours only
    uv run python run_experiment.py sim.laps=3 filter.n_particles=800
    uv run python run_experiment.py viz.animate=false viz.save_animation=true

Outputs (under viz.results_dir): metrics.json, *.png, log.npz, animation.mp4
"""

import importlib.util
import json
import os
import sys
import time
import contextlib

# JAX ships its own CUDA libraries via the jax[cuda12] wheels. A system CUDA in
# LD_LIBRARY_PATH (e.g. /usr/local/cuda-* from a ROS setup) shadows them and
# forces a silent CPU fallback. glibc's loader reads LD_LIBRARY_PATH only at
# startup, so we must strip those entries and re-exec before JAX is imported.
if os.environ.get("_PDE_SLAM_LD_CLEANED") != "1":
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    _clean = os.pathsep.join(
        p for p in _ld.split(os.pathsep) if p and "cuda" not in p.lower()
    )
    os.environ["_PDE_SLAM_LD_CLEANED"] = "1"
    if _clean:
        os.environ["LD_LIBRARY_PATH"] = _clean
    else:
        os.environ.pop("LD_LIBRARY_PATH", None)
    if _clean != _ld:
        os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pinn_map import QuadTreeMap, TrueFieldMap, DOMAIN
from rbpf_slam import RBPFSLAM


# ----------------------------------------------------------- setup helpers
def load_trajectory(dt):
    spec = importlib.util.spec_from_file_location(
        "SimTrajectory", os.path.join(HERE, "trajectory", "SimTrajectory.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    waypoints = mod.get_ellipse_points()
    U_ctrl = mod.gen_control_inputs(waypoints, dt)
    return waypoints, U_ctrl


def load_field(steps):
    sol = np.load(os.path.join(HERE, "data", "swe", "solutions.npy"))
    # sanitize solver blow-ups near dry cells
    sol[:, 0] = np.clip(sol[:, 0], -1.0, 20.0)
    sol[:, 1:] = np.clip(sol[:, 1:], -3.0, 3.0)
    assert sol.shape[0] >= steps, "not enough field frames"
    return sol


def motion(x, u, dt):
    return RBPFSLAM._motion_model(np.asarray(x, float), np.asarray(u, float), dt)


def _filter_kwargs(fcfg):
    """Translate optional YAML noise specs into RBPFSLAM constructor kwargs.
    All null by default -> RBPFSLAM uses its own defaults (unchanged behavior).
    Config lists are standard deviations; process is [v, omega_deg]."""
    kw = {}
    if fcfg.process_noise is not None:
        v, w_deg = fcfg.process_noise
        kw["process_noise"] = np.diag([v, np.deg2rad(w_deg)]) ** 2
    if fcfg.measurement_noise is not None:
        kw["measurement_noise"] = np.diag(np.array(list(fcfg.measurement_noise)) ** 2)
    if fcfg.lin_process_noise is not None:
        kw["lin_process_noise"] = np.diag(np.array(list(fcfg.lin_process_noise)) ** 2)
    if fcfg.p0_lin is not None:
        kw["p0_lin"] = np.diag(np.array(list(fcfg.p0_lin)) ** 2)
    return kw


# ---------------------------------------------------------------- simulation
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    import jax
    print(f"JAX backend: {jax.default_backend()} "
          f"(devices: {[d.device_kind for d in jax.devices()]})", flush=True)
    print(OmegaConf.to_yaml(cfg), flush=True)

    enabled = {"slam": bool(cfg.method.pde_slam),
               "dr": bool(cfg.method.dead_reckoning),
               "oracle": bool(cfg.method.oracle)}
    if not any(enabled.values()):
        raise ValueError("no method enabled; set at least one in method.*")

    dt = float(cfg.sim.dt)
    seed = int(cfg.sim.seed)

    results_dir = cfg.viz.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(HERE, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # -- trajectory + field
    waypoints, U_ctrl = load_trajectory(dt)
    LAP = U_ctrl.shape[0]
    steps = int(cfg.sim.steps) if cfg.sim.steps else int(cfg.sim.laps) * LAP
    sol = load_field(steps)

    act_noise = np.array([cfg.sim.act_noise_v, np.deg2rad(cfg.sim.act_noise_w_deg)])
    meas_std = np.array(list(cfg.sim.meas_std), float)
    rng = np.random.default_rng(seed)

    def sense_true(x, y, frame):
        ix = int(np.clip(round(x), 0, 39))
        iy = int(np.clip(round(y), 0, 39))
        val = np.array([sol[frame, 1, ix, iy], sol[frame, 2, ix, iy],
                        sol[frame, 0, ix, iy]])
        return val + rng.normal(0, meas_std)

    # -- initial state
    theta0 = np.arctan2(waypoints[1, 1] - waypoints[0, 1],
                        waypoints[1, 0] - waypoints[0, 0])
    x0 = [waypoints[0, 0], waypoints[0, 1], theta0, 0.0]
    s = list(cfg.filter.init_std)
    init_std = [s[0], s[1], np.deg2rad(s[2]), s[3]]
    filt_kw = _filter_kwargs(cfg.filter)

    # -- estimators
    slam = slam_map = oracle = None
    if enabled["slam"]:
        map_seed = int(cfg.map.seed) if cfg.map.seed is not None else seed
        slam_map = QuadTreeMap(seed=map_seed, max_depth=cfg.map.max_depth,
                               max_leaves=cfg.map.max_leaves,
                               split_min_obs=cfg.map.split_min_obs,
                               split_h_rmse=cfg.map.split_h_rmse)
        slam = RBPFSLAM(slam_map, cfg.filter.n_particles, seed=seed, **filt_kw)
        slam.initialize(x0, init_std)
    if enabled["oracle"]:
        oracle = RBPFSLAM(TrueFieldMap(sol), cfg.filter.n_particles,
                          seed=seed + 1, **filt_kw)
        oracle.initialize(x0, init_std)

    x_true = np.array(x0, float)
    x_dr = np.array(x0, float)

    keys = ["true"]
    if enabled["slam"]:
        keys += ["slam", "n_active", "n_leaves", "err_slam"]
    if enabled["dr"]:
        keys += ["dr", "err_dr"]
    if enabled["oracle"]:
        keys += ["oracle", "err_oracle"]
    log = {k: [] for k in keys}

    train_iters = int(cfg.map.train_iters)
    split_every = int(cfg.map.split_check_every)
    leafstep_full = 0   # hypothetical cost of training ALL leaves every step

    # -- visualization setup (live window and/or saved animation)
    has_display = bool(os.environ.get("DISPLAY"))
    live_window = bool(cfg.viz.animate) and has_display
    save_anim = bool(cfg.viz.save_animation)
    need_fig = live_window or save_anim
    stride = max(1, int(cfg.viz.anim_stride))

    import matplotlib
    if not live_window:
        matplotlib.use("Agg")          # headless: no interactive backend
    import matplotlib.pyplot as plt
    fig = axes = None
    if need_fig:
        fig, axes = _setup_live(plt, enabled, live_window)

    writer = anim_path = None
    if save_anim:
        writer, ext = _make_writer(cfg)
        anim_path = os.path.join(results_dir, f"animation.{ext}")

    t_start = time.time()
    active = None
    est_s = None

    saving = (writer.saving(fig, anim_path, cfg.viz.dpi)
              if save_anim else contextlib.nullcontext())
    with saving:
        for step in range(steps):
            t_now = step * dt
            control = U_ctrl[step % LAP]

            # -- ground truth motion + sensing
            u_act = control + rng.normal(0, act_noise)
            x_true = motion(x_true, u_act, dt)
            y = sense_true(x_true[0], x_true[1], step)

            # -- dead reckoning (commanded controls only)
            x_dr = motion(x_dr, control, dt)

            # -- oracle RBPF (known field)
            if enabled["oracle"]:
                oracle.predict(control, dt)
                oracle.update(y, t_now)
                oracle.resample()
                est_o, _ = oracle.estimate()

            # -- PDE SLAM
            if enabled["slam"]:
                slam.predict(control, dt)
                slam.update(y, t_now)
                slam.resample()
                est_s, cov_s = slam.estimate()

                # map update: register the observation at the ESTIMATED pose,
                # but only when the pose is confident and the cell is still a
                # frontier -- a drifting estimate must not corrupt the map.
                if cov_s[0, 0] + cov_s[1, 1] < 2 * 2.5 ** 2:
                    ix = int(np.clip(round(est_s[0]), 0, 39))
                    iy = int(np.clip(round(est_s[1]), 0, 39))
                    anchored = (cov_s[0, 0] + cov_s[1, 1] < 2 * 1.5 ** 2
                                or slam.trusted_frac > 0.2)
                    if slam_map.obs_counts[ix, iy] < 3 and anchored:
                        slam_map.add_observation(est_s[0], est_s[1], t_now, y)

                # uncertainty-gated training (Method 2)
                active = slam_map.active_leaves(est_s[:2], cov_s[:2, :2])
                if step % split_every == 0:   # periodic refinement check
                    if any(slam_map.maybe_split(lf) for lf in list(active)):
                        active = slam_map.active_leaves(est_s[:2], cov_s[:2, :2])
                for lf in active:
                    slam_map.train_leaf(lf, t_now, train_iters)

                n_leaves = len(slam_map.leaves())
                leafstep_full += n_leaves * train_iters

            # -- logging
            log["true"].append(x_true.copy())
            if enabled["slam"]:
                log["slam"].append(est_s.copy())
                log["n_active"].append(len(active))
                log["n_leaves"].append(n_leaves)
                log["err_slam"].append(np.hypot(*(est_s[:2] - x_true[:2])))
            if enabled["dr"]:
                log["dr"].append(x_dr.copy())
                log["err_dr"].append(np.hypot(*(x_dr[:2] - x_true[:2])))
            if enabled["oracle"]:
                log["oracle"].append(est_o.copy())
                log["err_oracle"].append(np.hypot(*(est_o[:2] - x_true[:2])))

            if step % 50 == 0:
                msg = f"step {step:4d}"
                if enabled["slam"]:
                    msg += (f"  slam={log['err_slam'][-1]:.2f}"
                            f" leaves={n_leaves} active={len(active)}")
                if enabled["dr"]:
                    msg += f"  dr={log['err_dr'][-1]:.2f}"
                if enabled["oracle"]:
                    msg += f"  oracle={log['err_oracle'][-1]:.2f}"
                print(msg, flush=True)

            # -- animation frame (same loop; rbpf_swe.py style)
            if need_fig and step % stride == 0:
                _draw_live(axes, log, enabled, slam, slam_map, active,
                           x_true, step, t_now, sol, dt)
                if live_window:
                    plt.pause(0.001)
                if save_anim:
                    writer.grab_frame()

        # ensure the final state is captured
        if need_fig:
            _draw_live(axes, log, enabled, slam, slam_map, active,
                       x_true, steps - 1, (steps - 1) * dt, sol, dt)
            if save_anim:
                writer.grab_frame()

    if save_anim:
        print(f"saved animation -> {anim_path}", flush=True)

    wall = time.time() - t_start
    for k in log:
        log[k] = np.asarray(log[k])

    # ------------------------------------------------------------- metrics
    metrics = compute_metrics(cfg, log, enabled, slam_map, sol, dt, steps,
                              LAP, leafstep_full, wall)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    if cfg.viz.static_plots:
        if live_window:
            plt.ioff()                   # stop static figures from auto-showing
        make_plots(log, enabled, slam_map, sol, dt, results_dir)
    np.savez(os.path.join(results_dir, "log.npz"), **log)

    if live_window:
        print("Simulation done - close the figure window to exit.", flush=True)
        plt.show()


def _make_writer(cfg):
    from matplotlib.animation import FFMpegWriter, PillowWriter
    if str(cfg.viz.anim_format).lower() == "gif":
        return PillowWriter(fps=int(cfg.viz.fps)), "gif"
    return (FFMpegWriter(fps=int(cfg.viz.fps),
                         metadata={"title": "PDE SLAM", "artist": "pde_slam"}),
            "mp4")


# ---------------------------------------------------------------- metrics
def compute_metrics(cfg, log, enabled, slam_map, sol, dt, steps, LAP,
                    leafstep_full, wall):
    def rmse(e):
        return float(np.sqrt(np.mean(np.asarray(e) ** 2)))

    metrics = {"steps": steps, "laps": int(cfg.sim.laps),
               "particles": int(cfg.filter.n_particles),
               "methods": [k for k, v in enabled.items() if v],
               "wall_time_s": round(wall, 1)}
    for name in ["slam", "dr", "oracle"]:
        if not enabled[name]:
            continue
        e = log[f"err_{name}"]
        metrics[name] = {
            "rmse": rmse(e), "final_error": float(e[-1]),
            "rmse_lap1": rmse(e[:LAP]), "rmse_lap2": rmse(e[LAP:]),
        }

    if not enabled["slam"]:
        return metrics

    # field reconstruction over visited cells at final time
    frame_eval = steps - 1
    t_eval = frame_eval * dt
    visited = np.zeros((40, 40), bool)
    for p in log["true"]:
        ix, iy = int(round(p[0])), int(round(p[1]))
        visited[max(0, ix - 2):ix + 3, max(0, iy - 2):iy + 3] = True
    cells = np.argwhere(visited)
    pts = np.column_stack([cells[:, 0], cells[:, 1],
                           np.full(len(cells), t_eval)])
    pred, _ = slam_map.predict_with_var(pts)
    truth = np.column_stack([sol[frame_eval, 1, cells[:, 0], cells[:, 1]],
                             sol[frame_eval, 2, cells[:, 0], cells[:, 1]],
                             sol[frame_eval, 0, cells[:, 0], cells[:, 1]]])
    metrics["field_rmse"] = {
        "u": rmse(pred[:, 0] - truth[:, 0]),
        "v": rmse(pred[:, 1] - truth[:, 1]),
        "h": rmse(pred[:, 2] - truth[:, 2]),
        "h_field_std": float(truth[:, 2].std()),
        "n_cells": int(len(cells)),
    }
    metrics["efficiency"] = {
        "train_steps_gated": int(slam_map.total_train_steps),
        "train_steps_all_leaves": int(leafstep_full),
        "saving_factor": round(leafstep_full / max(1, slam_map.total_train_steps), 2),
        "mean_active_leaves": float(np.mean(log["n_active"])),
        "final_leaves": int(log["n_leaves"][-1]),
    }
    return metrics


# ------------------------------------------------------- live animation
def _setup_live(plt, enabled, live_window):
    """2x2 figure (map + error + leaf-count), rbpf_swe.py style. Reused for
    both the live window and the saved animation."""
    import matplotlib.gridspec as gridspec
    if live_window:
        plt.ion()
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.4, 1])
    ax_map = fig.add_subplot(gs[:, 0])    # trajectory + quadtree (left)
    ax_err = fig.add_subplot(gs[0, 1])    # position error (top-right)
    ax_leaf = fig.add_subplot(gs[1, 1])   # active vs total leaves (bottom-right)
    return fig, (ax_map, ax_err, ax_leaf)


def _draw_live(axes, log, enabled, slam, slam_map, active, x_true,
               frame, t_now, sol, dt):
    from matplotlib.patches import Rectangle
    ax_map, ax_err, ax_leaf = axes
    true_arr = np.asarray(log["true"])
    n = len(true_arr)
    t_axis = np.arange(n) * dt

    # --- map: field background, quadtree, particles, paths ---
    ax_map.clear()
    h_bg = np.where(sol[frame, 0] > 1e-3, sol[frame, 0], np.nan)
    ax_map.imshow(h_bg.T, origin="lower", cmap="Blues", alpha=0.5,
                  extent=[-0.5, 39.5, -0.5, 39.5])

    if enabled["slam"]:
        active_ids = {id(lf) for lf in (active or [])}
        for lf in slam_map.leaves():
            x0, x1, y0, y1 = lf.bounds
            if id(lf) in active_ids:
                ec, lw = "tab:red", 1.6           # trained THIS step (gated)
            elif lf.train_steps > 0:
                ec, lw = "tab:orange", 1.0        # trained earlier
            else:
                ec, lw = "0.7", 0.5               # untouched
            ax_map.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, ec=ec, lw=lw))
        ax_map.scatter(slam.xn[0], slam.xn[1], s=6, c="red", alpha=0.25,
                       label="Particles")

    ax_map.plot(true_arr[:, 0], true_arr[:, 1], "b-", lw=2, label="Ground truth")
    if enabled["dr"]:
        dr_arr = np.asarray(log["dr"])
        ax_map.plot(dr_arr[:, 0], dr_arr[:, 1], "--", c="0.4", lw=1,
                    label="Dead reckoning")
    if enabled["oracle"]:
        orc_arr = np.asarray(log["oracle"])
        ax_map.plot(orc_arr[:, 0], orc_arr[:, 1], "g-", lw=1, label="Oracle RBPF")
    if enabled["slam"]:
        slam_arr = np.asarray(log["slam"])
        ax_map.plot(slam_arr[:, 0], slam_arr[:, 1], "r-", lw=1.5, label="PDE SLAM")
        ax_map.plot(slam_arr[-1, 0], slam_arr[-1, 1], "ro", ms=8, mfc="none")
    ax_map.plot(x_true[0], x_true[1], "b*", ms=13)          # true pose
    ax_map.set_xlim(0, 40); ax_map.set_ylim(0, 40); ax_map.set_aspect("equal")
    ax_map.set_xlabel("X [cells]"); ax_map.set_ylabel("Y [cells]")
    ax_map.legend(loc="upper left", fontsize=8, ncol=2)

    title = f"PDE SLAM  t={t_now:.1f}s"
    if enabled["slam"]:
        title += (f" | err={log['err_slam'][-1]:.2f} cells | "
                  f"leaves={log['n_leaves'][-1]} (active {len(active or [])})\n"
                  "red = active leaves, orange = trained, gray = untouched")
    ax_map.set_title(title)

    # --- position error over time ---
    ax_err.clear()
    if enabled["dr"]:
        ax_err.plot(t_axis, log["err_dr"], c="0.4", ls="--", lw=1,
                    label="Dead reckoning")
    if enabled["oracle"]:
        ax_err.plot(t_axis, log["err_oracle"], "g-", lw=1, label="Oracle RBPF")
    if enabled["slam"]:
        ax_err.plot(t_axis, log["err_slam"], "r-", lw=1.2, label="PDE SLAM")
    ax_err.set_xlabel("Time [s]"); ax_err.set_ylabel("Position error [cells]")
    ax_err.set_title("Localization error"); ax_err.grid(alpha=0.3)
    ax_err.legend(fontsize=8)

    # --- uncertainty-gated training: active vs total leaves ---
    ax_leaf.clear()
    if enabled["slam"]:
        ax_leaf.plot(t_axis, log["n_leaves"], "k-", label="Total leaves")
        ax_leaf.plot(t_axis, log["n_active"], "r-", lw=1, label="Active leaves")
        ax_leaf.fill_between(t_axis, 0, log["n_active"], color="r", alpha=0.15)
        ax_leaf.set_xlabel("Time [s]"); ax_leaf.set_ylabel("# leaves")
        ax_leaf.set_title("Uncertainty-gated training"); ax_leaf.grid(alpha=0.3)
        ax_leaf.legend(fontsize=8)
    else:
        ax_leaf.axis("off")
        ax_leaf.text(0.5, 0.5, "PDE SLAM disabled", ha="center", va="center",
                     transform=ax_leaf.transAxes, fontsize=11, color="0.5")


# ---------------------------------------------------------------- plotting
def make_plots(log, enabled, slam_map, sol, dt, results_dir):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(log["true"])
    frame_eval = n - 1
    t_ax = np.arange(n) * dt

    # 1 — trajectories (+ final quadtree when SLAM is on)
    fig, ax = plt.subplots(figsize=(8, 8))
    h_bg = np.where(sol[frame_eval, 0] > 1e-3, sol[frame_eval, 0], np.nan)
    ax.imshow(h_bg.T, origin="lower", cmap="Blues", alpha=0.5,
              extent=[-0.5, 39.5, -0.5, 39.5])
    if enabled["slam"]:
        for lf in slam_map.leaves():
            x0, x1, y0, y1 = lf.bounds
            trained = lf.train_steps > 0
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec="tab:orange" if trained else "0.6",
                                   lw=1.5 if trained else 0.7))
    ax.plot(log["true"][:, 0], log["true"][:, 1], "b-", lw=2, label="Ground truth")
    if enabled["dr"]:
        ax.plot(log["dr"][:, 0], log["dr"][:, 1], "--", c="0.4", lw=1.2,
                label="Dead reckoning")
    if enabled["oracle"]:
        ax.plot(log["oracle"][:, 0], log["oracle"][:, 1], "g-", lw=1.2,
                label="RBPF known field (oracle)")
    if enabled["slam"]:
        ax.plot(log["slam"][:, 0], log["slam"][:, 1], "r-", lw=1.5,
                label="PDE SLAM (ours)")
    ax.set_xlim(0, 40); ax.set_ylim(0, 40); ax.set_aspect("equal")
    ax.set_xlabel("X [cells]"); ax.set_ylabel("Y [cells]")
    ax.set_title("PDE SLAM with adaptive quadtree PINN map\n"
                 "orange = trained leaves, gray = untouched leaves")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "trajectories_quadtree.png"), dpi=150)
    plt.close(fig)

    # 2 — error vs time
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if enabled["dr"]:
        ax.plot(t_ax, log["err_dr"], c="0.4", ls="--", label="Dead reckoning")
    if enabled["oracle"]:
        ax.plot(t_ax, log["err_oracle"], "g-", lw=1, label="Oracle RBPF")
    if enabled["slam"]:
        ax.plot(t_ax, log["err_slam"], "r-", lw=1.2, label="PDE SLAM")
    if n // 2 > 0:
        ax.axvline(n // 2 * dt, c="k", lw=0.8, ls=":")
        ax.text(n // 2 * dt, ax.get_ylim()[1] * 0.9, " lap 2", fontsize=9)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Position error [cells]")
    ax.set_title("Localization error"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "error_vs_time.png"), dpi=150)
    plt.close(fig)

    if not enabled["slam"]:
        return

    # 3 — field reconstruction (h channel) over visited cells
    visited = np.zeros((40, 40), bool)
    for p in log["true"]:
        ix, iy = int(round(p[0])), int(round(p[1]))
        visited[max(0, ix - 2):ix + 3, max(0, iy - 2):iy + 3] = True
    cells = np.argwhere(visited)
    pts = np.column_stack([cells[:, 0], cells[:, 1],
                           np.full(len(cells), frame_eval * dt)])
    pred, _ = slam_map.predict_with_var(pts)
    truth = np.column_stack([sol[frame_eval, 1, cells[:, 0], cells[:, 1]],
                             sol[frame_eval, 2, cells[:, 0], cells[:, 1]],
                             sol[frame_eval, 0, cells[:, 0], cells[:, 1]]])
    true_h = np.full((40, 40), np.nan)
    pred_h = np.full((40, 40), np.nan)
    err_h = np.full((40, 40), np.nan)
    for (ix, iy), p, tr in zip(cells, pred, truth):
        true_h[ix, iy] = tr[2]
        pred_h[ix, iy] = p[2]
        err_h[ix, iy] = abs(p[2] - tr[2])
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    for a, img, title in zip(
            axs, [true_h, pred_h, err_h],
            ["True h (visited cells)", "PINN map h", "|error|"]):
        im = a.imshow(img.T, origin="lower", cmap="viridis")
        a.plot(log["true"][:, 0], log["true"][:, 1], "r-", lw=0.7, alpha=0.7)
        a.set_title(title)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"Field reconstruction at t={frame_eval * dt:.0f}s")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "field_reconstruction.png"), dpi=150)
    plt.close(fig)

    # 4 — computational efficiency
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_ax, log["n_leaves"], "k-", label="Total leaves")
    ax.plot(t_ax, log["n_active"], "r-", lw=1, label="Active (trained) leaves")
    ax.fill_between(t_ax, 0, log["n_active"], color="r", alpha=0.15)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("# leaves")
    ax.set_title("Uncertainty-gated training: active vs total PINN models")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "active_leaves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
