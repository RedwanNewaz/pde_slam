"""
run_experiment.py
-----------------
PDE SLAM with PINN quadtree map (Method 2), evaluated on the elliptical
trajectory from pde_localization/trajectory/SimTrajectory.py.

Setup
  - Ground truth: unicycle robot tracking get_ellipse_points() waypoints for
    2 laps, with actuation noise; senses (u, v, h) of the true SWE field
    (data/swe/solutions.npy) at its position + measurement noise.
  - PDE SLAM (ours): RBPF pose filter whose measurement model queries the
    online-learned quadtree PINN map; only leaves inside the 3-sigma pose
    covariance ellipse are trained each step (uncertainty gating); leaves
    split adaptively on persistent data misfit.
  - Baselines: dead reckoning (commanded controls only) and an oracle RBPF
    that knows the true field (the IROS-2026 localization setting).

Outputs: results/*.png, results/metrics.json
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pinn_map import QuadTreeMap, TrueFieldMap, DOMAIN
from rbpf_slam import RBPFSLAM

HERE = os.path.dirname(os.path.abspath(__file__))
LOC_ROOT = os.path.join(HERE, "..", "pde_localization")
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# ---------------------------------------------------------------- parameters
DT = 0.1
LAPS = 2
N_PARTICLES = 500
SEED = int(os.environ.get("PDESLAM_SEED", 42))
ACT_NOISE = np.array([0.2, np.deg2rad(8.0)])       # actuation noise (v, w)
MEAS_STD = np.array([0.2, 0.2, 0.1])               # sensor noise (u, v, h)
TRAIN_ITERS = 8                                     # grad steps per active leaf
T_EVAL = None                                       # set after STEPS known

# ------------------------------------------------- trajectory (repo import)
spec = importlib.util.spec_from_file_location(
    "SimTrajectory", os.path.join(LOC_ROOT, "trajectory", "SimTrajectory.py"))
SimTrajectory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SimTrajectory)

waypoints = SimTrajectory.get_ellipse_points()
U_ctrl = SimTrajectory.gen_control_inputs(waypoints, DT)
LAP = U_ctrl.shape[0]
STEPS = LAPS * LAP
if os.environ.get("PDESLAM_STEPS"):          # smoke-test override
    STEPS = int(os.environ["PDESLAM_STEPS"])

# ---------------------------------------------------------------- field data
sol = np.load(os.path.join(LOC_ROOT, "data", "swe", "solutions.npy"))
# sanitize solver blow-ups near dry cells
sol[:, 0] = np.clip(sol[:, 0], -1.0, 20.0)
sol[:, 1:] = np.clip(sol[:, 1:], -3.0, 3.0)
assert sol.shape[0] >= STEPS, "not enough field frames"


def sense_true(x, y, frame, rng):
    ix = int(np.clip(round(x), 0, 39))
    iy = int(np.clip(round(y), 0, 39))
    h = sol[frame, 0, ix, iy]
    u = sol[frame, 1, ix, iy]
    v = sol[frame, 2, ix, iy]
    return np.array([u, v, h]) + rng.normal(0, MEAS_STD)


def motion(x, u, dt=DT):
    return RBPFSLAM._motion_model(np.asarray(x, float), np.asarray(u, float), dt)


# ---------------------------------------------------------------- simulation
def main():
    import jax
    print(f"JAX backend: {jax.default_backend()} "
          f"(devices: {[d.device_kind for d in jax.devices()]})", flush=True)
    rng = np.random.default_rng(SEED)
    theta0 = np.arctan2(waypoints[1, 1] - waypoints[0, 1],
                        waypoints[1, 0] - waypoints[0, 0])
    x0 = [waypoints[0, 0], waypoints[0, 1], theta0, 0.0]
    init_std = [1.0, 1.0, np.deg2rad(15.0), 0.5]

    # estimators
    slam_map = QuadTreeMap(seed=SEED)
    slam = RBPFSLAM(slam_map, N_PARTICLES, seed=SEED)
    slam.initialize(x0, init_std)

    oracle = RBPFSLAM(TrueFieldMap(sol), N_PARTICLES, seed=SEED + 1)
    oracle.initialize(x0, init_std)

    x_true = np.array(x0, float)
    x_dr = np.array(x0, float)

    log = {k: [] for k in ["true", "slam", "dr", "oracle",
                           "n_active", "n_leaves", "err_slam",
                           "err_dr", "err_oracle"]}
    t_start = time.time()
    leafstep_full = 0   # hypothetical cost of training ALL leaves every step

    for step in range(STEPS):
        t_now = step * DT
        control = U_ctrl[step % LAP]

        # -- ground truth motion + sensing
        u_act = control + rng.normal(0, ACT_NOISE)
        x_true = motion(x_true, u_act)
        y = sense_true(x_true[0], x_true[1], step, rng)

        # -- dead reckoning
        x_dr = motion(x_dr, control)

        # -- oracle RBPF (known field)
        oracle.predict(control, DT)
        oracle.update(y, t_now)
        oracle.resample()
        est_o, _ = oracle.estimate()

        # -- PDE SLAM
        slam.predict(control, DT)
        slam.update(y, t_now)
        slam.resample()
        est_s, cov_s = slam.estimate()

        # map update: observation registered at ESTIMATED pose, but only
        # when (a) the pose is confident (a drifting estimate must not
        # corrupt the map) and (b) the observation is consistent with an
        # already-trusted map value there -- a large innovation on a mapped
        # cell signals mislocalization, and writing it in would poison the
        # good map built on the previous pass
        if cov_s[0, 0] + cov_s[1, 1] < 2 * 2.5 ** 2:
            # consistency check strictly CELL-level: only reject when THIS
            # cell has aged data (a neighborhood-level check rejects every
            # frontier observation, starving the map)
            ix = int(np.clip(round(est_s[0]), 0, 39))
            iy = int(np.clip(round(est_s[1]), 0, 39))
            # WRITE-ONCE policy: only frontier cells receive observations.
            # Re-writing already-mapped cells at a drifted estimate poisons
            # the anchor the filter relies on (drift-consistent corruption).
            # ... and only while the filter is ANCHORED: either the pose
            # covariance is tight or a good fraction of particles just
            # received trusted map corrections. Unanchored frontier writes
            # are registered at a drifting pose and poison the map.
            anchored = (cov_s[0, 0] + cov_s[1, 1] < 2 * 1.5 ** 2
                        or slam.trusted_frac > 0.2)
            if slam_map.obs_counts[ix, iy] < 3 and anchored:
                slam_map.add_observation(est_s[0], est_s[1], t_now, y)

        # uncertainty-gated training (Method 2)
        active = slam_map.active_leaves(est_s[:2], cov_s[:2, :2])
        if step % 25 == 0:   # periodic refinement check (validation pass)
            if any(slam_map.maybe_split(lf) for lf in list(active)):
                active = slam_map.active_leaves(est_s[:2], cov_s[:2, :2])
        for lf in active:
            slam_map.train_leaf(lf, t_now, TRAIN_ITERS)

        n_leaves = len(slam_map.leaves())
        leafstep_full += n_leaves * TRAIN_ITERS

        # -- logging
        log["true"].append(x_true.copy())
        log["slam"].append(est_s.copy())
        log["dr"].append(x_dr.copy())
        log["oracle"].append(est_o.copy())
        log["n_active"].append(len(active))
        log["n_leaves"].append(n_leaves)
        log["err_slam"].append(np.hypot(*(est_s[:2] - x_true[:2])))
        log["err_dr"].append(np.hypot(*(x_dr[:2] - x_true[:2])))
        log["err_oracle"].append(np.hypot(*(est_o[:2] - x_true[:2])))

        if step % 50 == 0:
            print(f"step {step:4d}  err slam={log['err_slam'][-1]:.2f} "
                  f"dr={log['err_dr'][-1]:.2f} oracle={log['err_oracle'][-1]:.2f} "
                  f"leaves={n_leaves} active={len(active)}", flush=True)

    wall = time.time() - t_start
    for k in log:
        log[k] = np.asarray(log[k])

    # ------------------------------------------------------------- metrics
    def rmse(e):
        return float(np.sqrt(np.mean(np.asarray(e) ** 2)))

    metrics = {"steps": STEPS, "laps": LAPS, "particles": N_PARTICLES,
               "wall_time_s": round(wall, 1)}
    for name in ["slam", "dr", "oracle"]:
        e = log[f"err_{name}"]
        metrics[name] = {
            "rmse": rmse(e), "final_error": float(e[-1]),
            "rmse_lap1": rmse(e[:LAP]), "rmse_lap2": rmse(e[LAP:]),
        }

    # field reconstruction over visited cells at final time
    frame_eval = STEPS - 1
    t_eval = frame_eval * DT
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
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    make_plots(log, slam_map, cells, pred, truth, frame_eval, visited)
    np.savez(os.path.join(RESULTS, "log.npz"), **log)


# ---------------------------------------------------------------- plotting
def make_plots(log, slam_map, cells, pred, truth, frame_eval, visited):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # 1 — trajectories + final quadtree
    fig, ax = plt.subplots(figsize=(8, 8))
    h_bg = np.where(sol[frame_eval, 0] > 1e-3, sol[frame_eval, 0], np.nan)
    ax.imshow(h_bg.T, origin="lower", cmap="Blues", alpha=0.5,
              extent=[-0.5, 39.5, -0.5, 39.5])
    for lf in slam_map.leaves():
        x0, x1, y0, y1 = lf.bounds
        trained = lf.train_steps > 0
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               ec="tab:orange" if trained else "0.6",
                               lw=1.5 if trained else 0.7))
    ax.plot(log["true"][:, 0], log["true"][:, 1], "b-", lw=2, label="Ground truth")
    ax.plot(log["dr"][:, 0], log["dr"][:, 1], "--", c="0.4", lw=1.2,
            label="Dead reckoning")
    ax.plot(log["oracle"][:, 0], log["oracle"][:, 1], "g-", lw=1.2,
            label="RBPF known field (oracle)")
    ax.plot(log["slam"][:, 0], log["slam"][:, 1], "r-", lw=1.5,
            label="PDE SLAM (ours)")
    ax.set_xlim(0, 40); ax.set_ylim(0, 40); ax.set_aspect("equal")
    ax.set_xlabel("X [cells]"); ax.set_ylabel("Y [cells]")
    ax.set_title("PDE SLAM with adaptive quadtree PINN map (2 laps)\n"
                 "orange = trained leaves, gray = untouched leaves")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "trajectories_quadtree.png"), dpi=150)

    # 2 — error vs time
    LAP = len(log["true"]) // 2
    t_ax = np.arange(len(log["err_slam"])) * DT
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t_ax, log["err_dr"], c="0.4", ls="--", label="Dead reckoning")
    ax.plot(t_ax, log["err_oracle"], "g-", lw=1, label="Oracle RBPF")
    ax.plot(t_ax, log["err_slam"], "r-", lw=1.2, label="PDE SLAM")
    ax.axvline(LAP * DT, c="k", lw=0.8, ls=":")
    ax.text(LAP * DT, ax.get_ylim()[1] * 0.9, " lap 2", fontsize=9)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Position error [cells]")
    ax.set_title("Localization error"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "error_vs_time.png"), dpi=150)

    # 3 — field reconstruction (h channel)
    true_h = np.full((40, 40), np.nan)
    pred_h = np.full((40, 40), np.nan)
    err_h = np.full((40, 40), np.nan)
    for (ix, iy), p, tr in zip(cells, pred, truth):
        true_h[ix, iy] = tr[2]
        pred_h[ix, iy] = p[2]
        err_h[ix, iy] = abs(p[2] - tr[2])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for a, img, title in zip(
            axes, [true_h, pred_h, err_h],
            ["True h (visited cells)", "PINN map h", "|error|"]):
        im = a.imshow(img.T, origin="lower", cmap="viridis")
        a.plot(log["true"][:, 0], log["true"][:, 1], "r-", lw=0.7, alpha=0.7)
        a.set_title(title)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"Field reconstruction at t={frame_eval * DT:.0f}s")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "field_reconstruction.png"), dpi=150)

    # 4 — computational efficiency
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_ax, log["n_leaves"], "k-", label="Total leaves")
    ax.plot(t_ax, log["n_active"], "r-", lw=1, label="Active (trained) leaves")
    ax.fill_between(t_ax, 0, log["n_active"], color="r", alpha=0.15)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("# leaves")
    ax.set_title("Uncertainty-gated training: active vs total PINN models")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "active_leaves.png"), dpi=150)


if __name__ == "__main__":
    main()
