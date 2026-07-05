# PDE SLAM with Adaptive Quadtree PINN Map (Method 2 prototype)

Extension of *Localization in Spatiotemporal Fields via Environmental PDEs*
(IROS 2026) from localization to SLAM: the robot jointly estimates its pose
and the environmental field, with the field map represented as an **adaptive
quadtree of physics-informed neural networks (PINNs)** and only the leaves
inside the pose-covariance ellipse trained each step (uncertainty gating) —
Method 2 of `Manuscript/PDE_SLAM_OnePager.md`.

## Layout

- `pinn_map.py` — JAX PINN leaves (MLP 3->64x3->3, linearized-SWE residuals with
  per-leaf learnable coefficients), quadtree with capacity/misfit splitting,
  covariance-gated active set, trust model (observation density + age).
- `rbpf_slam.py` — vectorized Rao-Blackwellized PF (pose particles + per-particle
  scalar bias KFs, as in the paper) whose measurement model queries the learned map.
- `run_experiment.py` — 2 laps of the elliptical trajectory from
  `pde_localization/trajectory/SimTrajectory.py::get_ellipse_points()`, SWE field
  from `pde_localization/data/swe/solutions.npy` (1000x3x40x40, channels h/u/v).
  Baselines: dead reckoning and an oracle RBPF with the true field (= the paper's
  localization setting). Outputs to `results/`.

## Setup & run (uv)

```bash
uv sync                    # CPU (any OS)
uv sync --extra cuda       # Linux + NVIDIA GPU (CUDA 12); JAX uses it automatically
uv run python run_experiment.py
```

The first line printed reports the JAX backend (`cpu` / `gpu`). Needs the
sibling directory `pde_localization` for trajectory + data. On Windows, native
JAX is CPU-only — use WSL2 for GPU. Env overrides: `PDESLAM_STEPS`,
`PDESLAM_SEED`.

## Key design decisions (each fixes a observed failure mode)

1. **Trust = data density + age.** The map is only trusted where >=5 observations
   exist in a 5x5-cell window AND the oldest predates the query by 8 s. Trusting
   the pass currently being mapped creates a self-confirming drift loop.
2. **Neutral weighting for untrusted particles.** A wide Gaussian still has a far
   lower peak than a trusted one; naive weighting drags the posterior back into
   the mapped band regardless of fit. Untrusted particles receive the *average*
   trusted likelihood.
3. **Write-once, anchored mapping.** Observations are registered at the estimated
   pose only for frontier cells (<3 obs) and only while the filter is anchored
   (tight covariance or a trusted correction this step). Re-writing mapped cells
   at a drifted estimate poisons the anchor (drift-consistent corruption).
4. **Temporal clamping.** Queries clamp t into the leaf's data support; the field
   is quasi-static and PINN extrapolation in t drifts.
5. **Refinement signal.** A leaf splits (up to depth 3) when its buffer exceeds a
   capacity budget or its validation h-RMSE over the whole buffer exceeds 0.3 —
   EMA-of-recent misfit alone never triggers because one net can fit the recent
   band while forgetting the rest of its block. Children inherit the parent's
   weights exactly (shared global input normalization) and are consolidated with
   30 gradient steps at split time.

## Results (998 steps = 2 laps, 500 particles, seed 42)

| Method | RMSE | Lap 1 | Lap 2 |
|---|---|---|---|
| Dead reckoning | 2.13 | 0.63 | 2.94 |
| **PDE SLAM (ours)** | **1.68** | **0.50** | **2.32** |
| Oracle RBPF (known field) | 0.38 | 0.51 | 0.14 |

(units: grid cells)

- **Loop closure works:** mid-lap-2 SLAM error drops to 0.2-0.8 cells while dead
  reckoning reaches 4-4.6 (see `results/error_vs_time.png`).
- **Field reconstruction:** h-RMSE 3.44 m over visited cells vs field std 4.31 m
  (u: 0.71, v: 0.86); best along the mapped band (`results/field_reconstruction.png`).
- **Gating efficiency:** 10.7k gated leaf-updates vs 16.9k for train-everything
  (1.58x saving at only 4 leaves; grows with leaf count — earlier 16-leaf runs
  reached 4.6-9x), mean 1.8 active leaves (`results/active_leaves.png`).
- Runtime: ~21 s for 998 steps on CPU (~21 ms/step including PINN training).

### Seed sensitivity (same config)

| Seed | SLAM | DR | Oracle |
|---|---|---|---|
| 42 | 1.68 | 2.13 | 0.38 |
| 7 | 3.47 | 4.94 | 0.30 |
| 3 | 4.23 | 3.43 | 5.59 |

Seed 3 defeats even the known-field oracle: under that noise realization the
trajectory crosses weakly-informative parts of the field (h iso-contours),
consistent with the variance the paper reports for the advection-diffusion case.
SLAM beats dead reckoning whenever the underlying localization problem is
observable.

## Known limitations / next steps

- Single shared map registered at the mean pose; a FastSLAM-style per-particle
  map (or pose-graph correction of map anchor points) would remove the residual
  drift lock-in visible late in lap 2.
- Physics loss uses linearized SWE with learnable coefficients as regularization;
  the full nonlinear residuals (and the pde_model solver as a data source) are a
  natural upgrade.
- Merging of sibling leaves and covariance-driven (not only data-driven)
  refinement are implemented in the one-pager's design but not yet exercised here.
