# From PDE-Based Localization to PDE SLAM via Uncertainty-Gated Local Models

**One-pager — extension of *Localization in Spatiotemporal Fields via Environmental PDEs* (IROS 2026)**

## Problem

The IROS 2026 paper localizes a vehicle against *known* PDE-governed fields (SWE, advection-diffusion) using an RBPF: particles carry the nonlinear pose state, per-particle Kalman filters track linear sensor bias. The natural extension, flagged in the paper's conclusion, is SLAM: jointly estimate the vehicle state **and** the field itself — in practice, the PDE's initial conditions and forcing parameters — online. The obstacle is cost: fitting one global PDE model over the full t×f×w×h tensor (1000×4×40×40) is prohibitive for online, per-particle map updates.

## Key idea: the field map factorizes spatially

Rao-Blackwellization already gives us the right structure. Extend the factorization from bias-only to bias + map, FastSLAM-style:

p(x_{1:t}, Θ | y_{1:t}) = p(x_{1:t} | y_{1:t}) · Π_k p(Θ_k | x_{1:t}, y_{1:t})

where Θ = {Θ_1, …, Θ_K} are the parameters (initial conditions, source/forcing terms) of **K local PDE models**, each owning a spatiotemporal block of the domain. Concretely: decompose 1000×4×40×40 into K = 10 local models of 100×4×4×4 each. Feature count f = 4 is fixed; only t, w, h shrink. Each local model is a small solver (or PINN/neural surrogate) with its own boundary interface to its neighbors.

**Hypothesis:** given the robot state estimate and its covariance, only the local models whose blocks intersect the pose uncertainty ellipse need to be updated at time t. A robot with a 3σ ellipse covering 2–3 blocks trains 2–3 models jointly instead of all 10 — an expected 3–5× reduction in map-update cost with negligible accuracy loss, because distant blocks receive no observations and their posteriors are unchanged by construction.

## Method (iteration 1: fixed decomposition)

The pipeline per time step: (1) RBPF pose prediction as in the current paper; (2) **gate selection** — compute the set A_t of blocks overlapping the pose covariance ellipse (Mahalanobis threshold on block centers); (3) measurement update of pose particles against the *current* local field predictions in A_t; (4) **sequential map update** — refit/assimilate only the models in A_t, conditioning on particle-weighted observations; models sharing a boundary with an active block exchange boundary values so continuity is preserved (Schwarz-style coupling). Blocks outside A_t just propagate their PDE forward in time — cheap, no fitting. This yields a SLAM system where the map is physical (PDE parameters, not occupancy), and consistency between blocks comes from shared boundary conditions rather than a global solve.

## Method (iteration 2: adaptive decomposition)

Fixed 4×4 tiles waste capacity where the field is smooth and under-resolve where it is sharp or where the robot lingers. Replace the fixed grid with a **quadtree**: split a block when (a) pose covariance conditioned on it stays high (field locally uninformative → need finer model) or (b) PDE residual/data misfit exceeds a threshold; merge siblings when both are low. The covariance matrix drives refinement, so resolution follows the robot's uncertainty rather than a uniform budget. Quadtree neighbors-finding gives O(log n) gate selection, and the boundary-exchange machinery from iteration 1 carries over to hanging nodes.

## Evaluation plan

Reuse both case studies from the paper (SWE coastal wave, advection-diffusion water quality) plus the ASV field data. Compare: (i) global joint PDE fit (baseline), (ii) fixed 10-block gated SLAM, (iii) adaptive quadtree SLAM. Metrics: pose RMSE/final error (as in the paper), field reconstruction error (per-channel RMSE against held-out grid), wall-clock per step, and number of active models per step. Ablations: gate threshold (1σ–3σ), block count K, with/without boundary exchange.

## Risks and mitigations

Block-boundary artifacts can bias the measurement likelihood → overlap blocks by one cell and blend. Loop closures touch stale blocks → trigger a one-shot refit of re-entered blocks (their stored posterior is the prior, so this is cheap). Initial-condition estimation is ill-posed for short residence times → per-block priors from historical data (GEBCO bathymetry, recorded wind), consistent with the paper's forcing-term setup.

**Deliverable claim for the paper:** the first SLAM formulation in which the map is a set of locally coupled environmental PDE models, with robot-uncertainty-gated model training providing the computational tractability that a global PDE fit lacks.
