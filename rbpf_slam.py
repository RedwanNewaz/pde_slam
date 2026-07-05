"""
rbpf_slam.py
------------
Rao-Blackwellized particle filter for PDE SLAM (vectorized).

Ports pf/RaoBlackwellizedPF.py from the pde_localization repo, with changes
that turn localization into SLAM:

1. The nominal field is queried from the *learned* map (QuadTreeMap of PINN
   leaves) instead of the ground-truth field.
2. The innovation covariance is inflated by the per-particle map uncertainty
   (leaf EMA misfit + observation-density gate), so untrained regions yield a
   flat likelihood and the filter degrades gracefully to dead reckoning until
   the map is learned.

Because R, Q_lin, P0 are diagonal and H = I, the per-particle Kalman filter
is exactly per-channel scalar; P is stored as (N, 3) and all updates are
vectorized (identical math to the loop version in the repo).

State partition (as in the paper):
   xn = [x, y, theta, v]        particles
   xl = [bias_u, bias_v, bias_h]  per-particle scalar KFs
"""

import math
import numpy as np


class RBPFSLAM:
    def __init__(self, field_map, n_particles=500,
                 process_noise=None, measurement_noise=None,
                 lin_process_noise=None, P0_lin=None, seed=0):
        self.map = field_map
        self.N = n_particles
        self.rng = np.random.default_rng(seed)

        self.Q_nl = process_noise if process_noise is not None \
            else np.diag([0.25, np.deg2rad(9.0)]) ** 2
        # diagonals only (KF is per-channel scalar)
        self.r = np.array([0.2, 0.2, 0.1]) ** 2 if measurement_noise is None \
            else np.diag(measurement_noise)
        self.q_lin = np.array([1e-4, 1e-4, 1e-4]) ** 2 if lin_process_noise is None \
            else np.diag(lin_process_noise)
        # keep P0 small so S ~= R and weights stay sharp (repo docstring)
        self.p0 = np.array([0.05, 0.05, 0.02]) ** 2 if P0_lin is None \
            else np.diag(P0_lin)

        self.xn = np.zeros((4, self.N))
        self.xl = np.zeros((self.N, 3))
        self.P = np.tile(self.p0, (self.N, 1))     # (N, 3) diagonal covs
        self.weights = np.ones(self.N) / self.N
        self.trusted_frac = 1.0

    def initialize(self, initial_state, std_dev):
        for k in range(4):
            self.xn[k] = initial_state[k] + self.rng.normal(0, std_dev[k], self.N)
        self.xl[:] = 0.0
        self.P = np.tile(self.p0, (self.N, 1))
        self.weights[:] = 1.0 / self.N

    @staticmethod
    def _motion_model(x, u, dt):
        """Single-state unicycle step (kept for external use)."""
        F = np.array([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0], [0, 0, 0, 0.]])
        B = np.array([[dt * math.cos(x[2]), 0.],
                      [dt * math.sin(x[2]), 0.],
                      [0., dt],
                      [1., 0.]])
        return (F @ x + B @ u).ravel()

    def predict(self, control, dt):
        ancestors = self._systematic_resample_indices()
        xn = self.xn[:, ancestors]
        self.xl = self.xl[ancestors]
        self.P = self.P[ancestors] + self.q_lin

        noise_std = np.sqrt(np.diag(self.Q_nl))
        v_cmd = control[0] + self.rng.normal(0, noise_std[0], self.N)
        w_cmd = control[1] + self.rng.normal(0, noise_std[1], self.N)
        x, y, th = xn[0], xn[1], xn[2]
        self.xn = np.vstack([x + dt * np.cos(th) * v_cmd,
                             y + dt * np.sin(th) * v_cmd,
                             th + dt * w_cmd,
                             v_cmd])
        self.weights[:] = 1.0 / self.N

    def update(self, measurement, t_now):
        y = np.asarray(measurement, dtype=float)
        pts = np.column_stack([self.xn[0], self.xn[1], np.full(self.N, t_now)])
        h_base, map_var = self.map.predict_with_var(pts)   # (N,3), (N,3)

        e = y[None, :] - h_base - self.xl                  # (N,3)
        S = self.P + self.r[None, :] + map_var             # (N,3) diagonal
        log_w = -0.5 * np.sum(np.log(S) + e * e / S, axis=1) \
                - 1.5 * math.log(2 * math.pi)

        # Mixed-variance correction: particles in untrusted map regions must
        # be weighted NEUTRALLY. A wide Gaussian still has a much lower peak
        # than a trusted one, which would systematically drag the posterior
        # back into the mapped band regardless of fit quality.
        untrusted = map_var[:, 0] >= 24.99
        trusted = ~untrusted
        self.trusted_frac = float(trusted.mean())   # anchoring diagnostic
        if trusted.any() and untrusted.any():
            lw_t = log_w[trusted]
            c0 = lw_t.max()
            neutral = c0 + math.log(np.mean(np.exp(lw_t - c0)))
            log_w[untrusted] = neutral
        elif not trusted.any():
            log_w[:] = 0.0

        # scalar Kalman updates (K ~ 0 for untrusted particles anyway)
        K = self.P / S
        self.xl = self.xl + K * e
        self.P = self.P * (1.0 - K)

        c = np.max(log_w)
        w = np.exp(log_w - c)
        self.weights = w / w.sum()

    def resample(self, threshold_ratio=0.5):
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < threshold_ratio * self.N:
            idx = self._systematic_resample_indices()
            self.xn = self.xn[:, idx]
            self.xl = self.xl[idx]
            self.P = self.P[idx]
            self.weights[:] = 1.0 / self.N

    def _systematic_resample_indices(self):
        cumsum = np.cumsum(self.weights)
        step = 1.0 / self.N
        u0 = self.rng.uniform(0.0, step)
        positions = np.arange(self.N) * step + u0
        return np.clip(np.searchsorted(cumsum, positions), 0, self.N - 1)

    def estimate(self):
        xn_mean = self.xn @ self.weights
        diff = self.xn - xn_mean[:, None]
        cov = (diff * self.weights[None, :]) @ diff.T
        return xn_mean, cov
