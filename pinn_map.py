"""
pinn_map.py
-----------
Adaptive quadtree field map with PINN leaves (Method 2 of the PDE-SLAM one-pager).

The map over the domain W = [0,40]x[0,40] is a quadtree. Each leaf owns a small
physics-informed neural network (PINN) phi_theta(x, y, t) -> (u, v, h) trained
online from the observations that fall inside its block. Physics residuals use
the *linearized* shallow water equations with per-leaf learnable coefficients
(system-identification flavor, robust to unit mismatch):

    r1 = h_t + a1 (u_x + v_y)
    r2 = u_t + a2 h_x
    r3 = v_t + a2 h_y      with a1 = exp(c1), a2 = exp(c2) learnable.

Leaves are split when their recent data misfit (EMA) stays high despite enough
observations -> resolution follows information content, per Method 2.

Backend: JAX (CPU). All leaves share one global input normalization so a child
can inherit its parent's weights exactly on split.
"""

import functools
import numpy as np
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------- constants
DOMAIN = (0.0, 40.0, 0.0, 40.0)      # x0, x1, y0, y1  (grid-cell units)
T_MAX = 100.0                        # [s] normalization horizon for t input
OUT_SCALE = np.array([1.0, 1.0, 5.0])  # u, v, h output scales
IN_CENTER = np.array([20.0, 20.0, T_MAX / 2])
IN_SCALE = np.array([20.0, 20.0, T_MAX / 2])

HIDDEN = 64
N_LAYERS = 3
LR = 2e-3
BATCH = 256          # data minibatch (sampled with replacement -> fixed shape)
N_COL = 128          # physics collocation points
W_PHYS = 1e-2        # physics loss weight
PAD_QUERY = 512      # fixed query batch (jit shape stability, >= n_particles)
VAR_MAX = 25.0       # variance of an untrained leaf (=> flat likelihood)
EMA = 0.98

# ---------------------------------------------------------------- MLP + Adam

def init_params(key):
    sizes = [3] + [HIDDEN] * N_LAYERS + [3]
    params = {"c1": jnp.array(0.0), "c2": jnp.array(0.0)}
    for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, sub = jax.random.split(key)
        params[f"W{i}"] = jax.random.normal(sub, (m, n)) * jnp.sqrt(2.0 / m)
        params[f"b{i}"] = jnp.zeros(n)
    return params


def apply_net(params, xyt):
    """xyt: (3,) physical coords -> (u, v, h) physical."""
    z = (xyt - IN_CENTER) / IN_SCALE
    for i in range(N_LAYERS):
        z = jnp.tanh(z @ params[f"W{i}"] + params[f"b{i}"])
    out = z @ params[f"W{N_LAYERS}"] + params[f"b{N_LAYERS}"]
    return out * OUT_SCALE


apply_batch = jax.jit(jax.vmap(apply_net, in_axes=(None, 0)))


def _loss(params, Xb, Yb, Xc):
    # data term (normalized residuals)
    pred = jax.vmap(apply_net, in_axes=(None, 0))(params, Xb)
    data = jnp.mean(((pred - Yb) / OUT_SCALE) ** 2)
    # physics term: J[n, out_k, in_j]
    J = jax.vmap(jax.jacfwd(apply_net, argnums=1), in_axes=(None, 0))(params, Xc)
    a1, a2 = jnp.exp(params["c1"]), jnp.exp(params["c2"])
    r1 = J[:, 2, 2] + a1 * (J[:, 0, 0] + J[:, 1, 1])
    r2 = J[:, 0, 2] + a2 * J[:, 2, 0]
    r3 = J[:, 1, 2] + a2 * J[:, 2, 1]
    phys = jnp.mean(r1 ** 2 + r2 ** 2 + r3 ** 2)
    return data + W_PHYS * phys


@jax.jit
def adam_step(params, m, v, t, Xb, Yb, Xc):
    loss, g = jax.value_and_grad(_loss)(params, Xb, Yb, Xc)
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = jax.tree.map(lambda a, b: b1 * a + (1 - b1) * b, m, g)
    v = jax.tree.map(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
    mh = jax.tree.map(lambda a: a / (1 - b1 ** t), m)
    vh = jax.tree.map(lambda a: a / (1 - b2 ** t), v)
    params = jax.tree.map(lambda p, a, b: p - LR * a / (jnp.sqrt(b) + eps),
                          params, mh, vh)
    return params, m, v, loss


# ---------------------------------------------------------------- quadtree

class Leaf:
    _uid = 0

    def __init__(self, bounds, depth, params, rng):
        self.bounds = bounds            # (x0, x1, y0, y1)
        self.depth = depth
        self.children = None
        self.params = params
        self.m = jax.tree.map(jnp.zeros_like, params)
        self.v = jax.tree.map(jnp.zeros_like, params)
        self.t_adam = 0
        self.rng = rng
        self.buf_X = []                 # (x, y, t)
        self.buf_Y = []                 # (u, v, h)
        self.ema_var = np.full(3, VAR_MAX)   # per-channel EMA of sq. error
        self.t_range = [np.inf, -np.inf]     # time support of the data
        self.train_steps = 0
        self.id = Leaf._uid
        Leaf._uid += 1

    @property
    def n_obs(self):
        return len(self.buf_X)

    def contains(self, x, y):
        x0, x1, y0, y1 = self.bounds
        return x0 <= x < x1 and y0 <= y < y1


class QuadTreeMap:
    """Adaptive quadtree of PINN leaves with covariance-gated training."""

    def __init__(self, seed=0, max_depth=3, max_leaves=48,
                 split_min_obs=60, split_h_rmse=0.3):
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.split_min_obs = split_min_obs
        self.split_h_rmse = split_h_rmse
        # capacity trigger: one small PINN should not own more than
        # this many observations -- refine so capacity follows data
        self.split_capacity_obs = 150
        key = jax.random.PRNGKey(seed)
        self.np_rng = np.random.default_rng(seed)
        self.root = Leaf(DOMAIN, 0, init_params(key), key)
        self.total_train_steps = 0
        # observation density grid: the PINN is only trusted NEAR data.
        # A leaf-level variance alone lets a leaf extrapolate confidently
        # into unvisited parts of its own block.
        self.obs_counts = np.zeros((40, 40), dtype=int)
        self.obs_first_t = np.full((40, 40), np.inf)
        self.min_local_obs = 5
        # a region is only TRUSTED once its data is older than this age:
        # correcting against a map being built from the current pass
        # creates a self-confirming drift loop
        self.trust_age = 8.0

    # -- tree ops ----------------------------------------------------------
    def leaf_at(self, x, y):
        x = float(np.clip(x, DOMAIN[0], DOMAIN[1] - 1e-6))
        y = float(np.clip(y, DOMAIN[2], DOMAIN[3] - 1e-6))
        node = self.root
        while node.children is not None:
            node = next(c for c in node.children if c.contains(x, y))
        return node

    def leaves(self):
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.children is None:
                out.append(n)
            else:
                stack.extend(n.children)
        return out

    def _split(self, leaf):
        x0, x1, y0, y1 = leaf.bounds
        xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        quads = [(x0, xm, y0, ym), (xm, x1, y0, ym),
                 (x0, xm, ym, y1), (xm, x1, ym, y1)]
        children = []
        for q in quads:
            leaf.rng, sub = jax.random.split(leaf.rng)
            c = Leaf(q, leaf.depth + 1,
                     jax.tree.map(lambda a: a.copy(), leaf.params), sub)
            # children inherit the parent's function, so its misfit is the
            # right prior for theirs
            c.ema_var = leaf.ema_var.copy()
            c.t_range = list(leaf.t_range)
            children.append(c)
        # partition observation buffer among children
        for X, Y in zip(leaf.buf_X, leaf.buf_Y):
            for c in children:
                if c.contains(X[0], X[1]):
                    c.buf_X.append(X)
                    c.buf_Y.append(Y)
                    break
        leaf.children = children
        leaf.buf_X, leaf.buf_Y = [], []

    def block_misfit(self, leaf, n_val=256):
        """Validation h-RMSE over the leaf's WHOLE buffer. The EMA misfit only
        tracks fit on recent data, which one net can keep low while forgetting
        the rest of its block; this measures whether the block's full data
        still fits a single PINN (the Method-2 refinement signal)."""
        if leaf.n_obs < 8:
            return 0.0
        idx = self.np_rng.integers(0, leaf.n_obs, min(n_val, PAD_QUERY))
        X = np.asarray(leaf.buf_X)[idx]
        Y = np.asarray(leaf.buf_Y)[idx]
        padded = np.zeros((PAD_QUERY, 3))
        padded[:len(X)] = X
        pred = np.asarray(apply_batch(leaf.params, jnp.asarray(padded)))[:len(X)]
        return float(np.sqrt(np.mean((pred[:, 2] - Y[:, 2]) ** 2)))

    def maybe_split(self, leaf):
        if (leaf.children is None
                and leaf.depth < self.max_depth
                and len(self.leaves()) + 3 <= self.max_leaves
                and leaf.n_obs >= self.split_min_obs
                and (leaf.n_obs >= self.split_capacity_obs
                     or self.block_misfit(leaf) > self.split_h_rmse)):
            self._split(leaf)
            # consolidate children immediately so the refined map is at
            # least as good as the parent before it is trusted again
            for c in leaf.children:
                if c.n_obs > 0:
                    self.train_leaf(c, t_now=c.buf_X[-1][2], n_iter=30)
            return True
        return False

    # -- observations ------------------------------------------------------
    def add_observation(self, x, y, t, meas):
        """meas = (u, v, h) measured at estimated pose (x, y), time t [s]."""
        leaf = self.leaf_at(x, y)
        pt = np.array([np.clip(x, *DOMAIN[:2]), np.clip(y, *DOMAIN[2:]), t])
        pred = np.asarray(apply_batch(leaf.params, pt[None, :]))[0]
        err2 = (np.asarray(meas) - pred) ** 2
        leaf.ema_var = EMA * leaf.ema_var + (1 - EMA) * err2
        leaf.buf_X.append(pt)
        leaf.buf_Y.append(np.asarray(meas, dtype=float))
        leaf.t_range[0] = min(leaf.t_range[0], t)
        leaf.t_range[1] = max(leaf.t_range[1], t)
        ix = int(np.clip(round(pt[0]), 0, 39))
        iy = int(np.clip(round(pt[1]), 0, 39))
        self.obs_counts[ix, iy] += 1
        self.obs_first_t[ix, iy] = min(self.obs_first_t[ix, iy], t)
        return leaf

    def _trusted(self, x, y, t):
        """Trust the map at (x, y) only if enough observations exist in the
        3x3 neighborhood AND the oldest of them predates t by trust_age."""
        ix = int(np.clip(round(x), 0, 39))
        iy = int(np.clip(round(y), 0, 39))
        sl = (slice(max(0, ix - 2), ix + 3), slice(max(0, iy - 2), iy + 3))
        return (self.obs_counts[sl].sum() >= self.min_local_obs
                and t - self.obs_first_t[sl].min() >= self.trust_age)

    # -- training ----------------------------------------------------------
    def train_leaf(self, leaf, t_now, n_iter=5):
        if leaf.n_obs < 8:
            return
        X = np.asarray(leaf.buf_X)
        Y = np.asarray(leaf.buf_Y)
        n = len(X)
        x0, x1, y0, y1 = leaf.bounds
        for _ in range(n_iter):
            # half recent, half uniform history
            i_rec = self.np_rng.integers(max(0, n - 200), n, BATCH // 2)
            i_uni = self.np_rng.integers(0, n, BATCH - BATCH // 2)
            idx = np.concatenate([i_rec, i_uni])
            Xc = np.column_stack([
                self.np_rng.uniform(x0, x1, N_COL),
                self.np_rng.uniform(y0, y1, N_COL),
                self.np_rng.uniform(max(0.0, t_now - 20.0), t_now + 1e-3, N_COL),
            ])
            leaf.t_adam += 1
            leaf.params, leaf.m, leaf.v, _ = adam_step(
                leaf.params, leaf.m, leaf.v, leaf.t_adam,
                jnp.asarray(X[idx]), jnp.asarray(Y[idx]), jnp.asarray(Xc))
            leaf.train_steps += 1
            self.total_train_steps += 1

    # -- covariance-gated active set (Method 2 gating) ----------------------
    def active_leaves(self, mean_xy, cov_xy, n_sigma=3.0):
        """Leaves whose block intersects the n-sigma pose ellipse."""
        cov = cov_xy + 1e-6 * np.eye(2)
        cinv = np.linalg.inv(cov)
        act = []
        for leaf in self.leaves():
            x0, x1, y0, y1 = leaf.bounds
            # closest point of rect to mean, then Mahalanobis distance
            px = np.clip(mean_xy[0], x0, x1)
            py = np.clip(mean_xy[1], y0, y1)
            d = np.array([px, py]) - mean_xy
            if d @ cinv @ d <= n_sigma ** 2:
                act.append(leaf)
        return act

    # -- queries -------------------------------------------------------------
    def predict_with_var(self, pts):
        """pts: (N,3) [x,y,t] -> (uvh (N,3), var (N,3)) from the owning leaves."""
        pts = np.asarray(pts, dtype=float)
        N = len(pts)
        out = np.zeros((N, 3))
        var = np.zeros((N, 3))
        # group points by leaf
        groups = {}
        for i, p in enumerate(pts):
            leaf = self.leaf_at(p[0], p[1])
            groups.setdefault(leaf.id, (leaf, []))[1].append(i)
        for leaf, idx in groups.values():
            idx = np.asarray(idx)
            block = pts[idx].copy()
            # quasi-static field: clamp query time into the leaf's data
            # support -- querying a PINN outside its temporal training
            # range extrapolates and drifts
            if leaf.t_range[0] <= leaf.t_range[1]:
                block[:, 2] = np.clip(block[:, 2], leaf.t_range[0], leaf.t_range[1])
            # pad to fixed shape for jit stability
            padded = np.zeros((PAD_QUERY, 3))
            padded[:len(idx)] = block
            pred = np.asarray(apply_batch(leaf.params, jnp.asarray(padded)))
            out[idx] = pred[:len(idx)]
            for j in idx:
                if self._trusted(pts[j, 0], pts[j, 1], pts[j, 2]):
                    var[j] = np.minimum(leaf.ema_var, VAR_MAX)
                else:
                    var[j] = VAR_MAX   # not enough aged data -> don't trust map
        return out, var


class TrueFieldMap:
    """Oracle map wrapper (known field) — used for the upper-bound baseline."""

    def __init__(self, fields):
        self.fields = fields  # (T, 3, 40, 40), channels (h, u, v)

    def predict_with_var(self, pts):
        pts = np.asarray(pts, dtype=float)
        T = self.fields.shape[0]
        it = np.clip((pts[:, 2] / 0.1).astype(int), 0, T - 1)
        ix = np.clip(np.round(pts[:, 0]).astype(int), 0, 39)
        iy = np.clip(np.round(pts[:, 1]).astype(int), 0, 39)
        h = self.fields[it, 0, ix, iy]
        u = self.fields[it, 1, ix, iy]
        v = self.fields[it, 2, ix, iy]
        return np.column_stack([u, v, h]), np.zeros((len(pts), 3))
