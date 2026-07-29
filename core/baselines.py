"""Baselines the mesh has to beat.

Without these, a falling error curve means nothing -- the world is predictable enough
that copying the last frame already does well. Every baseline predicts exactly the same
target as the mesh: the RETINA x RETINA frame at t+tau, scored by frame MSE.

Three of them, in ascending order of what they would prove:
  CopyLast   -- the floor. Losing to this ends the project.
  LinearAR   -- a closed-form linear extrapolator. Beating it means the mesh has found
                something non-linear about the world.
  GRU        -- backprop-through-time with a matched parameter count. This is the real
                comparison, and the mesh is not expected to win it outright; the
                question is whether local-only learning lands in the same league.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import ErrorTracker, frame_mse

HORIZONS = (1, 4, 16)


def _flat(frames: np.ndarray) -> np.ndarray:
    return frames.reshape(frames.shape[0], -1).astype(np.float64)


def as_batch(frames: np.ndarray, n_batch: int) -> list[np.ndarray]:
    """Split one long rollout into `n_batch` contiguous segments.

    Batched truncated BPTT over segments of a single stream, so the GRU sees exactly
    the same frames the mesh does -- no extra data, just better arithmetic intensity.
    """
    seg = len(frames) // n_batch
    if seg < 64:
        return [frames]
    return [frames[k * seg : (k + 1) * seg] for k in range(n_batch)]


# --------------------------------------------------------------------- copy-last


@dataclass
class CopyLast:
    """Predict that nothing changes. The floor."""

    horizons: tuple[int, ...] = HORIZONS

    def evaluate(self, frames: np.ndarray) -> ErrorTracker:
        tr = ErrorTracker(self.horizons)
        n = len(frames)
        for t in range(n):
            for tau in self.horizons:
                if t + tau < n:
                    tr.add(tau, frame_mse(frames[t], frames[t + tau]))
        return tr


# --------------------------------------------------------------------- linear AR


@dataclass
class LinearAR:
    """Ridge regression predicting each pixel from its own spatiotemporal neighbourhood.

    A whole-frame linear map would need far more frames than pixels to be identifiable
    (1024 outputs from 2049 inputs), so instead one filter is shared across all pixel
    positions: pixel (y, x) at t+tau is predicted from the K x K windows around (y, x)
    at t and t-1. That is translation-invariant, gives 1024 training rows per frame
    instead of one, and can represent constant-velocity motion of anything smaller than
    the window.

    This makes it a genuinely hard baseline rather than a straw man -- which is the
    point. If the mesh only ties this, "it learned the physics" is not a claim anyone
    should accept.
    """

    ksize: int = 7
    lags: int = 2
    ridge: float = 1e-2
    horizons: tuple[int, ...] = HORIZONS
    W: dict = field(default_factory=dict, repr=False)

    def _windows(self, frames: np.ndarray, t: int) -> np.ndarray:
        """(n_pixels, n_features) neighbourhood features describing time t."""
        k, pad = self.ksize, self.ksize // 2
        cols = []
        for lag in range(self.lags):
            f = frames[t - lag]
            fp = np.pad(f, pad, mode="edge")
            win = np.lib.stride_tricks.sliding_window_view(fp, (k, k))
            cols.append(win.reshape(-1, k * k))
        cols.append(np.ones((cols[0].shape[0], 1)))
        return np.concatenate(cols, axis=1).astype(np.float64)

    @property
    def n_features(self) -> int:
        return self.lags * self.ksize**2 + 1

    def fit(self, frames: np.ndarray) -> "LinearAR":
        frames = np.asarray(frames, dtype=np.float64)
        n, d = len(frames), self.n_features
        xtx = np.zeros((d, d))
        xty = {tau: np.zeros(d) for tau in self.horizons}
        start = self.lags - 1
        for t in range(start, n - max(self.horizons)):
            f = self._windows(frames, t)
            xtx += f.T @ f
            for tau in self.horizons:
                xty[tau] += f.T @ frames[t + tau].reshape(-1)
        chol = np.linalg.cholesky(xtx + self.ridge * np.trace(xtx) / d * np.eye(d))
        for tau in self.horizons:
            z = np.linalg.solve(chol, xty[tau])
            self.W[tau] = np.linalg.solve(chol.T, z)
        return self

    def predict(self, frames: np.ndarray, t: int, tau: int) -> np.ndarray:
        f = self._windows(np.asarray(frames, dtype=np.float64), t)
        return (f @ self.W[tau]).reshape(frames[0].shape)

    def evaluate(self, frames: np.ndarray) -> ErrorTracker:
        frames = np.asarray(frames, dtype=np.float64)
        tr = ErrorTracker(self.horizons)
        for t in range(self.lags - 1, len(frames)):
            f = self._windows(frames, t)
            for tau in self.horizons:
                if t + tau < len(frames):
                    pred = f @ self.W[tau]
                    tr.add(tau, float(np.mean((pred - frames[t + tau].reshape(-1)) ** 2)))
        return tr


# ------------------------------------------------------------ memoryful linear


@dataclass
class MemoryLinear:
    """Ridge from a long, coarse temporal window of the whole frame.

    The other baselines are memoryless and local, so they physically cannot know about
    an object that is behind the occluder. This one can: its taps reach back further
    than a typical occlusion lasts, so the object is visible in its input before it
    disappears.

    It is fitted in closed form on purpose. The GRU is the natural stateful model, but a
    GRU that fails to beat a memoryless baseline is ambiguous -- the world might not
    reward memory, or the optimiser might simply have failed. A closed-form solution has
    no optimiser to blame, so it turns the gate into a statement about the world.
    """

    down: int = 16              # frame is coarsened to down x down before lagging
    n_taps: int = 16
    tap_stride: int = 2         # taps reach back n_taps * tap_stride ticks
    ridge: float = 1e-2
    horizons: tuple[int, ...] = HORIZONS
    W: dict = field(default_factory=dict, repr=False)

    @property
    def reach(self) -> int:
        return self.n_taps * self.tap_stride

    def _coarse(self, frames: np.ndarray) -> np.ndarray:
        f = frames.shape[1] // self.down
        if f <= 1:
            return frames.reshape(len(frames), -1)
        return frames.reshape(len(frames), self.down, f, self.down, f).mean(axis=(2, 4)) \
                     .reshape(len(frames), -1)

    def _features(self, coarse: np.ndarray, t: int) -> np.ndarray:
        taps = [coarse[max(0, t - k * self.tap_stride)] for k in range(self.n_taps)]
        return np.concatenate(taps + [np.ones(1)])

    def fit(self, frames: np.ndarray) -> "MemoryLinear":
        frames = np.asarray(frames, dtype=np.float64)
        coarse = self._coarse(frames)
        flat = frames.reshape(len(frames), -1)
        lo, hi = self.reach, len(frames) - max(self.horizons)
        X = np.stack([self._features(coarse, t) for t in range(lo, hi)])
        A = X.T @ X
        A[np.diag_indices_from(A)] += self.ridge * np.trace(A) / A.shape[0]
        chol = np.linalg.cholesky(A)
        for tau in self.horizons:
            Y = flat[np.arange(lo, hi) + tau]
            z = np.linalg.solve(chol, X.T @ Y)
            self.W[tau] = np.linalg.solve(chol.T, z)
        self._shape = frames[0].shape
        return self

    def predict(self, frames: np.ndarray, t: int, tau: int) -> np.ndarray:
        # coarsening is O(stream); cache it so per-frame prediction stays O(1)
        if getattr(self, "_cache_key", None) is not id(frames):
            self._cache_key = id(frames)
            self._cache = self._coarse(np.asarray(frames, dtype=np.float64))
        return (self._features(self._cache, t) @ self.W[tau]).reshape(self._shape)

    def evaluate(self, frames: np.ndarray) -> ErrorTracker:
        frames = np.asarray(frames, dtype=np.float64)
        coarse = self._coarse(frames)
        flat = frames.reshape(len(frames), -1)
        tr = ErrorTracker(self.horizons)
        for t in range(self.reach, len(frames) - max(self.horizons)):
            f = self._features(coarse, t)
            for tau in self.horizons:
                tr.add(tau, float(np.mean((f @ self.W[tau] - flat[t + tau]) ** 2)))
        return tr


# ---------------------------------------------------------------- local MLP


@dataclass
class LocalMLP:
    """LinearAR's features, a non-linear function on top. The headroom instrument.

    This exists to answer one question cleanly: *does this world contain predictive
    structure that a linear model provably cannot express?* Because it consumes exactly
    the same K x K x lags neighbourhood as `LinearAR` and predicts exactly the same
    target, any gap between them is attributable to non-linearity alone -- not to
    capacity, not to receptive field, not to optimisation of a different architecture.

    A GRU is a poor instrument for this. On world v1 it lost to the linear filter too,
    so a GRU-based headroom test cannot distinguish "the world is linear" from "the GRU
    was badly optimised". This can.
    """

    ksize: int = 7
    lags: int = 2
    hidden: int = 64
    horizons: tuple[int, ...] = HORIZONS
    lr: float = 3e-3
    rows_per_frame: int = 96
    seed: int = 0

    def __post_init__(self) -> None:
        self._ar = LinearAR(ksize=self.ksize, lags=self.lags, horizons=self.horizons)
        rng = np.random.default_rng(self.seed)
        d, h, o = self._ar.n_features, self.hidden, len(self.horizons)
        self.p = {
            "W1": rng.normal(0, 1 / np.sqrt(d), (d, h)),
            "b1": np.zeros(h),
            "W2": rng.normal(0, 1 / np.sqrt(h), (h, o)),
            "b2": np.zeros(o),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._step = 0
        self.mu = None
        self.sd = None

    @property
    def n_params(self) -> int:
        return sum(v.size for v in self.p.values())

    def _forward(self, X):
        z = X @ self.p["W1"] + self.p["b1"]
        a = np.tanh(z)
        return a, a @ self.p["W2"] + self.p["b2"]

    def _adam(self, g):
        self._step += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        bc1, bc2 = 1 - b1**self._step, 1 - b2**self._step
        for k, gk in g.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * gk
            self._v[k] = b2 * self._v[k] + (1 - b2) * gk * gk
            self.p[k] -= self.lr * (self._m[k] / bc1) / (np.sqrt(self._v[k] / bc2) + eps)

    def _xy(self, frames, t, rng=None):
        X = self._ar._windows(frames, t)
        Y = np.stack([frames[t + tau].reshape(-1) for tau in self.horizons], axis=1)
        if rng is not None and self.rows_per_frame < X.shape[0]:
            sel = rng.choice(X.shape[0], self.rows_per_frame, replace=False)
            X, Y = X[sel], Y[sel]
        return X, Y

    def fit(self, frames, epochs: int = 8, logger=None, log_every: int = 0):
        frames = np.asarray(frames, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        lo, hi = self.lags - 1, len(frames) - max(self.horizons)

        sample = np.concatenate([self._ar._windows(frames, t)
                                 for t in range(lo, min(lo + 40, hi))])
        self.mu, self.sd = sample.mean(0), sample.std(0) + 1e-8
        self.mu[-1], self.sd[-1] = 0.0, 1.0          # keep the bias column at 1

        it = 0
        for _ in range(epochs):
            for t in rng.permutation(np.arange(lo, hi)):
                X, Y = self._xy(frames, int(t), rng)
                X = (X - self.mu) / self.sd
                a, pred = self._forward(X)
                diff = pred - Y
                n = diff.size
                g = {}
                g["W2"] = a.T @ (2 * diff / n)
                g["b2"] = (2 * diff / n).sum(0)
                da = (2 * diff / n) @ self.p["W2"].T * (1 - a * a)
                g["W1"] = X.T @ da
                g["b1"] = da.sum(0)
                self._adam(g)
                it += 1
                if logger is not None and log_every and it % log_every == 0:
                    logger.log(kind="mlp_train", iter=it,
                               loss=float((diff * diff).mean()))
        return self

    def predict(self, frames, t: int, tau: int) -> np.ndarray:
        X = (self._ar._windows(np.asarray(frames, dtype=np.float64), t) - self.mu) / self.sd
        _, pred = self._forward(X)
        return pred[:, self.horizons.index(tau)].reshape(frames[0].shape)

    def evaluate(self, frames) -> ErrorTracker:
        frames = np.asarray(frames, dtype=np.float64)
        tr = ErrorTracker(self.horizons)
        for t in range(self.lags - 1, len(frames) - max(self.horizons)):
            X = (self._ar._windows(frames, t) - self.mu) / self.sd
            _, pred = self._forward(X)
            for k, tau in enumerate(self.horizons):
                tr.add(tau, float(np.mean((pred[:, k] - frames[t + tau].reshape(-1)) ** 2)))
        return tr


# --------------------------------------------------------------------------- GRU


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


@dataclass
class GRU:
    """Single-layer GRU with hand-written BPTT and Adam. numpy only.

    Parameter count at the defaults is ~1.3M against the mesh's ~1.5M, so this is a
    like-for-like comparison of learning rules rather than of capacity.
    """

    n_in: int
    hidden: int = 192
    horizons: tuple[int, ...] = HORIZONS
    lr: float = 2e-3
    seed: int = 0
    residual: bool = True
    """Predict the *change* from the current frame rather than the frame itself.

    Without this the recurrent state has to re-encode 1024 pixels through a 192-unit
    bottleneck before it can say anything about motion, and it loses to copy-last at
    tau=1 for reasons that have nothing to do with world modelling. The mesh's sensory
    units read their own patches directly, so giving the GRU the current frame too is
    what makes the comparison about the learning rule instead of about plumbing.
    """

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        h, i = self.hidden, self.n_in
        sx, sh = 1.0 / np.sqrt(i), 1.0 / np.sqrt(h)
        self.p: dict[str, np.ndarray] = {}
        for g in "zrn":
            self.p[f"W{g}"] = rng.normal(0, sx, (i, h))
            self.p[f"U{g}"] = rng.normal(0, sh, (h, h))
            self.p[f"b{g}"] = np.zeros(h)
        for tau in self.horizons:
            self.p[f"Wo{tau}"] = rng.normal(0, sh, (h, i))
            self.p[f"bo{tau}"] = np.zeros(i)
        self._m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._step = 0

    @property
    def n_params(self) -> int:
        return sum(v.size for v in self.p.values())

    # -- forward / backward over one chunk ----------------------------------

    def _forward(self, xs: np.ndarray, h0: np.ndarray) -> tuple[list, np.ndarray]:
        p, cache, h = self.p, [], h0
        for x in xs:
            z = _sigmoid(x @ p["Wz"] + h @ p["Uz"] + p["bz"])
            r = _sigmoid(x @ p["Wr"] + h @ p["Ur"] + p["br"])
            rh = r * h
            n = np.tanh(x @ p["Wn"] + rh @ p["Un"] + p["bn"])
            h_new = (1 - z) * h + z * n
            cache.append((x, h, z, r, rh, n, h_new))
            h = h_new
        return cache, h

    def _loss_and_grads(self, cache: list, targets: dict) -> tuple[float, dict]:
        p = self.p
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dh_next = np.zeros_like(cache[0][1])
        loss, count = 0.0, 0

        # the loss is the mean over targets, so every gradient carries 1/n_targets
        n_targets = max(sum(len(targets[tau]) for tau in self.horizons), 1)

        dh_out = [np.zeros_like(c[6]) for c in cache]
        for tau in self.horizons:
            Wo, bo = p[f"Wo{tau}"], p[f"bo{tau}"]
            for t, (idx, y) in targets[tau]:
                hn = cache[t][6]
                pred = hn @ Wo + bo
                diff = pred - y
                loss += float(np.mean(diff * diff))
                count += 1
                scale = 2.0 / (diff.shape[0] * diff.shape[1] * n_targets)
                g[f"Wo{tau}"] += hn.T @ (diff * scale)
                g[f"bo{tau}"] += (diff * scale).sum(axis=0)
                dh_out[t] += (diff * scale) @ Wo.T

        for t in range(len(cache) - 1, -1, -1):
            x, h, z, r, rh, n, _ = cache[t]
            dh = dh_next + dh_out[t]
            dz = dh * (n - h)
            dn = dh * z
            dh_prev = dh * (1 - z)

            dn_raw = dn * (1 - n * n)
            g["Wn"] += x.T @ dn_raw
            g["Un"] += rh.T @ dn_raw
            g["bn"] += dn_raw.sum(axis=0)
            drh = dn_raw @ p["Un"].T
            dr = drh * h
            dh_prev += drh * r

            dz_raw = dz * z * (1 - z)
            g["Wz"] += x.T @ dz_raw
            g["Uz"] += h.T @ dz_raw
            g["bz"] += dz_raw.sum(axis=0)
            dh_prev += dz_raw @ p["Uz"].T

            dr_raw = dr * r * (1 - r)
            g["Wr"] += x.T @ dr_raw
            g["Ur"] += h.T @ dr_raw
            g["br"] += dr_raw.sum(axis=0)
            dh_prev += dr_raw @ p["Ur"].T

            dh_next = dh_prev

        return loss / max(count, 1), g

    def _adam(self, grads: dict, clip: float = 5.0) -> None:
        self._step += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        total = np.sqrt(sum(float((v * v).sum()) for v in grads.values()))
        scale = min(1.0, clip / (total + 1e-12))
        bc1 = 1 - b1**self._step
        bc2 = 1 - b2**self._step
        for k, gk in grads.items():
            gk = gk * scale
            self._m[k] = b1 * self._m[k] + (1 - b1) * gk
            self._v[k] = b2 * self._v[k] + (1 - b2) * gk * gk
            self.p[k] -= self.lr * (self._m[k] / bc1) / (np.sqrt(self._v[k] / bc2) + eps)

    # -- training / evaluation ----------------------------------------------

    def fit(
        self,
        streams: list[np.ndarray],
        epochs: int = 3,
        chunk: int = 24,
        log_every: int = 0,
        logger=None,
    ) -> "GRU":
        """`streams` is a list of (T, R, R) rollouts; they form the batch dimension."""
        xs = np.stack([_flat(s) for s in streams])  # (B, T, D)
        B, T, _ = xs.shape
        max_tau = max(self.horizons)
        h = np.zeros((B, self.hidden))
        it = 0
        for _ in range(epochs):
            h[:] = 0.0
            for start in range(0, T - max_tau - chunk, chunk):
                seq = xs[:, start : start + chunk].transpose(1, 0, 2)
                cache, h_end = self._forward(seq, h)
                base = xs[:, start : start + chunk] if self.residual else None
                targets = {
                    tau: [
                        (
                            t,
                            (
                                start + t,
                                xs[:, start + t + tau]
                                - (base[:, t] if base is not None else 0.0),
                            ),
                        )
                        for t in range(chunk)
                        if start + t + tau < T
                    ]
                    for tau in self.horizons
                }
                loss, grads = self._loss_and_grads(cache, targets)
                self._adam(grads)
                h = h_end
                it += 1
                if logger is not None and log_every and it % log_every == 0:
                    logger.log(kind="gru_train", iter=it, loss=loss)
        return self

    def predict_stream(self, frames: np.ndarray) -> dict[int, np.ndarray]:
        """{tau: (T, R, R)} where entry t is the prediction of frame t+tau made at t.

        The GRU carries state across the whole stream, so unlike a local filter it can
        in principle track an object while it is invisible -- which is exactly what the
        occlusion-locked gate needs to measure.
        """
        shape = frames[0].shape
        x = _flat(frames)[None]
        cache, _ = self._forward(x.transpose(1, 0, 2), np.zeros((1, self.hidden)))
        out = {}
        for tau in self.horizons:
            Wo, bo = self.p[f"Wo{tau}"], self.p[f"bo{tau}"]
            preds = np.stack([c[6][0] @ Wo + bo for c in cache])
            if self.residual:
                preds = preds + x[0]
            out[tau] = preds.reshape(-1, *shape)
        return out

    def evaluate(self, frames: np.ndarray) -> ErrorTracker:
        tr = ErrorTracker(self.horizons)
        x = _flat(frames)[None]  # batch of 1
        seq = x.transpose(1, 0, 2)
        cache, _ = self._forward(seq, np.zeros((1, self.hidden)))
        n = len(frames)
        for tau in self.horizons:
            Wo, bo = self.p[f"Wo{tau}"], self.p[f"bo{tau}"]
            for t in range(n):
                if t + tau >= n:
                    continue
                pred = cache[t][6] @ Wo + bo
                if self.residual:
                    pred = pred + x[:, t]
                tr.add(tau, float(np.mean((pred[0] - x[0, t + tau]) ** 2)))
        return tr
