"""The per-unit energy landscape (SPEC_RPDU section 2).

Each unit's state descends its own energy

    E_i(h) = -1/2 h' W_i h  -  b_i . h  +  (alpha/4) ||h||^4 ,    W_i = U_i U_i'

so the dynamics are a gradient flow. That is a stronger commitment than "a recurrent
update": it guarantees convergence to fixed points under frozen drive, and it makes the
valleys of E the unit's stable hypotheses rather than a metaphor. The `linear` flow kind
here exists only so E3 can ablate exactly that property while holding capacity fixed.
"""

from __future__ import annotations

import numpy as np


def wh(h: np.ndarray, U: np.ndarray) -> np.ndarray:
    """W_i h with W_i = U_i U_i', computed without ever forming W. (N, d)."""
    return np.einsum("nr,ndr->nd", np.einsum("nd,ndr->nr", h, U), U)


def energy(h: np.ndarray, U: np.ndarray, b: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """(N,) energy per unit. Used by tests to verify the flow really descends it."""
    quad = np.einsum("nd,nd->n", h, wh(h, U))
    sq = np.einsum("nd,nd->n", h, h)
    return -0.5 * quad - np.einsum("nd,nd->n", b, h) + 0.25 * alpha * sq * sq


def flow(
    h: np.ndarray,
    U: np.ndarray,
    b: np.ndarray,
    alpha: float = 1.0,
    kind: str = "energy",
) -> np.ndarray:
    """-grad E, i.e. the direction the state moves. (N, d)."""
    if kind == "energy":
        sq = np.einsum("nd,nd->n", h, h)[:, None]
        return wh(h, U) + b - alpha * sq * h
    if kind == "linear":
        # E3 ablation: same parameters, but a generic bounded recurrent map instead of
        # a gradient flow. No energy function, so no guaranteed settling.
        return np.tanh(wh(h, U) + b) - h
    raise ValueError(f"unknown flow kind {kind!r}")


def relax(
    h: np.ndarray,
    U: np.ndarray,
    b: np.ndarray,
    *,
    alpha: float = 1.0,
    eta: float = 0.15,
    steps: int = 2,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
    h_max: float = 10.0,
    kind: str = "energy",
) -> tuple[np.ndarray, int]:
    """Euler-integrate the flow for `steps` sub-steps. Returns (h, n_clipped)."""
    for _ in range(steps):
        h = h + eta * flow(h, U, b, alpha, kind)
        if noise and rng is not None:
            h = h + noise * rng.standard_normal(h.shape)
    norm = np.linalg.norm(h, axis=1, keepdims=True)
    over = norm > h_max
    n_clipped = int(over.sum())
    if n_clipped:
        h = np.where(over, h * (h_max / np.maximum(norm, 1e-12)), h)
    return h, n_clipped


def rollout(
    h: np.ndarray,
    U: np.ndarray,
    b: np.ndarray,
    horizons: tuple[int, ...],
    *,
    alpha: float = 1.0,
    eta: float = 0.15,
    steps: int = 2,
    h_max: float = 10.0,
    kind: str = "energy",
) -> dict[int, np.ndarray]:
    """Self-prediction: roll the landscape forward with the drive held frozen.

    This *is* the self-prediction head -- there is no separate weight matrix for it.
    If there were, the landscape would be ornamental and the E3 ablation meaningless.
    """
    out: dict[int, np.ndarray] = {}
    cur = h
    done = 0
    for tau in sorted(horizons):
        for _ in range(tau - done):
            cur, _ = relax(
                cur, U, b, alpha=alpha, eta=eta, steps=steps, h_max=h_max, kind=kind
            )
        done = tau
        out[tau] = cur.copy()
    return out


def fixed_points_1d(u: np.ndarray, b_scalar: float = 0.0, alpha: float = 1.0) -> np.ndarray:
    """Analytic fixed points along a rank-1 landscape's own direction, for tests.

    With W = lam * e e' and b = 0 the flow along e is lam*x - alpha*x^3, giving
    x = 0 and x = +/- sqrt(lam/alpha): one unstable ridge between two valleys.
    """
    lam = float(u @ u)
    if lam <= 0:
        return np.array([0.0])
    root = np.sqrt(lam / alpha)
    return np.array([-root, 0.0, root])
