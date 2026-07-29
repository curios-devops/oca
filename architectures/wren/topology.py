"""Building and rewiring the mesh graph (SPEC_PMP sections 1 and 4).

The lattice is a torus, so there are no boundary units with impoverished neighbourhoods.
Links are stored by (receiver, slot): `src[i, k]` is the unit that feeds slot k of unit
i. Every per-link array shares that indexing, which is what lets the whole protocol run
as a few gathers and one einsum.
"""

from __future__ import annotations

import numpy as np


def local_offsets(n_local: int) -> np.ndarray:
    """The `n_local` nearest lattice offsets, excluding self, ordered by distance."""
    offs = [
        (dy, dx)
        for dy in range(-3, 4)
        for dx in range(-3, 4)
        if (dy, dx) != (0, 0)
    ]
    offs.sort(key=lambda o: (o[0] ** 2 + o[1] ** 2, o))
    if n_local > len(offs):
        raise ValueError(f"n_local={n_local} exceeds the radius-3 neighbourhood")
    return np.array(offs[:n_local], dtype=np.int64)


def lattice_distance(i: np.ndarray, j: np.ndarray, side: int) -> np.ndarray:
    """Toroidal Chebyshev distance between flat lattice indices."""
    iy, ix = np.divmod(i, side)
    jy, jx = np.divmod(j, side)
    dy = np.minimum(np.abs(iy - jy), side - np.abs(iy - jy))
    dx = np.minimum(np.abs(ix - jx), side - np.abs(ix - jx))
    return np.maximum(dy, dx)


def build_topology(cfg, rng: np.random.Generator) -> dict:
    """Returns src (N, D), is_long (N, D) bool, and the executive unit ids.

    Executive units are the last `n_exec_units` lattice indices rather than a separate
    population. They are ordinary RPDUs; what makes them executive is only that every
    unit holds one in-link to one of them, so they are the mesh's shared bus.
    """
    side, N = cfg.lattice_side, cfg.n_units
    idx = np.arange(N)
    iy, ix = np.divmod(idx, side)

    offs = local_offsets(cfg.n_local)
    ny = (iy[:, None] + offs[None, :, 0]) % side
    nx = (ix[:, None] + offs[None, :, 1]) % side
    local = ny * side + nx                                    # (N, n_local)

    exec_ids = np.arange(N - cfg.n_exec_units, N)

    cols = [local]
    is_long_cols = [np.zeros_like(local, dtype=bool)]

    if cfg.use_long_range and cfg.n_long > 0:
        long = np.empty((N, cfg.n_long), dtype=np.int64)
        for i in range(N):
            dist = lattice_distance(np.full(N, i), idx, side)
            cand = idx[dist > 3]
            long[i] = rng.choice(cand, size=cfg.n_long, replace=False)
        cols.append(long)
        is_long_cols.append(np.ones_like(long, dtype=bool))

    if cfg.n_exec_links > 0:
        ex = rng.choice(exec_ids, size=(N, cfg.n_exec_links))
        cols.append(ex)
        is_long_cols.append(np.zeros_like(ex, dtype=bool))

    return {
        "src": np.concatenate(cols, axis=1),
        "is_long": np.concatenate(is_long_cols, axis=1),
        "exec_ids": exec_ids,
    }


def novelty_correlation(nov_hist: np.ndarray) -> np.ndarray:
    """(N, N) correlation of novelty traces. `nov_hist` is (window, N)."""
    x = nov_hist - nov_hist.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    x = x / np.maximum(sd, 1e-8)
    return (x.T @ x) / x.shape[0]


def rewire(state, cfg, rng: np.random.Generator, nov_hist: np.ndarray) -> dict:
    """Prune the least useful long-range links and grow replacements.

    Growth is biased toward units whose novelty co-varies with the receiver's: units
    that are surprised at the same moments are probably looking at the same thing, which
    is a purely local heuristic for finding a useful distant correspondent.
    """
    if not cfg.use_rewiring or not cfg.use_long_range or cfg.n_long == 0:
        return {"n_rewired": 0}

    N, side = state.n_units, cfg.lattice_side
    corr = novelty_correlation(nov_hist)
    np.fill_diagonal(corr, -np.inf)
    med_credit = float(np.median(state.credit))
    n_rewired = 0

    for i in range(N):
        slots = np.flatnonzero(state.is_long[i])
        if slots.size == 0:
            continue
        worst = slots[np.argsort(state.credit[i, slots])[: cfg.rewire_prune]]

        logits = corr[i] / cfg.sample_temp
        logits[state.src[i]] = -np.inf                     # no duplicate links
        dist = lattice_distance(np.full(N, i), np.arange(N), side)
        logits[dist <= 3] = -np.inf                        # long-range means long-range
        if not np.isfinite(logits).any():
            continue
        p = np.exp(logits - np.nanmax(logits[np.isfinite(logits)]))
        p[~np.isfinite(logits)] = 0.0
        total = p.sum()
        if total <= 0:
            continue
        p /= total

        picks = rng.choice(N, size=worst.size, replace=False, p=p)
        # widths come from the arrays, not from cfg.d: v2's read vector lives in rotor
        # space while its projector lives in the narrower amplitude space
        d_a = state.a.shape[-1]
        d_u = state.u_proj.shape[-1]
        for slot, new_src in zip(worst, picks):
            state.src[i, slot] = new_src
            state.w[i, slot] = cfg.new_link_gain
            state.a[i, slot] = rng.normal(0, 1.0 / np.sqrt(d_a), d_a)
            state.u_proj[i, slot] = rng.normal(0, 1.0 / np.sqrt(d_u), d_u)
            state.credit[i, slot] = med_credit             # grace period
            n_rewired += 1

    return {"n_rewired": n_rewired}
