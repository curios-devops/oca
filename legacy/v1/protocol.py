"""The Predictive Mesh Protocol (SPEC_PMP).

Four scalars per directed link per tick: confidence, expectation, novelty, importance.
The expectation is a coefficient along the link's own learned 1-D channel -- the basis
lives in the link, so the wire format stays four floats while the units remain
16-dimensional dynamical systems.
"""

from __future__ import annotations

import numpy as np


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def assemble_messages(state, h_pred1: np.ndarray) -> dict[str, np.ndarray]:
    """Build the (N, D) message arrays that will be read on the next tick.

    Indexed by (receiver, slot): entry [i, k] is what src[i, k] sends to i.
    """
    cfg = state.cfg
    src = state.src
    hs = h_pred1[src]                                   # (N, D, d) sender's next state
    e = np.einsum("nkd,nkd->nk", state.u_proj, hs)

    if cfg.msg_width <= 1:
        ones = np.ones_like(e)
        return {"e": e, "c": ones, "n": np.zeros_like(e), "i": np.zeros_like(e)}

    c = state.conf[src]
    n = state.novelty[src]
    iota = np.tanh(state.kappa[src] * n * c)
    return {"e": e, "c": c, "n": n, "i": iota}


def receiver_gate(state, msg: dict[str, np.ndarray]) -> np.ndarray:
    """gamma = confidence * (1 + importance), boosted inside a coalition."""
    cfg = state.cfg
    if cfg.msg_width <= 1:
        gamma = np.ones_like(msg["e"])
    else:
        gamma = msg["c"] * (1.0 + msg["i"])
    if cfg.use_coalition_feedback:
        same = state.coalition[:, None] == state.coalition[state.src]
        gamma = gamma * np.where(same, 1.0 + cfg.coalition_boost, 1.0)
    return gamma


def drive_from_messages(state, msg: dict, gamma: np.ndarray) -> np.ndarray:
    """Sum of read vectors weighted by gain, gate and expectation. (N, d)."""
    coeff = state.w * gamma * msg["e"]
    return np.einsum("nk,nkd->nd", coeff, state.a), coeff


def global_context(state, h: np.ndarray) -> np.ndarray:
    """The executive broadcast g(t): a fixed random summary of the population."""
    return np.tanh(state.R_proj @ h.mean(axis=0))


# ------------------------------------------------------------------ coalitions


def detect_coalitions(state, window: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Label propagation over the coherence graph. Returns (N,) integer labels.

    Nothing here creates coalitions -- if unit states do not actually co-vary, the
    threshold keeps no edges and every unit stays its own singleton label. That is the
    honest null result this detector is designed to be able to report.
    """
    cfg = state.cfg
    N = state.n_units
    W, _, d = window.shape

    z = window - window.mean(axis=0, keepdims=True)
    z = z / np.maximum(z.std(axis=0, keepdims=True), 1e-8)
    z = z.transpose(1, 0, 2).reshape(N, W * d) / np.sqrt(W * d)

    coh = np.einsum("nd,nkd->nk", z, z[state.src]) * gamma
    keep = coh > cfg.coalition_theta

    # Record where the threshold sits inside the actual coherence distribution. If
    # coalitions turn out not to form, this is what distinguishes "units do not
    # synchronise" from "theta was set too high", and the two deserve different
    # conclusions.
    state.coh_pct = {
        "p50": float(np.percentile(coh, 50)),
        "p90": float(np.percentile(coh, 90)),
        "p99": float(np.percentile(coh, 99)),
        "max": float(coh.max()),
        "frac_kept": float(keep.mean()),
    }

    labels = np.arange(N)
    for _ in range(cfg.label_prop_rounds):
        changed = 0
        for i in np.random.default_rng(state.t).permutation(N):
            nb = state.src[i][keep[i]]
            if nb.size == 0:
                continue
            vals, counts = np.unique(labels[nb], return_counts=True)
            best = vals[np.argmax(counts)]
            if best != labels[i]:
                labels[i] = best
                changed += 1
        if changed == 0:
            break
    return labels
