"""v2 protocol: phase-gated messages and phase-locking coalitions.

Two changes from v1, both aimed at making synchronisation do something instead of merely
being logged.

**Phase gating.** A message's influence is scaled by how well sender and receiver phases
agree. In v1 coalition membership multiplied a gain by 1.2x, which is a nudge; here two
units in antiphase are effectively disconnected. That is the design document's Step 7 --
"when synchronised, information flows; otherwise almost disconnected" -- as a mechanism.

**Phase-locking coalitions.** v1 detected coalitions by correlating raw state, which is
meaningless between units that have both settled to fixed points, and it duly found
almost nothing (median coherence 0.00). Phase-locking value is the standard measure for
oscillators and is well defined here.
"""

from __future__ import annotations

import numpy as np

from . import dynamics as dyn


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def assemble_messages(state, h_pred1: np.ndarray) -> dict[str, np.ndarray]:
    """(N, D) message arrays for the next tick, indexed by (receiver, slot)."""
    cfg = state.cfg
    hs = h_pred1[state.src]                                  # (N, D, d)
    e = np.einsum("nkd,nkd->nk", state.u_proj, hs)
    if cfg.msg_width <= 1:
        o = np.ones_like(e)
        return {"e": e, "c": o, "n": np.zeros_like(e), "i": np.zeros_like(e)}
    c = state.conf[state.src]
    n = state.novelty[state.src]
    return {"e": e, "c": c, "n": n, "i": np.tanh(state.kappa[state.src] * n * c)}


def receiver_gate(state, msg: dict[str, np.ndarray]) -> np.ndarray:
    """gamma = confidence * (1 + importance) * phase agreement."""
    cfg = state.cfg
    if cfg.msg_width <= 1:
        gamma = np.ones_like(msg["e"])
    else:
        gamma = msg["c"] * (1.0 + msg["i"])
    if cfg.use_phase_gate and cfg.phase_gate > 0:
        align = dyn.phase_alignment(state.z, state.src)       # (N, D) in [-1, 1]
        gamma = gamma * np.power(np.clip(0.5 * (1.0 + align), 0.0, 1.0), cfg.phase_gate)
    return gamma


def drive_from_messages(state, msg: dict, gamma: np.ndarray):
    """(N, d) drive in rotor space, plus the per-link coefficient the rule needs.

    Messages carry *amplitude* coefficients (content), but they are injected along a
    read vector in rotor space, so a message can shift both a receiver's amplitude and
    its phase. Content on the wire, binding in the geometry."""
    coeff = state.w * gamma * msg["e"]
    return np.einsum("nk,nkd->nd", coeff, state.a), coeff


def global_context(state, h: np.ndarray) -> np.ndarray:
    return np.tanh(state.R_proj @ h.mean(axis=0))


def phase_coherence(state, window: np.ndarray) -> np.ndarray:
    """(N, D) sustained in-phase agreement over a window of rotor states.

    Deliberately `mean_t cos(dphi_t)` and *not* the usual PLV `|mean_t exp(i dphi_t)|`.
    PLV rewards a phase difference that is merely *constant*, so two units whose random
    natural frequencies happen to match score 1.0 while sitting permanently in
    antiphase, and the measured coalitions turn out to be a fixed artefact of the
    frequency draw -- verified directly: changing the coupling strength tenfold left the
    PLV distribution identical, because coupling was not what produced it.

    Mean cosine is high only when units are actually aligned, which is the thing that
    has to be produced by interaction rather than by initialisation.
    """
    phases = np.stack([dyn.phase(z) for z in window])         # (W, N, m)
    d = phases[:, :, None, :] - phases[:, state.src, :]       # (W, N, D, m)
    return np.cos(d).mean(axis=(0, 3))


# --------------------------------------------------------------------- voting


def vote_update(state, gamma: np.ndarray) -> np.ndarray:
    """One round of Thousand-Brains-style voting. Returns the new (N, k) votes.

    Each unit proposes a vote from its own content, reconciles it with a
    confidence-weighted average of its neighbours' votes, and renormalises onto the unit
    sphere. Normalisation is doing real work: without it the whole population can satisfy
    the agreement pressure by decaying to zero, which is agreement about nothing.

    The `vote_self` mix is the interesting parameter. Pure consensus has one stable
    outcome -- every unit voting identically, a single global coalition, which is what
    increment 1's phase coupling produced when it was too strong. Pure self-proposal
    never agrees with anyone. Binding requires the intermediate regime.
    """
    cfg = state.cfg
    proposal = np.einsum("nkm,nm->nk", state.V, state.h_amp)     # (N, k)

    weight = np.maximum(gamma * state.conf[state.src], 0.0)      # (N, D)
    total = weight.sum(axis=1, keepdims=True) + 1e-8
    consensus = np.einsum("nd,ndk->nk", weight, state.y[state.src]) / total

    target = cfg.vote_self * proposal + (1.0 - cfg.vote_self) * consensus
    y = (1.0 - cfg.vote_rate) * state.y + cfg.vote_rate * target
    if cfg.vote_sharpen > 1.0:
        # push votes toward the corners of the sphere so they behave like
        # discrete hypotheses. Blended votes average smoothly toward a single
        # global compromise; discrete ones have to be chosen between, which is
        # what lets neighbourhoods disagree at all.
        y = np.sign(y) * np.abs(y) ** cfg.vote_sharpen
    return y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)


def vote_consensus_error(state, gamma: np.ndarray) -> np.ndarray:
    """(N, k) gap between a unit's own proposal and what its neighbours actually voted.

    This is the learning signal for the proposal map: a unit learns to predict the vote
    its neighbourhood will settle on. It is local -- a unit sees only the votes that
    arrived on its own links -- and it is the voting analogue of every other rule here,
    which is prediction of something the unit can observe.
    """
    proposal = np.einsum("nkm,nm->nk", state.V, state.h_amp)
    weight = np.maximum(gamma * state.conf[state.src], 0.0)
    total = weight.sum(axis=1, keepdims=True) + 1e-8
    consensus = np.einsum("nd,ndk->nk", weight, state.y[state.src]) / total
    return consensus - proposal


def vote_coalitions(state) -> np.ndarray:
    """Coalitions defined by agreement of votes rather than of phase.

    Increment 1 offered only phase-based coalitions. Running both lets the experiment ask
    which candidate binding mechanism -- synchrony or consensus -- actually carries
    object information, rather than assuming either does.
    """
    cfg = state.cfg
    N = state.n_units
    yn = state.y / (np.linalg.norm(state.y, axis=1, keepdims=True) + 1e-8)
    sim = np.einsum("nk,ndk->nd", yn, yn[state.src])
    keep = sim > cfg.vote_theta

    state.vote_pct = {
        "p50": float(np.percentile(sim, 50)),
        "p90": float(np.percentile(sim, 90)),
        "frac_kept": float(keep.mean()),
    }

    labels = np.arange(N)
    rng = np.random.default_rng(state.t + 1)
    for _ in range(cfg.label_prop_rounds):
        changed = 0
        for i in rng.permutation(N):
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


def detect_coalitions(state, window: np.ndarray) -> np.ndarray:
    """Label propagation over phase-locked links. Returns (N,) labels."""
    cfg = state.cfg
    N = state.n_units
    coh = phase_coherence(state, window)
    keep = coh > cfg.plv_theta

    state.plv_pct = {
        "p50": float(np.percentile(coh, 50)),
        "p90": float(np.percentile(coh, 90)),
        "p99": float(np.percentile(coh, 99)),
        "max": float(coh.max()),
        "frac_kept": float(keep.mean()),
    }

    labels = np.arange(N)
    rng = np.random.default_rng(state.t)
    for _ in range(cfg.label_prop_rounds):
        changed = 0
        for i in rng.permutation(N):
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
