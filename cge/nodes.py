"""Phase 1 — component benchmarks for **level 2**, the Dynamic Cortical Node.

Level 1 measured precision against efficiency. That is the right question for a neuron and
the wrong one for a node: a node is not a channel, it is the thing the design says holds
knowledge. So level 2 measures what a node claims to do — form concepts, keep a
representation whose relations survive compression, predict at the timescale it integrates
over, and say all of it through a five-scalar mouthpiece.

Every gate here has a control that can kill the mechanism it tests. That is not a stylistic
preference: the legacy line published three findings that turned out to be the measurement
rather than the model, and each time the missing piece was a control at matched budget.

The two that can kill this level outright:

* **L2-2** — if the relational working state does not beat mean pooling at matched
  capacity, then the strongest result the legacy line produced did not transfer, and the
  node is an aggregation convenience rather than a level of abstraction.
* **L2-1** — if the knowledge model's concepts carry no more about the world than a
  shuffled null, then "memory lives in the DCN" is still a location and not a mechanism.

Lives in `bench/` rather than in `dcn/`, so that a future architecture inherits the whole
battery and "the same test" keeps meaning the same thing.
"""

from __future__ import annotations

import numpy as np

from core.probes import fit_ridge, whiten

# Every representation is scored on the same target and at the same capacity, because
# the legacy line measured what happens otherwise: an unmatched comparison handed a
# 3456-dimensional state a decode error of 993,925 against a chance of 7.8.
N_COMPONENTS = 32
TARGET_COMPONENTS = 24


# --------------------------------------------------------------- shared target


def retina_target(retina: np.ndarray, cut: int, n_components: int = TARGET_COMPONENTS):
    """A fixed low-dimensional view of the world, shared by every representation.

    The prediction target has to be architecture-neutral or the comparison is rigged: if
    each model is scored on predicting its own state, the model with the most predictable
    state wins by being the most boring. Principal components of the raw frame belong to
    the world, not to any model, and every version is asked for the same thing.
    """
    R = np.asarray(retina, dtype=np.float64).reshape(len(retina), -1)
    reduce = whiten(R, cut, n_components)
    return reduce(R)


def predictive_gain(X: np.ndarray, Y: np.ndarray, tau: int, split: float = 0.6,
                    n_components: int | None = N_COMPONENTS) -> dict:
    """Predict `Y` at t+tau from `X` at t, relative to assuming nothing changes.

    Ratios below 1.0 beat persistence. Reported as a ratio rather than an error because
    persistence is the baseline that actually matters at these timescales — the legacy line
    spent a full cycle discovering that no function of a mesh state beat "assume no change"
    one tick ahead, while at 64 ticks a 44% gain was sitting there unclaimed.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = min(len(X), len(Y)) - tau
    if n < 200:
        return {"ratio": None, "n": int(n)}
    Xa, Ya, Yb = X[:n], Y[tau:tau + n], Y[:n]

    predict, te = fit_ridge(Xa, Ya, split=split, n_components=n_components)
    err_model = float(((predict(Xa[te]) - Ya[te]) ** 2).mean())
    err_persist = float(((Yb[te] - Ya[te]) ** 2).mean())
    return {"ratio": err_model / max(err_persist, 1e-12),
            "mse": err_model, "persistence_mse": err_persist, "n": int(n)}


# ------------------------------------------------------- L2-1 concept formation


def normalised_mi(a: np.ndarray, b: np.ndarray) -> float:
    """Mutual information between two labellings, normalised to [0, 1]."""
    a, b = np.asarray(a), np.asarray(b)
    ua, ub = np.unique(a), np.unique(b)
    if len(ua) < 2 or len(ub) < 2:
        return 0.0
    joint = np.zeros((len(ua), len(ub)))
    for i, va in enumerate(ua):
        for j, vb in enumerate(ub):
            joint[i, j] = np.mean((a == va) & (b == vb))
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    return mi / max(np.sqrt(ha * hb), 1e-12)


def gate_concept_formation(concepts: np.ndarray, labels: np.ndarray,
                           patch_clusters: np.ndarray | None = None,
                           seed: int = 0, n_null: int = 20) -> dict:
    """Do the knowledge model's concepts track the world, above two controls?

    `concepts` and `labels` are (T, n_nodes): what each node currently believes it is
    seeing, and what is actually in its receptive field.

    Two controls, and the second is the one that matters. A **shuffled null** catches
    mutual information that is an artefact of label frequency — it caught exactly that in
    the legacy line, where an apparent MI of 0.449 turned out to be entirely bias. And
    **clustering the node's own raw input patch** at the same number of clusters asks the
    only question worth asking of a representation: is there anything here that was not
    already in the pixels?
    """
    rng = np.random.default_rng(seed)
    rows, nulls, ctrl = [], [], []
    for i in range(concepts.shape[1]):
        c, y = concepts[:, i], labels[:, i]
        seen = y >= 0
        if seen.sum() < 50 or len(np.unique(y[seen])) < 2:
            continue
        rows.append(normalised_mi(c[seen], y[seen]))
        nulls.append(float(np.mean([normalised_mi(rng.permutation(c[seen]), y[seen])
                                    for _ in range(n_null)])))
        if patch_clusters is not None:
            ctrl.append(normalised_mi(patch_clusters[seen, i], y[seen]))

    if not rows:
        return {"valid": False, "reason": "no node saw two distinct labels often enough"}
    excess = float(np.mean(rows) - np.mean(nulls))
    out = {
        "mi": float(np.mean(rows)),
        "shuffled_null": float(np.mean(nulls)),
        "excess_over_null": excess,
        "n_nodes_scored": len(rows),
        "sem": (float(np.std(np.array(rows) - np.array(nulls), ddof=1) / np.sqrt(len(rows)))
                if len(rows) > 1 else None),
        "valid": True,
    }
    if ctrl:
        out["raw_patch_clusters_mi"] = float(np.mean(ctrl))
        out["excess_over_pixels"] = float(np.mean(rows) - np.mean(ctrl))
    return out


def kmeans_labels(X: np.ndarray, k: int, seed: int = 0, iters: int = 25) -> np.ndarray:
    """Minimal k-means, so the pixel control has the same number of categories."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    cent = X[rng.choice(len(X), size=min(k, len(X)), replace=False)]
    lab = np.zeros(len(X), dtype=np.int64)
    for _ in range(iters):
        d = ((X[:, None, :] - cent[None]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == lab).all():
            break
        lab = new
        for j in range(len(cent)):
            if (lab == j).any():
                cent[j] = X[lab == j].mean(0)
    return lab


# ------------------------------------------- L2-2 relational vs pooled (kill gate)


def gate_relational(work: np.ndarray, pooled: np.ndarray, target: np.ndarray,
                    tau: int) -> dict:
    """Does keeping relations between members beat summarising over them?

    The legacy line's most transferable finding, restated as a gate: a mean over members
    never beat persistence, while pairwise co-activation over the *same* members reached
    0.79x. Both representations here are built from the identical member states, projected
    to the identical number of components, and scored on the identical target — so the
    only difference left is the operator.
    """
    rel = predictive_gain(work, target, tau)
    pool = predictive_gain(pooled, target, tau)
    if rel["ratio"] is None or pool["ratio"] is None:
        return {"valid": False, "reason": "not enough frames"}
    return {
        "relational_ratio": rel["ratio"],
        "pooled_ratio": pool["ratio"],
        "gain_over_pooling": 1.0 - rel["ratio"] / max(pool["ratio"], 1e-12),
        "beats_persistence": bool(rel["ratio"] < 1.0),
        "control": {"persistence": 1.0, "pooled": pool["ratio"]},
        "valid": True,
    }


# ---------------------------------------------------------------- L2-3 horizon


def gate_horizon(work: np.ndarray, target: np.ndarray, taus=(1, 4, 16, 64),
                 declared: int = 16, baseline: np.ndarray | None = None) -> dict:
    """Where does this level actually beat persistence, and is that where it claims to?

    A level that declares a horizon and then wins somewhere else has not been understood,
    even if the number is good. The contract in `dcn/contract.py` makes the declaration
    mandatory precisely so this gate can contradict it.

    **Scored against a baseline representation at the same tau, not against the raw ratio.**
    The first version of this gate compared ratios directly and would have concluded that
    every model on the bench is a 64-tick model -- because persistence degrades faster than
    anything else as tau grows, so *any* representation, including the raw pixels, looks
    better the further out it is asked to predict. That trend is a property of the baseline,
    not of the model. Dividing by what the raw frame achieves at the identical tau removes
    it, and what is left is the timescale where this representation adds something.
    """
    curve = {str(t): predictive_gain(work, target, t) for t in taus}
    ratios = {t: curve[str(t)]["ratio"] for t in taus if curve[str(t)]["ratio"] is not None}
    if not ratios:
        return {"valid": False}

    rel = None
    if baseline is not None:
        base = {t: predictive_gain(baseline, target, t)["ratio"] for t in taus}
        rel = {t: ratios[t] / base[t] for t in ratios if base.get(t)}

    scored = rel or ratios
    best = min(scored, key=scored.get)
    return {
        "curve": {k: v["ratio"] for k, v in curve.items()},
        "curve_vs_pixels": {str(k): v for k, v in (rel or {}).items()} or None,
        "declared_horizon": declared,
        "best_horizon": int(best),
        "ratio_at_declared": ratios.get(declared),
        "vs_pixels_at_declared": (rel or {}).get(declared),
        # within a factor of two of the declared horizon, and actually adding something
        # there rather than merely riding the baseline's decay
        "declaration_holds": bool(scored.get(declared) is not None
                                  and scored[declared] < 1.0
                                  and abs(np.log2(max(best, 1) / max(declared, 1))) <= 1.0),
        "valid": True,
    }


# --------------------------------------------------- L2-4 publication bottleneck


def gate_publication(publication: np.ndarray, full_state: np.ndarray,
                     target: np.ndarray, tau: int) -> dict:
    """"A DCN does not send information; it publishes only its state." Is that enough?

    The design's boldest claim, and the one most likely to be wrong — the legacy version of
    it (five scalars out of an assembly) is where I would still put my money against. A
    reader gets only prediction error, confidence, activation, novelty, energy and the
    phase spectrum, and is scored against a reader given the whole internal state at the
    same capacity. If the bottleneck costs little, the claim survives.
    """
    pub = predictive_gain(publication, target, tau)
    full = predictive_gain(full_state, target, tau)
    if pub["ratio"] is None or full["ratio"] is None:
        return {"valid": False}
    return {
        "published_only_ratio": pub["ratio"],
        "full_state_ratio": full["ratio"],
        "cost_of_the_bottleneck": pub["ratio"] / max(full["ratio"], 1e-12),
        "published_beats_persistence": bool(pub["ratio"] < 1.0),
        "valid": True,
    }


# ------------------------------------------------- L2-5 phase and channel contention


def gate_channel_contention(dense: np.ndarray, events_gated: np.ndarray,
                            events_ungated: np.ndarray,
                            budgets=(0.02, 0.04, 0.08, 0.16), seed: int = 0) -> dict:
    """Where oscillation is supposed to earn its place: many neurons, one channel.

    Level 1 found that phase gating costs a lone neuron precision, and said so — a gate can
    delay an emission and can never improve it. That was the honest prediction, stated
    before the run. The claim was always that phase pays off in *coordination*, and
    coordination needs more than one neuron, so this is the first place the claim can be
    tested rather than asserted.

    The setup is contention: a node's channel carries at most `budget` of its members per
    tick, and anything over that is dropped. If phase staggers emissions across the cycle,
    fewer collide and less is dropped at the same total rate. If it does not, axiom 3 has
    no operational payoff and should be said to have none.

    **Swept over budgets rather than reported at one.** A single operating point is a
    choice, and a choice made after seeing the numbers is not a measurement. The curve also
    shows where the mechanism matters: severe contention is where scheduling can help, and
    a channel with room to spare is where it cannot.
    """
    from .components import nrmse, zero_order_hold

    out = {"curve": {}}
    for b in budgets:
        row = {}
        for name, ev in (("phase_gated", events_gated), ("ungated", events_ungated)):
            kept, dropped = _apply_budget(ev, b, seed)
            row[name] = {
                "offered_rate": float(ev.mean()),
                "delivered_rate": float(kept.mean()),
                "dropped_frac": float(dropped),
                "nrmse": nrmse(zero_order_hold(dense, kept), dense),
            }
        g, u = row["phase_gated"], row["ungated"]
        row["collision_reduction"] = 1.0 - g["dropped_frac"] / max(u["dropped_frac"], 1e-12)
        row["nrmse_gain"] = 1.0 - g["nrmse"] / max(u["nrmse"], 1e-12)
        out["curve"][f"{b:g}"] = row

    gains = [r["nrmse_gain"] for r in out["curve"].values()]
    out["best_nrmse_gain"] = float(max(gains))
    out["mean_nrmse_gain"] = float(np.mean(gains))
    out["earns_its_place"] = bool(out["mean_nrmse_gain"] > 0.05)
    return out


def _apply_budget(events: np.ndarray, budget: float, seed: int = 0):
    """Drop emissions past a per-tick channel budget, choosing survivors at random.

    Random, not first-come. Keeping the lowest indices systematically starves the
    high-numbered neurons, which would show up as a scheduling effect that is really an
    artefact of array order -- and both variants would be biased the same way, so the
    comparison would look fair while measuring the wrong thing.
    """
    rng = np.random.default_rng(seed)
    n_t, n = events.shape
    cap = max(int(round(budget * n)), 1)
    kept = np.zeros_like(events)
    dropped = 0
    for t in range(n_t):
        idx = np.flatnonzero(events[t])
        if len(idx) > cap:
            dropped += len(idx) - cap
            idx = rng.choice(idx, size=cap, replace=False)
        kept[t, idx] = True
    return kept, dropped / max(events.sum(), 1)
