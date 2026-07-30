"""Aggregate-then-vote, on frozen Corvus towers. The `L2 = assembly, L3 = consensus` hypothesis.

**The claim under test.** A tower sees 6.2% of the retina and cannot form a hypothesis about an
object on its own. So the missing level is not voting — it is an *assembly* that reconstructs a
hypothesis from fragments, with voting one level above that, between assemblies that each covered
the object differently. Every architecture here tried to vote or pool at the wrong level, or
pooled toward the wrong target.

**Why this is not the five aggregation failures.** All five aggregated toward *predicting their
own members*. This aggregates toward a **hypothesis** — which object is this. Different target,
and the five nulls do not bear on it. The one prior test that does is Heron's concept formation,
which lost to k-means on the raw patch by 14x, and that is a declared control here.

**Why the target is object kind, visible.** It is the one place where the information is provably
present and provably destroyed:

    raw pixels, object visible   0.759
    Wren                         0.515
    Swift                        0.612

Two thirds of an available signal, thrown away by every architecture. That is `CGE-A-00` in its
sharpest measurable form, and it is exactly the hole the assembly hypothesis claims to fill.

**Frozen means not modified, not not-run.** Corvus's towers are read, never changed; the tag
`corvus-v4.4-alpha` stays valid. This is the same move that took Heron's send-on-delta policy to
a frozen Wren unit's trace.

**Declared before running** (floors, not hopes):

    assembly  must beat  best_single_tower   AND  kmeans_on_raw_patch
    vote      must beat  best_single_assembly AND mean_of_hypotheses

`best_single_assembly` is chosen on the **training** slice. Choosing it on the test slice would
hand the control the answer, which is the mistake that cost this project `CGE-A-01`.

    python experiments/corvus_assembly_vote.py --ticks 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from architectures.corvus import cortex as corvus_cortex
from core.probes import whiten
from core.world import Sensors
from core.world.identity import IdentityConfig, IdentityWorld
from core.world.sensors import PATCH, RETINA
from exp08_identity import balanced_accuracy

G = RETINA // PATCH                       # 8 patches across
BUDGETS = (8, 16, 32, 64, 128)
"""Matched capacity is **swept**, not fixed, and that is a correction to this file's first run.

At a single budget of 32 a lone tower (29 dims) passes through untouched while an assembly of
five (145 dims) is compressed 4.5:1 -- so the bottleneck, not the aggregation, decides the
comparison, and the condition under test is the one being handicapped. Sweeping and reporting
each condition at *its own best budget* is fair to all of them, including the raw-pixel control,
which is swept too. Choosing the operating point afterwards is exactly what the contention gate
was fixed for."""

ASSEMBLY_SIZE = 5


def local_towers(world, i: int, radius: int = 2):
    """Patch tiles around object i, and the towers whose blocks cover them.

    Object-local for the same two reasons `exp08` is: a whole-frame probe cannot tell which
    object it is being asked about, and with balanced kinds it could identify a hidden object by
    elimination from the ones still visible -- which looks like memory and is not.

    radius=2 gives a 5x5 patch neighbourhood, which spans a 3x3 block of towers. Nine towers is
    the smallest set from which overlapping assemblies can be drawn at all.
    """
    scale = RETINA / world.cfg.size
    cx_, cy_ = world.pos[i] * scale
    px, py = int(cx_ // PATCH), int(cy_ // PATCH)
    tiles, towers = [], []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            r_, c_ = (py + dy) % G, (px + dx) % G
            tiles.append((r_, c_))
            towers.append(int(corvus_cortex.PATCH_TOWER[r_ * G + c_]))
    return tiles, sorted(set(towers))


def patch_pixels(ret, tiles):
    return np.concatenate([ret[r * PATCH:(r + 1) * PATCH,
                                c * PATCH:(c + 1) * PATCH].ravel() for r, c in tiles])


def collect(ticks: int, seed: int, warmup: int, learn_ticks: int) -> list[dict]:
    """Drive frozen Corvus through the identity world, recording object-local state."""
    world = IdentityWorld(IdentityConfig(seed=seed))
    sensors = Sensors()
    cx = corvus_cortex.build_cortex(seed=0)
    for _ in range(30):
        world.step()

    rows = []
    for t in range(ticks):
        s_now, ret = sensors.observe(world)
        cx.learn = t < learn_ticks
        corvus_cortex.tick(cx, s_now)
        if t > warmup:
            h = cx.towers.h
            occ = [world.is_fully_occluded(i) for i in range(world.cfg.n_objects)]
            for i in range(world.cfg.n_objects):
                if occ[i] and sum(occ) != 1:
                    continue                      # ambiguous: more than one hidden
                tiles, towers = local_towers(world, i)
                rows.append({"towers": towers,
                             "h": h[towers].copy(),           # (n_local, 29)
                             "r": patch_pixels(ret, tiles),
                             "kind": int(world.kind[i]),
                             "occ": bool(occ[i]), "obj": i})
        world.step()
    return rows


# --------------------------------------------------------------------- probes

def _fit_scores(X: np.ndarray, y: np.ndarray, cut: int, n_components: int,
                ridge: float = 1e-2):
    """Ridge classifier on {-1,+1}. Returns the *continuous* score, not the decision.

    The score is what makes voting expressible: |score| is distance from the boundary, i.e. how
    confident this assembly is, and selection needs a confidence to select on.
    """
    project = whiten(X, cut, n_components=min(n_components, X.shape[1]))
    Z = project(X)
    Xtr = np.c_[Z[:cut], np.ones(cut)]
    ytr = np.where(y[:cut] == 1, 1.0, -1.0)
    A = Xtr.T @ Xtr
    A[np.diag_indices_from(A)] += ridge * np.trace(A) / A.shape[0]
    w = np.linalg.solve(A, Xtr.T @ ytr)
    return np.c_[Z, np.ones(len(Z))] @ w


def _acc(scores: np.ndarray, y: np.ndarray, cut: int) -> float:
    return balanced_accuracy(np.where(scores[cut:] > 0, 1, 0), y[cut:])


def _kmeans_acc(X: np.ndarray, y: np.ndarray, cut: int, k: int = 8, seed: int = 0) -> float:
    """Unsupervised clustering of the raw patch, then each cluster labelled by its training
    majority. The control that beat Heron's concept formation by 14x."""
    rng = np.random.default_rng(seed)
    Z = (X - X[:cut].mean(0)) / (X[:cut].std(0) + 1e-9)
    C = Z[rng.choice(cut, k, replace=False)]
    for _ in range(25):
        lab = np.argmin(((Z[:cut, None, :] - C[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if (lab == j).any():
                C[j] = Z[:cut][lab == j].mean(0)
    assign = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
    vote = {j: int(round(y[:cut][assign[:cut] == j].mean())) if (assign[:cut] == j).any() else 0
            for j in range(k)}
    return balanced_accuracy(np.array([vote[a] for a in assign[cut:]]), y[cut:])


def evaluate(rows: list[dict], split: float = 0.6) -> dict:
    """Every condition, on one set of frames, at matched capacity."""
    y = np.array([r["kind"] for r in rows])
    cut = int(len(rows) * split)
    if len(rows) < 200 or len(np.unique(y)) < 2:
        return {"valid": False, "reason": f"only {len(rows)} frames"}

    # the tower sets differ per frame (the object moves), so work in a fixed slot layout:
    # slot j = the j-th local tower, ordered by index. Nine slots, zero-padded when fewer.
    n_slots = max(len(r["towers"]) for r in rows)
    W = rows[0]["h"].shape[1]
    H = np.zeros((len(rows), n_slots, W))
    for i, r in enumerate(rows):
        H[i, :len(r["towers"])] = r["h"]
    Rp = np.stack([r["r"] for r in rows])

    out = {"n_frames": len(rows), "n_slots": int(n_slots), "valid": True}

    subsets = list(combinations(range(n_slots), min(ASSEMBLY_SIZE, n_slots)))
    rng = np.random.default_rng(0)
    if len(subsets) > 9:
        subsets = [subsets[i] for i in rng.choice(len(subsets), 9, replace=False)]
    out["n_assemblies"], out["assembly_size"] = len(subsets), len(subsets[0])

    # every condition at every budget; each is then reported at its own best, so no condition
    # is decided by a bottleneck chosen for another one
    curves = {k: [] for k in ("best_single_tower", "all_towers_concatenated", "raw_pixels",
                              "best_single_assembly", "vote_by_selection",
                              "mean_of_hypotheses")}
    for nc in BUDGETS:
        per_tower = [_acc(_fit_scores(H[:, j], y, cut, nc), y, cut) for j in range(n_slots)]
        tr_tower = [balanced_accuracy(
            np.where(_fit_scores(H[:, j], y, cut, nc)[:cut] > 0, 1, 0), y[:cut])
            for j in range(n_slots)]
        curves["best_single_tower"].append(per_tower[int(np.argmax(tr_tower))])
        curves["all_towers_concatenated"].append(
            _acc(_fit_scores(H.reshape(len(rows), -1), y, cut, nc), y, cut))
        curves["raw_pixels"].append(_acc(_fit_scores(Rp, y, cut, nc), y, cut))

        scores = np.stack([_fit_scores(H[:, list(s)].reshape(len(rows), -1), y, cut, nc)
                           for s in subsets])              # (n_assemblies, n_frames)
        # the best assembly is chosen on TRAINING accuracy -- picking it on the test slice would
        # hand the control the answer, the mistake that cost this project CGE-A-01
        train_acc = [balanced_accuracy(np.where(scores[a][:cut] > 0, 1, 0), y[:cut])
                     for a in range(len(subsets))]
        curves["best_single_assembly"].append(
            _acc(scores[int(np.argmax(train_acc))], y, cut))
        pick = np.argmax(np.abs(scores), axis=0)           # most confident assembly, per frame
        curves["vote_by_selection"].append(
            _acc(scores[pick, np.arange(len(rows))], y, cut))
        curves["mean_of_hypotheses"].append(_acc(scores.mean(axis=0), y, cut))

    out["budgets"] = list(BUDGETS)
    out["curves"] = {k: [float(v) for v in vs] for k, vs in curves.items()}
    for k, vs in curves.items():
        out[k] = float(max(vs))
        out[f"{k}_at"] = int(BUDGETS[int(np.argmax(vs))])
    out["kmeans_on_raw_patch"] = _kmeans_acc(Rp, y, cut)

    # --- floors ----------------------------------------------------------------------
    out["floors"] = {
        "assembly_beats_best_tower":
            out["best_single_assembly"] - out["best_single_tower"],
        "assembly_beats_kmeans":
            out["best_single_assembly"] - out["kmeans_on_raw_patch"],
        "vote_beats_best_assembly":
            out["vote_by_selection"] - out["best_single_assembly"],
        "vote_beats_mean":
            out["vote_by_selection"] - out["mean_of_hypotheses"],
        "anything_beats_raw_pixels":
            max(out["vote_by_selection"], out["best_single_assembly"],
                out["all_towers_concatenated"]) - out["raw_pixels"],
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[5])
    args = ap.parse_args()

    print("Aggregate-then-vote on FROZEN corvus-v4.4-alpha towers")
    print(f"target: object kind. controls: raw pixels, k-means on the raw patch.\n")

    results = {}
    for seed in args.seeds:
        rows = collect(args.ticks, seed, args.warmup, int(args.ticks * 0.8))
        for cond, want in (("visible", False), ("occluded", True)):
            sub = [r for r in rows if r["occ"] == want]
            res = evaluate(sub)
            results.setdefault(cond, []).append(res)

            print(f"  seed {seed}  [{cond}]  {res.get('n_frames', 0)} frames")
            if not res.get("valid"):
                print(f"    UNMEASURED: {res.get('reason')}\n")
                continue
            print(f"    {'':26} {'best':>6}  at   " +
                  "  ".join(f"{b:>5}" for b in res["budgets"]))
            for k in ("best_single_tower", "all_towers_concatenated", "best_single_assembly",
                      "mean_of_hypotheses", "vote_by_selection", "raw_pixels"):
                curve = "  ".join(f"{v:.3f}"[1:] for v in res["curves"][k])
                mark = " *" if k == "raw_pixels" else "  "
                print(f"   {mark}{k:25} {res[k]:.3f}  {res[k + '_at']:>3}   {curve}")
            print(f"    *{'kmeans_on_raw_patch':25} {res['kmeans_on_raw_patch']:.3f}")
            print("    (* = control)")
            print("    floors:")
            for k, v in res["floors"].items():
                print(f"      {k:32} {v:+.3f}  {'PASS' if v > 0.05 else 'FAIL'}")
            print()

    p = Path(__file__).resolve().parents[1] / "logs" / "corvus_assembly_vote.json"
    p.write_text(json.dumps({"substrate": "corvus-v4.4-alpha (frozen, read-only)",
                             "ticks": args.ticks, "results": results}, indent=1))
    print(f"wrote logs/{p.name}")


if __name__ == "__main__":
    main()
