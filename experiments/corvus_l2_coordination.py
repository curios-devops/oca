"""Corvus Layer 2 — `CGE-B-05`, the coordination floor. Q7 option B.

Layer 2 does two jobs. Compression is optional, off by default, and measured at −0.004 ± 0.006
(`CGE-B-03`). **Coordination runs on every tick in both modes and had nothing to beat** — the
Heron failure reproduced in the contract instead of the implementation. This gate is the floor
that fixes that.

The question, stated so it can fail: **does a tower's cluster-mates tell you anything about that
tower's future that the tower does not already tell you about itself?**

Target: a tower's own published state, one cluster-horizon ahead (64 ticks — a layer scored at
the wrong horizon is competing with persistence and cannot win). Predictors: the tower's current
state plus its `m-1` cluster-mates, under four membership rules.

| condition | members | role |
|---|---|---|
| `independent_towers` | the tower alone | **control 1** — the null. No coordination at all. |
| `random` | seed + random others | sanity: is the affinity estimate measuring anything? |
| `fixed_proximity_membership` | seed + contiguous neighbours | **control 2** — what we hard-coded, never measured |
| `connectivity` | seed + the towers it co-varies with | **the mechanism** |

`connectivity` must beat **both** declared controls by the floor's margin. Beating the null
alone would only show that more inputs help; beating proximity is what tests the claim the
cortical-column literature actually makes.

Three design points, each of which would invalidate the run if got wrong.

**One trajectory, four memberships.** The cluster does not feed back into the towers, so
membership cannot change the publications. Running four times would inject seed noise into a
comparison that has none. The tower trace is collected once and replayed through four stacks.

**The layer's own rule, not a copy.** Membership comes from stepping a real `ClusterStack`, so
the gate cannot drift away from the code it judges.

**Matched capacity.** A tower alone offers `width` numbers; a cluster offers `m * width`. Left
unmatched, this gate would reward width and say nothing about grouping — the error that made
relational aggregation look like a result until mean pooling matched it. Every condition is
projected to the same number of components.

    python experiments/corvus_l2_coordination.py --ticks 14000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from architectures.corvus import cluster as L2
from architectures.corvus import cortex as corvus_cortex
from architectures.corvus.contract import LAYERS
from core.probes import fit_ridge
from core.world import Sensors
from core.world.maze import MazeConfig, MazeWorld

MODES = ("independent_towers", "random", "fixed_proximity_membership", "connectivity")


def collect_publications(seed: int, ticks: int, warmup: int) -> np.ndarray:
    """(T, n_towers, width) — what the towers published, once, for every condition to share."""
    world = MazeWorld(MazeConfig(seed=seed))
    sensors = Sensors()
    cx = corvus_cortex.build_cortex(seed=0)
    learn_until = int(ticks * 0.8)

    pub = []
    for t in range(ticks):
        s_now, _ = sensors.observe(world)
        cx.learn = t < learn_until
        corvus_cortex.tick(cx, s_now)
        if t > warmup:
            pub.append(cx.towers.publication().copy())
        world.step()
    return np.asarray(pub, dtype=np.float64)


def membership_for(mode: str, pub: np.ndarray, seed: int) -> np.ndarray | None:
    """Replay the trace through a real cluster and take the membership it settled on.

    `independent_towers` has no membership by definition -- it is the absence of the layer.
    """
    if mode == "independent_towers":
        return None
    rule = {"random": "random", "fixed_proximity_membership": "proximity",
            "connectivity": "connectivity"}[mode]
    n_towers = pub.shape[1]
    stack = L2.build_stack(L2.ClusterConfig(seed=seed, membership=rule), n_towers=n_towers)
    for t in range(len(pub)):
        L2.step(stack, pub[t])
    return stack.membership


def score(pub: np.ndarray, membership: np.ndarray | None, horizon: int,
          n_components: int, split: float = 0.6) -> float:
    """Mean held-out NRMSE of predicting each tower's own state `horizon` ticks ahead.

    Every tower is scored, not only the cluster seeds: a rule that helps four towers and hurts
    thirteen has not helped.
    """
    T, n_towers, width = pub.shape
    X_end = T - horizon
    cut = int(X_end * split)

    errs = []
    for i in range(n_towers):
        if membership is None:
            src = pub[:X_end, i, :]
        else:
            # the cluster this tower belongs to; a tower can sit in more than one, so take the
            # first, and fall back to the tower alone if no cluster claimed it
            rows = np.where((membership == i).any(axis=1))[0]
            if len(rows) == 0:
                src = pub[:X_end, i, :]
            else:
                src = pub[:X_end][:, membership[rows[0]], :].reshape(X_end, -1)
        Y = pub[horizon:, i, :]
        predict, test = fit_ridge(src, Y, split=split, n_components=n_components)
        pred, truth = predict(src[test]), Y[test]
        denom = truth.std() + 1e-12
        errs.append(float(np.sqrt(((pred - truth) ** 2).mean()) / denom))
    return float(np.mean(errs))


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of members two membership rules agree on, cluster by cluster.

    The diagnostic that separates two very different failures. If connectivity scores no better
    than proximity *and* picks different towers, grouping does not matter. If it scores no
    better because it picks the **same** towers, then in this world co-variation simply is
    adjacency -- which would be a finding about retinotopic routing, not about coordination.
    """
    return float(np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(a, b)]))


def run_seed(seed: int, ticks: int, warmup: int, horizon: int) -> dict:
    pub = collect_publications(seed, ticks, warmup)
    width = pub.shape[2]
    row = {"seed": seed, "n_frames": int(len(pub)), "n_components": width}
    members = {m: membership_for(m, pub, seed) for m in MODES}
    for mode in MODES:
        row[mode] = score(pub, members[mode], horizon, n_components=width)

    prox, conn, rand = (members["fixed_proximity_membership"], members["connectivity"],
                        members["random"])
    row["overlap_connectivity_proximity"] = overlap(conn, prox)
    row["overlap_random_proximity"] = overlap(rand, prox)

    # lower NRMSE is better, so a positive delta means the mechanism reconstructs better
    for ctl in ("independent_towers", "fixed_proximity_membership"):
        row[f"vs_{ctl}"] = 1.0 - row["connectivity"] / max(row[ctl], 1e-12)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=14000)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    layer = LAYERS["cluster"]
    floors = [f for f in layer.floors if f.job == "coordination"]
    horizon = layer.horizon

    print(f"CGE-B-05 -- Corvus Layer 2 coordination, {args.ticks} ticks, seeds {args.seeds}")
    print(f"target: a tower's own state {horizon} ticks ahead, matched capacity")
    for f in floors:
        print(f"declared floor: beat {f.beats} by {f.margin:+.2f}")
    print()

    rows = [run_seed(s, args.ticks, args.warmup, horizon) for s in args.seeds]

    print(f"  {'seed':>4}  " + "  ".join(f"{m[:14]:>14}" for m in MODES))
    for r in rows:
        print(f"  {r['seed']:>4}  " + "  ".join(f"{r[m]:>14.4f}" for m in MODES))
    print("  (NRMSE, lower is better)\n")

    ov = np.mean([r["overlap_connectivity_proximity"] for r in rows])
    ovr = np.mean([r["overlap_random_proximity"] for r in rows])
    same = ("the two rules pick nearly the same towers" if ov > 0.8
            else "the two rules pick genuinely different towers")
    print(f"  membership overlap with proximity: connectivity {ov:.2f}, random {ovr:.2f}"
          f"   ({same})\n")

    verdicts = {}
    for f in floors:
        d = np.array([r[f"vs_{f.beats}"] for r in rows])
        ok = bool(d.min() > f.margin)
        verdicts[f.beats] = {"mean": float(d.mean()),
                             "std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                             "min": float(d.min()), "passed": ok}
        print(f"  connectivity vs {f.beats:28} {d.mean():+.3f} +/- "
              f"{verdicts[f.beats]['std']:.3f}   worst {d.min():+.3f}  "
              f"{'PASS' if ok else 'FAIL'}")

    passed = all(v["passed"] for v in verdicts.values())
    print(f"\n  => coordination floor: {'PASS' if passed else 'FAIL'}"
          + ("" if passed else "  -- grouping by connectivity does not beat every control"))

    out = {"gate": "CGE-B-05", "layer": "cluster", "job": "coordination",
           "horizon": horizon, "ticks": args.ticks,
           "verdict": "PASS" if passed else "FAIL",
           "floors": {f.beats: f.margin for f in floors},
           "per_control": verdicts, "per_seed": rows}
    p = Path(__file__).resolve().parents[1] / "logs" / "corvus_l2_coordination.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote logs/{p.name}")


if __name__ == "__main__":
    main()
