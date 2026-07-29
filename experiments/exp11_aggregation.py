"""Where does the predictive information disappear?

The previous section left an unusually clean failure. Predicting an assembly's members 64
ticks ahead, relative to a persistence baseline:

    full mesh state      0.69x   -- the information exists
    assembly workspace   6.56x   -- the assembly destroys it

The RPDUs are not the bottleneck and neither is the hierarchy. The *compression operator*
is. So this holds everything else fixed -- same mesh, same members, same target, same
protocol -- and varies only how member states become an assembly state.

Deliberately measured **offline, at best case**, with a ridge fit rather than the online
learning rule. That separates two questions that were tangled together before: does the
representation *contain* the predictive information, and can a local rule *find* it? This
experiment answers only the first, which is the one that decides whether the operator is
the problem.

The operators, in increasing order of what they preserve:

* **mean** -- the spec's obvious reading, and the current baseline. Four units reporting
  "edge moving right", "left", "right", "left" average to nothing, while the assembly
  plainly contains two coherent hypotheses.
* **attention** -- a confidence-weighted mean. Still a mean, so still one vector, but it
  can at least ignore members that are not contributing.
* **pairwise** -- co-activation between members. Identity survives, and so do relations:
  "42 active while 77 is silent" is a different state from the reverse, where a mean
  cannot tell them apart.
* **synchrony** -- who is phase-locked with whom, read from v2's own rotor phases. If
  transient synchrony is the code, this is the representation.
* **reservoir** -- an autonomous recurrent field that members *perturb* rather than
  rebuild each tick, so the trajectory persists instead of being recomputed.

Two references bracket everything: the full mesh state (the ceiling, since the assembly
cannot know more than its input) and the members kept apart with no compression at all.

    python experiments/exp11_aggregation.py --ticks 16000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.metrics import JsonlLogger
from core.probes import whiten
from core.world import Sensors
from core.world.gridworld import GridWorld, WorldConfig
from architectures.swift import Config2, build_mesh2, tick2
from architectures.swift import dynamics as dyn
from architectures.swift.pa import AssemblyConfig, build_assembly


def _upper(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]


class Reservoir:
    """An autonomous recurrent field, perturbed by members rather than rebuilt.

    Spectral radius below 1 gives the echo-state property: the field forgets its initial
    condition but keeps a fading memory of the whole input trajectory, which is what
    "the trajectory predicts, not the state" needs to be testable.
    """

    def __init__(self, n_in: int, n: int = 256, radius: float = 0.95, seed: int = 0):
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 1, (n, n))
        self.W = W * (radius / max(abs(np.linalg.eigvals(W)).max(), 1e-9))
        self.W_in = rng.normal(0, 1 / np.sqrt(n_in), (n, n_in))
        self.x = np.zeros(n)

    def step(self, u: np.ndarray) -> np.ndarray:
        self.x = np.tanh(self.W @ self.x + self.W_in @ u)
        return self.x


def collect(ticks, seed, warmup, world_cfg, side=12, res_size=256):
    world = GridWorld(world_cfg)
    sensors = Sensors()
    mesh = build_mesh2(Config2(lattice_side=side, seed=0, eta_head=0.01))
    asm = build_assembly(mesh, AssemblyConfig(manager=False))
    idx = asm.member_idx[0]                      # one assembly, fixed membership
    reservoir = Reservoir(n_in=len(idx) * mesh.h.shape[1], n=res_size)
    for _ in range(30):
        world.step()

    rows = {k: [] for k in ("mesh", "members", "mean", "attention", "pairwise",
                            "synchrony", "reservoir")}
    for t in range(ticks):
        s_now, _ = sensors.observe(world)
        mesh.learn = t < int(ticks * 0.75)
        tick2(mesh, s_now)

        h = mesh.h[idx]                          # (m, d) member readouts
        conf = mesh.conf[idx]
        flat = h.ravel()
        res = reservoir.step(flat)

        if t > warmup:
            rows["mesh"].append(mesh.h.ravel().copy())
            rows["members"].append(flat.copy())
            rows["mean"].append(h.mean(axis=0))
            w = conf / (conf.sum() + 1e-8)
            rows["attention"].append(w @ h)
            act = np.linalg.norm(h, axis=1)      # per-member activity
            rows["pairwise"].append(_upper(np.outer(act, act)))
            ph = dyn.phase(mesh.z[idx])          # (m, rotors) real phases from v2
            rows["synchrony"].append(_upper(np.cos(ph[:, None, :] - ph[None, :, :]).mean(2)))
            rows["reservoir"].append(res.copy())
        world.step()
    return {k: np.array(v) for k, v in rows.items()}


LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def predictive_score(X, target, horizon, n_components, split=0.6):
    """Best-case offline fit of the change `horizon` ticks ahead, over persistence.

    The ridge strength is chosen on a validation slice of the *training* half, never on
    the test half. Without it a low-dimensional, nearly-constant input like the mean
    vector is left effectively unregularised and returns absurd scores -- measured 3174x
    persistence, which is a conditioning failure being read as a result.
    """
    Y = target[horizon:] - target[:-horizon]
    X = X[:-horizon]
    cut = int(len(X) * split)
    inner = int(cut * 0.8)
    reduce = whiten(X, inner, n_components)
    Z = reduce(X)

    def fit(rows, lam):
        A = np.c_[Z[rows], np.ones(len(rows))]
        G = A.T @ A
        G[np.diag_indices_from(G)] += lam * np.trace(G) / G.shape[0]
        return np.linalg.solve(G, A.T @ Y[rows])

    def err(W, rows):
        P = np.c_[Z[rows], np.ones(len(rows))] @ W
        return float(((P - Y[rows]) ** 2).mean())

    tr, va, te = np.arange(inner), np.arange(inner, cut), np.arange(cut, len(Z))
    lam = min(LAMBDAS, key=lambda l: err(fit(tr, l), va))
    mse = err(fit(np.arange(cut), lam), te)
    pers = float((Y[te] ** 2).mean())
    return mse / pers, mse, pers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=16000)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--components", type=int, nargs="*", default=[48, 128])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--objects", type=int, default=1)
    ap.add_argument("--occluder", action="store_true")
    ap.add_argument("--side", type=int, default=12)
    ap.add_argument("--out", default="logs/exp11_aggregation.jsonl")
    args = ap.parse_args()

    wc = WorldConfig(seed=args.seed, n_objects=args.objects, occluder=args.occluder)
    print(f"collecting {args.ticks} ticks ...")
    data = collect(args.ticks, args.seed, args.warmup, wc, args.side)

    # The target is the members kept apart, so it does not presuppose that a mean is the
    # right summary -- the question is whether their future is predictable at all.
    target = data["members"]
    order = ["mesh", "members", "reservoir", "synchrony", "pairwise", "attention", "mean"]
    res = {}
    for k in order:
        res[k] = {"dim": int(data[k].shape[1])}
        for nc in args.components:
            ratio, mse, pers = predictive_score(data[k], target, args.horizon, nc)
            res[k][f"nc{nc}"] = ratio

    with JsonlLogger(args.out, meta=vars(args)) as log:
        log.log(kind="aggregation", horizon=args.horizon, **res)
    Path("logs/exp11_summary.json").write_text(json.dumps(res, indent=2))
    _report(res, order, args)


def _report(res, order, args):
    labels = {
        "mesh": "full mesh state (ceiling)",
        "members": "members, no compression",
        "reservoir": "autonomous reservoir",
        "synchrony": "synchrony graph (who locks with whom)",
        "pairwise": "pairwise co-activation",
        "attention": "attention-weighted mean",
        "mean": "mean vector (current baseline)",
    }
    print(f"\npredicting the members {args.horizon} ticks ahead, best-case offline fit")
    print("relative to persistence -- below 1.00 means the representation predicts\n")
    head = "  ".join(f"{'PCA ' + str(nc):>9s}" for nc in args.components)
    print(f"{'representation':40s} {'dim':>6s}  {head}")
    for k in order:
        r = res[k]
        cells = "  ".join(f"{r[f'nc{nc}']:9.2f}" for nc in args.components)
        print(f"{labels[k]:40s} {r['dim']:6d}  {cells}")

    best_nc = args.components[-1]
    ceiling = res["mesh"][f"nc{best_nc}"]
    mean = res["mean"][f"nc{best_nc}"]
    print(f"\nceiling {ceiling:.2f}   mean-vector baseline {mean:.2f}")
    print("\nhow much of the gap each operator recovers:")
    for k in order:
        if k in ("mesh", "mean"):
            continue
        v = res[k][f"nc{best_nc}"]
        rec = (mean - v) / max(mean - ceiling, 1e-9)
        print(f"  {labels[k]:40s} {v:6.2f}   {rec*100:6.1f}%")
    print("\nthe question is not which scores best but where the information survives:")
    print("an operator that recovers most of the gap shows the mean was destroying it.")


if __name__ == "__main__":
    main()
