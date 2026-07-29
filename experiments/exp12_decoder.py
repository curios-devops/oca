"""Is the learner the bottleneck now, or is the representation still?

The aggregation experiment left a clean split: pairwise co-activation carries the signal
offline (0.79x persistence) while the online delta rule extracts little of it (1.72x). So
this freezes the representation and varies only the decoder.

Crossed with the representation, which is the part that decides the interpretation. If a
non-linear decoder on *raw members* matches a non-linear decoder on *pairwise features*,
then the pairwise operator was only doing the learner's job by hand, and the finding is
"you need a non-linear learner" rather than "you need a relational representation". Those
two conclusions point at completely different next architectures, and a decoder-only
comparison cannot tell them apart.

The three questions the 2 x N grid answers:

* **ridge on pairwise vs ridge on members** -- does the operator help a linear learner?
* **mlp on pairwise vs mlp on members** -- does it still help once the learner can form
  products itself? This is the control.
* **mlp on pairwise vs ridge on pairwise** -- is there structure beyond second order?
  Pairwise features are quadratic in member activity, so a two-layer network over them
  reaches higher orders. If that helps, interactions above pairwise matter.

    python experiments/exp12_decoder.py --ticks 16000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from exp11_aggregation import LAMBDAS, collect
from core.metrics import JsonlLogger
from core.probes import whiten
from core.world.gridworld import WorldConfig


def _splits(n, split=0.6, val=0.8):
    cut = int(n * split)
    inner = int(cut * val)
    return np.arange(inner), np.arange(inner, cut), np.arange(cut, n)


def ridge_decoder(Z, Y, tr, va, te):
    def fit(rows, lam):
        A = np.c_[Z[rows], np.ones(len(rows))]
        G = A.T @ A
        G[np.diag_indices_from(G)] += lam * np.trace(G) / G.shape[0]
        return np.linalg.solve(G, A.T @ Y[rows])

    def err(W, rows):
        return float((((np.c_[Z[rows], np.ones(len(rows))] @ W) - Y[rows]) ** 2).mean())

    lam = min(LAMBDAS, key=lambda l: err(fit(tr, l), va))
    return err(fit(np.concatenate([tr, va]), lam), te)


def mlp_decoder(Z, Y, tr, va, te, hidden=64, epochs=200, lr=3e-3, seed=0, batch=256):
    """Two hidden layers over the frozen representation, early-stopped on validation."""
    rng = np.random.default_rng(seed)
    d_in, d_out = Z.shape[1], Y.shape[1]
    p = {
        "W1": rng.normal(0, 1 / np.sqrt(d_in), (d_in, hidden)), "b1": np.zeros(hidden),
        "W2": rng.normal(0, 1 / np.sqrt(hidden), (hidden, hidden)), "b2": np.zeros(hidden),
        "W3": rng.normal(0, 1 / np.sqrt(hidden), (hidden, d_out)), "b3": np.zeros(d_out),
    }
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    step = 0

    def fwd(X):
        a1 = np.tanh(X @ p["W1"] + p["b1"])
        a2 = np.tanh(a1 @ p["W2"] + p["b2"])
        return a1, a2, a2 @ p["W3"] + p["b3"]

    def loss(rows):
        return float(((fwd(Z[rows])[2] - Y[rows]) ** 2).mean())

    best, best_p, patience = loss(va), {k: x.copy() for k, x in p.items()}, 0
    for _ in range(epochs):
        for i in range(0, len(tr), batch):
            rows = tr[i:i + batch]
            X, T = Z[rows], Y[rows]
            a1, a2, out = fwd(X)
            n = out.size
            g = {}
            d3 = 2 * (out - T) / n
            g["W3"], g["b3"] = a2.T @ d3, d3.sum(0)
            d2 = (d3 @ p["W3"].T) * (1 - a2 * a2)
            g["W2"], g["b2"] = a1.T @ d2, d2.sum(0)
            d1 = (d2 @ p["W2"].T) * (1 - a1 * a1)
            g["W1"], g["b1"] = X.T @ d1, d1.sum(0)

            step += 1
            bc1, bc2 = 1 - 0.9**step, 1 - 0.999**step
            for k, gk in g.items():
                m[k] = 0.9 * m[k] + 0.1 * gk
                v[k] = 0.999 * v[k] + 0.001 * gk * gk
                p[k] -= lr * (m[k] / bc1) / (np.sqrt(v[k] / bc2) + 1e-8)

        cur = loss(va)
        if cur < best - 1e-9:
            best, best_p, patience = cur, {k: x.copy() for k, x in p.items()}, 0
        else:
            patience += 1
            if patience >= 15:
                break
    p = best_p
    return loss(te)


def delta_decoder(Z, Y, tr, va, te, passes=1, lr=0.02, seed=0):
    """The assembly's own rule: local, incremental, normalised LMS.

    Run with `passes=1` it is exactly what the assembly does online. Run with many passes
    over the same data it becomes an optimiser question rather than a function-class one:
    if extra passes close the gap to ridge, the delta rule was simply converging too
    slowly on correlated inputs, which is the classic weakness of plain LMS and the reason
    recursive least squares exists.
    """
    rng = np.random.default_rng(seed)
    W = np.zeros((Z.shape[1] + 1, Y.shape[1]))
    A_tr = np.c_[Z[tr], np.ones(len(tr))]
    best, best_W = np.inf, W.copy()
    for _ in range(passes):
        for i in rng.permutation(len(tr)):
            x = A_tr[i]
            err = Y[tr[i]] - x @ W
            W += lr * np.outer(x, err) / (x @ x + 1e-8)
        cur = float((((np.c_[Z[va], np.ones(len(va))] @ W) - Y[va]) ** 2).mean())
        if cur < best:
            best, best_W = cur, W.copy()
    return float((((np.c_[Z[te], np.ones(len(te))] @ best_W) - Y[te]) ** 2).mean())


def with_context(Z, lags=(0, 8, 16, 32)):
    """Stack the representation at several lags, to test temporal integration."""
    out = []
    for lag in lags:
        shifted = np.roll(Z, lag, axis=0)
        shifted[:lag] = Z[0]
        out.append(shifted)
    return np.concatenate(out, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=16000)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--components", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--objects", type=int, default=2)
    ap.add_argument("--occluder", action="store_true", default=True)
    ap.add_argument("--out", default="logs/exp12_decoder.jsonl")
    args = ap.parse_args()

    wc = WorldConfig(seed=args.seed, n_objects=args.objects, occluder=args.occluder)
    print(f"collecting {args.ticks} ticks ...")
    data = collect(args.ticks, args.seed, args.warmup, wc)

    tau = args.horizon
    target = data["members"]
    Y = target[tau:] - target[:-tau]
    n = len(Y)
    tr, va, te = _splits(n)
    pers = float((Y[te] ** 2).mean())

    reps = {"members (raw)": data["members"], "pairwise": data["pairwise"]}
    res = {}
    for rname, raw in reps.items():
        X = raw[:-tau]
        reduce = whiten(X, len(tr), args.components)
        Z = reduce(X)
        Zc = with_context(Z)
        res[rname] = {}
        for dname, fn, feats, kw in (
                ("delta rule, 1 pass", delta_decoder, Z, {"passes": 1}),
                ("delta rule, 40 passes", delta_decoder, Z, {"passes": 40}),
                ("ridge (linear)", ridge_decoder, Z, {}),
                ("mlp (2 hidden)", mlp_decoder, Z, {}),
                ("mlp + temporal context", mlp_decoder, Zc, {})):
            mse = fn(feats, Y, tr, va, te, **kw)
            res[rname][dname] = mse / pers
            print(f"  {rname:16s} {dname:24s} {mse / pers:6.3f}", flush=True)

    with JsonlLogger(args.out, meta=vars(args)) as log:
        log.log(kind="decoder", horizon=tau, **res)
    Path("logs/exp12_summary.json").write_text(json.dumps(res, indent=2))
    _report(res)


def _report(res):
    decoders = list(next(iter(res.values())))
    print(f"\npredicting members 64 ticks ahead, relative to persistence "
          f"(below 1.00 predicts)\n")
    print(f"{'decoder':26s}" + "".join(f"{r:>18s}" for r in res))
    for d in decoders:
        print(f"{d:26s}" + "".join(f"{res[r][d]:18.3f}" for r in res))

    conv = (res["pairwise"]["delta rule, 1 pass"]
            - res["pairwise"]["delta rule, 40 passes"])
    opt = (res["pairwise"]["delta rule, 40 passes"] - res["pairwise"]["ridge (linear)"])
    lin_gap = res["members (raw)"]["ridge (linear)"] - res["pairwise"]["ridge (linear)"]
    mlp_gap = res["members (raw)"]["mlp (2 hidden)"] - res["pairwise"]["mlp (2 hidden)"]
    order = res["pairwise"]["ridge (linear)"] - res["pairwise"]["mlp (2 hidden)"]
    ctx = (res["pairwise"]["mlp (2 hidden)"]
           - res["pairwise"]["mlp + temporal context"])

    print("\nreading:")
    print(f"  operator helps a LINEAR learner       {lin_gap:+.3f}")
    print(f"  operator helps a NON-LINEAR learner   {mlp_gap:+.3f}   "
          + ("<- the representation is doing real work"
             if mlp_gap > 0.02 else
             "<- the operator was standing in for the learner"))
    print(f"  structure beyond second order         {order:+.3f}   "
          + ("<- higher-order interactions matter" if order > 0.02
             else "<- pairwise already captures it"))
    print(f"  temporal context adds                 {ctx:+.3f}")
    print(f"  extra passes of the delta rule buy    {conv:+.3f}   "
          + ("<- the online rule was just converging slowly" if conv > 0.05
             else "<- more data passes do not help it"))
    print(f"  delta rule (converged) still behind ridge by {opt:+.3f}")


if __name__ == "__main__":
    main()
