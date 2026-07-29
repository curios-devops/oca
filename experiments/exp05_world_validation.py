"""G0 -- can world v2 discriminate between architectures at all?

World v1 could not. A 7x7x2 linear filter was very nearly Bayes-optimal there, so no
architecture and no ablation of one could demonstrate value above it, and every v1 null
result is consistent with that single fact. Before spending any effort on architecture,
the new world has to be shown to reward the things the architecture is supposed to
provide. Two separate questions, two separate instruments:

**G0a -- is there non-linear structure?**  `LocalMLP` versus `LinearAR`. Identical
features, identical target, only the function class differs, so any gap is
non-linearity and nothing else. (A GRU is useless for this: it lost to the linear filter
on world v1 too, so a GRU-based test cannot separate "the world is linear" from "the GRU
was badly optimised".)

**G0b -- does the world reward memory?**  This is the one that matters, and it is a
question about the *world*, so it is answered without any vision model in the way: from
ground truth, how much of an object's re-emergence position is determined by the state
it had when it disappeared? A memoryless observer must guess from the marginal; an
observer that tracked the object can do better by exactly the amount this measures. If
that gap is small, no architecture could ever demonstrate permanence here, however good
its vision -- there would be nothing to demonstrate.

**G0c (reported, not a gate)** -- how much of that gap the current pixel baselines
actually close. Expected to be very little: a local filter is structurally incapable of
it, since at the moment the prediction is issued the object is not inside any window.
That remaining gap is the target Stage 2's architecture has to hit.

    python experiments/exp05_world_validation.py --train 8000 --test 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.baselines import (HORIZONS, GRU, LinearAR, LocalMLP, MemoryLinear,
                            as_batch)
from core.data import rollout
from core.metrics import JsonlLogger, frame_mse
from core.world.physics import PhysicsConfig, PhysicsWorld, make_physics_world

CRITERION = 0.25


def occlusion_episodes(snaps, min_hidden):
    """(t_start, t_end, object) for every episode of full occlusion."""
    out, hidden = [], {}
    for t, s in enumerate(snaps):
        for i, occ in enumerate(s.get("fully_occluded", [])):
            if occ and i not in hidden:
                hidden[i] = t
            elif not occ and i in hidden:
                t0 = hidden.pop(i)
                if t - t0 >= min_hidden:
                    out.append((t0, t, i))
    return out


def occlusion_predictability(seeds=range(8), n_ticks=30000, min_hidden=8):
    """G0b: from ground truth, how determined is re-emergence by the entry state?

    Deliberately model-free. This asks whether the *world* rewards having tracked a
    hidden object, which is a property of the world's dynamics and must not be confused
    with whether some particular network can see well enough to exploit it.

    Vertical exit position is the quantity that matters. Horizontal exit position is
    pinned by the geometry of the band -- an object leaves where the occluder ends
    whatever it did inside -- so scoring it would flatter any model.
    """
    X, Y = [], []
    for seed in seeds:
        w = PhysicsWorld(PhysicsConfig(seed=int(seed)))
        hidden = {}
        for t in range(n_ticks):
            for i in range(w.cfg.n_objects):
                occ = w.is_fully_occluded(i)
                if occ and i not in hidden:
                    hidden[i] = (t, w.pos[i].copy(), w.vel[i].copy(), w.regime, w.radius[i])
                elif not occ and i in hidden:
                    t0, p0, v0, reg, rad = hidden.pop(i)
                    if t - t0 >= min_hidden:
                        X.append([*p0, *v0, reg, rad, t - t0])
                        Y.append([*w.pos[i], *w.vel[i]])
            w.step()
    X, Y = np.array(X), np.array(Y)
    cut = int(len(X) * 0.7)

    def ridge_err(col):
        A_ = np.c_[X[:cut], np.ones(cut)]
        B_ = np.c_[X[cut:], np.ones(len(X) - cut)]
        A = A_.T @ A_
        A[np.diag_indices_from(A)] += 1e-6 * np.trace(A) / A.shape[0]
        P = B_ @ np.linalg.solve(A, A_.T @ Y[:cut])
        err = float(np.abs(P[:, col] - Y[cut:, col]).mean())
        chance = float(np.abs(Y[cut:, col] - Y[:cut, col].mean()).mean())
        return err, chance

    ey, cy = ridge_err(1)
    ex, cx = ridge_err(0)
    return {
        "n_episodes": len(X),
        "median_hidden_ticks": float(np.median(X[:, 6])),
        "exit_y_err": ey, "exit_y_chance": cy, "exit_y_gain": 1 - ey / cy,
        "exit_x_err": ex, "exit_x_chance": cx, "exit_x_gain": 1 - ex / cx,
    }


def local_box(frame_shape, pos, size, pad=4):
    scale = frame_shape[0] / size
    cx, cy = pos * scale
    r0, r1 = int(max(0, cy - pad)), int(min(frame_shape[0], cy + pad + 1))
    c0, c1 = int(max(0, cx - pad)), int(min(frame_shape[1], cx + pad + 1))
    return r0, r1, c0, c1


def score(predict, frames, snaps, episodes, world_size, after=(2, 9)):
    """Whole-frame MSE, and error on re-emergence predicted from behind the occluder.

    The re-emergence number only counts a (target frame, horizon) pair when the
    prediction was issued at a tick where the object was still *fully invisible*. That
    is the whole point: it measures what a model knew about something it could not see.
    """
    n = len(frames)
    whole = {tau: [] for tau in HORIZONS}
    local = {tau: [] for tau in HORIZONS}
    n_scored = 0

    for tau in HORIZONS:
        for t in range(2, n - tau):
            p = predict(t, tau)
            if p is not None:
                whole[tau].append(frame_mse(p, frames[t + tau]))

        for (t0, te, i) in episodes:
            for f in range(te + after[0], min(te + after[1], n)):
                t = f - tau
                if not (t0 <= t < te):          # must have been issued while hidden
                    continue
                p = predict(t, tau)
                if p is None:
                    continue
                r0, r1, c0, c1 = local_box(frames[0].shape, snaps[f]["pos"][i], world_size)
                if r1 > r0 and c1 > c0:
                    local[tau].append(frame_mse(p[r0:r1, c0:c1], frames[f][r0:r1, c0:c1]))
                    n_scored += 1
    return {
        "whole": {f"mse_t{t}": float(np.mean(whole[t])) for t in HORIZONS},
        "reemerge": {f"mse_t{t}": (float(np.mean(local[t])) if local[t] else None)
                     for t in HORIZONS},
        "n_scored": n_scored,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlp-epochs", type=int, default=10)
    ap.add_argument("--gru-epochs", type=int, default=250)
    ap.add_argument("--gru-hidden", type=int, default=192)
    ap.add_argument("--pred-seeds", type=int, default=8)
    ap.add_argument("--pred-ticks", type=int, default=30000)
    ap.add_argument("--out", default="logs/exp05_world.jsonl")
    args = ap.parse_args()

    tr, _, _ = rollout(args.train, seed=args.seed, world_factory=make_physics_world)
    te, _, te_snaps = rollout(args.test, seed=9000, world_factory=make_physics_world)
    world_size = 64
    episodes = occlusion_episodes(te_snaps, min_hidden=max(HORIZONS) // 2)
    print(f"train {tr.shape}  test {te.shape}  occlusion episodes: {len(episodes)}")

    results = {}
    with JsonlLogger(args.out, meta=vars(args)) as log:
        print("measuring how determined re-emergence is by the entry state ...")
        results["_occlusion"] = occlusion_predictability(
            seeds=range(args.pred_seeds), n_ticks=args.pred_ticks)
        log.log(kind="occlusion_predictability", **results["_occlusion"])

        print("fitting baselines ...")
        lin = LinearAR().fit(tr)
        mlp = LocalMLP(seed=args.seed).fit(tr, epochs=args.mlp_epochs,
                                           logger=log, log_every=2000)
        mem = MemoryLinear().fit(tr)
        gru = GRU(n_in=tr[0].size, hidden=args.gru_hidden, seed=args.seed)
        gru.fit(as_batch(tr, 8), epochs=args.gru_epochs, log_every=200, logger=log)
        gru_pred = gru.predict_stream(te)

        models = {
            "copy_last": lambda t, tau: te[t],
            "linear_ar": lambda t, tau: lin.predict(te, t, tau),
            "local_mlp": lambda t, tau: mlp.predict(te, t, tau),
            "memory_linear": lambda t, tau: mem.predict(te, t, tau),
            "gru": lambda t, tau: gru_pred[tau][t],
        }
        for name, fn in models.items():
            results[name] = score(fn, te, te_snaps, episodes, world_size)
            log.log(kind="world_v2", model=name, **results[name])

    _report(results)
    Path("logs/exp05_summary.json").write_text(json.dumps(results, indent=2))


def _fmt(d):
    return "  ".join(
        f"t{t}=" + (f"{d[f'mse_t{t}']:.5f}" if d[f"mse_t{t}"] is not None else "  n/a")
        for t in HORIZONS)


def _report(res: dict) -> None:
    print(f"\n{'model':14s} {'whole frame':38s} re-emergence events")
    for name, r in res.items():
        if name.startswith("_"):
            continue
        print(f"{name:14s} {_fmt(r['whole']):38s} {_fmt(r['reemerge'])}")

    pred = res["_occlusion"]
    g0a = 1 - res["local_mlp"]["whole"]["mse_t16"] / res["linear_ar"]["whole"]["mse_t16"]
    g0b = pred["exit_y_gain"]
    memoryless = min(res[m]["reemerge"]["mse_t16"]
                     for m in ("linear_ar", "local_mlp", "copy_last"))
    stateful = min(res[m]["reemerge"]["mse_t16"] for m in ("memory_linear", "gru"))
    g0c = 1 - stateful / memoryless

    print(f"\nG0a  non-linear structure : local MLP beats linear by {g0a*100:+5.1f}% "
          f"(whole frame, 16-step)")
    print(f"G0b  memory is rewarded   : re-emergence height is {g0b*100:5.1f}% "
          f"determined by the entry state")
    print(f"       exit y error {pred['exit_y_err']:.2f} against {pred['exit_y_chance']:.2f} "
          f"at chance, {pred['n_episodes']} episodes, "
          f"median {pred['median_hidden_ticks']:.0f} ticks hidden")
    print(f"       exit x only {pred['exit_x_gain']*100:+.1f}% -- pinned by the band "
          f"edge, as expected, so scoring it would flatter any model")
    print(f"G0c  currently exploited  : best stateful pixel baseline beats best "
          f"memoryless by {g0c*100:+5.1f}% at re-emergence")

    ok = g0b >= CRITERION
    print(f"\nG0 {'PASS' if ok else 'FAIL'}  (need G0b >= {CRITERION*100:.0f}%)")
    if ok:
        print(f"     -> the world rewards tracking a hidden object. The distance between "
              f"the {g0b*100:.0f}%\n        available and the {g0c*100:.0f}% today's "
              f"pixel baselines capture is the gap Stage 2 targets.")
    else:
        print("     -> re-emergence is not determined by anything a tracker could have\n"
              "        known, so no architecture could demonstrate permanence here.")


if __name__ == "__main__":
    main()
