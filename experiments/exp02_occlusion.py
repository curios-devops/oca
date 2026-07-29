"""E2 -- does the mesh maintain a representation of an object it cannot see?

An object passes fully behind the occluder and emerges. On half the trials its vertical
velocity is flipped *while it is hidden*, so the world it re-emerges into contradicts any
forward prediction made at the moment it disappeared. Horizontal motion is left alone, so
both conditions stay hidden for the same number of ticks and emerge on the same side at
the same moment -- only the height is wrong.

The measurement is the mesh's prediction error in the frames right after re-emergence:

  no representation of the hidden object  ->  equally surprised either way
  some representation of it               ->  more surprised when it was perturbed

That asymmetry is the whole test, and copy-last is run through the identical scoring
path as a zero-permanence reference: it has no state at all, so if the harness reports
an asymmetry for copy-last, the measure is broken rather than the mesh being clever.
An untrained mesh is scored too, so any effect can be attributed to learning rather
than to architecture.

    python experiments/exp02_occlusion.py --train 8000 --trials 60
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data import rollout
from legacy.v1.mesh import build_mesh, predicted_retina, tick
from core.metrics import JsonlLogger, frame_mse
from legacy.v1.run import run_stream
from legacy.v1.state import Config
from core.world import GridWorld, Sensors, WorldConfig

# A wider occluder than the default so an object is fully hidden for ~10 ticks;
# smaller objects for the same reason.
E2_WORLD = WorldConfig(
    occluder_rect=(24, 4, 16, 56),
    radius_range=(3.0, 4.0),
    speed_range=(0.5, 0.9),
    seed=0,
)


def local_mse(pred: np.ndarray, target: np.ndarray, world, i: int, pad: int = 4) -> float:
    """MSE restricted to a box around object i's true position, in retina coordinates.

    Whole-frame MSE is dominated by the two objects that are *not* under test and by
    static background, which dilutes the emergence signal to nothing. Scoring locally is
    what makes the measurement sensitive enough to detect an effect if one exists.
    """
    scale = target.shape[0] / world.cfg.size
    cx, cy = world.pos[i] * scale
    half = world.radius[i] * 1.35 * scale + pad
    r0, r1 = int(max(0, cy - half)), int(min(target.shape[0], cy + half + 1))
    c0, c1 = int(max(0, cx - half)), int(min(target.shape[1], cx + half + 1))
    if r1 <= r0 or c1 <= c0:
        return frame_mse(pred, target)
    return frame_mse(pred[r0:r1, c0:c1], target[r0:r1, c0:c1])


def run_trials(mesh, sensors, n_ticks, seed, window=4, min_hidden=4, logger=None,
               tag="trial", probe_tau=4):
    """Drive the world continuously, perturbing every other occlusion event.

    The mesh is never reset between trials: it stays in the same continuous regime it
    was trained in, which is the only regime the hypothesis is about.

    The headline measure uses the `probe_tau`-step prediction, which was made while the
    object was still fully hidden. That is the actual permanence question -- did the
    mesh, seeing nothing, expect an object to appear? -- whereas the 1-step prediction
    is made once the object is already back in view and mostly restates copy-last.
    """
    cfg = WorldConfig(**{**E2_WORLD.__dict__, "seed": seed})
    world = GridWorld(cfg)
    rng = np.random.default_rng(seed + 777)
    for _ in range(30):
        world.step()

    horizons = mesh.cfg.horizons if mesh is not None else (1, 4, 16)
    hidden_since, condition = {}, {}
    next_condition = 0
    pending, trials = [], []
    pred_hist = deque(maxlen=max(horizons) + 1)
    frame_hist = deque(maxlen=2)

    for t in range(n_ticks):
        s_now, ret = sensors.observe(world)
        if mesh is not None:
            tick(mesh, s_now)

        # score any trial whose emergence window covers this tick
        for rec in pending:
            if not (rec["start"] < t <= rec["start"] + window):
                continue
            i = rec["object"]
            if len(frame_hist) == 2:
                rec["copy"].append(local_mse(frame_hist[-1], ret, world, i))
            if mesh is not None:
                for tau in horizons:
                    if len(pred_hist) >= tau:
                        rec[f"mesh{tau}"].append(
                            local_mse(pred_hist[-tau][tau], ret, world, i)
                        )

        done = [r for r in pending if t > r["start"] + window]
        for r in done:
            pending.remove(r)
            if not r["copy"]:
                continue
            rec = {
                "condition": r["condition"],
                "object": r["object"],
                "hidden_ticks": r["hidden"],
                "copy_mse": float(np.mean(r["copy"])),
            }
            for tau in horizons:
                vals = r[f"mesh{tau}"]
                rec[f"mesh_mse_t{tau}"] = float(np.mean(vals)) if vals else None
            rec["mesh_mse"] = rec.get(f"mesh_mse_t{probe_tau}")
            trials.append(rec)
            if logger is not None:
                logger.log(kind=tag, **rec)

        frame_hist.append(ret)
        if mesh is not None:
            pred_hist.append({tau: predicted_retina(mesh, tau, sensors)
                              for tau in horizons})

        for i in range(cfg.n_objects):
            now_hidden = world.is_fully_occluded(i)
            was_hidden = i in hidden_since
            if now_hidden and not was_hidden:
                hidden_since[i] = t
                condition[i] = "perturbed" if next_condition % 2 else "preserved"
                next_condition += 1
                if condition[i] == "perturbed":
                    world.perturb(i, rng, "flip_y")
            elif was_hidden and not now_hidden:
                dur = t - hidden_since.pop(i)
                cond = condition.pop(i)
                if dur >= min_hidden:
                    rec = {"start": t, "condition": cond, "object": i,
                           "hidden": dur, "copy": []}
                    rec.update({f"mesh{tau}": [] for tau in horizons})
                    pending.append(rec)

        world.step()

    return trials


def summarise(trials, key):
    out = {}
    for cond in ("preserved", "perturbed"):
        vals = [t[key] for t in trials if t["condition"] == cond and t[key] is not None]
        out[cond] = {
            "n": len(vals),
            "mean": float(np.mean(vals)) if vals else None,
            "sem": float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else None,
        }
    a = [t[key] for t in trials if t["condition"] == "perturbed" and t[key] is not None]
    b = [t[key] for t in trials if t["condition"] == "preserved" and t[key] is not None]
    if a and b:
        out["asymmetry"] = float(np.mean(a) - np.mean(b))
        out["ratio"] = float(np.mean(a) / max(np.mean(b), 1e-12))
        out["p_permutation"] = permutation_p(a, b)
    return out


def permutation_p(a, b, n_perm=20000, seed=0):
    """Two-sided permutation test on the difference of means."""
    rng = np.random.default_rng(seed)
    pool = np.array(a + b)
    obs = abs(np.mean(a) - np.mean(b))
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--ticks", type=int, default=12000)
    ap.add_argument("--side", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--out", default="logs/exp02_occlusion.jsonl")
    args = ap.parse_args()

    sensors = Sensors()
    cfg = Config(lattice_side=args.side, seed=args.seed, eta_head=0.08)

    with JsonlLogger(args.out, meta=vars(args)) as log:
        print("training on the occluded world ...")
        _, sen, _ = rollout(args.train, seed=args.seed, world_cfg=E2_WORLD)
        ret, _, _ = rollout(args.train, seed=args.seed, world_cfg=E2_WORLD)
        trained = build_mesh(cfg)
        run_stream(trained, sen, ret, learn=True, sensors=sensors, logger=log,
                   tag="e2_train", log_every=500)
        trained.learn = False

        untrained = build_mesh(cfg)
        untrained.learn = False

        print(f"running trials over {args.ticks} ticks ...")
        tr_trained = run_trials(trained, sensors, args.ticks, seed=4242,
                                window=args.window, logger=log, tag="trial_trained")
        tr_untrained = run_trials(untrained, sensors, args.ticks, seed=4242,
                                  window=args.window, logger=log, tag="trial_untrained")

        results = {
            "trained_mesh": summarise(tr_trained, "mesh_mse"),
            "untrained_mesh": summarise(tr_untrained, "mesh_mse"),
            "copy_last_reference": summarise(tr_trained, "copy_mse"),
            "n_trials": len(tr_trained),
        }
        log.log(kind="result", **results)

    Path("logs/exp02_summary.json").write_text(json.dumps(results, indent=2))
    _report(results)


def _report(res: dict) -> None:
    print(f"\n{res['n_trials']} scored occlusion events\n")
    print(f"{'model':22s} {'preserved':>12s} {'perturbed':>12s} {'ratio':>7s} {'p':>8s}")
    for name in ("trained_mesh", "untrained_mesh", "copy_last_reference"):
        r = res[name]
        if r.get("ratio") is None:
            continue
        print(f"{name:22s} {r['preserved']['mean']:12.5f} {r['perturbed']['mean']:12.5f} "
              f"{r['ratio']:7.2f} {r['p_permutation']:8.4f}")
    t = res["trained_mesh"]
    ok = t.get("ratio", 0) > 1 and t.get("p_permutation", 1) < 0.05
    print("\nE2 " + ("SUPPORTS" if ok else "DOES NOT SUPPORT")
          + " a maintained representation of the hidden object")


if __name__ == "__main__":
    main()
