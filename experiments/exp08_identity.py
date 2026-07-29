"""Increment 3 -- will the mesh build an object representation when it has to?

Every failure so far reduces to one: the mesh never represented objects, because a local
motion field is cheaper and the prediction objective never rewarded anything more. Making
rare events count for more does not fix that -- local trajectory extrapolation still
solves occlusion. The question that matters is what prediction is *impossible* without
object identity.

The identity world answers it. A passer crosses the occluder band, a bouncer reverses
inside it, the two are visually distinct in the open and indistinguishable while hidden,
and at the moment of disappearance their velocities are identical. So which side an
object re-emerges from is determined only by which object went in.

The measurement is a decode of the hidden object's kind from the model's state during
occlusion, reported as **balanced accuracy** (chance 50% for any constant strategy --
necessary here because bouncers re-enter the band repeatedly and make the raw base rate
69%). Two controls decide whether the number means anything:

* **raw retina during occlusion** must be at chance. If pixels reveal the kind, the task
  does not require memory and the whole experiment is void.
* **raw retina while visible** must be well above chance, proving the probe works and the
  cue is there to be picked up.

    python experiments/exp08_identity.py --ticks 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from architectures.wren.mesh import build_mesh, tick
from core.metrics import JsonlLogger
from architectures.wren.state import Config
from core.world import Sensors
from core.world.sensors import PATCH, RETINA
from core.world.identity import IdentityConfig, IdentityWorld
from architectures.swift import Config2, build_mesh2, tick2


def local_units(world, i, side, radius=1):
    """Visual units whose receptive fields tile the neighbourhood of object i.

    The probe has to be *local to the object*, for two independent reasons. A whole-frame
    probe cannot tell which of four objects it is being asked about -- the assignment
    problem a linear decoder cannot solve. And with a fixed, balanced set of kinds, a
    whole-frame probe can identify the hidden object by elimination from the three still
    visible, which would look like memory and is not.

    A local patch shows the object's brightness when it is in the open and flat grey when
    it is behind the band, which is exactly the contrast the experiment needs.
    """
    g = RETINA // PATCH
    scale = RETINA / world.cfg.size
    cx, cy = world.pos[i] * scale
    px, py = int(cx // PATCH), int(cy // PATCH)
    tiles, mesh_idx = [], []
    slots = Sensors.lattice_slots(side)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            r_, c_ = (py + dy) % g, (px + dx) % g
            tiles.append((r_, c_))
            mesh_idx.append(slots[r_ * g + c_])
    return tiles, np.array(mesh_idx)


def patch_pixels(ret, tiles):
    return np.concatenate([ret[r * PATCH:(r + 1) * PATCH,
                               c * PATCH:(c + 1) * PATCH].ravel() for r, c in tiles])


def collect(kind, state, n_ticks, seed, warmup, learn_ticks, side):
    """Record object-local model state and the object's kind, tick by tick.

    Only frames with exactly one fully occluded object are kept for the occluded
    condition, so "the hidden object's kind" is unambiguous.
    """
    world = IdentityWorld(IdentityConfig(seed=seed))
    sensors = Sensors()
    for _ in range(30):
        world.step()

    rows = []
    for t in range(n_ticks):
        s_now, ret = sensors.observe(world)
        if state is not None:
            state.learn = t < learn_ticks
            (tick2 if kind == "v2" else tick)(state, s_now)

        if t > warmup:
            occ = [world.is_fully_occluded(i) for i in range(world.cfg.n_objects)]
            for i in range(world.cfg.n_objects):
                if occ[i] and sum(occ) != 1:
                    continue                     # ambiguous: more than one hidden
                tiles, mesh_idx = local_units(world, i, side)
                rows.append({
                    "h": state.h[mesh_idx].ravel().copy() if state is not None else None,
                    "r": patch_pixels(ret, tiles),
                    "kind": int(world.kind[i]),
                    "occ": bool(occ[i]),
                    "obj": i,
                })
        world.step()
    return rows


def balanced_accuracy(pred, true):
    accs = []
    for c in np.unique(true):
        sel = true == c
        if sel.sum():
            accs.append(float((pred[sel] == c).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def decode_kind(X, y, split=0.6, ridge=1e-2):
    """Ridge classifier on {-1, +1}, scored by balanced accuracy on a time split."""
    n = len(X)
    if n < 40 or len(np.unique(y)) < 2:
        return None
    cut = int(n * split)
    Xtr = np.c_[X[:cut], np.ones(cut)]
    Xte = np.c_[X[cut:], np.ones(n - cut)]
    ytr = np.where(y[:cut] == 1, 1.0, -1.0)
    A = Xtr.T @ Xtr
    A[np.diag_indices_from(A)] += ridge * np.trace(A) / A.shape[0]
    w = np.linalg.solve(A, Xtr.T @ ytr)
    pred = np.where(Xte @ w > 0, 1, 0)
    return {"balanced_acc": balanced_accuracy(pred, y[cut:]), "n_test": int(n - cut)}


def probe_all(rows, obj=None):
    """Decode kind from each representation, split by whether the object is visible."""
    sel = [r for r in rows if obj is None or r["obj"] == obj]
    out = {}
    for cond, want_occ in (("visible", False), ("occluded", True)):
        sub = [r for r in sel if r["occ"] == want_occ]
        if len(sub) < 40:
            continue
        y = np.array([r["kind"] for r in sub])
        out[cond] = {}
        for src in ("h", "r"):
            if sub[0][src] is None:
                continue
            X = np.stack([r[src] for r in sub])
            res = decode_kind(X, y)
            if res:
                out[cond]["model_state" if src == "h" else "raw_retina"] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--learn-ticks", type=int, default=15000)
    ap.add_argument("--side", type=int, default=12)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="logs/exp08_identity.jsonl")
    args = ap.parse_args()

    models = {
        "v1_mesh": ("v1", build_mesh(Config(lattice_side=args.side, seed=0, eta_head=0.01))),
        "v2_mesh": ("v2", build_mesh2(Config2(lattice_side=args.side, seed=0, eta_head=0.01))),
    }

    results = {}
    with JsonlLogger(args.out, meta=vars(args)) as log:
        for name, (kind, state) in models.items():
            print(f"running {name} on the identity world ...")
            rows = collect(kind, state, args.ticks, args.seed, args.warmup,
                           args.learn_ticks, args.side)
            results[name] = probe_all(rows)
            log.log(kind="identity", model=name, **results[name])

    Path("logs/exp08_summary.json").write_text(json.dumps(results, indent=2))
    _report(results)


def _report(res: dict) -> None:
    print(f"\ndecoding the hidden object's kind (balanced accuracy, chance = 50%)\n")
    print(f"{'model':10s} {'state:visible':>14s} {'state:OCCLUDED':>15s} "
          f"{'retina:visible':>15s} {'retina:occluded':>16s}")
    for name, r in res.items():
        def g(cond, src):
            v = r.get(cond, {}).get(src)
            return f"{v['balanced_acc']*100:.1f}%" if v else "n/a"
        print(f"{name:10s} {g('visible','model_state'):>14s} "
              f"{g('occluded','model_state'):>15s} {g('visible','raw_retina'):>15s} "
              f"{g('occluded','raw_retina'):>16s}")

    ctrl = next((r.get("occluded", {}).get("raw_retina") for r in res.values()
                 if r.get("occluded", {}).get("raw_retina")), None)
    vis = next((r.get("visible", {}).get("raw_retina") for r in res.values()
                if r.get("visible", {}).get("raw_retina")), None)
    print("\ncontrols:")
    if vis:
        ok = vis["balanced_acc"] > 0.65
        print(f"  retina while visible {vis['balanced_acc']*100:5.1f}%  "
              + ("the cue is present and the probe can read it"
                 if ok else "PROBE BROKEN: even visible kind is not decodable"))
    if ctrl:
        ok = abs(ctrl["balanced_acc"] - 0.5) < 0.1
        print(f"  retina while hidden  {ctrl['balanced_acc']*100:5.1f}%  "
              + ("nothing leaks through the occluder, so memory is required"
                 if ok else "LEAK: the kind is visible during occlusion; experiment void"))

    print("\nreading:")
    for name, r in res.items():
        v = r.get("occluded", {}).get("model_state")
        if not v:
            continue
        acc = v["balanced_acc"]
        verdict = ("REMEMBERS which object is hidden" if acc > 0.65 else
                   "partial" if acc > 0.57 else
                   "at chance -- no object identity in the state")
        print(f"  {name}: {acc*100:5.1f}% -> {verdict}")


if __name__ == "__main__":
    main()
