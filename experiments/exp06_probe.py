"""G1 -- what does a representation actually contain?

The v1 occlusion experiment (E2) asked a behavioural question: is the mesh more
surprised when a hidden object's trajectory is violated? It answered no. But a
behavioural null is ambiguous -- it cannot distinguish "the representation is not
maintained" from "it is maintained but the readout does not use it", and it is far less
sensitive than looking directly at the state.

This decodes object position, velocity and identity straight out of a model's state
with a ridge probe, scored separately while the object is visible and while it is
hidden, always against the raw retina as a control. On world v1 that control was
decisive: the mesh state decoded visible position at 16.3 world units where the raw
retina it had just been fed managed 7.5, and during occlusion the mesh carried nothing
beyond what a static frame implies (14.7 versus 14.1). The mesh had no object
representation to maintain in the first place, in full view -- which is the real reason
E2 was null.

Two properties make this measurement trustworthy:

* **The retina control.** If the probe cannot decode from raw pixels either, the probe
  is the limitation and the run says nothing. Every table reports it.
* **One object by default.** With several identical-looking objects a *linear* probe
  cannot solve the assignment problem even in principle, so a null would be
  uninterpretable. `--n-objects` raises it deliberately, for the identity test.

    python experiments/exp06_probe.py --model mesh --ticks 9000
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
from core.world.physics import PhysicsConfig, PhysicsWorld


def collect(model_state, n_ticks, seed, n_objects, warmup, learn=True):
    """Run the world, recording the model state, the retina, and ground truth."""
    world = PhysicsWorld(PhysicsConfig(seed=seed, n_objects=n_objects))
    sensors = Sensors()
    for _ in range(30):
        world.step()

    H, R, POS, VEL, OCC, BRI = [], [], [], [], [], []
    for t in range(n_ticks):
        s_now, ret = sensors.observe(world)
        if model_state is not None:
            model_state.learn = learn
            tick(model_state, s_now)
        if t > warmup:
            if model_state is not None:
                H.append(model_state.h.ravel().copy())
            R.append(ret.ravel().copy())
            POS.append(world.pos.ravel().copy())
            VEL.append(world.vel.ravel().copy())
            OCC.append([world.is_fully_occluded(i) for i in range(n_objects)])
            BRI.append(world.bright.copy())
        world.step()

    return {
        "state": np.array(H) if H else None,
        "retina": np.array(R),
        "pos": np.array(POS), "vel": np.array(VEL),
        "occ": np.array(OCC), "bright": np.array(BRI),
    }


def ridge_probe(X, Y, occ, ridge=1e-2, split=0.6):
    """Decode Y from X, held out in time, scored separately by occlusion status."""
    n = len(X)
    cut = int(n * split)
    Xtr = np.c_[X[:cut], np.ones(cut)]
    Xte = np.c_[X[cut:], np.ones(n - cut)]
    Ytr, Yte = Y[:cut], Y[cut:]
    A = Xtr.T @ Xtr
    A[np.diag_indices_from(A)] += ridge * np.trace(A) / A.shape[0]
    P = Xte @ np.linalg.solve(A, Xtr.T @ Ytr)

    err = np.sqrt(((P - Yte) ** 2).sum(1))
    chance = np.sqrt(((Yte - Ytr.mean(0)) ** 2).sum(1))
    o = occ[cut:]
    return {
        "visible": float(err[~o].mean()) if (~o).any() else None,
        "occluded": float(err[o].mean()) if o.any() else None,
        "chance": float(chance.mean()),
        "n_occluded": int(o.sum()),
    }


def run_probes(data, obj=0):
    """Every probe, for every state representation available."""
    sources = {"raw_retina (control)": data["retina"]}
    if data["state"] is not None:
        sources["model_state"] = data["state"]

    targets = {
        "position": data["pos"][:, 2 * obj:2 * obj + 2],
        "velocity": data["vel"][:, 2 * obj:2 * obj + 2],
    }
    occ = data["occ"][:, obj]

    out = {}
    for tname, Y in targets.items():
        out[tname] = {sname: ridge_probe(X, Y, occ) for sname, X in sources.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mesh", "none"], default="mesh")
    ap.add_argument("--ticks", type=int, default=9000)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--n-objects", type=int, default=1)
    ap.add_argument("--side", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--eta-head", type=float, default=0.08)
    ap.add_argument("--out", default="logs/exp06_probe.jsonl")
    args = ap.parse_args()

    state = None
    if args.model == "mesh":
        cfg = Config(lattice_side=args.side, seed=0, eta_head=args.eta_head)
        state = build_mesh(cfg)

    print(f"collecting {args.ticks} ticks ({args.n_objects} object(s)) ...")
    data = collect(state, args.ticks, args.seed, args.n_objects, args.warmup)
    res = run_probes(data)

    with JsonlLogger(args.out, meta=vars(args)) as log:
        for target, bysrc in res.items():
            for src, r in bysrc.items():
                log.log(kind="probe", target=target, source=src, **r)

    Path("logs/exp06_summary.json").write_text(json.dumps(res, indent=2))
    _report(res, data)


def _report(res: dict, data: dict) -> None:
    print(f"\nfrom {len(data['retina'])} scored frames, "
          f"{int(data['occ'][:, 0].sum())} of them fully occluded")
    for target, bysrc in res.items():
        print(f"\n  {target} decode error (world units, lower is better):")
        for src, r in bysrc.items():
            vis = f"{r['visible']:6.2f}" if r["visible"] is not None else "   n/a"
            occ = f"{r['occluded']:6.2f}" if r["occluded"] is not None else "   n/a"
            print(f"    {src:22s} visible {vis} | occluded {occ} "
                  f"| chance {r['chance']:6.2f}")

    pos = res["position"]
    ctrl = pos["raw_retina (control)"]
    if ctrl["visible"] is None or ctrl["visible"] >= ctrl["chance"] * 0.8:
        print("\n  PROBE INVALID: the control cannot decode position from raw pixels, "
              "so\n  nothing here constrains any model. Fix the probe, not the model.")
        return
    if "model_state" not in pos:
        return

    m = pos["model_state"]
    print("\n  reading:")
    print(f"    visible  -- model {m['visible']:.2f} vs retina {ctrl['visible']:.2f}: "
          + ("the state preserves positional information"
             if m["visible"] <= ctrl["visible"] * 1.2
             else "the state DESTROYS positional information it was fed"))
    if m["occluded"] is not None and ctrl["occluded"] is not None:
        gain = 1 - m["occluded"] / ctrl["occluded"]
        print(f"    occluded -- model {m['occluded']:.2f} vs retina "
              f"{ctrl['occluded']:.2f} ({gain*100:+.0f}%): "
              + ("the model tracks what it cannot see" if gain > 0.15
                 else "the model carries NOTHING beyond the static frame"))


if __name__ == "__main__":
    main()
