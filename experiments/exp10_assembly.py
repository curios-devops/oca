"""Gates PA1-PA4 -- is the Predictive Assembly a level of abstraction, or just pooling?

Run on the tunnel maze, because it has the cleanest control this project has produced:
inside a covered corridor every frame is byte-identical, so a raw-pixel probe is at
chance *by construction* rather than by argument. Position while blind can only come from
integrating your own moves, which is exactly what a slow shared field is for.

* **PA1 -- does the field add anything?** Workspace against a plain mean of the same
  members. This is the gate that can kill the idea: if the field never beats pooling,
  "the object emerges in the shared field" is unsupported and the assembly is a
  convenience.
* **PA2 -- persistence.** Error against how long the agent has been blind. The assembly
  should decay more slowly than its members; equal decay means no abstraction.
* **PA3 -- does the coalition manager earn its keep?** Dynamic membership against fixed.
* **PA4 -- are five scalars enough?** Decode from the executive interface alone.

    python experiments/exp10_assembly.py --ticks 14000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.metrics import JsonlLogger
from core.probes import decode_error
from core.world import Sensors
from core.world.maze import MazeConfig, MazeWorld
from legacy.v2 import Config2, build_mesh2, tick2
from legacy.pa import AssemblyConfig, build_assembly, export, pooled_members, step_assembly


def collect(cfg_a, ticks, seed, warmup, learn_ticks, side=12):
    world = MazeWorld(MazeConfig(seed=seed, tunnels=True))
    sensors = Sensors()
    mesh = build_mesh2(Config2(lattice_side=side, seed=0, eta_head=0.01))
    asm = build_assembly(mesh, cfg_a)

    rows = []
    for t in range(ticks):
        s_now, ret = sensors.observe(world)
        mesh.learn = t < learn_ticks
        tick2(mesh, s_now)
        step_assembly(asm, mesh, learn=t < learn_ticks)

        if t > warmup and world.in_tunnel():
            rows.append({
                "workspace": asm.w.ravel().copy(),
                "pooled": pooled_members(asm, mesh).ravel(),
                "mesh": mesh.h.ravel().copy(),
                "exports": export(asm, mesh).ravel(),
                "retina": ret.ravel().copy(),
                "pos": world.pos.astype(float).copy(),
                "entry": np.array(world._tunnel_entry, dtype=float),
                "steps": world.steps_in_tunnel(),
            })
        world.step()
    return rows, asm


N_COMPONENTS = 48
"""Every representation is projected to this many components before decoding, so the
comparison is about information rather than about width. The mesh state is 3456-dimensional
and the pooled vector 96; without equalising, the wide one simply overfits."""


def probe(rows, key, split=0.6, ridge=1e-2):
    """Decode true position from one representation, at matched capacity."""
    return decode_error(np.stack([r[key] for r in rows]),
                        np.stack([r["pos"] for r in rows]), split, ridge,
                        n_components=N_COMPONENTS)


def references(rows, split=0.6):
    Y = np.stack([r["pos"] for r in rows])
    entry = np.stack([r["entry"] for r in rows])
    cut = int(len(Y) * split)
    return {
        "frozen_at_entry": np.linalg.norm(entry[cut:] - Y[cut:], axis=1),
        "chance": np.linalg.norm(Y[:cut].mean(0) - Y[cut:], axis=1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=14000)
    ap.add_argument("--warmup", type=int, default=2500)
    ap.add_argument("--learn-ticks", type=int, default=11000)
    ap.add_argument("--side", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/exp10_assembly.jsonl")
    args = ap.parse_args()

    print("collecting (dynamic membership) ...")
    rows, asm = collect(AssemblyConfig(), args.ticks, args.seed, args.warmup,
                        args.learn_ticks, args.side)
    if len(rows) < 300:
        raise SystemExit(f"only {len(rows)} blind frames -- raise --ticks")

    refs = references(rows)
    errs = {k: probe(rows, k) for k in ("workspace", "pooled", "mesh", "exports", "retina")}
    errs.update(refs)
    steps = np.array([r["steps"] for r in rows])[int(len(rows) * 0.6):]

    print("collecting (fixed membership, for PA3) ...")
    rows_fixed, _ = collect(AssemblyConfig(manager=False, dynamic_membership=False),
                            args.ticks, args.seed, args.warmup, args.learn_ticks,
                            args.side)
    fixed_err = probe(rows_fixed, "workspace")

    res = {
        "n_blind_frames": len(rows),
        "mean": {k: float(v.mean()) for k, v in errs.items()},
        "fixed_membership": float(fixed_err.mean()),
        "n_dissolved": int(asm.n_dissolved),
        "by_blind_duration": {},
    }
    for lo, hi in ((1, 2), (3, 4), (5, 99)):
        sel = (steps >= lo) & (steps <= hi)
        if sel.sum() >= 20:
            res["by_blind_duration"][f"{lo}-{hi if hi < 99 else '+'}"] = {
                "n": int(sel.sum()),
                **{k: float(v[sel].mean()) for k, v in errs.items()},
            }

    with JsonlLogger(args.out, meta=vars(args)) as log:
        log.log(kind="assembly_gates", **{k: v for k, v in res.items()
                                          if k != "by_blind_duration"})
    Path("logs/exp10_summary.json").write_text(json.dumps(res, indent=2))
    _report(res)


def _report(res: dict) -> None:
    m = res["mean"]
    print(f"\nposition error while blind, in maze cells, from {res['n_blind_frames']} "
          f"frames (lower is better)\n")
    order = ["retina", "chance", "frozen_at_entry", "mesh", "pooled", "workspace",
             "exports"]
    labels = {
        "retina": "raw retina (control)", "chance": "chance",
        "frozen_at_entry": "frozen at tunnel mouth", "mesh": "full mesh state",
        "pooled": "members, mean-pooled (PA1 control)",
        "workspace": "assembly workspace", "exports": "5 exported scalars (PA4)",
    }
    for k in order:
        print(f"  {labels[k]:36s} {m[k]:6.2f}")

    print("\ncontrol: pixels must be at chance inside a tunnel -- "
          + ("OK" if abs(m["retina"] - m["chance"]) < 0.15 * m["chance"]
             else "LEAK, experiment void"))

    pa1 = 1 - m["workspace"] / m["pooled"]
    print(f"\nPA1  workspace vs mean-pooled members: {pa1*100:+5.1f}%  "
          + ("the field adds something pooling does not"
             if pa1 > 0.05 else "NO -- the assembly is pooling"))

    pa3 = 1 - m["workspace"] / res["fixed_membership"]
    print(f"PA3  dynamic vs fixed membership:      {pa3*100:+5.1f}%  "
          f"({res['n_dissolved']} assemblies re-formed)")

    pa4 = 1 - m["exports"] / m["pooled"]
    print(f"PA4  five scalars vs pooled members:   {pa4*100:+5.1f}%  "
          + ("the bottleneck is sufficient" if pa4 > -0.05
             else "the five-scalar interface loses the information"))

    if res["by_blind_duration"]:
        print("\nPA2  persistence -- error against how long the agent has been blind:")
        print(f"    {'blind':>7s} {'n':>5s} {'workspace':>10s} {'pooled':>8s} "
              f"{'mesh':>8s} {'frozen':>8s}")
        for k, b in res["by_blind_duration"].items():
            print(f"    {k:>7s} {b['n']:5d} {b['workspace']:10.2f} {b['pooled']:8.2f} "
                  f"{b['mesh']:8.2f} {b['frozen_at_entry']:8.2f}")
        print("    the assembly should decay more slowly than its members; "
              "equal decay means no abstraction")


if __name__ == "__main__":
    main()
