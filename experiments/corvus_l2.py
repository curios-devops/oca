"""Corvus Layer 2 — `CGE-B-03`, the cluster's own declared floor.

Required by rule R1. Layer 2 is built, it runs on every tick, and it has **never been asked to
beat anything.** That is precisely the failure mode the Corvus contract was written to prevent:
Heron's Layer 2 satisfied its contract completely while being worse than the layer below it,
because no contract ever asked.

The floor is declared in `architectures/corvus/cluster.py` and predates this file:

    Floor(beats="pass_through", margin=0.05)

    "A cluster must beat the concatenation of its own members before it is permitted to
     summarise them; the burden of proof belongs on the compression step."

So the question is exact: **does the cluster's summary retain more about the world than its own
members do, at matched capacity?** If it does not, the compliant behaviour is to keep passing
members through — which is the default — and Layer 2 contributes coordination but no
abstraction.

Two design points that decide whether the answer means anything.

**One run, both readouts.** The cluster does not feed back into the towers, so the summary is a
deterministic linear map of the same tick's pass-through. Running twice would add seed noise to
a comparison that has none. Both readouts come from the same trajectory.

**Matched capacity, or the result is about width.** The summary is 4x16 = 64 numbers; the
pass-through is 4x5xwidth, far more. Comparing them raw would reward the wider one and say
nothing about compression — the error that made relational aggregation look like a result until
mean pooling matched its width. The pass-through is therefore projected to the *same* number of
components before either is probed.

    python experiments/corvus_l2.py --ticks 12000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from architectures.corvus import cortex as corvus_cortex
from architectures.corvus.contract import LAYERS
from core.probes import whiten
from core.world import Sensors
from core.world.maze import MazeConfig, MazeWorld
from exp09_maze import RADIUS, visible_mask, wall_probe


def collect(seed: int, ticks: int, warmup: int, side: int = 12):
    """Drive the full cortex through the maze, recording both cluster readouts per tick."""
    world = MazeWorld(MazeConfig(seed=seed))
    sensors = Sensors()
    cx = corvus_cortex.build_cortex(seed=0, cluster_mode="summarise")
    learn_until = int(ticks * 0.8)

    PT, SM, TW, RT, W = [], [], [], [], []
    for t in range(ticks):
        s_now, ret = sensors.observe(world)
        cx.learn = t < learn_until
        corvus_cortex.tick(cx, s_now)
        if t > warmup:
            PT.append(cx.clusters._passthrough.ravel().copy())
            SM.append(cx.clusters.summary.ravel().copy())
            TW.append(cx.towers.publication().ravel().copy())
            RT.append(ret.ravel().copy())
            W.append(world.surrounding_walls(RADIUS))
        world.step()
    return (np.array(PT), np.array(SM), np.array(TW), np.array(RT), np.array(W))


def probe_at(X, W, mask, n_components, split=0.6):
    """Wall decode from `X` reduced to a fixed number of components. Matched capacity is the
    whole point: an unmatched probe measures width, not representation."""
    cut = int(len(X) * split)
    # fit the projection on the training slice only, then apply it to the whole trace
    project = whiten(X, cut, n_components=n_components)
    return wall_probe(project(X), W, mask, split=split)


def run_seed(seed: int, ticks: int, warmup: int) -> dict:
    PT, SM, TW, RT, W = collect(seed, ticks, warmup)
    mask = visible_mask(MazeConfig().view)
    k = SM.shape[1]                       # the summary's own width; everything matches to it

    summary = probe_at(SM, W, mask, k)
    floor = probe_at(PT, W, mask, k)      # the declared control: its own members
    towers = probe_at(TW, W, mask, k)     # the layer below, for context
    retina = probe_at(RT, W, mask, k)     # Mirror, at the same capacity

    # Validity is a question about the *probe*, not about capacity: can pixels read the walls
    # that are actually on screen? That is asked of the full retina, as `gate_maze` asks it.
    # Asking it of a 64-component retina would let a capacity choice invalidate the run.
    full = probe_at(RT, W, mask, None)

    return {"seed": seed, "n_components": int(k), "n_frames": int(len(SM)),
            "summary_hidden": summary["hidden"], "passthrough_hidden": floor["hidden"],
            "towers_hidden": towers["hidden"], "retina_hidden": retina["hidden"],
            "retina_visible_full": full["visible"], "retina_hidden_full": full["hidden"],
            "headline": summary["hidden"] - floor["hidden"],
            "valid": bool(full["visible"] is not None and full["visible"] > 0.8)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=12000)
    ap.add_argument("--warmup", type=int, default=2500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    fl = LAYERS["cluster"].floor
    print(f"CGE-B-03 -- Corvus Layer 2, {args.ticks} ticks, seeds {args.seeds}")
    print(f"declared floor: beat {fl.beats} by {fl.margin:+.2f}")
    print("target: out-of-view wall decode, matched capacity\n")

    rows = [run_seed(s, args.ticks, args.warmup) for s in args.seeds]
    d = np.array([r["headline"] for r in rows])

    print(f"  {'seed':>4}  {'summary':>8} {'pass-thru':>10} {'towers':>8} {'retina':>8}"
          f" {'delta':>8}  valid")
    for r in rows:
        print(f"  {r['seed']:>4}  {r['summary_hidden']:>8.3f} {r['passthrough_hidden']:>10.3f}"
              f" {r['towers_hidden']:>8.3f} {r['retina_hidden']:>8.3f}"
              f" {r['headline']:>+8.3f}  {r['valid']}")

    mean, std = float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0
    print(f"\n  summary vs its own members: {mean:+.3f} +/- {std:.3f}"
          f"   (worst seed {d.min():+.3f})")

    if not all(r["valid"] for r in rows):
        verdict, why = "UNMEASURED", "the retina control cannot read visible walls"
    elif d.min() > fl.margin:
        verdict, why = "PASS", "compression retains more than the members it replaced"
    else:
        verdict, why = "FAIL", ("compression does not beat passing members through; "
                                "pass_through remains the compliant mode")
    print(f"  => {verdict}: {why}")

    out = {"gate": "CGE-B-03", "layer": "cluster", "ticks": args.ticks,
           "floor": {"beats": fl.beats, "margin": fl.margin},
           "verdict": verdict, "reason": why,
           "mean": mean, "std": std, "per_seed": rows}
    p = Path(__file__).resolve().parents[1] / "logs" / "corvus_l2.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote logs/{p.name}")


if __name__ == "__main__":
    main()
