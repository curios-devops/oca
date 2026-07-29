"""The race: RPDU v1, RPDU v2 and DCN v3 solving the same maze, side by side.

Every architecture gets the identical maze from the identical seed, the identical number of
training ticks, the identical probe and the identical planner. The only thing that differs
is the model supplying the wall map — which is exactly the division of labour the maze was
built to expose:

    the architecture supplies the map, including the cells the agent cannot see;
    a nine-line breadth-first planner turns that map into a move.

So reaching the exit more often means one thing only: a better estimate of what is out of
view. That is the same quantity the wall-decode gate measures, put in a form a person can
watch — and, honestly, the reason a benchmark ever gets looked at twice.

    python experiments/maze_race.py --steps 900
    python experiments/make_maze_demo.py && make serve

The score is exits reached in a fixed number of steps. It is a downstream task rather than
a probe, so it is the noisier measurement of the two; the scorecard is what settles
anything, and this is what makes the settling watchable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from bench.registry import get
from core.world import Sensors
from core.world.maze import MazeConfig, MazeWorld
from exp09_maze import RADIUS, _plan, fit_probe, visible_mask, wall_probe

ENTRANTS = (("raw", "Raw pixels"), ("v1", "RPDU v1"), ("v2", "RPDU v2"), ("dcn", "DCN v3"))
"""Raw pixels races too, rather than sitting in a caption as the control.

It has no memory and no parameters: at every step it decodes the wall map from the frame in
front of it and forgets. Any architecture that cannot beat it around this maze is not
remembering anything the image did not already contain — and putting it in the same grid,
under the same clock, is a much harder thing to look past than a number in a legend.
"""


def train(version, seed: int, side: int, ticks: int, warmup: int):
    """Learn the maze, then fit the wall probe on what the model ended up holding."""
    world = MazeWorld(MazeConfig(seed=seed))
    sensors = Sensors()
    state = version.new(seed=0, side=side)
    learn_until = int(ticks * 0.8)

    H, R, W = [], [], []
    for t in range(ticks):
        s_now, ret = sensors.observe(world)
        state.learn = t < learn_until
        version.tick(state, s_now)
        if t > warmup:
            H.append(version.readout(state).ravel().copy())
            R.append(ret.ravel().copy())
            W.append(world.surrounding_walls(RADIUS))
        world.step()

    H, R, W = np.array(H), np.array(R), np.array(W)
    mask = visible_mask(MazeConfig().view)
    return {
        "state": state,
        "probe": fit_probe(H, W),
        "wall_decode": wall_probe(H, W, mask),
        "retina_decode": wall_probe(R, W, mask),
    }


def race(version, state, probe_w, seed: int, steps: int, warmup: int):
    """Run the agent for `steps` moves, planning over the map the model decodes.

    The warm-up moves under the maze's own exploration policy so every entrant enters the
    scored stretch with a settled state; scoring starts only afterwards, and the trace that
    the visualisation replays is exactly the scored stretch.
    """
    world = MazeWorld(MazeConfig(seed=seed))
    sensors = Sensors()
    n = world.cfg.size
    belief = np.full((n, n), 0.5)
    trace, reached, hits = [], 0, []

    for t in range(warmup + steps):
        s_now, _ = sensors.observe(world)
        state.learn = False
        version.tick(state, s_now)

        est = (np.r_[version.readout(state).ravel(), 1.0] @ probe_w).reshape(
            2 * RADIUS + 1, -1)
        r0, c0 = world.pos
        for i, dr in enumerate(range(-RADIUS, RADIUS + 1)):
            for j, dc in enumerate(range(-RADIUS, RADIUS + 1)):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < n and 0 <= c < n:
                    belief[r, c] = 0.7 * belief[r, c] + 0.3 * float(np.clip(est[i, j], 0, 1))

        if t < warmup:
            world.step()
            continue

        action = _plan(belief, world.pos, world.exit_pos)
        trace.append({"pos": world.pos.tolist(),
                      "belief": np.round(belief, 2).tolist()})
        before = world.n_reached
        world.step(action)
        if world.n_reached > before:
            reached += world.n_reached - before
            hits.append(len(trace) - 1)

    return {"trace": trace, "reached": reached, "hits": hits,
            "grid": world.grid.tolist(), "exit": world.exit_pos.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=14000)
    ap.add_argument("--warmup", type=int, default=2500)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--side", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/maze_race.json")
    args = ap.parse_args()

    results, payload = {}, {"entrants": [], "steps": args.steps}
    for key, label in ENTRANTS:
        version = get(key)
        print(f"training {label} ...", flush=True)
        tr = train(version, args.seed, args.side, args.ticks, args.warmup)
        print(f"  wall decode out of view {tr['wall_decode']['hidden']*100:.1f}% "
              f"(retina {tr['retina_decode']['hidden']*100:.1f}%)")
        print(f"  racing {label} ...", flush=True)
        r = race(version, tr["state"], tr["probe"], args.seed, args.steps, 400)
        print(f"  reached the exit {r['reached']} times in {args.steps} steps")

        results[key] = {"label": label,
                        "exits": r["reached"],
                        "hidden_acc": tr["wall_decode"]["hidden"],
                        "visible_acc": tr["wall_decode"]["visible"],
                        "retina_hidden": tr["retina_decode"]["hidden"],
                        "n_params": int(tr["state"].n_params())}
        payload["entrants"].append({"key": key, "label": label,
                                    "exits": r["reached"], "hits": r["hits"],
                                    "trace": r["trace"],
                                    "hidden_acc": tr["wall_decode"]["hidden"],
                                    "retina_hidden": tr["retina_decode"]["hidden"],
                                    "n_params": int(tr["state"].n_params())})
        payload["grid"] = r["grid"]
        payload["exit"] = r["exit"]

    Path(args.out).write_text(json.dumps(payload))
    Path("logs/maze_race_summary.json").write_text(json.dumps(results, indent=2))
    _report(results, args)


def _report(res: dict, args) -> None:
    print(f"\n{'='*64}\nTHE RACE -- same maze, same planner, {args.steps} steps each"
          f"\n{'='*64}\n")
    print(f"  {'model':10s} {'params':>10s} {'out-of-view walls':>19s} {'exits':>7s}")
    for key, r in res.items():
        acc = f"{r['hidden_acc']*100:.1f}%" if r["hidden_acc"] else "n/a"
        print(f"  {r['label']:10s} {r['n_params']:10,d} {acc:>19s} {r['exits']:7d}")

    ctrl = next(iter(res.values()))["retina_hidden"]
    print(f"\n  the raw-pixel control decodes out-of-view walls at {ctrl*100:.1f}%.")
    print("  a model below that line is not remembering anything the image "
          "did not already show.")

    best = max(res.values(), key=lambda r: r["exits"])
    tied = [r["label"] for r in res.values() if r["exits"] == best["exits"]]
    print(f"\n  trophy: {' and '.join(tied)} ({best['exits']} exits)")
    print("\n  exits reached is a downstream task and the noisier of the two numbers.")
    print("  the wall-decode column is what the scorecard settles; this is what makes")
    print("  it watchable.")


if __name__ == "__main__":
    main()
