"""Increment 4 -- does an active sensorimotor loop change what gets represented?

Every earlier world was passive. Here the agent moves, receives an efference copy of its
own action, and sees only a 5x5 window, so most of the maze is out of sight at any
moment. That makes spatial memory *necessary* rather than merely rewarded: you cannot
navigate a maze you cannot remember.

The measurement mirrors the occlusion probe. Decode the walls in a 7x7 neighbourhood from
the model's state, and split the cells by whether they are currently visible:

* the **inner 5x5** is in view, so the raw retina should decode it near-perfectly -- this
  is the control proving the probe works;
* the **outer ring** is *not* in view, so the retina must be near chance there, and any
  model that beats it is holding structure it cannot currently see.

The outer-ring number is the whole experiment. Reaching the exit is not the metric; it is
the visualisation, and it comes afterwards.

    python experiments/exp09_maze.py --ticks 20000
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
from core.world.maze import ACTIONS, MazeConfig, MazeWorld
from architectures.swift import Config2, build_mesh2, tick2

RADIUS = 3


def visible_mask(view: int, radius: int = RADIUS) -> np.ndarray:
    """Which cells of the (2r+1)^2 neighbourhood fall inside the agent's window."""
    h = view // 2
    m = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            m.append(abs(dr) <= h and abs(dc) <= h)
    return np.array(m)


def collect(kind, state, n_ticks, seed, warmup, learn_ticks, trace_from=None,
            trace_len=0):
    world = MazeWorld(MazeConfig(seed=seed))
    sensors = Sensors()

    H, R, W, trace = [], [], [], []
    for t in range(n_ticks):
        s_now, ret = sensors.observe(world)
        if state is not None:
            state.learn = t < learn_ticks
            (tick2 if kind == "v2" else tick)(state, s_now)
        if t > warmup:
            if state is not None:
                H.append(state.h.ravel().copy())
            R.append(ret.ravel().copy())
            W.append(world.surrounding_walls(RADIUS))
        if trace_from is not None and trace_from <= t < trace_from + trace_len:
            trace.append({
                "pos": world.pos.tolist(),
                "action": int(world.last_action),
                "blocked": bool(world.last_blocked),
                "view": world.local_view().tolist(),
            })
        world.step()

    return {
        "state": np.array(H) if H else None,
        "retina": np.array(R),
        "walls": np.array(W),
        "visited": None,
        "grid": world.grid.tolist(),
        "exit": world.exit_pos.tolist(),
        "trace": trace,
        "reached": world.n_reached,
    }


def wall_probe(X, Y, mask, split=0.6, ridge=1e-2):
    """Decode the wall map, scored separately on visible and out-of-view cells."""
    n = len(X)
    cut = int(n * split)
    Xtr = np.c_[X[:cut], np.ones(cut)]
    Xte = np.c_[X[cut:], np.ones(n - cut)]
    A = Xtr.T @ Xtr
    A[np.diag_indices_from(A)] += ridge * np.trace(A) / A.shape[0]
    P = Xte @ np.linalg.solve(A, Xtr.T @ Y[:cut])
    pred = (P > 0.5).astype(float)
    truth = Y[cut:]

    def acc(sel):
        if not sel.any():
            return None
        # balanced across wall/free so a constant guess scores 50%
        p, y = pred[:, sel].ravel(), truth[:, sel].ravel()
        return float(np.mean([(p[y == c] == c).mean() for c in (0.0, 1.0)
                              if (y == c).any()]))

    return {"visible": acc(mask), "hidden": acc(~mask),
            "n_test": int(n - cut), "n_hidden_cells": int((~mask).sum())}


def navigate(kind, state, seed, probe_w, mask, n_steps=600, warmup=1500):
    """Exit-finding driven by the mesh's own decoded wall map.

    Honest about the division of labour: the RPDU supplies the map -- at each step the
    trained probe reads a 7x7 wall estimate out of the mesh state, including the cells
    the agent cannot currently see -- and a two-line planner picks whichever legal move
    most reduces distance to the exit. The agent is given the *direction* of the exit,
    the way an animal following a scent would be; what it is not given is where the walls
    are, which is the part that has to come from memory.
    """
    world = MazeWorld(MazeConfig(seed=seed))
    sensors = Sensors()
    n = world.cfg.size
    belief = np.full((n, n), 0.5)          # accumulated wall belief, 0 free .. 1 wall
    trace, reached = [], 0

    for t in range(warmup + n_steps):
        s_now, ret = sensors.observe(world)
        state.learn = t < warmup
        (tick2 if kind == "v2" else tick)(state, s_now)

        est = (np.r_[state.h.ravel(), 1.0] @ probe_w).reshape(2 * RADIUS + 1, -1)
        r0, c0 = world.pos
        for i, dr in enumerate(range(-RADIUS, RADIUS + 1)):
            for j, dc in enumerate(range(-RADIUS, RADIUS + 1)):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < n and 0 <= c < n:
                    # accumulate rather than overwrite: a single glance is noisy, and
                    # stitching successive estimates is what turns local wall beliefs
                    # into a map worth planning over
                    belief[r, c] = 0.7 * belief[r, c] + 0.3 * float(np.clip(est[i, j], 0, 1))

        if t < warmup:
            world.step()
            continue

        action = _plan(belief, world.pos, world.exit_pos)
        before = world.n_reached
        trace.append({"pos": world.pos.tolist(), "action": int(action),
                      "view": world.local_view().tolist(),
                      "belief": np.round(belief, 2).tolist()})
        world.step(action)
        reached += world.n_reached - before

    return {"trace": trace, "reached": reached, "grid": world.grid.tolist(),
            "exit": world.exit_pos.tolist()}


def _plan(belief, pos, goal, wall_thresh=0.55):
    """Breadth-first search to the exit over cells the mesh believes are free.

    Greedy distance-reduction was tried first and reached the exit zero times in 800
    steps -- worse than the random walk -- because it walks into concave dead ends and
    oscillates. The map is the RPDU's contribution; the planner just has to be good
    enough not to waste it.
    """
    n = belief.shape[0]
    start, goal = tuple(pos), tuple(goal)
    prev = {start: None}
    queue = [start]
    while queue:
        cur = queue.pop(0)
        if cur == goal:
            break
        for a, (dr, dc) in enumerate(ACTIONS):
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or nxt in prev:
                continue
            if belief[nxt] > wall_thresh and nxt != goal:
                continue
            prev[nxt] = cur
            queue.append(nxt)

    if goal not in prev:                     # no believed route: fall back to exploring
        free = [a for a, (dr, dc) in enumerate(ACTIONS)
                if 0 <= pos[0] + dr < n and 0 <= pos[1] + dc < n
                and belief[pos[0] + dr, pos[1] + dc] <= wall_thresh]
        return int(free[0]) if free else 0

    step = goal
    while prev[step] != start and prev[step] is not None:
        step = prev[step]
    d = (step[0] - start[0], step[1] - start[1])
    return int(ACTIONS.index(d)) if d in ACTIONS else 0


def fit_probe(X, Y, split=0.6, ridge=1e-2):
    cut = int(len(X) * split)
    Xtr = np.c_[X[:cut], np.ones(cut)]
    A = Xtr.T @ Xtr
    A[np.diag_indices_from(A)] += ridge * np.trace(A) / A.shape[0]
    return np.linalg.solve(A, Xtr.T @ Y[:cut])


def tunnel_gate(kind, state, ticks, seed, warmup, learn_ticks):
    """Can the mesh say where it is while it cannot see anything?

    Inside a covered corridor every frame is byte-identical, so pixels carry exactly zero
    positional information and the raw-retina control is at chance by construction -- which
    is the point. The open maze could not make that claim: there a 5x5 view largely
    identifies where you are, so its control sat at 77% and the headroom was small.

    What is left inside a tunnel is dead reckoning. Three references bracket the result:

    * **frozen at entry** -- remember only the mouth you walked into. This is the bar. A
      model that merely stores "I went in at (r,c)" scores exactly this, so beating it is
      the definition of integrating your own moves.
    * **raw retina** -- chance, by construction.
    * **perfect dead-reckoner** -- zero error, the ceiling.
    """
    world = MazeWorld(MazeConfig(seed=seed, tunnels=True))
    sensors = Sensors()
    rows = []

    for t in range(ticks):
        s_now, ret = sensors.observe(world)
        state.learn = t < learn_ticks
        (tick2 if kind == "v2" else tick)(state, s_now)
        if t > warmup and world.in_tunnel():
            rows.append({
                "h": state.h.ravel().copy(),
                "r": ret.ravel().copy(),
                "pos": world.pos.astype(float).copy(),
                "entry": np.array(world._tunnel_entry, dtype=float),
                "steps": world.steps_in_tunnel(),
            })
        world.step()

    if len(rows) < 200:
        return {"error": f"only {len(rows)} in-tunnel frames"}

    Y = np.stack([r["pos"] for r in rows])
    entry = np.stack([r["entry"] for r in rows])
    steps = np.array([r["steps"] for r in rows])
    cut = int(len(rows) * 0.6)

    def probe(key):
        return decode_error(np.stack([r[key] for r in rows]), Y)

    err_model, err_retina = probe("h"), probe("r")
    err_frozen = np.linalg.norm(entry[cut:] - Y[cut:], axis=1)
    err_chance = np.linalg.norm(Y[:cut].mean(0) - Y[cut:], axis=1)
    te_steps = steps[cut:]

    buckets = {}
    for lo, hi in ((1, 2), (3, 4), (5, 8), (9, 99)):
        sel = (te_steps >= lo) & (te_steps <= hi)
        if sel.sum() >= 20:
            buckets[f"{lo}-{hi if hi < 99 else '+'}"] = {
                "n": int(sel.sum()),
                "model": float(err_model[sel].mean()),
                "frozen_at_entry": float(err_frozen[sel].mean()),
                "retina": float(err_retina[sel].mean()),
            }

    return {
        "n_frames": len(rows),
        "model": float(err_model.mean()),
        "frozen_at_entry": float(err_frozen.mean()),
        "retina": float(err_retina.mean()),
        "chance": float(err_chance.mean()),
        "beats_frozen_by": float(1 - err_model.mean() / err_frozen.mean()),
        "buckets": buckets,
    }


def _report_tunnel(res: dict) -> None:
    print("\nposition error while blind, in maze cells (lower is better)\n")
    print(f"{'model':10s} {'mesh':>8s} {'frozen@entry':>13s} {'retina':>8s} "
          f"{'chance':>8s} {'vs frozen':>10s}")
    for name, r in res.items():
        if "error" in r:
            print(f"{name:10s} {r['error']}")
            continue
        print(f"{name:10s} {r['model']:8.2f} {r['frozen_at_entry']:13.2f} "
              f"{r['retina']:8.2f} {r['chance']:8.2f} {r['beats_frozen_by']*100:+9.1f}%")

    first = next((r for r in res.values() if "error" not in r), None)
    if not first:
        return
    print("\ncontrol: the retina should sit at chance -- inside a tunnel every frame is")
    print(f"identical, so pixels cannot encode position. retina {first['retina']:.2f} "
          f"vs chance {first['chance']:.2f}  "
          + ("OK" if abs(first["retina"] - first["chance"]) < 0.15 * first["chance"]
             else "LEAK: pixels are carrying position after all"))

    print("\nerror against how long the agent has been blind:")
    for name, r in res.items():
        if "error" in r or not r.get("buckets"):
            continue
        print(f"  {name}")
        for k, b in r["buckets"].items():
            better = b["frozen_at_entry"] - b["model"]
            print(f"    {k:>5s} steps (n={b['n']:4d})  mesh {b['model']:5.2f}  "
                  f"frozen {b['frozen_at_entry']:5.2f}  -> {better:+.2f}")

    print("\nreading:")
    for name, r in res.items():
        if "error" in r:
            continue
        g = r["beats_frozen_by"]
        print(f"  {name}: " + ("integrates its own moves while blind" if g > 0.10
                               else "no better than remembering the tunnel mouth"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--learn-ticks", type=int, default=15000)
    ap.add_argument("--side", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", type=int, default=400)
    ap.add_argument("--demo", type=int, default=600,
                    help="steps of mesh-guided exit seeking for the visualisation")
    ap.add_argument("--tunnels", action="store_true",
                    help="run the covered-corridor path-integration gate instead")
    ap.add_argument("--out", default="logs/exp09_maze.jsonl")
    args = ap.parse_args()

    mask = visible_mask(MazeConfig().view)
    models = {
        "v1_mesh": ("v1", build_mesh(Config(lattice_side=args.side, seed=0, eta_head=0.01))),
        "v2_mesh": ("v2", build_mesh2(Config2(lattice_side=args.side, seed=0, eta_head=0.01))),
    }

    if args.tunnels:
        res = {}
        with JsonlLogger(args.out, meta=vars(args)) as log:
            for name, (kind, state) in models.items():
                print(f"running {name} in the tunnel maze ...")
                res[name] = tunnel_gate(kind, state, args.ticks, args.seed,
                                        args.warmup, args.learn_ticks)
                log.log(kind="tunnel", mesh=name, **{k: v for k, v in res[name].items() if k != "buckets"})
        Path("logs/exp09_tunnel_summary.json").write_text(json.dumps(res, indent=2))
        _report_tunnel(res)
        return

    results, trace_payload = {}, None
    with JsonlLogger(args.out, meta=vars(args)) as log:
        for name, (kind, state) in models.items():
            print(f"running {name} in the maze ...")
            data = collect(kind, state, args.ticks, args.seed, args.warmup,
                           args.learn_ticks,
                           trace_from=args.ticks - args.trace - 1,
                           trace_len=args.trace)
            res = {"model_state": wall_probe(data["state"], data["walls"], mask),
                   "raw_retina": wall_probe(data["retina"], data["walls"], mask),
                   "exit_reached": data["reached"]}
            results[name] = res
            log.log(kind="maze", model=name, **res)
            if name == "v1_mesh" and args.demo:
                print("  running mesh-guided exit seeking ...")
                pw = fit_probe(data["state"], data["walls"])
                demo = navigate(kind, state, args.seed, pw, mask, n_steps=args.demo)
                res["demo_exits_reached"] = demo["reached"]
                trace_payload = {"grid": demo["grid"], "exit": demo["exit"],
                                 "trace": demo["trace"], "model": name}
                print(f"  reached the exit {demo['reached']} times "
                      f"in {args.demo} guided steps")

    Path("logs/exp09_summary.json").write_text(json.dumps(results, indent=2))
    if trace_payload:
        Path("logs/maze_trace.json").write_text(json.dumps(trace_payload))
    _report(results, mask)


def _report(res: dict, mask) -> None:
    print(f"\ndecoding the 7x7 wall map (balanced accuracy, chance = 50%)")
    print(f"inner 5x5 = {int(mask.sum())} cells in view, "
          f"outer ring = {int((~mask).sum())} cells NOT in view\n")
    print(f"{'model':10s} {'source':14s} {'visible':>9s} {'OUT OF VIEW':>13s}")
    for name, r in res.items():
        for src in ("raw_retina", "model_state"):
            v = r[src]
            vis = f"{v['visible']*100:.1f}%" if v["visible"] else "n/a"
            hid = f"{v['hidden']*100:.1f}%" if v["hidden"] else "n/a"
            print(f"{name:10s} {src:14s} {vis:>9s} {hid:>13s}")

    ctrl = next(iter(res.values()))["raw_retina"]
    print("\ncontrols:")
    print(f"  retina on visible cells {ctrl['visible']*100:.1f}%  "
          + ("probe works" if ctrl["visible"] > 0.8 else "PROBE BROKEN"))
    print(f"  retina on hidden cells  {ctrl['hidden']*100:.1f}%  "
          + ("out-of-view cells are genuinely not in the image"
             if ctrl["hidden"] < 0.65 else "LEAK: hidden cells are visible after all"))

    print("\nreading:")
    for name, r in res.items():
        m, c = r["model_state"]["hidden"], r["raw_retina"]["hidden"]
        gain = (m - c) * 100
        print(f"  {name}: {m*100:5.1f}% on out-of-view cells vs retina {c*100:5.1f}% "
              f"({gain:+.1f} pts) -> "
              + ("holds a map of what it cannot see" if gain > 5
                 else "no spatial memory beyond the current view"))


if __name__ == "__main__":
    main()
