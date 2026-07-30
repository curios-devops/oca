"""Jay Layer 1 — its declared floors, in the pose world.

The question, in the first setting this project has had where the raw input cannot simply
answer it: **does a representation identify an object at an orientation it was never shown?**

Fitting happens on trained poses only. Scoring happens on the held-out pose only. The raw
control is measured at **0.158** there, against a chance of 0.250 — memorised appearances do not
merely fail to transfer, they mislead.

Floors, declared in `architectures/jay/tower.py` before this file existed:

    object identity   beats  nearest_template_on_held_out_poses   (0.158)
    object identity   beats  raw_frame                            (CGE-A-00)

Controls, because a designed readout needs them:

    use_binding=False     same features, same capacity, no place. If this ties, binding is
                          decoration and the layer is retired.
    shuffle_locations     same features bound to permuted places. Destroys arrangement,
                          keeps everything else.
    features_only         distances over every visited place regardless of feature. Tests
                          whether the binding matters or only where the sensor went.

    python experiments/jay_l1.py --ticks 60000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from architectures.jay import cortex as jay_cortex
from architectures.jay.contract import LAYERS
from core.world import Sensors
from core.world.pose import SHAPES, PoseConfig, PoseWorld
from validate_pose_world import balanced_accuracy, decode, nearest_template

CHANCE = 1.0 / len(SHAPES)

VARIANTS = {
    "binding": {},
    "no_binding": {"use_binding": False},
    "shuffled_locations": {"shuffle_locations": True},
    "features_only": {"features_only": True},
}


def collect(seed: int, ticks: int, **tower_kw) -> dict:
    """Drive Jay through the pose world, recording the readout at the END of each episode.

    **One row per presentation, not per tick.** The tower accumulates a feature-at-location map
    as the fovea moves; asking it to identify an object halfway through a sweep is asking it
    about a fragment it has not finished collecting. The gate reads the model the layer built.
    """
    world = PoseWorld(PoseConfig(seed=seed))
    sensors = Sensors()
    cx = jay_cortex.build_cortex(seed=0, **tower_kw)
    cx.new_episode()

    rows, frames = [], []
    prev_ep = world.episode_t
    for _ in range(ticks):
        sen, ret = sensors.observe(world)
        jay_cortex.tick(cx, sen)
        frames.append(ret.ravel().copy())
        kind, pose, held = world.kind, world.pose, world.is_held_out()
        world.step()
        if world.episode_t < prev_ep:                 # the episode just rolled over
            rows.append({"h": cx.h.ravel().copy(), "r": np.mean(frames, axis=0),
                         "kind": kind, "pose": pose, "held_out": held})
            frames = []
            cx.new_episode()
        prev_ep = world.episode_t
    return rows


def evaluate(rows: list[dict], n_components: int = 64) -> dict:
    H = np.stack([r["h"] for r in rows])
    R = np.stack([r["r"] for r in rows])
    y = np.array([r["kind"] for r in rows])
    ho = np.array([r["held_out"] for r in rows])
    tr, te = np.flatnonzero(~ho), np.flatnonzero(ho)
    if len(tr) < 120 or len(te) < 40:
        return {"valid": False, "reason": f"{len(tr)} trained / {len(te)} held-out episodes"}

    cut = int(len(tr) * 0.75)
    fit, val = tr[:cut], tr[cut:]
    out = {"valid": True, "n_trained": int(len(tr)), "n_heldout": int(len(te)),
           "n_episodes": len(rows)}

    out["model_trained"] = decode(H[fit], y[fit], H[val], y[val], n_components=n_components)
    out["model_heldout"] = decode(H[fit], y[fit], H[te], y[te], n_components=n_components)
    out["raw_trained"] = decode(R[fit], y[fit], R[val], y[val], n_components=n_components)
    out["raw_heldout"] = decode(R[fit], y[fit], R[te], y[te], n_components=n_components)
    out["template_trained"] = nearest_template(R[fit], y[fit], R[val], y[val])
    out["template_heldout"] = nearest_template(R[fit], y[fit], R[te], y[te])
    # the raw control is the best cheap use of the input, chosen on TRAINED poses
    best = "template" if out["template_trained"] >= out["raw_trained"] else "raw"
    out["raw_control"] = best
    out["control_heldout"] = out[f"{best}_heldout"]
    return out


def invariance(seed: int = 0, ticks_per_sweep: int = 48) -> dict:
    """**The diagnostic that decides the layer, and it needs no probe at all.**

    A pose-invariant object representation must make the *same* object at two orientations look
    more alike than *two* objects at the same orientation. Comparing readouts directly answers
    that in seconds, where an accuracy number takes an hour and confounds the question with how
    well a ridge probe happens to fit.

    Every sweep uses an identical fovea path from an identical start, so the only thing that
    differs between two readouts is what was on the canvas.
    """
    def sweep(kind: int, pose: int) -> np.ndarray:
        w = PoseWorld(PoseConfig(seed=seed))
        sensors, cx = Sensors(), jay_cortex.build_cortex(seed=0)
        w.kind, w.pose = kind, pose
        w._canvas, w.fovea = w._draw(), np.array([16.0, 16.0])
        cx.new_episode()
        for a in ([3] * 6 + [1] * 6 + [2] * 6 + [0] * 6) * (ticks_per_sweep // 24):
            sen, _ = sensors.observe(w)
            jay_cortex.tick(cx, sen)
            w.step(action=a)
        return cx.h.ravel()

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    n_p = PoseConfig().n_poses
    same_obj = [cos(sweep(k, 0), sweep(k, p)) for k in range(len(SHAPES))
                for p in range(1, n_p)]
    diff_obj = [cos(sweep(a, p), sweep(b, p)) for p in range(n_p)
                for a in range(len(SHAPES)) for b in range(a + 1, len(SHAPES))]
    return {"same_object_across_poses": float(np.mean(same_obj)),
            "different_objects_same_pose": float(np.mean(diff_obj)),
            "margin": float(np.mean(same_obj) - np.mean(diff_obj))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=60000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    floors = [f for f in LAYERS["tower"].floors if f.job == "object_identity"]
    print(f"Jay L1 — object identity at an unseen orientation. chance {CHANCE:.3f}")
    for f in floors:
        print(f"  declared floor: beat {f.beats} by {f.margin:+.2f}")
    print()

    inv = invariance()
    print(f"  invariance, measured directly on the readout:")
    print(f"    same object across poses      {inv['same_object_across_poses']:.3f}")
    print(f"    different objects, same pose  {inv['different_objects_same_pose']:.3f}")
    print(f"    margin                        {inv['margin']:+.3f}  "
          f"{'pose-invariant' if inv['margin'] > 0.05 else 'NOT POSE-INVARIANT'}")
    print("    (a representation more sensitive to pose than to identity cannot pass "
          "the floors below,\n     however the probe is fitted.)\n")

    results = {"invariance": inv}
    for name, kw in VARIANTS.items():
        per_seed = [evaluate(collect(s, args.ticks, **kw)) for s in args.seeds]
        ok = [r for r in per_seed if r.get("valid")]
        if not ok:
            print(f"  {name:20} UNMEASURED: {per_seed[0].get('reason')}")
            results[name] = {"valid": False}
            continue
        agg = {k: float(np.mean([r[k] for r in ok]))
               for k in ("model_trained", "model_heldout", "raw_trained", "raw_heldout",
                         "template_trained", "template_heldout", "control_heldout")}
        agg["model_heldout_min"] = float(min(r["model_heldout"] for r in ok))
        agg["per_seed"] = [r["model_heldout"] for r in ok]
        results[name] = agg

    print(f"  {'variant':20} {'trained':>8} {'HELD-OUT':>9}   per seed")
    for name, r in results.items():
        if not isinstance(r, dict) or not r.get("model_heldout"):
            continue
        per = "  ".join(f"{v:.3f}" for v in r["per_seed"])
        print(f"  {name:20} {r['model_trained']:>8.3f} {r['model_heldout']:>9.3f}   {per}")

    b = results.get("binding", {})
    if b.get("model_heldout"):
        print(f"\n  {'raw frame':20} {b['raw_trained']:>8.3f} {b['raw_heldout']:>9.3f}   <- control")
        print(f"  {'nearest-template':20} {b['template_trained']:>8.3f} "
              f"{b['template_heldout']:>9.3f}   <- control")
        print(f"  {'chance':20} {'':>8} {CHANCE:>9.3f}")

        # **A floor is not cleared by beating a control that is below chance.**
        #
        # The first run of this gate reported PASS on both floors with the model at 0.254 --
        # chance is 0.250 -- because nearest-template scores 0.057 on held-out poses and the raw
        # frame 0.188. A representation that knows nothing beat controls that are actively
        # misled, and the arithmetic said PASS. That is the CGE-A-01 shape exactly: a comparison
        # technically satisfied and measuring nothing.
        #
        # So the floor has two clauses now, and the first is not optional.
        above_chance = b["model_heldout_min"] > CHANCE + 0.05
        print(f"\n  above chance?  {b['model_heldout_min']:.3f} vs {CHANCE:.3f} + 0.05  "
              f"=> {'YES' if above_chance else 'NO -- every floor below is void'}")
        print("\n  floors, on held-out poses:")
        verdicts = {}
        for f in floors:
            ctl = b["template_heldout"] if "template" in f.beats else b["raw_heldout"]
            d = b["model_heldout_min"] - ctl
            ok = bool(above_chance and d > f.margin)
            verdicts[f.beats] = {"delta": d, "above_chance": bool(above_chance), "passed": ok}
            note = "" if above_chance else "   (VOID: model at chance)"
            print(f"    vs {f.beats:38} {d:+.3f}  {'PASS' if ok else 'FAIL'}{note}")

        print("\n  ablation and controls, on held-out poses:")
        for name in ("no_binding", "shuffled_locations", "features_only"):
            r = results.get(name, {})
            if not r.get("model_heldout"):
                continue
            d = b["model_heldout"] - r["model_heldout"]
            print(f"    binding vs {name:24} {d:+.3f}  "
                  f"{'binding earns it' if d > 0.05 else 'BINDING IS DECORATION'}")
        results["floor_verdicts"] = verdicts

    p = Path(__file__).resolve().parents[1] / "logs" / "jay_l1.json"
    p.write_text(json.dumps({"ticks": args.ticks, "chance": CHANCE, "results": results},
                            indent=1, default=float))
    print(f"\nwrote logs/{p.name}")


if __name__ == "__main__":
    main()
