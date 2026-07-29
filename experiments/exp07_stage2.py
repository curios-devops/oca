"""Increment-1 gate: does architecture v2 fix what the diagnostics identified?

Runs the v1 and v2 meshes over the *same* world-v2 stream and asks three questions,
each tied to a specific v1 failure:

1. **Does the state hold an object at all?**  Ridge probe for object position, scored
   while visible and while fully occluded, always against the raw retina as a control.
   On v1 this was the decisive diagnostic: the mesh state decoded visible position worse
   than the retina it had just been fed, and carried nothing extra during occlusion.
2. **Do coalitions form, and do they mean anything?**  v1 found 1.7% of units in
   coalitions because gradient-flow units cannot synchronise. v2 units are oscillators,
   so formation is expected -- the real question is the second half, measured as mutual
   information between coalition labels and which object a unit is actually looking at.
   Coalitions that form but carry no object information would be a negative result.
3. **Is it still a working predictor?**  A model that hallucinates structure while
   getting worse at prediction has not improved.

    python experiments/exp07_stage2.py --ticks 12000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legacy.v1.mesh import build_mesh, predicted_retina, tick
from core.metrics import JsonlLogger, coalition_stats, frame_mse
from legacy.v1.state import Config
from core.world import Sensors
from core.world.physics import PhysicsConfig, PhysicsWorld
from core.world.sensors import N_VISUAL, PATCH, RETINA
from legacy.v2 import Config2, build_mesh2, predicted_retina2, tick2

from exp06_probe import ridge_probe


def unit_object_labels(world, n_objects, frame_shape=(RETINA, RETINA), reach=6.0):
    """Which object (if any) each visual unit is currently looking at.

    A visual unit owns a PATCH x PATCH tile of the retina and is assigned to the nearest
    visible object whose centre lies within `reach` retina pixels of its tile centre.
    Assigning only the single tile containing an object leaves 2-3 labelled units per
    frame, and mutual information over 2 points is noise dressed as a result; a radius
    gives each object a neighbourhood of tiles. Units seeing nothing get -1 and are
    excluded, since "we both see empty space" is not a coalition.
    """
    g = RETINA // PATCH
    labels = np.full(N_VISUAL, -1)
    scale = frame_shape[0] / world.cfg.size
    centres = np.array([[(c + 0.5) * PATCH, (r + 0.5) * PATCH]
                        for r in range(g) for c in range(g)])
    best = np.full(N_VISUAL, np.inf)
    for i in range(n_objects):
        if world.is_fully_occluded(i):
            continue
        p = world.pos[i] * scale
        dist = np.linalg.norm(centres - p, axis=1)
        take = (dist < reach) & (dist < best)
        labels[take] = i
        best[take] = dist[take]
    return labels


def normalised_mi(a, b):
    """Normalised mutual information between two label arrays."""
    a, b = np.asarray(a), np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    if ua.size < 2 or ub.size < 2:
        return 0.0
    joint = np.zeros((ua.size, ub.size))
    np.add.at(joint, (ia, ib), 1.0)
    joint /= joint.sum()
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    mi = float((joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum())

    def ent(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    denom = 0.5 * (ent(pa.ravel()) + ent(pb.ravel()))
    return mi / denom if denom > 0 else 0.0


def _binding_stats(mi, null):
    """Per-coalition-definition: excess MI over a paired shuffled null, with a SEM.

    Paired because the null depends on the label partition and the object layout of that
    same snapshot; comparing pooled means would throw that pairing away.
    """
    out = {}
    for key in mi:
        d = np.array(mi[key]) - np.array(null[key])
        out[key] = {
            "mi": float(np.mean(mi[key])),
            "null": float(np.mean(null[key])),
            "excess": float(d.mean()),
            "sem": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
            "n": len(d),
        }
    return out


def drive(kind, state, n_ticks, seed, n_objects, warmup, learn_ticks):
    """Run one model over the world, collecting everything both questions need."""
    world = PhysicsWorld(PhysicsConfig(seed=seed, n_objects=n_objects))
    sensors = Sensors()
    for _ in range(30):
        world.step()

    H, R, POS, OCC = [], [], [], []
    coal_rows, mi_rows, null_rows = [], {}, {}
    reemerge = {1: [], 4: [], 16: []}
    prev_labels = None
    frames, preds = [], []
    hidden = {}

    for t in range(n_ticks):
        s_now, ret = sensors.observe(world)
        if state is not None:
            state.learn = t < learn_ticks
            if kind == "v2":
                tick2(state, s_now)
                pr = {tau: predicted_retina2(state, tau, sensors)
                      for tau in state.cfg.horizons}
            else:
                tick(state, s_now)
                pr = {tau: predicted_retina(state, tau, sensors)
                      for tau in state.cfg.horizons}
        else:
            pr = None

        frames.append(ret)
        preds.append(pr)

        if t > warmup:
            if state is not None:
                H.append(state.h.ravel().copy())
            R.append(ret.ravel().copy())
            POS.append(world.pos.ravel().copy())
            OCC.append([world.is_fully_occluded(i) for i in range(n_objects)])

            if state is not None and t % 50 == 0:
                cs = coalition_stats(state.coalition, prev_labels)
                prev_labels = state.coalition.copy()
                coal_rows.append(cs)
                obj = unit_object_labels(world, n_objects)
                seen = obj >= 0
                if seen.sum() >= 6 and len(set(obj[seen].tolist())) >= 2:
                    rs = np.random.default_rng(t)
                    for key, labels in (("phase", state.coalition),
                                        ("vote", getattr(state, "vote_coalition", None))):
                        if labels is None:
                            continue
                        lab = labels[:N_VISUAL][seen]
                        mi = normalised_mi(lab, obj[seen])
                        # normalised MI is biased upward on small samples, so the only
                        # interpretable number is the excess over a shuffled null,
                        # paired snapshot by snapshot
                        null = np.mean([normalised_mi(rs.permutation(lab), obj[seen])
                                        for _ in range(20)])
                        mi_rows.setdefault(key, []).append(mi)
                        null_rows.setdefault(key, []).append(null)

        # re-emergence scoring: prediction issued while the object was invisible
        for i in range(n_objects):
            occ = world.is_fully_occluded(i)
            if occ and i not in hidden:
                hidden[i] = t
            elif not occ and i in hidden:
                t0 = hidden.pop(i)
                if t - t0 >= 8 and preds[0] is not None:
                    for f in range(t + 2, min(t + 9, n_ticks)):
                        pass  # scored in the second pass below, once frames exist
                    reemerge.setdefault("_events", []).append((t0, t, i))
        world.step()

    # second pass for re-emergence, now that all frames and predictions exist
    events = reemerge.pop("_events", [])
    scored = {tau: [] for tau in (1, 4, 16)}
    if preds[0] is not None:
        for (t0, te, i) in events:
            for tau in (1, 4, 16):
                for f in range(te + 2, min(te + 9, len(frames))):
                    src_t = f - tau
                    if not (t0 <= src_t < te) or preds[src_t] is None:
                        continue
                    p = preds[src_t][tau]
                    scored[tau].append(frame_mse(p, frames[f]))

    return {
        "state": np.array(H) if H else None,
        "retina": np.array(R),
        "pos": np.array(POS),
        "occ": np.array(OCC),
        "coalitions": coal_rows,
        "mi": mi_rows,
        "mi_null": null_rows,
        "reemerge": {tau: (float(np.mean(v)) if v else None) for tau, v in scored.items()},
        "n_events": len(events),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=12000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--learn-ticks", type=int, default=9000)
    ap.add_argument("--n-objects", type=int, default=1)
    ap.add_argument("--side", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="logs/exp07_stage2.jsonl")
    args = ap.parse_args()

    models = {
        "v1_mesh": ("v1", build_mesh(Config(lattice_side=args.side, seed=0,
                                            eta_head=0.01))),
        "v2_mesh": ("v2", build_mesh2(Config2(lattice_side=args.side, seed=0,
                                              eta_head=0.01))),
    }

    results = {}
    with JsonlLogger(args.out, meta=vars(args)) as log:
        for name, (kind, state) in models.items():
            print(f"running {name} for {args.ticks} ticks ...")
            data = drive(kind, state, args.ticks, args.seed, args.n_objects,
                         args.warmup, args.learn_ticks)
            probe = {
                "model_state": ridge_probe(data["state"], data["pos"], data["occ"][:, 0]),
                "raw_retina": ridge_probe(data["retina"], data["pos"], data["occ"][:, 0]),
            }
            coal = {k: float(np.mean([c[k] for c in data["coalitions"] if k in c]))
                    for k in ("n_coalitions", "frac_in_coalition", "largest", "churn")
                    if any(k in c for c in data["coalitions"])}
            if not data["coalitions"]:
                coal = {}
            results[name] = {
                "probe": probe,
                "coalitions": coal,
                "binding": _binding_stats(data["mi"], data["mi_null"]),
                "reemerge": data["reemerge"],
                "n_events": data["n_events"],
            }
            log.log(kind="stage2", model=name, **results[name])

    Path("logs/exp07_summary.json").write_text(json.dumps(results, indent=2))
    _report(results)


def _report(res: dict) -> None:
    print(f"\n{'':10s} {'position decode (world units)':44s}")
    print(f"{'model':10s} {'state:visible':>14s} {'state:occluded':>15s} "
          f"{'retina:occluded':>16s} {'chance':>8s}")
    for name, r in res.items():
        p, c = r["probe"]["model_state"], r["probe"]["raw_retina"]
        print(f"{name:10s} {p['visible']:14.2f} {p['occluded']:15.2f} "
              f"{c['occluded']:16.2f} {p['chance']:8.2f}")

    print(f"\n{'model':10s} {'coalitions':>12s} {'largest':>9s} {'bindMI':>8s} "
          f"{'reemerge t4':>12s} {'t16':>9s}")
    for name, r in res.items():
        c = r["coalitions"]
        rm = r["reemerge"]
        best = max((b["excess"] for b in r.get("binding", {}).values()), default=0.0)
        print(f"{name:10s} {c.get('frac_in_coalition', 0)*100:11.1f}% "
              f"{c.get('largest', 0):9.0f} {best:8.3f} "
              f"{rm[4]:12.5f} {rm[16]:9.5f}")

    v1, v2 = res.get("v1_mesh"), res.get("v2_mesh")
    if not (v1 and v2):
        return
    print("\nreading:")
    for name, r in (("v1", v1), ("v2", v2)):
        p, c = r["probe"]["model_state"], r["probe"]["raw_retina"]
        gain = 1 - p["occluded"] / c["occluded"]
        print(f"  {name}: state beats the retina during occlusion by {gain*100:+5.1f}%"
              + ("  <- tracks what it cannot see" if gain > 0.15 else
                 "  <- carries nothing extra"))
    print(f"  coalitions: v1 {v1['coalitions'].get('frac_in_coalition',0)*100:.1f}% of "
          f"units, v2 {v2['coalitions'].get('frac_in_coalition',0)*100:.1f}%")
    print("\n  coalition/object binding (MI excess over a paired shuffled null):")
    for name, r in (("v1", v1), ("v2", v2)):
        for key, b in r.get("binding", {}).items():
            sem = b["sem"] or 0.0
            sig = abs(b["excess"]) > 2 * sem and sem > 0
            print(f"    {name} {key:6s} excess {b['excess']:+.3f} +/- {sem:.3f} "
                  f"(n={b['n']})" + ("  <- tracks objects" if sig and b["excess"] > 0
                                     else "  <- not distinguishable from chance"))


if __name__ == "__main__":
    main()
