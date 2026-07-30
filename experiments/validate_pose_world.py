"""The pose world's five validity checks. **Run before any architecture touches it.**

A world is not a contribution until it has been shown to measure something. This project has
three retracted claims from measurements that were never checked this way, and one deprecated
gate that no architecture could ever have passed.

Check 2 is the whole specification: **raw pixels must be at chance on held-out poses.** If they
are not, the poses are too close together, memorising appearances is sufficient, and the world
is void — no result from it counts, however good it looks.

    python experiments/validate_pose_world.py --ticks 40000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.world import Sensors
from core.world.pose import ACTIONS, SHAPES, PoseConfig, PoseWorld

CHANCE = 1.0 / len(SHAPES)


def collect(ticks: int, seed: int) -> dict:
    world = PoseWorld(PoseConfig(seed=seed))
    sensors = Sensors()
    R, K, P, HO, EFF = [], [], [], [], []
    for _ in range(ticks):
        _, ret = sensors.observe(world)
        R.append(ret.ravel().copy())
        K.append(world.kind)
        P.append(world.pose)
        HO.append(world.is_held_out())
        EFF.append(world.efference().copy())
        world.step()
    return {"R": np.array(R), "kind": np.array(K), "pose": np.array(P),
            "held_out": np.array(HO), "eff": np.array(EFF)}


def _ridge_multiclass(Xtr, ytr, Xte, n_classes, ridge=1e-2):
    """One-vs-rest ridge. Returns predicted class for each test row."""
    A = Xtr.T @ Xtr
    A[np.diag_indices_from(A)] += ridge * np.trace(A) / A.shape[0]
    Y = np.full((len(ytr), n_classes), -1.0)
    Y[np.arange(len(ytr)), ytr] = 1.0
    W = np.linalg.solve(A, Xtr.T @ Y)
    return np.argmax(Xte @ W, axis=1)


def balanced_accuracy(pred, true, n_classes) -> float:
    accs = [float((pred[true == c] == c).mean()) for c in range(n_classes)
            if (true == c).any()]
    return float(np.mean(accs)) if accs else float("nan")


def decode(Xtr, ytr, Xte, yte, n_classes=len(SHAPES), n_components=64) -> float:
    """Fit on trained poses, score on whatever is handed in. Capacity matched throughout."""
    mu = Xtr.mean(0)
    _, s, vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    k = min(n_components, len(s))
    comp = vt[:k].T
    scale = np.maximum(s[:k] / np.sqrt(max(len(Xtr) - 1, 1)), 1e-9)
    ztr = np.c_[((Xtr - mu) @ comp) / scale, np.ones(len(Xtr))]
    zte = np.c_[((Xte - mu) @ comp) / scale, np.ones(len(Xte))]
    return balanced_accuracy(_ridge_multiclass(ztr, ytr, zte, n_classes), yte, n_classes)


def nearest_template(Xtr, ytr, Xte, yte, max_templates: int = 3000,
                     chunk: int = 512) -> float:
    """Explicit memorisation: the class of the closest training appearance. What an
    architecture must beat to have generalised rather than stored.

    Chunked, and the template bank is capped. The naive form materialises
    (n_test, n_train, n_pixels) -- at 40,000 ticks that is tens of billions of floats.
    """
    if len(Xtr) > max_templates:
        keep = np.linspace(0, len(Xtr) - 1, max_templates).astype(int)
        Xtr, ytr = Xtr[keep], ytr[keep]
    tn = (Xtr ** 2).sum(1)
    pred = np.empty(len(Xte), dtype=ytr.dtype)
    for i in range(0, len(Xte), chunk):
        blk = Xte[i:i + chunk]
        # ||a-b||^2 = ||a||^2 - 2a.b + ||b||^2; the ||a||^2 term is constant per row
        d = tn[None, :] - 2.0 * (blk @ Xtr.T)
        pred[i:i + chunk] = ytr[np.argmin(d, axis=1)]
    return balanced_accuracy(pred, yte, len(SHAPES))


def bag_of_features(X: np.ndarray, ret_side: int) -> np.ndarray:
    """Total mass and its radial histogram — every blob's appearance, positions discarded.

    Blobs are identical across kinds by construction, so this must be at chance. If it is not,
    the shapes leak identity through something other than arrangement.
    """
    img = X.reshape(len(X), ret_side, ret_side)
    c = (ret_side - 1) / 2.0
    yy, xx = np.mgrid[0:ret_side, 0:ret_side]
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2).ravel()
    bins = np.linspace(0, r.max() + 1e-6, 9)
    which = np.digitize(r, bins) - 1
    feats = [X.sum(1, keepdims=True)]
    for b in range(len(bins) - 1):
        feats.append(X[:, which == b].sum(1, keepdims=True))
    return np.concatenate(feats, axis=1)


def run(seed: int, ticks: int) -> dict:
    d = collect(ticks, seed)
    R, kind, ho = d["R"], d["kind"], d["held_out"]
    ret_side = int(round(np.sqrt(R.shape[1])))
    tr, te = ~ho, ho

    out = {"seed": seed, "n_frames": int(len(R)),
           "n_trained_frames": int(tr.sum()), "n_heldout_frames": int(te.sum()),
           "chance": CHANCE}
    if tr.sum() < 500 or te.sum() < 200:
        return {**out, "valid": False, "reason": "not enough frames"}

    # a held-out *time* split within the trained poses, so check 1 is not scored on frames the
    # probe was fitted to -- otherwise "pixels work on trained poses" is a tautology
    cut = int(tr.sum() * 0.7)
    tr_idx = np.flatnonzero(tr)
    fit, val = tr_idx[:cut], tr_idx[cut:]

    # The raw control is the **best cheap use of the input**, not one arbitrary estimator.
    # "Beat your own input" is not a claim about linear probes: a first pass here scored the
    # linear probe at 0.317 on trained poses and would have declared the world unmeasurable,
    # while nearest-template was at 0.680 on the same frames. Picking the weaker control and
    # calling the world void would have been the same error as picking it and calling an
    # architecture good.
    out["ridge_trained"] = decode(R[fit], kind[fit], R[val], kind[val])
    out["ridge_heldout"] = decode(R[fit], kind[fit], R[te], kind[te])
    out["template_trained"] = nearest_template(R[fit], kind[fit], R[val], kind[val])
    out["template_heldout"] = nearest_template(R[fit], kind[fit], R[te], kind[te])

    # choose the estimator on TRAINED poses, then ask whether *that* one generalises. Taking
    # the max of both columns independently would cherry-pick a control that does not exist.
    best = "template" if out["template_trained"] >= out["ridge_trained"] else "ridge"
    out["raw_control"] = best
    out["raw_trained"] = out[f"{best}_trained"]
    out["raw_heldout"] = out[f"{best}_heldout"]

    B = bag_of_features(R, ret_side)
    out["bag_trained"] = decode(B[fit], kind[fit], B[val], kind[val], n_components=9)

    # kinds 0 and 1 alone: identical angle multiset, identical radius multiset, paired
    # differently. This is the pair the world exists to make hard, so a feature bag must be at
    # chance *between these two specifically*, not merely across all four on average.
    pair_fit = fit[np.isin(kind[fit], (0, 1))]
    pair_val = val[np.isin(kind[val], (0, 1))]
    out["bag_confusable_pair"] = decode(B[pair_fit], kind[pair_fit],
                                        B[pair_val], kind[pair_val], n_components=9)

    out["efference_delivered"] = bool(np.allclose(d["eff"].sum(axis=1), 1.0))
    out["efference_entropy"] = float(-(lambda p: (p * np.log(p + 1e-12)).sum())(
        d["eff"].mean(axis=0)) / np.log(len(ACTIONS)))

    # Checks 1 and 2 are one test in two halves and neither means anything alone. Check 2
    # passing while check 1 fails is a world where *nothing* works, which reads as "the control
    # is at chance" and is really "there is nothing to measure" -- a vacuous pass, and this
    # project has been bitten by that exact shape twice.
    #
    # The bar on check 1 is twice chance rather than near-perfect, and the reason is structural:
    # the sensor is a fovea, so a single frame is a *fragment* by design. Requiring one fragment
    # to classify almost perfectly would be requiring the world not to have the property it was
    # built for. What has to be true is that the cue is decisively recoverable at trained poses
    # and gone at held-out ones -- a large gap, which is check 6.
    near = CHANCE
    out["gap"] = out["raw_trained"] - out["raw_heldout"]
    out["checks"] = {
        "1_raw_well_above_chance_on_trained": out["raw_trained"] > 2 * near,
        "2_raw_at_chance_on_heldout": out["raw_heldout"] < near + 0.10,
        "3_bag_of_features_at_chance": out["bag_trained"] < near + 0.10,
        "4_confusable_pair_at_chance_for_a_feature_bag": out["bag_confusable_pair"] < 0.60,
        "5_efference_delivered_every_tick": out["efference_delivered"],
        "6_memorisation_does_not_transfer": out["gap"] > 0.20,
    }
    out["valid"] = bool(out["checks"]["1_raw_well_above_chance_on_trained"]
                        and out["checks"]["2_raw_at_chance_on_heldout"]
                        and out["checks"]["6_memorisation_does_not_transfer"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=40000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    print(f"Pose world validity — {len(SHAPES)} kinds, chance {CHANCE:.3f}, "
          f"held-out poses {PoseConfig().held_out} of {PoseConfig().n_poses}\n")

    rows = [run(s, args.ticks) for s in args.seeds]
    for r in rows:
        print(f"  seed {r['seed']}  {r.get('n_trained_frames', 0)} trained / "
              f"{r.get('n_heldout_frames', 0)} held-out frames")
        if "raw_trained" not in r:
            print(f"    {r.get('reason')}\n")
            continue
        print(f"    raw control chosen on trained poses: {r['raw_control']}")
        for k in ("ridge_trained", "ridge_heldout", "template_trained", "template_heldout",
                  "bag_trained", "bag_confusable_pair", "gap"):
            print(f"    {k:24} {r[k]:.3f}")
        for name, ok in r["checks"].items():
            print(f"      {name:36} {'PASS' if ok else 'FAIL'}")
        print()

    valid = all(r.get("valid") for r in rows)
    print("=" * 78)
    if valid:
        print("  WORLD IS VALID. Raw pixels are at chance on poses they never saw, so the")
        print("  question 'does the representation beat its own input' can be asked here.")
    else:
        print("  WORLD IS VOID. Raw pixels generalise to held-out poses, so memorising")
        print("  appearances is sufficient and nothing measured here would mean anything.")
        print("  Fix the world -- do not run an architecture against it.")

    p = Path(__file__).resolve().parents[1] / "logs" / "pose_world_validity.json"
    p.write_text(json.dumps({"valid": valid, "rows": rows}, indent=1, default=float))
    print(f"\nwrote logs/{p.name}")


if __name__ == "__main__":
    main()
