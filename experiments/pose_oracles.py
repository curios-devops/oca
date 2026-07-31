"""Oracles for the pose world: how much invariance is available, and to what.

None of these is an architecture. Each is handed something no component here gets — the
perfectly reconstructed canvas — and asked whether the answer can be extracted from it. They
exist to separate two very different reasons for a null:

    the architecture is failing at a solvable problem
    the problem is not solvable in this world

Four levels, each one strictly more generous than the last.

1. **Ground truth.** The blob positions the simulator used. Confirms the invariant exists at all.
2. **Perfect integration.** Every fovea frame stitched back at the position it came from, then
   nearest-template. Integration with no invariance.
3. **Integration + a rotation-invariant descriptor.** The radial average of the power spectrum,
   which is invariant to rotation and translation by construction.
4. **Integration + part detection.** Peaks located, then the sorted pairwise distances between
   them — the exact invariant the world was designed around.

    python experiments/pose_oracles.py --ticks 60000
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.world.pose import SHAPES, PoseConfig
from validate_pose_world import CHANCE, collect, decode, nearest_template


def ground_truth_separation() -> dict:
    """Does the invariant exist at all? Computed from the simulator's own shape definitions."""
    d = {}
    for k, sh in enumerate(SHAPES):
        pos = np.array([[r * np.cos(np.deg2rad(a)), r * np.sin(np.deg2rad(a))] for a, r in sh])
        d[k] = sorted(float(np.linalg.norm(pos[i] - pos[j]))
                      for i, j in itertools.combinations(range(3), 2))
    gaps = {f"{a}v{b}": float(np.abs(np.array(d[a]) - np.array(d[b])).max())
            for a, b in itertools.combinations(range(len(SHAPES)), 2)}
    return {"pairwise_distances": d, "closest_gap": min(gaps.values()), "gaps": gaps,
            "in_pixels": {k: [round(x * PoseConfig().object_radius, 1) for x in v]
                          for k, v in d.items()}}


def radial_power(imgs: np.ndarray, side: int, n_bins: int = 28) -> np.ndarray:
    """Radial average of the power spectrum. Rotation- and translation-invariant by
    construction: rotating an image rotates its spectrum, and a radial average of a rotated
    function is unchanged."""
    X = imgs.reshape(len(imgs), side, side)
    P = np.abs(np.fft.fftshift(np.fft.fft2(X), axes=(1, 2))) ** 2
    c = (side - 1) / 2.0
    yy, xx = np.mgrid[0:side, 0:side]
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2).ravel()
    edges = np.linspace(0, r.max() + 1e-9, n_bins + 1)
    w = np.clip(np.digitize(r, edges) - 1, 0, n_bins - 1)
    F = P.reshape(len(X), -1)
    return np.stack([F[:, w == b].mean(1) for b in range(n_bins)], axis=1)


def part_distances(imgs: np.ndarray, side: int, n_parts: int = 3,
                   sup: float = 2.2) -> np.ndarray:
    """Locate the parts by peak-picking with non-max suppression, then take the **sorted**
    pairwise distances between them. Sorted so it carries no order, distances so it carries no
    orientation: the exact invariant this world was designed around.

    **This detector is weak and its weakness is measurable**, which is why the run reports the
    recovered distances next to the true ones rather than only an accuracy. Sharpening it is
    feature engineering, not architecture, and it is not the project.
    """
    radius = int(PoseConfig().blob_radius * sup)
    X = imgs.reshape(len(imgs), side, side)
    out = np.zeros((len(X), n_parts * (n_parts - 1) // 2))
    for n in range(len(X)):
        img = X[n].copy()
        pts = []
        for _ in range(n_parts):
            y, x = divmod(int(np.argmax(img)), side)
            pts.append((y, x))
            img[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1] = 0.0
        out[n] = sorted(float(np.hypot(a[0] - b[0], a[1] - b[1]))
                        for a, b in itertools.combinations(pts, 2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gt = ground_truth_separation()
    print("1. Does the invariant exist? Sorted pairwise distances from the simulator itself.\n")
    for k, v in gt["in_pixels"].items():
        print(f"     kind {k}   {v} px")
    print(f"\n   closest pair of kinds differ by {gt['closest_gap'] * PoseConfig().object_radius:.1f} px"
          f"  =>  {'THE INVARIANT EXISTS' if gt['closest_gap'] > 0.1 else 'THE WORLD IS BROKEN'}\n")

    d = collect(args.ticks, args.seed)
    I, y, ho = d["I"], d["kind"], d["held_out"]
    side = int(round(np.sqrt(I.shape[1])))
    tr, te = np.flatnonzero(~ho), np.flatnonzero(ho)
    cut = int(len(tr) * 0.7)
    fit, val = tr[:cut], tr[cut:]

    rows = {}
    rows["integration only"] = (nearest_template(I[fit], y[fit], I[val], y[val]),
                                nearest_template(I[fit], y[fit], I[te], y[te]))
    S = radial_power(I, side)
    rows["+ rotation-invariant spectrum"] = (decode(S[fit], y[fit], S[val], y[val], n_components=28),
                                             decode(S[fit], y[fit], S[te], y[te], n_components=28))
    G = part_distances(I, side)
    rows["+ part detection, pairwise distances"] = (
        decode(G[fit], y[fit], G[val], y[val], n_components=3),
        decode(G[fit], y[fit], G[te], y[te], n_components=3))

    print(f"2-4. Handed the perfect reconstruction. {len(tr)} trained / {len(te)} held-out "
          f"presentations, chance {CHANCE:.3f}\n")
    print(f"     {'oracle':40}{'trained':>9}{'HELD-OUT':>10}")
    for name, (a, b) in rows.items():
        print(f"     {name:40}{a:>9.3f}{b:>10.3f}")

    print(f"\n     recovered vs true pairwise distances (px) -- how good is the part detector?")
    for k in range(len(SHAPES)):
        rec = np.round(G[y == k].mean(0), 1)
        print(f"       kind {k}   recovered {rec}   true {gt['in_pixels'][k]}")

    best = max(b for _, b in rows.values())
    print(f"\n     best held-out score by any oracle: {best:.3f}  (chance {CHANCE:.3f})")

    p = Path(__file__).resolve().parents[1] / "logs" / "pose_oracles.json"
    p.write_text(json.dumps({"ground_truth": gt, "chance": CHANCE,
                             "oracles": {k: {"trained": a, "held_out": b}
                                         for k, (a, b) in rows.items()}},
                            indent=1, default=float))
    print(f"\nwrote logs/{p.name}")


if __name__ == "__main__":
    main()
