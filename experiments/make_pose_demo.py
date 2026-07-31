"""Record a pose-world run and build a self-contained page from it.

    python experiments/make_pose_demo.py
    python -m http.server 8080 --directory demo     # then open /pose.html

The page shows three things side by side, and the middle one is the point:

* **the world** — the whole object, which nothing in the system ever sees at once
* **the eye** — the 32-pixel fovea, which is all it ever gets
* **what it has pieced together** — the fragments put back where they came from

Watching the third panel fill in is the project in one animation. And the caption underneath is
the finding: a perfect reconstruction still cannot recognise the object at an angle it has never
been shown, so **invariance is not a consequence of integration.**

The page is generated from a recorded trace, exactly as the maze demo is, so it can never
disagree with a run. The viewer redraws everything from the canvas and the fovea positions, so
the payload is one image per presentation plus two numbers per tick.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.world.pose import PoseConfig, PoseWorld

KIND_NAMES = ("Ada", "Bo", "Cy", "Dot")
"""The four shapes, named so a viewer can follow one across the run. They are the same three
blobs every time -- only the arrangement differs, which is the whole difficulty."""


def record(n_trained: int = 5, n_held_out: int = 3, seed: int = 3) -> dict:
    """Record until there are enough of both kinds of presentation to see the contrast.

    **This selects for display and the real world does not.** One pose in six is held out, so a
    straight recording of eight presentations shows the interesting case roughly once and the
    contrast is invisible. Nothing about a presentation is altered -- they are drawn exactly as
    the world produces them, and only *how many of each are kept* is chosen here. No measurement
    reads this file.
    """
    cfg = PoseConfig(seed=seed)
    world = PoseWorld(cfg)
    episodes, want = [], {True: n_held_out, False: n_trained}
    while sum(want.values()) > 0:
        canvas = world.full_object()
        kind, pose, held = world.kind, world.pose, world.is_held_out()
        path = []
        for _ in range(cfg.episode_len):
            path.append([int(round(world.fovea[0])), int(round(world.fovea[1]))])
            world.step()
        if want[bool(held)] == 0:
            continue
        want[bool(held)] -= 1
        episodes.append({
            "kind": int(kind), "name": KIND_NAMES[kind], "pose": int(pose),
            "held_out": bool(held), "path": path,
            # one digit per pixel: a 120x120 float canvas is 115 KB of JSON and 14 KB this way,
            # with no visible loss at this size
            "canvas": "".join(str(min(9, max(0, int(round(v * 9))))) for v in canvas.ravel()),
        })
    return {
        "canvas_size": cfg.canvas, "fovea": cfg.size, "n_poses": cfg.n_poses,
        "held_out_poses": list(cfg.held_out), "episode_len": cfg.episode_len,
        "episodes": episodes,
    }


def main() -> None:
    payload = record()
    # Interleave so an unseen angle turns up early and regularly. Recorded in order they arrive
    # in two runs, which puts every unseen one at the end -- a viewer would give up before the
    # contrast the page is about ever appeared.
    fam = [e for e in payload["episodes"] if not e["held_out"]]
    new = [e for e in payload["episodes"] if e["held_out"]]
    mixed = []
    while fam or new:
        for _ in range(2):
            if fam:
                mixed.append(fam.pop(0))
        if new:
            mixed.append(new.pop(0))
    payload["episodes"] = mixed
    tpl = (ROOT / "experiments" / "pose_demo_template.html").read_text()
    out = ROOT / "demo" / "pose.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(tpl.replace("__DATA__", json.dumps(payload, separators=(",", ":"))))
    held = sum(e["held_out"] for e in payload["episodes"])
    print(f"wrote {out.relative_to(ROOT)}  "
          f"{len(payload['episodes'])} presentations, {held} at an unseen angle "
          f"(selected for display; the world produces one in six), "
          f"{out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
