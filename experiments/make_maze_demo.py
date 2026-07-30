"""Regenerate the maze demo pages from a run's trace.

Keeps the visualisation reproducible rather than a one-off artefact. Two pages, whichever
traces exist:

    python experiments/maze_race.py --steps 900     -> demo/index.html   (all three race)
    python experiments/exp09_maze.py --demo 900     -> demo/single.html  (one, in detail)
    python experiments/make_maze_demo.py
    python -m http.server 8080 --directory demo

The race is the front page because it is the comparison that carries the argument: same
maze, same planner, same steps, so the only difference is the map each architecture
decodes out of its own state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def compress(trace_path: Path) -> dict:
    d = json.loads(trace_path.read_text())
    grid, tr, ex = d["grid"], d["trace"], d["exit"]
    n = len(grid)
    pos = [s["pos"] for s in tr]
    # reaching the exit teleports the agent, so an exit event shows up as a jump in
    # position rather than as a frame standing on the exit cell
    hits = [i for i in range(1, len(pos))
            if abs(pos[i][0] - pos[i - 1][0]) + abs(pos[i][1] - pos[i - 1][1]) > 1]
    return {
        "n": n,
        "grid": "".join(str(v) for row in grid for v in row),
        "exit": ex,
        "pos": pos,
        # belief quantised to one digit per cell: 900 frames of 21x21 floats is 2.4 MB
        # of JSON and 400 KB this way, with no visible loss at this cell size
        "belief": ["".join(str(min(9, max(0, int(round(v * 9)))))
                           for row in s["belief"] for v in row) for s in tr],
        "hits": hits,
    }


def _quantise(belief, n) -> str:
    """One digit per cell. 900 frames of 21x21 floats is 2.4 MB of JSON and 400 KB this
    way, with no visible loss at this cell size."""
    return "".join(str(min(9, max(0, int(round(v * 9)))))
                   for row in belief for v in row)


def _path_integration() -> dict:
    """The other gate's result, keyed by architecture, if it has been run.

    Carried into the page because the two numbers *disagree*, and that disagreement is the most
    useful thing the demo can show: a trophy is always a trophy at one question.
    """
    f = ROOT / "logs" / "scorecard_pathint.json"
    if not f.exists():
        return {}
    cells = json.loads(f.read_text()).get("cells", {})
    return {k.split("|")[0]: v.get("mean") for k, v in cells.items()
            if v.get("mean") is not None}


def compress_race(path: Path) -> dict:
    d = json.loads(path.read_text())
    pathint = _path_integration()
    grid = d["grid"]
    n = len(grid)
    ents = []
    for e in d["entrants"]:
        ents.append({
            "key": e["key"], "label": e["label"], "exits": e["exits"],
            "hits": e["hits"], "hidden_acc": e["hidden_acc"],
            "retina_hidden": e["retina_hidden"], "n_params": e["n_params"],
            "path_integration": pathint.get(e["key"]),
            "pos": [s["pos"] for s in e["trace"]],
            "belief": [_quantise(s["belief"], n) for s in e["trace"]],
        })
    return {"n": n, "grid": "".join(str(v) for row in grid for v in row),
            "exit": d["exit"], "steps": min(len(e["pos"]) for e in ents),
            "entrants": ents}


def _render(payload: dict, template: str, out: Path) -> None:
    tpl = (ROOT / "experiments" / template).read_text()
    out.parent.mkdir(exist_ok=True)
    out.write_text(tpl.replace("__DATA__", json.dumps(payload, separators=(",", ":"))))


def main() -> None:
    logs = ROOT / "logs"
    wrote = []

    race = logs / "maze_race.json"
    if race.exists():
        p = compress_race(race)
        _render(p, "maze_race_template.html", ROOT / "demo" / "index.html")
        wrote.append(f"demo/index.html  the race, {p['steps']} steps, "
                     + ", ".join(f"{e['label']} {e['exits']}" for e in p["entrants"]))

    single = logs / "maze_trace.json"
    if single.exists():
        p = compress(single)
        name = "single.html" if race.exists() else "index.html"
        _render(p, "maze_demo_template.html", ROOT / "demo" / name)
        wrote.append(f"demo/{name}  one model, {len(p['pos'])} steps, "
                     f"{len(p['hits'])} exits")

    if not wrote:
        raise SystemExit("no traces in logs/ -- run maze_race.py or exp09_maze.py --demo")
    for line in wrote:
        print("wrote " + line)


if __name__ == "__main__":
    main()
