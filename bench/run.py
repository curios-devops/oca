"""Run every registered mesh version through every gate and print one table.

    python -m bench.run                       # all versions, all gates, 1 seed
    python -m bench.run --seeds 0 1 2         # with error bars
    python -m bench.run --gates maze --sides 12 24   # the scale stress

The point of the scorecard is to make regressions visible. v2 restored mechanisms that v1
could not have (coalitions) while costing accuracy elsewhere, and that trade is only
legible when both are scored on the same battery in the same run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.gates import GATES
from bench.registry import available, get


def run_cell(version, gate_name, seed, side, **kw):
    t0 = time.time()
    try:
        res = GATES[gate_name](version, seed=seed, side=side, **kw)
        res["seconds"] = round(time.time() - t0, 1)
        return res
    except Exception as exc:                      # a broken gate must not hide the rest
        return {"error": f"{type(exc).__name__}: {exc}", "valid": False,
                "headline": None, "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="*", default=None)
    ap.add_argument("--gates", nargs="*", default=list(GATES))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--sides", type=int, nargs="*", default=[12])
    ap.add_argument("--ticks", type=int, default=None)
    ap.add_argument("--out", default="logs/scorecard.json")
    args = ap.parse_args()

    versions = args.versions or available()
    extra = {"ticks": args.ticks} if args.ticks else {}
    card: dict = {"meta": vars(args), "cells": {}}

    for vname in versions:
        version = get(vname)
        for gate in args.gates:
            for side in args.sides:
                runs = []
                for seed in args.seeds:
                    print(f"  {vname:>4s} / {gate:<11s} side={side} seed={seed} ...",
                          flush=True)
                    runs.append(run_cell(version, gate, seed, side, **extra))
                card["cells"][f"{vname}|{gate}|{side}"] = _summarise(runs)

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(card, indent=2, default=str))
    _report(card, versions, args.gates, args.sides)


def _summarise(runs):
    heads = [r["headline"] for r in runs if r.get("headline") is not None]
    out = {"runs": runs, "n": len(runs), "valid": all(r.get("valid") for r in runs)}
    if heads:
        out["mean"] = float(np.mean(heads))
        out["sem"] = float(np.std(heads, ddof=1) / np.sqrt(len(heads))) if len(heads) > 1 else None
        out["higher_is_better"] = runs[0].get("higher_is_better", True)
    return out


def _fmt(cell):
    if not cell or "mean" not in cell:
        err = next((r.get("error") for r in cell.get("runs", []) if r.get("error")), None)
        return "  error " if err else "    n/a "
    s = f"{cell['mean']:+.3f}" if cell.get("higher_is_better") else f"{cell['mean']:.3f}"
    if cell.get("sem"):
        s += f"±{cell['sem']:.3f}"
    return s + ("" if cell["valid"] else " !")


def _report(card, versions, gates, sides):
    from bench.registry import get
    print(f"\n{'':10s}" + "".join(f"{g:>18s}" for g in gates))
    for vname in versions:
        for side in sides:
            code = get(vname).codename or vname
            label = code + (f"/{side}" if len(sides) > 1 else "")
            row = "".join(f"{_fmt(card['cells'].get(f'{vname}|{g}|{side}')):>18s}"
                          for g in gates)
            print(f"{label:10s}{row}")

    print("\nheadline per gate:")
    print("  prediction  16-step MSE as a fraction of copy-last (lower is better)")
    print("  maze        out-of-view wall decode minus the raw-pixel control (higher)")
    print("  identity    hidden-object kind, balanced accuracy, chance 0.50 (higher)")
    print("  coalitions  object MI above a shuffled null (higher)")
    print("  '!' marks a cell whose control failed, so its number means nothing.")

    reasons = []
    for key, cell in card["cells"].items():
        if cell.get("valid", True):
            continue
        why = next((r.get("reason") or r.get("error") for r in cell.get("runs", [])
                    if r.get("reason") or r.get("error")), None)
        reasons.append((key, why))
    if reasons:
        print("\n  cells marked '!' are unmeasured, not zero:")
        for key, why in reasons:
            print(f"    {key}: {why or 'control failed'}")


if __name__ == "__main__":
    main()
