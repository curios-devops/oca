"""E4 -- what does the mesh do when the world goes away?

The design document claims "the brain keeps changing even in silence". A network of
gradient-flow units has two obvious ways to fail that claim: collapse to a fixed point
(every unit sits in its nearest valley and nothing further happens) or diverge. Both are
informative, so this experiment is written to report whichever happens rather than to
confirm the claim.

Sensory input is cut at the halfway point and the mesh free-runs. Tracked throughout:
population state entropy, mean state norm, surprise, and coalition turnover.

    python experiments/exp04_silence.py --train 8000 --silent 3000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data import rollout
from architectures.wren.mesh import build_mesh, tick
from core.metrics import JsonlLogger, coalition_stats, state_entropy
from architectures.wren.run import run_stream
from architectures.wren.state import Config
from core.world import Sensors
from core.world.sensors import N_SENSORY, P


def probe(state, prev_labels):
    stats = coalition_stats(state.coalition, prev_labels)
    coh = getattr(state, "coh_pct", {})
    return {
        "entropy": state_entropy(state.h),
        "h_norm": float(np.linalg.norm(state.h, axis=1).mean()),
        "h_std": float(state.h.std()),
        "surprise": float(state.surprise.mean()),
        "coh_p90": coh.get("p90"),
        "coh_p99": coh.get("p99"),
        "coh_max": coh.get("max"),
        **stats,
    }


def phase(state, sensory, logger, tag, n, silent=False):
    """Run `n` ticks, either driven by `sensory` or on zero input."""
    rows, prev = [], None
    blank = np.zeros((N_SENSORY, P), dtype=np.float32)
    for t in range(n):
        tick(state, blank if silent else sensory[t])
        if t % 10 == 0 and state.t > state.cfg.coalition_window:
            row = probe(state, prev)
            prev = state.coalition.copy()
            row.update(kind=tag, phase_t=t)
            logger.log(**row)
            rows.append(row)
    return rows


def agg(rows, key, last_frac=0.5):
    vals = [r[key] for r in rows[int(len(rows) * (1 - last_frac)):] if r.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--driven", type=int, default=1500)
    ap.add_argument("--silent", type=int, default=3000)
    ap.add_argument("--side", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/exp04_silence.jsonl")
    args = ap.parse_args()

    cfg = Config(lattice_side=args.side, seed=args.seed, eta_head=0.08)
    sensors = Sensors()
    ret, sen, _ = rollout(args.train + args.driven, seed=args.seed)

    with JsonlLogger(args.out, meta=vars(args)) as log:
        state = build_mesh(cfg)
        print("training ...")
        run_stream(state, sen[: args.train], ret[: args.train], learn=True,
                   sensors=sensors, logger=log, tag="e4_train", log_every=1000)
        state.learn = False

        print("driven phase ...")
        driven = phase(state, sen[args.train:], log, "driven", args.driven)
        print("silent phase ...")
        silent = phase(state, None, log, "silent", args.silent, silent=True)

        results = {
            k: {"driven": agg(driven, k), "silent": agg(silent, k)}
            for k in ("entropy", "h_norm", "h_std", "surprise",
                      "n_coalitions", "frac_in_coalition", "churn",
                      "coh_p90", "coh_p99", "coh_max")
        }
        results["verdict"] = verdict(results)
        log.log(kind="result", **results)

    Path("logs/exp04_summary.json").write_text(json.dumps(results, indent=2))
    _report(results)


def verdict(r: dict) -> str:
    d, s = r["h_std"]["driven"], r["h_std"]["silent"]
    churn = r["churn"]["silent"]
    if not np.isfinite(s) or s > 5 * d:
        return "diverged"
    if s < 0.05 * d:
        return "collapsed to a fixed point"
    if churn is not None and churn > 0.05:
        return "keeps changing: state varies and coalitions still turn over"
    return "settled: state persists but the mesh stops reorganising"


def _report(r: dict) -> None:
    print(f"\n{'metric':20s} {'driven':>12s} {'silent':>12s}")
    for k, v in r.items():
        if k == "verdict":
            continue
        print(f"{k:20s} {v['driven']:12.4f} {v['silent']:12.4f}")
    print(f"\nE4 verdict: {r['verdict']}")


if __name__ == "__main__":
    main()
