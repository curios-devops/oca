"""E1 -- does the mesh learn a world model at all?

Trains the mesh on one world seed, then scores it with plasticity frozen on a held-out
seed, against the baselines from exp00. The gate is copy-last: if the mesh cannot beat
"nothing changes", the hypothesis has failed its first contact with data and the honest
move is to report that and stop.

    python experiments/exp00_baselines.py --train 8000 --test 1500
    python experiments/exp01_learning.py  --train 8000 --test 1500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data import train_test_streams
from core.metrics import JsonlLogger
from architectures.wren.run import train_and_eval
from architectures.wren.state import Config

HORIZONS = (1, 4, 16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--test", type=int, default=1500)
    ap.add_argument("--side", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eta-head", type=float, default=0.08)
    ap.add_argument("--out", default="logs/exp01_learning.jsonl")
    args = ap.parse_args()

    data = train_test_streams(args.train, args.test)
    cfg = Config(lattice_side=args.side, seed=args.seed, eta_head=args.eta_head)

    t0 = time.time()
    with JsonlLogger(args.out, meta=vars(args)) as log:
        result, state = train_and_eval(cfg, data, logger=log, log_every=250)
    result["seconds"] = round(time.time() - t0, 1)

    print(f"mesh: N={cfg.n_units} degree={cfg.degree} params={result['n_params']:,} "
          f"({result['seconds']}s, {state.n_clipped} clipped)")
    print("held-out MSE:", _fmt(result["test"]))

    verdict = _compare(result)
    Path("logs/exp01_summary.json").write_text(
        json.dumps({"mesh": result, "verdict": verdict}, indent=2)
    )
    for line in verdict:
        print(line)


def _fmt(d: dict) -> str:
    return "  ".join(f"t{tau}={d[f'mse_t{tau}']:.5f}" for tau in HORIZONS)


def _compare(result: dict) -> list[str]:
    path = Path("logs/baselines_summary.json")
    if not path.exists():
        return ["no baselines_summary.json -- run exp00_baselines.py first"]
    base = json.loads(path.read_text())
    lines = [""]
    for name, vals in base.items():
        ratios = [result["test"][f"mse_t{t}"] / vals[f"mse_t{t}"] for t in HORIZONS]
        verdict = "mesh better" if all(r < 1 for r in ratios) else (
            "mesh worse" if all(r > 1 for r in ratios) else "mixed")
        detail = " ".join(f"t{t}={r:.2f}x" for t, r in zip(HORIZONS, ratios))
        lines.append(f"vs {name:10s} {detail}   ({verdict})")
    gate = result["test"]["mse_t1"] < base["copy_last"]["mse_t1"]
    lines.append("")
    lines.append("GATE " + ("PASS" if gate else "FAIL") + ": mesh vs copy-last at tau=1")
    return lines


if __name__ == "__main__":
    main()
