"""E3b -- is the uncertainty head doing real work, or just setting the learning rate?

E3 finds that removing the uncertainty head is by far the most damaging ablation. But
precision `c = 1/(1+sigma_hat)` multiplies every learning rate, and switching it off sets
c to 1.0 where it otherwise sits near 0.35. So `no_uncertainty` also triples the
effective step size, and the damage could be nothing more than a badly tuned learning
rate.

This sweeps the learning rate for the ablated model. If some fixed rate matches the full
model, precision weighting is a learning-rate schedule and should be described as one.
If the full model still wins at every fixed rate, then what matters is that precision
*varies* -- per unit and over time -- which is the actual claim in the design document.

    python experiments/exp03b_uncertainty_control.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.data import train_test_streams
from core.metrics import JsonlLogger
from legacy.v1.run import train_and_eval
from legacy.v1.state import Config

HORIZONS = (1, 4, 16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--test", type=int, default=1500)
    ap.add_argument("--side", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--base-eta", type=float, default=0.08)
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.15, 0.3, 0.5, 1.0, 2.0])
    ap.add_argument("--out", default="logs/exp03b_uncertainty.jsonl")
    args = ap.parse_args()

    data = train_test_streams(args.train, args.test)
    results = {}

    with JsonlLogger(args.out, meta=vars(args)) as log:
        # reference: the full model, and the mean precision it actually used
        runs, confs = [], []
        for seed in args.seeds:
            cfg = Config(lattice_side=args.side, seed=seed, eta_head=args.base_eta)
            res, state = train_and_eval(cfg, data, logger=log, tag="full", log_every=0)
            runs.append(res["test"])
            confs.append(float(state.conf.mean()))
        results["full"] = _mean(runs)
        results["full"]["mean_precision"] = float(np.mean(confs))
        print(f"full (mean precision {np.mean(confs):.3f})  {_fmt(results['full'])}")

        for scale in args.scales:
            runs = []
            for seed in args.seeds:
                cfg = Config(lattice_side=args.side, seed=seed,
                             eta_head=args.base_eta * scale).variant(
                    use_uncertainty=False,
                    eta_landscape=Config().eta_landscape * scale,
                    eta_link=Config().eta_link * scale,
                )
                res, _ = train_and_eval(cfg, data, logger=log,
                                        tag=f"nounc_x{scale}", log_every=0)
                runs.append(res["test"])
            key = f"no_uncertainty_lr_x{scale}"
            results[key] = _mean(runs)
            log.log(kind="control", name=key, scale=scale, **results[key])
            print(f"{key:26s} {_fmt(results[key])}")

    Path("logs/exp03b_summary.json").write_text(json.dumps(results, indent=2))
    _report(results)


def _mean(runs):
    return {f"mse_t{t}": float(np.mean([r[f"mse_t{t}"] for r in runs])) for t in HORIZONS}


def _fmt(d):
    return "  ".join(f"t{t}={d[f'mse_t{t}']:.5f}" for t in HORIZONS)


def _report(results):
    full = results["full"]["mse_t16"]
    best = min((v["mse_t16"], k) for k, v in results.items() if k != "full")
    print(f"\nfull model              t16 = {full:.5f}")
    print(f"best ablated + retuned  t16 = {best[0]:.5f}  ({best[1]})")
    if best[0] <= full * 1.02:
        print("\nVERDICT: retuning the learning rate recovers the full model. The "
              "uncertainty head\nis acting as a learning-rate schedule, not as an "
              "estimate that has to adapt.")
    else:
        print(f"\nVERDICT: the full model still wins by {best[0]/full:.2f}x at the best "
              "fixed rate.\nWhat matters is that precision varies per unit and over "
              "time, not its average size.")


if __name__ == "__main__":
    main()
