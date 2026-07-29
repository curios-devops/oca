"""E3 -- which ingredients actually carry the effect?

Each ablation removes exactly one claim from the design documents and reruns E1. The
table this produces is the most useful output of the whole project: it says which parts
of the story are load-bearing and which are decoration.

Two of these are included specifically because they could embarrass the hypothesis:
`instant_delivery` removes the one-tick propagation delay that the settling story
depends on, and `no_landscape` replaces the energy geometry with a generic recurrent
map of identical capacity. If either *improves* the mesh, that is the finding.

    python experiments/exp03_ablation.py --train 8000 --test 1500
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
from core.world.physics import make_physics_world

HORIZONS = (1, 4, 16)

ABLATIONS = {
    "full":              {},
    "no_uncertainty":    dict(use_uncertainty=False),
    "no_rewiring":       dict(use_rewiring=False),
    "no_long_range":     dict(use_long_range=False),
    "no_landscape":      dict(use_landscape=False),
    "no_coalition_fb":   dict(use_coalition_feedback=False),
    "no_oscillator":     dict(use_oscillator=False),
    "msg_width_1":       dict(msg_width=1),
    "instant_delivery":  dict(instant_delivery=True),
    "no_plasticity":     dict(use_plasticity=False),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--test", type=int, default=1500)
    ap.add_argument("--side", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--world", choices=["v1", "physics"], default="v1")
    ap.add_argument("--eta-head", type=float, default=0.08,
                    help="0.08 suits world v1; world v2 needs ~0.01 "
                         "or the 16-step head overfits an unpredictable target")
    ap.add_argument("--out", default="logs/exp03_ablation.jsonl")
    args = ap.parse_args()

    names = args.only or list(ABLATIONS)
    factory = make_physics_world if args.world == "physics" else None
    data = train_test_streams(args.train, args.test, world_factory=factory)
    results: dict[str, dict] = {}

    with JsonlLogger(args.out, meta=vars(args)) as log:
        for name in names:
            runs = []
            t0 = time.time()
            for seed in args.seeds:
                cfg = Config(lattice_side=args.side, seed=seed, eta_head=args.eta_head)
                cfg = cfg.variant(**ABLATIONS[name])
                res, _ = train_and_eval(cfg, data, logger=log, tag=name, log_every=0)
                runs.append(res["test"])
                log.log(kind="ablation_run", name=name, seed=seed, **res["test"])
            results[name] = {
                f"mse_t{tau}": _mean([r[f"mse_t{tau}"] for r in runs]) for tau in HORIZONS
            }
            results[name]["sem_t16"] = _sem([r["mse_t16"] for r in runs])
            results[name]["n_seeds"] = len(runs)
            log.log(kind="ablation", name=name, **results[name])
            print(f"{name:18s} {_fmt(results[name])}  ({time.time()-t0:.0f}s)")

    suffix = "" if args.world == "v1" else f"_{args.world}"
    Path(f"logs/exp03_summary{suffix}.json").write_text(json.dumps(results, indent=2))
    _report(results)


def _mean(xs):
    return float(sum(xs) / len(xs))


def _sem(xs):
    if len(xs) < 2:
        return None
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return float((var / len(xs)) ** 0.5)


def _fmt(d):
    return "  ".join(f"t{tau}={d[f'mse_t{tau}']:.5f}" for tau in HORIZONS)


def _report(results: dict) -> None:
    if "full" not in results:
        return
    full = results["full"]
    print(f"\n{'ablation':18s} {'t1':>8s} {'t4':>8s} {'t16':>8s}   effect on t16")
    rows = []
    for name, r in results.items():
        if name == "full":
            continue
        ratio = r["mse_t16"] / full["mse_t16"]
        rows.append((ratio, name, r))
    for ratio, name, r in sorted(rows, reverse=True):
        sign = "hurts" if ratio > 1.02 else ("HELPS" if ratio < 0.98 else "no effect")
        print(f"{name:18s} {r['mse_t1']/full['mse_t1']:7.2f}x "
              f"{r['mse_t4']/full['mse_t4']:7.2f}x {ratio:7.2f}x   {sign}")
    print("\nRatios above 1 mean the full model was better, so the ablated ingredient "
          "was doing work.\nRatios below 1 mean removing it improved the mesh.")


if __name__ == "__main__":
    main()
