"""Gate for build step 2: establish the numbers the mesh has to beat.

Trains and evaluates all three baselines on the same train/held-out world seeds the
mesh will use, and writes their errors to logs/baselines.jsonl. Nothing later in the
project means anything without these.

    python experiments/exp00_baselines.py --train 4000 --test 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.baselines import HORIZONS, GRU, CopyLast, LinearAR, as_batch
from core.data import train_test_streams
from core.metrics import JsonlLogger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gru-epochs", type=int, default=40)
    ap.add_argument("--gru-hidden", type=int, default=192)
    ap.add_argument("--gru-batch", type=int, default=8)
    ap.add_argument("--out", default="logs/baselines.jsonl")
    args = ap.parse_args()

    streams = train_test_streams(args.train, args.test, train_seed=args.seed)
    tr, te = streams["train"]["retina"], streams["test"]["retina"]
    print(f"train {tr.shape}  test {te.shape}")

    results: dict[str, dict] = {}
    with JsonlLogger(args.out, meta=vars(args)) as log:
        copy = CopyLast().evaluate(te).cumulative()
        results["copy_last"] = copy
        log.log(kind="baseline", name="copy_last", **copy)
        print("copy_last  ", _fmt(copy))

        lin = LinearAR()
        lin.fit(tr)
        lin_r = lin.evaluate(te).cumulative()
        results["linear_ar"] = lin_r
        log.log(kind="baseline", name="linear_ar", n_features=lin.n_features, **lin_r)
        print("linear_ar  ", _fmt(lin_r))

        gru = GRU(n_in=tr[0].size, hidden=args.gru_hidden, seed=args.seed)
        batch = as_batch(tr, args.gru_batch)
        print(f"gru params {gru.n_params:,}  batch {len(batch)}x{len(batch[0])}")
        gru.fit(batch, epochs=args.gru_epochs, log_every=100, logger=log)
        gru_r = gru.evaluate(te).cumulative()
        results["gru"] = gru_r
        log.log(kind="baseline", name="gru", n_params=gru.n_params, **gru_r)
        print("gru        ", _fmt(gru_r))

    Path("logs/baselines_summary.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", args.out, "and logs/baselines_summary.json")


def _fmt(d: dict) -> str:
    return "  ".join(f"t{tau}={d[f'mse_t{tau}']:.5f}" for tau in HORIZONS)


if __name__ == "__main__":
    main()
