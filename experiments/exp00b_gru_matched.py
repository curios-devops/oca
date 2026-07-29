"""A capacity-matched GRU baseline.

The default GRU has ~1.29M parameters against the mesh's ~2.25M, so beating it could be
a capacity result rather than a learning-rule result. This trains a wider GRU sized to
the mesh and appends it to the baseline summary, so the E1 comparison can be read
without that caveat.

    python experiments/exp00b_gru_matched.py --hidden 288
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.baselines import HORIZONS, GRU, as_batch
from core.data import train_test_streams
from core.metrics import JsonlLogger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=8000)
    ap.add_argument("--test", type=int, default=1500)
    ap.add_argument("--hidden", type=int, default=288)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/baselines_matched.jsonl")
    args = ap.parse_args()

    streams = train_test_streams(args.train, args.test, train_seed=args.seed)
    tr, te = streams["train"]["retina"], streams["test"]["retina"]

    with JsonlLogger(args.out, meta=vars(args)) as log:
        gru = GRU(n_in=tr[0].size, hidden=args.hidden, seed=args.seed)
        print(f"gru_matched params {gru.n_params:,}")
        gru.fit(as_batch(tr, args.batch), epochs=args.epochs, log_every=200, logger=log)
        res = gru.evaluate(te).cumulative()
        res["n_params"] = gru.n_params
        log.log(kind="baseline", name="gru_matched", **res)

    path = Path("logs/baselines_summary.json")
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary["gru_matched"] = res
    path.write_text(json.dumps(summary, indent=2))
    print("gru_matched", "  ".join(f"t{t}={res[f'mse_t{t}']:.5f}" for t in HORIZONS))


if __name__ == "__main__":
    main()
