"""Metrics and logging.

Everything an experiment reports goes through here and lands in a JSONL file. Figures
are generated from the logs afterwards, never from live state, so a plot can always be
regenerated from a run that has already finished.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class JsonlLogger:
    """Append-only JSONL writer. numpy scalars and arrays are coerced to plain lists."""

    def __init__(self, path: str | Path, meta: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", buffering=1)
        self._t0 = time.time()
        if meta is not None:
            self.log(kind="meta", **meta)

    def log(self, **record) -> None:
        record.setdefault("wall", round(time.time() - self._t0, 3))
        self._fh.write(json.dumps(record, default=_coerce) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _coerce(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


@dataclass
class RunningMean:
    """Windowed mean, for smoothing noisy per-tick quantities into a learning curve."""

    window: int = 500
    _buf: deque = field(default_factory=deque, repr=False)

    def add(self, x: float) -> None:
        self._buf.append(float(x))
        while len(self._buf) > self.window:
            self._buf.popleft()

    @property
    def value(self) -> float:
        return float(np.mean(self._buf)) if self._buf else float("nan")

    @property
    def n(self) -> int:
        return len(self._buf)

    def reset(self) -> None:
        self._buf.clear()


def frame_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error over a whole predicted frame."""
    d = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(d * d))


def normalised_mse(pred: np.ndarray, target: np.ndarray, baseline: float) -> float:
    """Frame MSE expressed as a fraction of a baseline MSE. < 1 means better."""
    return frame_mse(pred, target) / max(baseline, 1e-12)


@dataclass
class ErrorTracker:
    """Accumulates per-horizon frame errors for one predictor."""

    horizons: tuple[int, ...] = (1, 4, 16)
    window: int = 500

    def __post_init__(self) -> None:
        self.means = {tau: RunningMean(self.window) for tau in self.horizons}
        self.total = {tau: 0.0 for tau in self.horizons}
        self.count = {tau: 0 for tau in self.horizons}

    def add(self, tau: int, mse: float) -> None:
        self.means[tau].add(mse)
        self.total[tau] += mse
        self.count[tau] += 1

    def windowed(self) -> dict[str, float]:
        return {f"mse_t{tau}": m.value for tau, m in self.means.items()}

    def cumulative(self) -> dict[str, float]:
        return {
            f"mse_t{tau}": (self.total[tau] / self.count[tau]) if self.count[tau] else float("nan")
            for tau in self.horizons
        }


def coalition_stats(labels: np.ndarray, prev_labels: np.ndarray | None = None) -> dict:
    """Summarise one coalition snapshot: how many, how big, how much churn."""
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    real = counts[counts > 1]  # singletons are not coalitions
    stats = {
        "n_coalitions": int(real.size),
        "largest": int(real.max()) if real.size else 0,
        "mean_size": float(real.mean()) if real.size else 0.0,
        "frac_in_coalition": float(real.sum() / labels.size),
    }
    if prev_labels is not None:
        same_prev = prev_labels[:, None] == prev_labels[None, :]
        same_now = labels[:, None] == labels[None, :]
        union = np.logical_or(same_prev, same_now).sum()
        inter = np.logical_and(same_prev, same_now).sum()
        stats["churn"] = float(1.0 - inter / max(union, 1))
    return stats


def state_entropy(h: np.ndarray, bins: int = 32) -> float:
    """Entropy of the population state distribution, projected to 1-D.

    Used by E4: a mesh that collapses to a fixed point in silence loses entropy; a mesh
    that diverges gains it without bound. Healthy free-running dynamics should hold
    roughly steady.
    """
    x = np.asarray(h)
    proj = x @ np.ones(x.shape[1]) / np.sqrt(x.shape[1])
    lo, hi = np.percentile(proj, [0.5, 99.5])
    if not np.isfinite(lo) or hi - lo < 1e-9:
        return 0.0
    counts, _ = np.histogram(proj, bins=bins, range=(lo, hi))
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())
