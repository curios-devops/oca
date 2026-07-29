"""Shared experiment runner: drive a mesh over a stream and score its predictions.

Every experiment goes through `run_stream`, so training and held-out evaluation differ
only by the `learn` flag and never by accident of harness code.
"""

from __future__ import annotations

import numpy as np

from .mesh import build_mesh, predicted_retina, tick
from core.metrics import ErrorTracker, frame_mse
from .state import Config
from core.world import Sensors


def run_stream(
    state,
    sensory: np.ndarray,
    retina: np.ndarray,
    *,
    learn: bool,
    sensors: Sensors | None = None,
    burn_in: int = 50,
    logger=None,
    tag: str = "run",
    log_every: int = 250,
) -> ErrorTracker:
    """Drive the mesh over one stream, scoring frame predictions at every horizon."""
    cfg: Config = state.cfg
    sensors = sensors or Sensors()
    state.learn = learn
    tracker = ErrorTracker(cfg.horizons)
    n = len(retina)

    for t in range(n):
        diag = tick(state, sensory[t])
        if t < burn_in:
            continue
        for tau in cfg.horizons:
            if t + tau < n:
                pred = predicted_retina(state, tau, sensors)
                tracker.add(tau, frame_mse(pred, retina[t + tau]))
        if logger is not None and log_every and t % log_every == 0:
            logger.log(kind=tag, learn=learn, **diag, **tracker.windowed())

    return tracker


def train_and_eval(
    cfg: Config,
    data: dict,
    *,
    logger=None,
    tag: str = "mesh",
    burn_in: int = 50,
    log_every: int = 250,
) -> dict:
    """Train on the training stream, then score with plasticity frozen on a held-out
    world seed. Freezing matters: otherwise the mesh keeps adapting during evaluation
    and the number stops being a generalisation measure."""
    sensors = Sensors()
    state = build_mesh(cfg)

    train = run_stream(
        state, data["train"]["sensory"], data["train"]["retina"],
        learn=True, sensors=sensors, burn_in=burn_in, logger=logger,
        tag=f"{tag}_train", log_every=log_every,
    )
    test = run_stream(
        state, data["test"]["sensory"], data["test"]["retina"],
        learn=False, sensors=sensors, burn_in=burn_in, logger=logger,
        tag=f"{tag}_test", log_every=log_every,
    )

    result = {
        "label": cfg.label(),
        "n_params": state.n_params(),
        "train_final": train.windowed(),
        "test": test.cumulative(),
        "n_clipped": state.n_clipped,
    }
    if logger is not None:
        logger.log(kind="result", **result)
    return result, state
