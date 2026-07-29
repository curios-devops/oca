"""Rollout generation. Every experiment draws its data through this module so the
mesh and the baselines see byte-identical streams.
"""

from __future__ import annotations

import numpy as np

from .world import GridWorld, Sensors, WorldConfig


def rollout(
    n_steps: int,
    seed: int = 0,
    world_cfg: WorldConfig | None = None,
    burn_in: int = 20,
    world_factory=None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Run the world forward and transduce it.

    Returns (retina (n_steps, R, R), sensory (n_steps, N_SENSORY, P), snapshots).
    Snapshots carry ground-truth object positions and occlusion flags -- used only by
    the analysis probes, never fed to any model.

    `world_factory(seed) -> world` selects a different world (see world/physics.py);
    any object exposing GridWorld's step/render/contact interface works.
    """
    if world_factory is not None:
        world = world_factory(seed)
    else:
        cfg = world_cfg if world_cfg is not None else WorldConfig(seed=seed)
        if world_cfg is not None:
            cfg = WorldConfig(**{**cfg.__dict__, "seed": seed})
        world = GridWorld(cfg)
    sensors = Sensors()

    for _ in range(burn_in):
        world.step()

    rets, sens, snaps = [], [], []
    for _ in range(n_steps):
        s, r = sensors.observe(world)
        rets.append(r)
        sens.append(s)
        snaps.append(world.state_snapshot())
        world.step()

    return np.stack(rets), np.stack(sens), snaps


def train_test_streams(
    n_train: int,
    n_test: int,
    train_seed: int = 0,
    test_seed: int = 9_000,
    world_cfg: WorldConfig | None = None,
    world_factory=None,
) -> dict:
    """Held-out evaluation uses a different world seed, not a later slice of the same
    trajectory -- otherwise a model could win by memorising this particular rollout."""
    kw = dict(world_cfg=world_cfg, world_factory=world_factory)
    tr_ret, tr_sen, tr_snap = rollout(n_train, seed=train_seed, **kw)
    te_ret, te_sen, te_snap = rollout(n_test, seed=test_seed, **kw)
    return {
        "train": {"retina": tr_ret, "sensory": tr_sen, "snaps": tr_snap},
        "test": {"retina": te_ret, "sensory": te_sen, "snaps": te_snap},
    }
