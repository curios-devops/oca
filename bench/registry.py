"""A common interface over mesh versions, so any two can be compared on equal terms.

Three experiments had grown their own copy of `(tick2 if kind == "v2" else tick)`, which
is fine for two versions and becomes a liability at three. More importantly, a comparison
is only worth anything if every version meets every gate through the *same* code path --
otherwise a difference in the table can always be a difference in the harness.

Registering a new version is one decorator and four methods. Meshes are always rebuilt
from a seed rather than loaded from disk: it costs a few seconds, and it means a
scorecard can never silently describe a checkpoint that no longer matches the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

_REGISTRY: dict[str, "MeshVersion"] = {}


@dataclass
class MeshVersion:
    """Everything a gate needs from a mesh, and nothing about how it works inside."""

    name: str
    build: Callable[..., object]
    tick: Callable[[object, np.ndarray], object]
    readout: Callable[[object], np.ndarray]
    coalitions: Callable[[object], np.ndarray | None]
    describe: Callable[[object], dict]

    unit_labels: Callable[[object], np.ndarray] | None = None
    """World -> one object label per element of `coalitions`, or -1 for "sees nothing".

    Architectures do not agree on what a unit is: a mesh has 64 retinotopic units, a DCN
    has 17 nodes. Asking both "does your grouping carry object identity" therefore needs
    each to say which part of the world each of its own units is looking at, or the
    comparison silently scores them at different granularities and the finer one wins for
    being finer. Defaults to the visual-unit labelling the meshes use."""

    object_units: Callable[..., np.ndarray] | None = None
    """(state, world, object_index, side) -> indices into `readout` local to that object.

    The identity gate has to probe *near the object*, because a whole-frame probe can name
    a hidden object by elimination from the visible ones -- which looks exactly like memory
    and is not. Each version says which of its own units are local."""

    predicted_frame: Callable[..., np.ndarray] | None = None

    def new(self, seed: int = 0, side: int = 12, **kw):
        return self.build(seed=seed, side=side, **kw)


def register(version: MeshVersion) -> MeshVersion:
    _REGISTRY[version.name] = version
    return version


def get(name: str) -> MeshVersion:
    if name not in _REGISTRY:
        raise KeyError(f"unknown mesh version {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return list(_REGISTRY)


# --------------------------------------------------------------------------- v1


def _v1_build(seed: int = 0, side: int = 12, eta_head: float = 0.01, **kw):
    from legacy.v1.mesh import build_mesh
    from legacy.v1.state import Config
    return build_mesh(Config(lattice_side=side, seed=seed, eta_head=eta_head, **kw))


def _v1_tick(state, s):
    from legacy.v1.mesh import tick
    return tick(state, s)


def _v1_describe(state) -> dict:
    return {"n_units": int(state.n_units), "readout_dim": int(state.h.shape[1]),
            "n_params": int(state.n_params()), "dynamics": "gradient flow"}


def _mesh_unit_labels(world) -> np.ndarray:
    from exp07_stage2 import unit_object_labels
    return unit_object_labels(world, world.cfg.n_objects)


def _mesh_object_units(state, world, i: int, side: int) -> np.ndarray:
    from exp08_identity import local_units
    return local_units(world, i, side)[1]


def _v1_frame(state, tau, sensors):
    from legacy.v1.mesh import predicted_retina
    return predicted_retina(state, tau, sensors)


register(MeshVersion(
    name="v1",
    build=_v1_build,
    tick=_v1_tick,
    readout=lambda s: s.h,
    coalitions=lambda s: s.coalition,
    describe=_v1_describe,
    unit_labels=_mesh_unit_labels,
    object_units=_mesh_object_units,
    predicted_frame=_v1_frame,
))


# --------------------------------------------------------------------------- v2


def _v2_build(seed: int = 0, side: int = 12, eta_head: float = 0.01, **kw):
    from legacy.v2 import Config2, build_mesh2
    return build_mesh2(Config2(lattice_side=side, seed=seed, eta_head=eta_head, **kw))


def _v2_tick(state, s):
    from legacy.v2 import tick2
    return tick2(state, s)


def _v2_describe(state) -> dict:
    return {"n_units": int(state.n_units), "readout_dim": int(state.h.shape[1]),
            "n_params": int(state.n_params()), "dynamics": "limit-cycle oscillators"}


def _v2_frame(state, tau, sensors):
    from legacy.v2 import predicted_retina2
    return predicted_retina2(state, tau, sensors)


register(MeshVersion(
    name="v2",
    build=_v2_build,
    tick=_v2_tick,
    readout=lambda s: s.h,
    coalitions=lambda s: s.coalition,
    describe=_v2_describe,
    unit_labels=_mesh_unit_labels,
    object_units=_mesh_object_units,
    predicted_frame=_v2_frame,
))


# -------------------------------------------------------------------------- dcn
#
# Architecture v2 under the same adapter as the frozen line. The adapter lives here, in
# `bench/`, and not in `dcn/` -- that is what lets a new architecture be scored without
# either side importing the other, and it is enforced by tests/test_dcn_contract.py.


def _dcn_build(seed: int = 0, side: int = 12, **kw):
    from dcn.cortex import build_cortex
    kw.pop("eta_head", None)                 # a legacy knob; the DCN has no head
    return build_cortex(seed=seed, **kw)


def _dcn_tick(state, s):
    from dcn.cortex import tick
    return tick(state, s)


def _dcn_frame(state, tau, sensors):
    from dcn.cortex import predicted_retina
    return predicted_retina(state, tau, sensors)


def _dcn_unit_labels(world) -> np.ndarray:
    """One label per node, from the node grid that tiles the image."""
    from dcn.cortex import NODE_SIDE, N_NODES
    lab = np.full(N_NODES, -1)
    cell = world.cfg.size / NODE_SIDE
    for i in range(world.cfg.n_objects):
        if world.is_fully_occluded(i):
            continue
        x, y = world.pos[i]
        nc = int(np.clip(x // cell, 0, NODE_SIDE - 1))
        nr = int(np.clip(y // cell, 0, NODE_SIDE - 1))
        lab[nr * NODE_SIDE + nc] = i
    return lab


def _dcn_object_units(state, world, i: int, side: int) -> np.ndarray:
    """The node containing the object, and its grid neighbours."""
    from dcn.cortex import NODE_SIDE, N_NODES
    cell = world.cfg.size / NODE_SIDE
    x, y = world.pos[i]
    nc = int(np.clip(x // cell, 0, NODE_SIDE - 1))
    nr = int(np.clip(y // cell, 0, NODE_SIDE - 1))
    idx = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = (nr + dr) % NODE_SIDE, (nc + dc) % NODE_SIDE
            idx.append(rr * NODE_SIDE + cc)
    return np.array(sorted(set(i for i in idx if i < N_NODES)))


def _dcn_describe(state) -> dict:
    d = state.describe()
    d["dynamics"] = "reservoir + resonance spectrum"
    d["readout_dim"] = int(state.h.shape[1])
    d["n_units"] = int(state.n_units)
    return d


register(MeshVersion(
    name="dcn",
    build=_dcn_build,
    tick=_dcn_tick,
    readout=lambda s: s.h,
    coalitions=lambda s: s.coalition,
    describe=_dcn_describe,
    unit_labels=_dcn_unit_labels,
    object_units=_dcn_object_units,
    predicted_frame=_dcn_frame,
))
