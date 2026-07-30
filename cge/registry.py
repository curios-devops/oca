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

_ALIASES = {"v1": "wren", "v2": "swift", "dcn": "heron", "raw": "mirror"}
"""The keys these architectures were registered under before they were named.

Every scorecard and log this project has produced is keyed `v1|prediction|12` and similar, and
several thousand lines of results reference them. Renaming the keys outright would silently
orphan that record, so the codename is now the key and the old key still resolves. A historical
log stays interpretable; new output uses the name.
"""


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

    Architectures do not agree on what a unit is: Wren and Swift have 64 retinotopic units,
    Heron has 17 nodes. Asking both "does your grouping carry object identity" therefore needs
    each to say which part of the world each of its own units is looking at, or the
    comparison silently scores them at different granularities and the finer one wins for
    being finer. Defaults to the visual-unit labelling the meshes use."""

    object_units: Callable[..., np.ndarray] | None = None
    """(state, world, object_index, side) -> indices into `readout` local to that object.

    The identity gate has to probe *near the object*, because a whole-frame probe can name
    a hidden object by elimination from the visible ones -- which looks exactly like memory
    and is not. Each version says which of its own units are local."""

    predicted_frame: Callable[..., np.ndarray] | None = None

    codename: str = ""
    """A distinct name, so three frozen architectures stop being "v1, v2 and the other one".

    Registry keys stay `v1`/`v2`/`dcn` forever -- they are written into every scorecard and
    log this project has produced, and renaming them would silently orphan the record. The
    codename is what appears in tables and figures. Corvids, because the newest architecture
    is named for the birds that pass object-permanence tests, which is the problem all of
    these failed."""

    mechanism: str = ""
    """One line: the defining computational commitment. What actually distinguishes them."""

    def new(self, seed: int = 0, side: int = 12, **kw):
        return self.build(seed=seed, side=side, **kw)


def register(version: MeshVersion) -> MeshVersion:
    _REGISTRY[version.name] = version
    return version


def get(name: str) -> MeshVersion:
    key = _ALIASES.get(name, name)
    if key not in _REGISTRY:
        raise KeyError(f"unknown architecture {name!r}; have {sorted(_REGISTRY)} "
                       f"(historical aliases: {sorted(_ALIASES)})")
    return _REGISTRY[key]


def available() -> list[str]:
    return list(_REGISTRY)


# ---------------------------------------------------------------------- wren


def _v1_build(seed: int = 0, side: int = 12, eta_head: float = 0.01, **kw):
    from architectures.wren.mesh import build_mesh
    from architectures.wren.state import Config
    return build_mesh(Config(lattice_side=side, seed=seed, eta_head=eta_head, **kw))


def _v1_tick(state, s):
    from architectures.wren.mesh import tick
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
    from architectures.wren.mesh import predicted_retina
    return predicted_retina(state, tau, sensors)


register(MeshVersion(
    name="wren",
    codename="Wren",
    mechanism="gradient flow on a learned energy landscape",
    build=_v1_build,
    tick=_v1_tick,
    readout=lambda s: s.h,
    coalitions=lambda s: s.coalition,
    describe=_v1_describe,
    unit_labels=_mesh_unit_labels,
    object_units=_mesh_object_units,
    predicted_frame=_v1_frame,
))


# --------------------------------------------------------------------- swift


def _v2_build(seed: int = 0, side: int = 12, eta_head: float = 0.01, **kw):
    from architectures.swift import Config2, build_mesh2
    return build_mesh2(Config2(lattice_side=side, seed=seed, eta_head=eta_head, **kw))


def _v2_tick(state, s):
    from architectures.swift import tick2
    return tick2(state, s)


def _v2_describe(state) -> dict:
    return {"n_units": int(state.n_units), "readout_dim": int(state.h.shape[1]),
            "n_params": int(state.n_params()), "dynamics": "limit-cycle oscillators"}


def _v2_frame(state, tau, sensors):
    from architectures.swift import predicted_retina2
    return predicted_retina2(state, tau, sensors)


register(MeshVersion(
    name="swift",
    codename="Swift",
    mechanism="Stuart-Landau limit-cycle oscillators, phase-gated coupling",
    build=_v2_build,
    tick=_v2_tick,
    readout=lambda s: s.h,
    coalitions=lambda s: s.coalition,
    describe=_v2_describe,
    unit_labels=_mesh_unit_labels,
    object_units=_mesh_object_units,
    predicted_frame=_v2_frame,
))


# ------------------------------------------------------------------------ heron
#
# Heron, under the same adapter as every other entrant. The adapter lives here, in `cge/`,
# and never inside an architecture -- that is what lets a new architecture be scored without
# either side importing the other, and it is enforced by tests/test_corvus_contract.py.


def _dcn_build(seed: int = 0, side: int = 12, **kw):
    from architectures.heron.cortex import build_cortex
    kw.pop("eta_head", None)                 # a Wren/Swift knob; Heron has no head
    return build_cortex(seed=seed, **kw)


def _dcn_tick(state, s):
    from architectures.heron.cortex import tick
    return tick(state, s)


def _dcn_frame(state, tau, sensors):
    from architectures.heron.cortex import predicted_retina
    return predicted_retina(state, tau, sensors)


def _dcn_unit_labels(world) -> np.ndarray:
    """One label per node, from the node grid that tiles the image."""
    from architectures.heron.cortex import NODE_SIDE, N_NODES
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
    from architectures.heron.cortex import NODE_SIDE, N_NODES
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
    name="heron",
    codename="Heron",
    mechanism="event-driven neurons into a reservoir with a resonance spectrum",
    build=_dcn_build,
    tick=_dcn_tick,
    readout=lambda s: s.h,
    coalitions=lambda s: s.coalition,
    describe=_dcn_describe,
    unit_labels=_dcn_unit_labels,
    object_units=_dcn_object_units,
    predicted_frame=_dcn_frame,
))


# ----------------------------------------------------------------------- corvus
#
# OCA v4, and the first entrant designed against the record rather than before it. Its Layer 1
# declares `trivial_memory` as its floor, which is gate CGE-A-01 -- the one nothing has ever
# passed.


def _corvus_build(seed: int = 0, side: int = 12, **kw):
    from architectures.corvus.cortex import build_cortex
    kw.pop("eta_head", None)                 # a Wren/Swift knob; Corvus has no head
    return build_cortex(seed=seed, **kw)


def _corvus_tick(state, s):
    from architectures.corvus.cortex import tick
    return tick(state, s)


def _corvus_frame(state, tau, sensors):
    from architectures.corvus.cortex import predicted_retina
    return predicted_retina(state, tau, sensors)


def _corvus_unit_labels(world) -> np.ndarray:
    from architectures.corvus.cortex import N_TOWERS, TOWER_SIDE
    lab = np.full(N_TOWERS, -1)
    cell = world.cfg.size / TOWER_SIDE
    for i in range(world.cfg.n_objects):
        if world.is_fully_occluded(i):
            continue
        x, y = world.pos[i]
        nc = int(np.clip(x // cell, 0, TOWER_SIDE - 1))
        nr = int(np.clip(y // cell, 0, TOWER_SIDE - 1))
        lab[nr * TOWER_SIDE + nc] = i
    return lab


def _corvus_object_units(state, world, i: int, side: int) -> np.ndarray:
    from architectures.corvus.cortex import N_TOWERS, TOWER_SIDE
    cell = world.cfg.size / TOWER_SIDE
    x, y = world.pos[i]
    nc = int(np.clip(x // cell, 0, TOWER_SIDE - 1))
    nr = int(np.clip(y // cell, 0, TOWER_SIDE - 1))
    idx = {(((nr + dr) % TOWER_SIDE) * TOWER_SIDE + ((nc + dc) % TOWER_SIDE))
           for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    return np.array(sorted(j for j in idx if j < N_TOWERS))


register(MeshVersion(
    name="corvus",
    codename="Corvus",
    mechanism="entities: a reference frame corrected when observable, propagated by a learned "
              "action model when not",
    build=_corvus_build,
    tick=_corvus_tick,
    readout=lambda s: s.h,
    coalitions=lambda s: s.coalition,
    describe=lambda s: s.describe(),
    unit_labels=_corvus_unit_labels,
    object_units=_corvus_object_units,
    predicted_frame=_corvus_frame,
))


# ----------------------------------------------------------------------- mirror
#
# The floor, registered as an entrant rather than kept as a footnote.
#
# It holds no state beyond the current frame, learns nothing, and has zero parameters. Every
# gate already computes it as a control, but computing a control and *entering* it are
# different things: as a control it is a number in a caption, and as an entrant it appears in
# the same table, in the same units, and cannot be skipped over. That matters here, because
# on this benchmark raw pixels currently beat every architecture in the project at predicting
# the world, and beat two of the three at decoding walls the agent cannot see.
#
# Anything that cannot beat this is not adding a representation. It is adding latency.


class _RawState:
    """A model with no model: the current frame, and nothing else."""

    class _Cfg:
        horizons = (1, 4, 16)
        lattice_side = 0

    def __init__(self):
        from core.world.sensors import RETINA
        self.cfg = self._Cfg()
        self.learn = True                 # accepted and ignored; there is nothing to learn
        self.frame = np.zeros((RETINA, RETINA), dtype=np.float64)
        self.coalition = None

    @property
    def h(self):
        return self.frame

    def n_params(self) -> int:
        return 0


def _raw_build(seed: int = 0, side: int = 12, **kw):
    return _RawState()


def _raw_tick(state, s):
    """Reassemble the frame from the sensory patches. No integration, no memory."""
    from core.world import Sensors
    from core.world.sensors import N_VISUAL
    state.frame = Sensors().from_patches(np.asarray(s)[:N_VISUAL]).astype(np.float64)
    return {}


def _raw_frame(state, tau, sensors):
    """Its forecast at every horizon is the current frame: copy-last, stated plainly."""
    return state.frame.astype(np.float32)


register(MeshVersion(
    name="mirror",
    codename="Mirror",
    mechanism="no state at all -- the current frame, and nothing else",
    build=_raw_build,
    tick=_raw_tick,
    readout=lambda s: s.frame,
    coalitions=lambda s: None,            # no grouping; the binding gate reports unmeasured
    describe=lambda s: {"n_units": 0, "readout_dim": int(s.frame.size),
                        "n_params": 0, "dynamics": "none — the current frame"},
    unit_labels=_mesh_unit_labels,
    object_units=lambda state, world, i, side: np.arange(state.frame.shape[0]),
    predicted_frame=_raw_frame,
))
