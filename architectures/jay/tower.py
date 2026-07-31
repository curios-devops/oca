"""Layer 1 — the Tower. OCA v5, build `jay-v5.2`.

`jay-v5.1` bound a feature to a *place* and read out by the distance between places. It was
refuted directly, without a probe: two different objects looked more alike (0.587) than one
object at two orientations (0.541). The full entry is in
`architecture-history/JAY_L1_BINDING.md`, and its conclusion is the premise of this file:

> **A reference frame needs an orientation as well as a position. That tower had only a
> position.**

Distance-binning removes a *global* rotation of the place field. It cannot remove a rotation that
has already changed what each feature is and where each feature sits relative to the others,
because the (feature, place) pairs a rotated object produces are not the rotated pairs of the
original — they are different pairs.

## What changes, and it is one thing

The tower now **derives its own reference direction from what it has seen**, and expresses every
location relative to it.

**Places are polar.** A location is `(r, phi)` about the observed mass centroid rather than a
point on a Cartesian lattice, so a rotation of the object is a *shift along one axis* of the
place code instead of an arbitrary permutation of cells.

**The reference direction is measured, not given.** From the accumulated mass map, weighted by
radius: the angular bin where the object extends furthest. That direction is **covariant** —
rotate the object and it rotates with it — which is exactly the property v5.1 had nothing of.

**Canonicalisation is a circular shift.** Rotating the angular axis so the reference direction
lands at bin zero makes the readout invariant by construction, exactly, at the bin resolution.
The pose world's orientations are 60 degrees apart and the bins are 30, so a pose change is a
shift of exactly two bins and nothing is resampled.

Under an object rotation of `delta`: every `phi` shifts by `delta`, the reference direction
shifts by `delta`, and the canonical map does not move. That is the whole mechanism.

## What it must beat, and the ablation that decides it

`canonicalise=False` is the new control and the important one: identical code, identical
capacity, no reference direction. **That is v5.1's failure mode reproduced inside v5.2**, so if
the two score the same then orientation was never the missing piece and this build is refuted the
way the last one was.

`use_binding=False` and `shuffle_locations` carry over unchanged.

**The invariance diagnostic runs first.** It answers in seconds, from the readouts alone, whether
the same object at two poses looks more alike than two objects at one pose. An hour of probe
fitting cannot rescue a representation that fails it, and v5.1 cost an hour finding that out.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import neuron as L0
from .contract import Floor, Layer, register


@dataclass(frozen=True)
class TowerConfig:
    n_towers: int = 17
    n_inputs: int = 16
    n_neurons: int = 64
    n_proprio: int = 80

    d_feat: int = 12
    """Width of the feature code a tower extracts from what its neurons published."""

    n_r: int = 5
    n_a: int = 12
    """Radial and angular bins of the polar place code. Twelve angular bins is 30 degrees each,
    and the pose world's orientations are 60 apart -- so a pose change is a shift of exactly two
    bins and canonicalisation is exact rather than approximate."""

    place_sigma_r: float = 0.30
    place_sigma_a: float = 0.9
    """Tuning widths, in bins. Wide enough that a location between two bins reaches both."""

    d_disp: int = 2
    frame_extent: float = 6.0
    """Half-range of the accumulated displacement in steps: the fovea traverses about six per
    axis, and the place field spans a unit radius."""

    horizon: int = 16

    canonicalise: bool = True
    """**The mechanism under test.** `False` skips the reference direction and leaves the map in
    the sensor's own frame -- which is v5.1's failure mode, reproduced as this build's control."""

    use_binding: bool = True
    """`False` stores the feature sum at every place, at identical capacity."""

    shuffle_locations: bool = False
    """Control: the place is re-drawn from another tower every tick, destroying the binding
    between a feature and where it was seen and keeping everything else."""

    decay: float = 0.999
    seed: int = 0

    def variant(self, **changes) -> "TowerConfig":
        return replace(self, **changes)

    @property
    def d_place(self) -> int:
        return self.n_r * self.n_a

    @property
    def d_readout(self) -> int:
        return self.d_feat * self.d_place


@dataclass
class TowerStack:
    cfg: TowerConfig
    rng: np.random.Generator
    neurons: L0.Population
    P: np.ndarray                 # (d_feat, n_neurons)   fixed sensory encoder
    M: np.ndarray                 # (2, n_proprio)        the body's action -> step map
    disp: np.ndarray              # (n_towers, 2)         accumulated displacement
    bound: np.ndarray             # (n_towers, d_feat, n_r, n_a)  feature at polar place
    mass: np.ndarray              # (n_towers,)
    centroid: np.ndarray          # (n_towers, 2)
    t: int = 0
    learn: bool = True

    @property
    def n_units(self) -> int:
        return self.cfg.n_towers

    def member_state(self) -> np.ndarray:
        return self.neurons.last_sent.reshape(self.cfg.n_towers, self.cfg.n_neurons)

    def n_params(self) -> int:
        return int(self.neurons.n_params() + self.P.size + self.M.size)

    def reference_direction(self) -> np.ndarray:
        """(n_towers,) — the angular bin the object extends furthest into.

        **Covariant with the object**, which is the property this build exists for: rotate the
        object and this rotates with it. Weighted by radius because the outermost mass is what
        distinguishes one orientation from another, and an inner blob is nearly uncommitted.
        """
        r_w = np.arange(1, self.cfg.n_r + 1, dtype=np.float64)
        profile = np.einsum("nfra,r->na", self.bound, r_w)         # (n, n_a)
        return np.argmax(profile, axis=1)

    def readout(self) -> np.ndarray:
        """(n_towers, d_readout) — the bound map, rotated into its own reference frame.

        A circular shift, so nothing is resampled and the invariance is exact at bin resolution.
        """
        cfg = self.cfg
        out = self.bound
        if cfg.canonicalise:
            ref = self.reference_direction()
            out = np.stack([np.roll(out[n], -int(ref[n]), axis=-1) for n in range(len(out))])
        out = out.reshape(len(out), -1)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)

    @property
    def h(self) -> np.ndarray:
        return self.readout()

    def describe(self) -> dict:
        return {"n_towers": self.cfg.n_towers, "d_feat": self.cfg.d_feat,
                "n_r": self.cfg.n_r, "n_a": self.cfg.n_a, "d_readout": self.cfg.d_readout,
                "canonicalise": self.cfg.canonicalise, "binding": self.cfg.use_binding,
                "n_params": self.n_params(),
                "mechanism": "feature bound to a polar place, rotated into a reference "
                             "direction the tower derives from its own observations"}


def _proprio_to_delta(n_proprio: int) -> np.ndarray:
    """(2, n_proprio) — the body's own map from an efference copy to the step it produced.

    **Body knowledge, not cognition**, and the distinction is load-bearing. A random projection
    is not a position: the efference is a one-hot over {up, down, left, right, hold}, and nothing
    in a random matrix makes up and down cancel. An organism is not required to *discover* that
    its own step cancels its opposite; its proprioception delivers that. What it must discover is
    what is out there.
    """
    from core.world.pose import ACTIONS
    from core.world.sensors import P as PATCH_WIDTH
    M = np.zeros((2, n_proprio))
    for a, (dy, dx) in enumerate(ACTIONS):
        row, col = divmod(a, 4)
        unit = (row // 2) * 2 + (col // 2)
        # `somatic()` masks the full 16-vector per unit rather than compacting the quadrant, so
        # within a unit the offset is the contact index itself. Computing a within-quadrant
        # offset instead put left and right on indices that are always zero, and horizontal
        # movement was never registered at all.
        idx = unit * PATCH_WIDTH + a
        if idx < n_proprio:
            M[0, idx], M[1, idx] = dy, dx
    return M


def build_stack(cfg: TowerConfig | None = None) -> TowerStack:
    cfg = cfg or TowerConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_towers
    neurons = L0.build(L0.NeuronConfig(seed=cfg.seed, n_inputs=cfg.n_inputs,
                                       n_neurons=cfg.n_neurons), n_groups=n)
    return TowerStack(
        cfg=cfg, rng=rng, neurons=neurons,
        P=rng.choice([-1.0, 1.0], (cfg.d_feat, cfg.n_neurons)) / np.sqrt(cfg.n_neurons),
        M=_proprio_to_delta(cfg.n_proprio),
        disp=np.zeros((n, 2)), bound=np.zeros((n, cfg.d_feat, cfg.n_r, cfg.n_a)),
        mass=np.zeros(n), centroid=np.zeros((n, 2)),
    )


def _polar_place(stack: TowerStack, rel: np.ndarray) -> np.ndarray:
    """(n_towers, 2) object-centred position -> (n_towers, n_r, n_a) polar place code.

    Polar rather than Cartesian so that an object rotation is a **shift along one axis** of the
    code, which a circular roll can undo exactly. On a Cartesian lattice a rotation is an
    arbitrary permutation of cells and nothing can undo it without resampling.
    """
    cfg = stack.cfg
    if cfg.shuffle_locations:
        rel = stack.rng.permutation(rel, axis=0)
    r = np.linalg.norm(rel, axis=1)
    a = np.arctan2(rel[:, 1], rel[:, 0]) % (2 * np.pi)

    r_c = (np.arange(cfg.n_r) + 0.5) / cfg.n_r
    gr = np.exp(-((r[:, None] - r_c[None]) ** 2) / (2 * cfg.place_sigma_r ** 2))

    a_c = (np.arange(cfg.n_a) + 0.5) * 2 * np.pi / cfg.n_a
    da = np.abs(((a[:, None] - a_c[None]) + np.pi) % (2 * np.pi) - np.pi)
    ga = np.exp(-(da ** 2) / (2 * (cfg.place_sigma_a * 2 * np.pi / cfg.n_a) ** 2))

    g = gr[:, :, None] * ga[:, None, :]
    return g / (g.sum(axis=(1, 2), keepdims=True) + 1e-9)


def new_episode(stack: TowerStack) -> None:
    """Clear the map. An object model belongs to one presentation."""
    stack.bound[:] = 0.0
    stack.disp[:] = 0.0
    stack.mass[:] = 0.0
    stack.centroid[:] = 0.0


def step(stack: TowerStack, visual: np.ndarray, proprio: np.ndarray) -> dict:
    cfg = stack.cfg
    L0.step(stack.neurons, np.asarray(visual, dtype=np.float64))

    feat = np.maximum(stack.member_state() @ stack.P.T, 0.0)          # (n, d_feat)
    m = feat.sum(axis=1)

    stack.disp += (stack.M @ np.asarray(proprio, dtype=np.float64))[:2][None, :]

    # object-centred: where am I relative to the mass I have actually seen
    stack.mass += m
    w = (m / (stack.mass + 1e-9))[:, None]
    stack.centroid += w * (stack.disp - stack.centroid)
    rel = (stack.disp - stack.centroid) / cfg.frame_extent

    place = _polar_place(stack, rel)                                  # (n, n_r, n_a)
    stack.bound *= cfg.decay
    if cfg.use_binding:
        stack.bound += np.einsum("nf,nra->nfra", feat, place)
    else:
        stack.bound += feat[:, :, None, None] / cfg.d_place

    stack.t += 1
    return {"t": stack.t, "mass": float(m.mean()),
            "disp": float(np.abs(stack.disp).mean())}


register(Layer(
    name="tower",
    horizon=TowerConfig().horizon,
    inputs_from="neuron",
    floor=(
        Floor(
            job="object_identity", always_on=True,
            beats="no_canonical_orientation", margin=0.05,
            why="The v5.1 refutation, reproduced as this build's own control. That tower had a "
                "position and no orientation and its readout was more sensitive to pose than to "
                "identity. If canonicalising changes nothing, orientation was not the missing "
                "piece and this build is refuted the same way.",
        ),
        Floor(
            job="object_identity", always_on=True,
            beats="nearest_template_on_held_out_poses", margin=0.05,
            why="Measured at 0.158 on held-out poses against a chance of 0.250 -- memorised "
                "appearances do not merely fail to transfer, they mislead. CGE-A-00 in the first "
                "setting this project has had where the input cannot simply answer.",
        ),
        Floor(
            job="object_identity", always_on=True,
            beats="raw_frame", margin=0.05,
            why="The standing floor of the whole project. Five entrants, none has cleared it.",
        ),
    ),
    build=lambda seed=0, **kw: build_stack(TowerConfig(seed=seed, **kw)),
    step=lambda s, u: step(s, u[0], u[1]),
    readout=lambda s: s.readout(),
    describe=lambda s: s.describe(),
))
