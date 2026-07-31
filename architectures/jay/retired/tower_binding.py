"""Layer 1 — the Tower, feature-at-place binding. **REFUTED 2026-07-31. Not in the live stack.**

Measured in the pose world, three seeds, with the invariance diagnostic that needs no probe:

    same object across poses        0.541
    different objects, same pose    0.587
    margin                         -0.047     NOT POSE-INVARIANT

**Two different objects look more alike than one object at two orientations.** The readout is
more sensitive to pose than to identity, which is exactly backwards, and no probe fitted on top
can repair it. Accuracy agrees: 0.209 on held-out poses and 0.255 on trained ones, against a
chance of 0.250 -- at chance even on orientations it was shown, while the raw frame reaches
0.442 and nearest-template 0.925.

All three declared controls tie with the mechanism (-0.004, +0.004, -0.050), so the kill
criterion in `docs/jay/SPEC_L1_TOWER.md` applies exactly as written.

**Why, and it is structural rather than a bug.** The feature code is a random projection of raw
patch activations and is **not rotation-covariant**. Binning away the rotation of *places* leaves
the rotation of *features* untouched, so the (feature, place) pairs a rotated object produces are
not the rotated pairs of the original -- they are different pairs. A reference frame needs an
orientation as well as a position, and this tower has only a position.

Kept executable so `experiments/jay_l1.py` still reproduces those numbers.

Original documentation follows.

---

Layer 1 — the Tower. OCA v5.

Corvus's tower held one anchor: *where am I*. It passes `CGE-A-09` at +0.572 and says nothing
about *what is there*, and the aggregate-then-vote experiment showed the cost — one tower and
nine towers scored the same 0.60 on object kind against 0.782 in the pixels, so the information
was already gone at this layer and nothing above could recover it.

This tower **binds a feature to a place**, which is the one operation this project has never
tried. Pooling has failed six times; binding has not been attempted.

Three steps, and each earns its place against a declared control.

**1. Displacement, integrated from the efference copy.** The efference is read through the body's
own map from an action to the step it produced -- **body knowledge, not cognition**, and the
distinction is load-bearing. Corvus used a *random* injective projection here and that is fine
when the question is whether displacement is recoverable (A-09 decodes it with a probe). It is
not fine when displacement is an *input* to an object model: a random projection of a one-hot
does not make up and down cancel, so the accumulator does not track the sensor. Measured, that
put every variant of this layer at chance even on poses it had seen.

**2. An object-centred frame.** The accumulator gives a *self*-centred position. Subtracting the
running centroid of where feature mass was actually observed makes it **object**-centred, and
therefore invariant to where the sensor happened to start. This is the step Corvus does not have
and the reason its tower could not represent an object.

**3. A feature-at-location map, read out through pairwise distances.** `M[f, l] += feature f at
place l`. Translation invariance comes from step 2; **rotation invariance comes from the
readout**, because distances between the places where features were seen do not change when the
object turns -- so the readout is *binned by distance*, never indexed by which cell. A Gram
matrix over cell identities is not pose-invariant at all, and building one was this file's third
defect.

The pose world was built so that this is exactly the quantity that separates its confusable
pair: kinds 0 and 1 share their angle multiset *and* their radius multiset and differ only in the
pairing, so their pairwise distances differ and nothing weaker tells them apart.

**A designed readout needs its controls, and it has three.** `use_binding=False` stores the
feature sum instead of the map, at identical capacity -- if that scores the same, binding is
decoration. `shuffle_locations=True` binds the same features to permuted places, destroying
arrangement and keeping everything else. `features_only=True` takes distances over every visited
place regardless of feature, testing whether the binding matters or only the sensor's trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .. import neuron as L0
from ..contract import Floor, Layer, register


@dataclass(frozen=True)
class TowerConfig:
    n_towers: int = 17
    n_inputs: int = 16
    n_neurons: int = 64
    n_proprio: int = 80

    d_feat: int = 12
    """Width of the feature code a tower extracts from what its neurons published."""

    n_bins: int = 14
    """Distance bins in the readout. The pose-invariant quantity is *how far apart* two occupied
    places are, not which cells they were."""

    d_place: int = 24
    """Number of place cells tiling the object-centred frame. A distributed code, so a location
    that was never visited still overlaps the ones that were."""

    place_sigma: float = 0.22
    """Tuning width as a fraction of the frame's extent. Wide enough that the code generalises
    across nearby locations, narrow enough that distant ones stay separable."""

    d_disp: int = 2
    frame_extent: float = 6.0
    """Half-range of the accumulated displacement, in steps, used to normalise it into the place
    code. The fovea traverses `(canvas - size) / fovea_step` steps per axis -- about six here --
    so the accumulator ranges roughly +/-6 and the place field spans [-1, 1].

    Set to 30 in the first version, which put the entire trajectory inside 24% of the place
    field and collapsed it onto a handful of central cells. A normalisation constant that hides
    the signal it normalises is a defect whether or not a gate exists."""

    horizon: int = 16

    use_binding: bool = True
    """The mechanism under test. `False` stores the feature sum at identical capacity -- the
    ablation that decides whether binding is doing the work."""

    shuffle_locations: bool = False
    """Control: bind the same features to a place drawn from another tower, re-drawn every tick.
    Destroys the feature-to-place binding and keeps everything else. Should collapse to chance."""

    features_only: bool = False
    """Control: distances over every visited place regardless of which feature was there."""

    decay: float = 0.999
    """The map forgets slowly, so it spans an episode without spanning two of them."""

    seed: int = 0

    def variant(self, **changes) -> "TowerConfig":
        return replace(self, **changes)

    @property
    def d_readout(self) -> int:
        return self.d_feat * self.d_feat * self.n_bins


@dataclass
class TowerStack:
    cfg: TowerConfig
    rng: np.random.Generator
    neurons: L0.Population
    P: np.ndarray                 # (d_feat, n_neurons)  fixed sensory encoder
    M: np.ndarray                 # (d_disp, n_proprio)  fixed injective action projection
    centres: np.ndarray           # (d_place, 2)         place-cell centres
    bins: np.ndarray              # (n_bins, d_place, d_place)  pair -> distance bin, one-hot
    perm: np.ndarray              # (d_place,)           the shuffle control's permutation
    disp: np.ndarray              # (n_towers, 2)        accumulated displacement
    bound: np.ndarray             # (n_towers, d_feat, d_place)  the feature-at-location map
    mass: np.ndarray              # (n_towers,)          total feature mass seen this episode
    centroid: np.ndarray          # (n_towers, 2)        running centroid of observed mass
    t: int = 0
    learn: bool = True

    @property
    def n_units(self) -> int:
        return self.cfg.n_towers

    def member_state(self) -> np.ndarray:
        return self.neurons.last_sent.reshape(self.cfg.n_towers, self.cfg.n_neurons)

    def n_params(self) -> int:
        return int(self.neurons.n_params() + self.P.size + self.M.size + self.centres.size)

    def readout(self) -> np.ndarray:
        """(n_towers, d_readout) — the pose-invariant summary of the bound map.

        **Binned by the distance between places, not indexed by which places.** Rotating the
        object moves every feature to a different place cell, so a Gram matrix over cell
        *identities* changes completely -- it is not invariant to pose at all. What survives a
        rotation is how far apart the occupied places are.

        This file's first version documented distances and implemented the Gram, which is the
        difference between a mechanism that can work and one that cannot. Measured, the Gram
        version sat at chance on trained poses as well as held-out ones.

        `R[f, g, b]` = how much feature `f` and feature `g` co-occurred at places separated by
        distance bin `b`. That is exactly the quantity the pose world was built to require: its
        confusable pair shares its angle multiset *and* its radius multiset, so only the
        distances between features tell the two apart.
        """
        cfg = self.cfg
        src = (self.bound.sum(axis=1, keepdims=True) if cfg.features_only else self.bound)
        out = np.stack([np.einsum("nfp,pq,ngq->nfg", src, self.bins[b], src)
                        for b in range(cfg.n_bins)], axis=-1)
        out = out.reshape(len(src), -1)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)

    @property
    def h(self) -> np.ndarray:
        return self.readout()

    def describe(self) -> dict:
        return {"n_towers": self.cfg.n_towers, "d_feat": self.cfg.d_feat,
                "d_place": self.cfg.d_place, "d_readout": self.cfg.d_readout,
                "binding": self.cfg.use_binding, "n_params": self.n_params(),
                "mechanism": "feature bound to object-centred place; "
                             "pairwise-distance readout, invariant to pose"}


def build_stack(cfg: TowerConfig | None = None) -> TowerStack:
    cfg = cfg or TowerConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_towers
    neurons = L0.build(L0.NeuronConfig(seed=cfg.seed, n_inputs=cfg.n_inputs,
                                       n_neurons=cfg.n_neurons), n_groups=n)
    # place cells on a jittered lattice: a regular grid would alias with the fovea's step size
    k = int(np.ceil(np.sqrt(cfg.d_place)))
    gy, gx = np.meshgrid(np.linspace(-1, 1, k), np.linspace(-1, 1, k), indexing="ij")
    cent = np.c_[gy.ravel(), gx.ravel()][:cfg.d_place]
    cent = cent + rng.normal(0, 0.08, cent.shape)
    assert cfg.d_disp == cent.shape[1], "place cells live in the displacement space"
    # pair -> distance bin. Precomputed because the readout is taken once per presentation and
    # the binning is the only thing standing between this and a pose-dependent Gram matrix.
    d = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=-1)
    edges = np.linspace(0.0, d.max() + 1e-9, cfg.n_bins + 1)
    bins = np.zeros((cfg.n_bins, cfg.d_place, cfg.d_place))
    which = np.clip(np.digitize(d, edges) - 1, 0, cfg.n_bins - 1)
    for b in range(cfg.n_bins):
        bins[b] = (which == b).astype(float)

    return TowerStack(
        cfg=cfg, rng=rng, neurons=neurons, bins=bins,
        P=rng.choice([-1.0, 1.0], (cfg.d_feat, cfg.n_neurons)) / np.sqrt(cfg.n_neurons),
        M=_proprio_to_delta(cfg.n_proprio),
        centres=cent,
        perm=rng.permutation(cfg.d_place),
        disp=np.zeros((n, 2)), bound=np.zeros((n, cfg.d_feat, cfg.d_place)),
        mass=np.zeros(n), centroid=np.zeros((n, 2)),
    )


def _proprio_to_delta(n_proprio: int) -> np.ndarray:
    """(2, n_proprio) — the body's own map from an efference copy to the step it produced.

    **This is body knowledge, not cognition, and the distinction is load-bearing.** A random
    projection here is not a position: the efference is a one-hot over {up, down, left, right,
    hold}, and a random 2-of-8 slice of a projection of *action counts* is generically not a
    function of where the sensor is. Up and down have to cancel, and nothing in a random matrix
    makes them.

    Measured, that defect put every variant of this layer at chance -- 0.242 on poses it had
    seen, where chance is 0.250 -- because the place code was indexing a quantity that did not
    track the sensor.

    An organism is not required to *discover* that its own step cancels its opposite; its
    proprioception delivers that. What it must discover is what is out there, and the
    `self_localisation` floor still has to be earned separately.

    The follow-up this world newly makes possible: the fovea moves over a *static* object, so
    view change is a consistent function of action here in a way it never was in the maze --
    where Corvus's v4.1 reached |A| = 0.0013 trying to learn exactly this. Learning the map
    instead of being given it is now plausible for the first time, and is the next question.
    """
    from core.world.pose import ACTIONS
    from core.world.sensors import P as PATCH_WIDTH
    M = np.zeros((2, n_proprio))
    # `Sensors.somatic` spreads the 16-value contact map across 4 units of PATCH_WIDTH each,
    # unit k carrying quadrant k. The efference one-hot sits in contact[:len(ACTIONS)], so
    # action a lands at a known offset in the flattened proprioceptive vector.
    for a, (dy, dx) in enumerate(ACTIONS):
        row, col = divmod(a, 4)                       # position in the 4x4 contact map
        unit = (row // 2) * 2 + (col // 2)            # which somatic unit carries that quadrant
        # `somatic()` masks the FULL 16-vector per unit rather than compacting the quadrant, so
        # within a unit the offset is the contact index itself. Computing a within-quadrant
        # offset instead -- which the first version did -- put left and right on indices that
        # are always zero, and **horizontal movement was never registered at all**. Every
        # variant then sat at chance, on held-out and trained poses alike, and the run would
        # have been published as a refutation of binding.
        idx = unit * PATCH_WIDTH + a
        if idx < n_proprio:
            M[0, idx], M[1, idx] = dy, dx
    return M


def _place_code(stack: TowerStack, rel: np.ndarray) -> np.ndarray:
    """(n_towers, 2) object-centred position -> (n_towers, d_place) distributed place code."""
    cfg = stack.cfg
    if cfg.shuffle_locations:
        # **Re-drawn every tick.** A *fixed* permutation of the place cells was the first
        # version of this control and it was no control at all: the readout is a set of
        # pairwise products, so a fixed permutation only reorders the readout's dimensions,
        # which any probe that learns a weight per dimension cannot see. Measured, it scored
        # identically to the mechanism to three decimals. Re-drawing each tick is what actually
        # destroys the binding between a feature and a place.
        rel = stack.rng.permutation(rel, axis=0)
    d2 = ((rel[:, None, :] - stack.centres[None]) ** 2).sum(-1)
    g = np.exp(-d2 / (2.0 * cfg.place_sigma ** 2))
    return g / (g.sum(axis=1, keepdims=True) + 1e-9)


def new_episode(stack: TowerStack) -> None:
    """Clear the map. An object model belongs to one presentation; carrying it across two would
    be measuring the sequence rather than the object."""
    stack.bound[:] = 0.0
    stack.disp[:] = 0.0
    stack.mass[:] = 0.0
    stack.centroid[:] = 0.0


def step(stack: TowerStack, visual: np.ndarray, proprio: np.ndarray) -> dict:
    """One tick. `visual` is (n_towers, n_inputs); `proprio` is the efference copy."""
    cfg = stack.cfg
    L0.step(stack.neurons, np.asarray(visual, dtype=np.float64))

    # 1. what is here -- a feature code from what the neurons published, never their activation
    feat = np.maximum(stack.member_state() @ stack.P.T, 0.0)      # (n, d_feat)
    m = feat.sum(axis=1)                                          # (n,) feature mass

    # 2. where am I -- displacement integrated from the efference copy alone. Nothing in the
    #    image says how far the sensor moved, because the object does not move.
    d = (stack.M @ np.asarray(proprio, dtype=np.float64))[:2]
    stack.disp += d[None, :]

    # 3. where am I *on the object* -- self-centred position minus the running centroid of where
    #    mass was actually seen. This is what makes the map invariant to where the sensor began.
    stack.mass += m
    w = (m / (stack.mass + 1e-9))[:, None]
    stack.centroid += w * (stack.disp - stack.centroid)
    rel = (stack.disp - stack.centroid) / cfg.frame_extent

    # 4. bind: this feature, at this place
    place = _place_code(stack, rel)                               # (n, d_place)
    stack.bound *= cfg.decay
    if cfg.use_binding:
        stack.bound += np.einsum("nf,np->nfp", feat, place)
    else:
        # the ablation: same features, same capacity, no place. Every place gets the same
        # feature sum, so arrangement cannot survive.
        stack.bound += feat[:, :, None] / cfg.d_place

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
            beats="nearest_template_on_held_out_poses", margin=0.05,
            why="Measured at 0.158 on held-out poses against a chance of 0.250 -- memorised "
                "appearances do not merely fail to transfer, they mislead. This is CGE-A-00 in "
                "the first setting this project has had where the input cannot simply answer.",
        ),
        Floor(
            job="object_identity", always_on=True,
            beats="raw_frame", margin=0.05,
            why="The standing floor of the whole project. Five entrants, none has cleared it.",
        ),
        Floor(
            job="self_localisation", always_on=True,
            beats="no_integration", margin=0.05,
            why="Carried from CGE-A-09, which Corvus passes at +0.572. A v5 tower that loses "
                "this has traded a working mechanism for a hypothesis.",
        ),
    ),
    build=lambda seed=0, **kw: build_stack(TowerConfig(seed=seed, **kw)),
    step=lambda s, u: step(s, u[0], u[1]),
    readout=lambda s: s.readout(),
    describe=lambda s: s.describe(),
))
