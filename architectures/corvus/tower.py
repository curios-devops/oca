"""Layer 1 — Cortical Tower.

The primary cognitive building block: a local world model built through perception and action,
holding prediction, a reference frame, persistent state about referents it cannot currently
observe, confidence, and sensorimotor state.

**This is the layer where all three predecessors died**, and the design is written around exactly
why. Wren, Swift and Heron each had a layer here with 2304, 3456 and 1309 dimensions of internal
state, and all three lost to storing two numbers. State was never absent. *Persistence* was.

So the mechanism is not another way of transforming a signal. It is one loop:

    observable      -> correct the entity toward what is seen, and shrink uncertainty
    not observable  -> advance the entity by a predicted displacement, and grow uncertainty

That is the whole idea, and three things make it a claim rather than a tautology.

**The tower is not told whether it can see.** It infers it, from signals it already has: if it
acted and the view did not change, then the view is not informative about where it is. Inside a
covered corridor every frame is byte-identical, so that condition is exactly true; outside, moving
changes the view. Being handed an `in_tunnel` flag would make this gate meaningless, so it is not
handed one.

**The frame has two parts, and they live in different spaces.** An *anchor* -- an encoding of the
last view that was informative -- and a *displacement accumulator*, integrated from the efference
copy since that anchor. The entity's attributes are both.

This is the second design of this layer, and the first one failed for a reason worth recording,
because it is a mistake that looks correct on paper.

The first version had a single frame: a random projection of what the neurons published, with a
*learned* linear model of how actions move it. The intuition was that learning beats writing --
a hand-written dead reckoner would pass the gate and prove nothing. Measured, the model never
learned: `|A|` reached 0.0013 after 14,000 ticks. The reason is structural, not a matter of
learning rate. **Actions do not move a view-encoding consistently.** The same step changes the
whole 5x5 view, and by a completely different amount depending on where you are, so there is no
stable action -> view-delta mapping for a linear rule to find. I had asked it to learn a function
that does not exist.

Displacement, on the other hand, composes additively by construction. So it is integrated in its
own space through a fixed injective projection of the efference copy, and *nothing about it needs
supervision*: the accumulator is an exact linear function of the action sequence, which is
affinely related to true displacement. The reader still has to learn to combine anchor and
displacement, the anchor still has to be captured at the right moment, and a blocked move still
has to be accounted for -- so the mechanism can be wrong in several ways and the gate can still
fail. What it is not is a function that cannot be learned because it is not a function.

**The anchor requires two consecutive sighted ticks, not one.** This is the other thing the first
version got wrong. Tunnels here are short -- roughly four steps -- so about a quarter of in-tunnel
frames are *entry* frames, where the view legitimately just changed and the naive detector reports
"observable". Measured: 26% of in-tunnel ticks were treated as sighted, and each one pulled the
belief 35% toward an encoding that is identical everywhere inside a corridor. The entry position
was being destroyed on the way in, which is exactly the information `frozen-at-entry` keeps.
Requiring the previous tick to have been sighted too excludes the transition.

**Stated before running, so it can be scored honestly:** this is expected to pass `CGE-A-01`
(beat frozen-at-entry while blind) and is *not* expected to fix `CGE-A-00` (beat the raw frame at
predicting an observable world). Those are different problems and only the first one is what an
entity solves. If it passes A-01 and still fails A-00, that is the informative outcome, not a
disappointment.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import neuron as L0
from .contract import Floor, Layer, register
from .primitives import Entity, Event, Registry

PUBLISHED = ("prediction_error", "confidence", "novelty", "activity", "uncertainty")
"""What a tower says about itself. Five scalars, plus one entity reference per event.

Heron's version of this narrow interface was its only passing gate -- a reader given five scalars
and a phase spectrum came within 1.50x of a reader given the whole internal state. The difference
here is the entity reference: a scalar cannot say what it is about, and being unable to say that
is precisely how persistence failed to cross a layer boundary.
"""


@dataclass(frozen=True)
class TowerConfig:
    n_towers: int = 17
    n_neurons: int = 64
    n_inputs: int = 64
    """Visual width per tower."""

    n_proprio: int = 64
    """Width of the proprioceptive channel -- the efference copy of the tower's own action.

    **Broadcast to every tower, not routed to one.** Heron sent it to exactly one node of
    seventeen, which left sixteen towers unable to distinguish "I moved" from "the world moved" --
    the distinction the entire sensorimotor story rests on. Proprioception is not localised in the
    visual field, so it is not routed like vision."""

    d_anchor: int = 16
    """Width of the anchor: an encoding of the last view that was informative."""

    d_disp: int = 8
    """Width of the displacement accumulator. Small, because displacement is low-dimensional --
    it only has to distinguish the moves the body can make."""

    horizon: int = 16

    # -- observability inference -------------------------------------------
    view_change_floor: float = 1e-4
    """Below this, an acted-upon view has not changed and is therefore uninformative."""

    act_floor: float = 1e-6

    # -- the entity loop ---------------------------------------------------
    trust: float = 0.35
    """How hard an observation corrects the belief. Below 1 because a single glance is noisy."""
    uncertainty_growth: float = 0.02

    # -- displacement integration -------------------------------------------
    use_displacement: bool = True
    """Integrate the efference copy while blind. Off, the tower holds its anchor still and is
    exactly `frozen-at-entry` -- which is the ablation that shows whether the mechanism works."""

    # -- prediction --------------------------------------------------------
    eta_pred: float = 0.05

    use_entities: bool = True
    """With entities off the tower keeps its anchor but never propagates it while blind -- which
    is what all three frozen architectures did, and the comparison that shows whether the
    primitive is doing the work."""

    seed: int = 0

    def variant(self, **changes) -> "TowerConfig":
        return replace(self, **changes)

    @property
    def d_frame(self) -> int:
        return self.d_anchor + self.d_disp

    @property
    def d_publication(self) -> int:
        return len(PUBLISHED)


@dataclass
class TowerStack:
    """All the towers of one cortex. Arrays with a leading tower axis."""

    cfg: TowerConfig
    rng: np.random.Generator
    neurons: L0.Population

    P: np.ndarray                 # (d_anchor, n_neurons) fixed sensory encoder
    M: np.ndarray                 # (d_disp, n_proprio) fixed injective action projection
    W: np.ndarray                 # (n_towers, n_inputs, d_frame) sensory prediction

    frame: np.ndarray             # (n_towers, d_frame) = anchor ++ displacement
    err: np.ndarray               # (n_towers,)
    conf: np.ndarray              # (n_towers,)
    novelty: np.ndarray           # (n_towers,)
    uncertainty: np.ndarray       # (n_towers,)

    registries: list = field(default_factory=list)
    selves: list = field(default_factory=list)

    prev_visual: np.ndarray | None = None
    prev_anchor_obs: np.ndarray | None = None
    prev_frame: np.ndarray | None = None
    prev_proprio: np.ndarray | None = None
    prev_observable: np.ndarray | None = None
    observable: np.ndarray | None = None
    hist: list = field(default_factory=list, repr=False)

    t: int = 0
    learn: bool = True
    _events: list = field(default_factory=list, repr=False)

    # -- readout -----------------------------------------------------------

    @property
    def n_units(self) -> int:
        return self.cfg.n_towers

    def publication(self) -> np.ndarray:
        """(n_towers, 5). Scalars only -- no frame, no attributes, no member activity."""
        act = np.abs(self.member_state()).mean(axis=1)
        return np.stack([self.err, self.conf, self.novelty, act, self.uncertainty], axis=1)

    def member_state(self) -> np.ndarray:
        """What each tower's neurons published, held between events. Never their activation."""
        return self.neurons.last_sent.reshape(self.cfg.n_towers, self.cfg.n_neurons)

    def entity_state(self) -> np.ndarray:
        """(n_towers, d_frame) — each tower's self-entity attributes.

        This is the persistent part, and the part no frozen architecture had. Outside a tunnel it
        tracks what is seen; inside one it is the entry encoding plus integrated action.
        """
        if not self.selves:
            return self.frame
        return np.stack([s.attributes for s in self.selves])

    @property
    def h(self) -> np.ndarray:
        """(n_towers, d_frame + 5) the readout the layer above and the gates see."""
        return np.concatenate([self.entity_state(), self.publication()], axis=1)

    @property
    def coalition(self) -> np.ndarray:
        """Towers grouped by which of them currently believe they can see.

        Not synchrony and not a concept index. Observability is the only grouping this layer
        actually computes, and claiming a richer one would be inventing a mechanism to have
        something to report.
        """
        if self.observable is None:
            return np.zeros(self.cfg.n_towers, dtype=np.int64)
        return self.observable.astype(np.int64)

    def events(self) -> list[Event]:
        return list(self._events)

    def n_params(self) -> int:
        return int(self.neurons.w.size + self.M.size + self.W.size + self.P.size)

    def describe(self) -> dict:
        return {"n_towers": self.cfg.n_towers, "n_neurons_total": self.neurons.n_neurons,
                "n_params": self.n_params(), "horizon": self.cfg.horizon,
                "d_anchor": self.cfg.d_anchor, "d_disp": self.cfg.d_disp,
                "entities": self.cfg.use_entities,
                "n_entities": sum(len(r) for r in self.registries),
                "mechanism": "anchor re-set on two consecutive sighted ticks; displacement "
                             "integrated from the efference copy while blind"}


def build_stack(cfg: TowerConfig | None = None) -> TowerStack:
    cfg = cfg or TowerConfig()
    rng = np.random.default_rng(cfg.seed)
    n, D = cfg.n_towers, cfg.d_frame

    neurons = L0.build(L0.NeuronConfig(seed=cfg.seed, n_neurons=cfg.n_neurons,
                                       n_inputs=cfg.n_inputs + cfg.n_proprio),
                       n_groups=n)

    stack = TowerStack(
        cfg=cfg, rng=rng, neurons=neurons,
        P=rng.choice([-1.0, 1.0], (cfg.d_anchor, cfg.n_neurons)) / np.sqrt(cfg.n_neurons),
        # injective over the actions the body can take: distinct actions must produce distinct
        # displacements, or the accumulator cannot be inverted by any reader
        M=rng.normal(0, 1.0, (cfg.d_disp, cfg.n_proprio)),
        W=np.zeros((n, cfg.n_inputs, D)),
        frame=np.zeros((n, D)),
        err=np.ones(n), conf=np.zeros(n), novelty=np.ones(n), uncertainty=np.ones(n),
    )
    stack.registries = [Registry(owner="tower") for _ in range(n)]
    return stack


def step(stack: TowerStack, visual: np.ndarray, proprio: np.ndarray) -> dict:
    """One tick.

    `visual` is (n_towers, n_inputs) -- each tower's own patch. `proprio` is (n_proprio,) -- the
    efference copy, broadcast to every tower.
    """
    cfg = stack.cfg
    n, D = cfg.n_towers, cfg.d_frame
    visual = np.asarray(visual, dtype=np.float64).reshape(n, cfg.n_inputs)
    proprio = np.asarray(proprio, dtype=np.float64).reshape(-1)[:cfg.n_proprio]
    if proprio.size < cfg.n_proprio:
        proprio = np.pad(proprio, (0, cfg.n_proprio - proprio.size))

    # -- 1. neurons integrate vision and proprioception together ------------
    L0.step(stack.neurons, np.concatenate(
        [visual, np.broadcast_to(proprio, (n, cfg.n_proprio))], axis=1))
    members = stack.member_state()

    # -- 2. can this tower see where it is? ---------------------------------
    # Inferred, never told. If it acted and the view did not change, the view carries no
    # information about position -- which is exactly true inside a covered corridor.
    acting = float(np.abs(proprio).sum()) > cfg.act_floor
    if stack.prev_visual is None:
        view_change = np.full(n, np.inf)
    else:
        view_change = np.abs(visual - stack.prev_visual).mean(axis=1)
    observable = ~(acting & (view_change < cfg.view_change_floor))
    stack.observable = observable

    # -- 3. the two parts of the frame ---------------------------------------
    anchor_obs = members @ stack.P.T                      # (n, d_anchor) what is seen now
    step_disp = stack.M @ proprio                         # (d_disp,) this tick's displacement

    # The anchor lags by one tick, and that is not a detail.
    #
    # You cannot tell that a view was informative until you have seen whether the next one
    # differs. At the moment of entering a covered corridor the view has just changed, so every
    # one-tick test calls it sighted -- and anchoring there overwrites the last real position with
    # an encoding that is identical everywhere inside the corridor. Measured: 26% of in-tunnel
    # ticks were treated as sighted, and requiring two *consecutive* sighted ticks did not help,
    # because at the entry tick the previous tick genuinely was sighted. That rule excludes the
    # tick after going blind, which is the wrong one.
    #
    # Anchoring to t-1 instead of t is what actually excludes the transition: when the view stops
    # being informative, the anchor is already the last one that was.
    confirmed = observable & (stack.prev_observable
                              if stack.prev_observable is not None else observable)
    anchor_src = (stack.prev_anchor_obs if stack.prev_anchor_obs is not None else anchor_obs)

    # -- 4. the entity loop -------------------------------------------------
    if not stack.selves:
        for i in range(n):
            stack.selves.append(stack.registries[i].create(
                np.concatenate([anchor_obs[i], np.zeros(cfg.d_disp)]), now=stack.t))

    for i, ent in enumerate(stack.selves):
        if confirmed[i]:
            # re-anchor to the *previous* encoding, and zero the accumulator
            ent.observe("tower", np.concatenate([anchor_src[i], np.zeros(cfg.d_disp)]),
                        now=stack.t, trust=cfg.trust)
        elif cfg.use_entities:
            # The operation the whole primitive exists for. Nothing is observed; the belief
            # advances because the referent is expected to have moved, and uncertainty records
            # that the movement was inferred rather than seen.
            delta = np.concatenate([
                np.zeros(cfg.d_anchor),
                step_disp if cfg.use_displacement else np.zeros(cfg.d_disp)])
            ent.propagate("tower", delta, growth=cfg.uncertainty_growth)
        else:
            # The ablation: hold everything still while blind. This is what all three frozen
            # architectures effectively did, and it is exactly `frozen-at-entry`.
            ent.propagate("tower", np.zeros(cfg.d_frame), growth=cfg.uncertainty_growth)

    stack.frame = np.stack([e.attributes for e in stack.selves])
    stack.uncertainty = np.array([e.uncertainty for e in stack.selves])

    # -- 6. predict this tower's own patch at the declared horizon ----------
    tau = cfg.horizon
    stack.hist.append((stack.frame.copy(), visual.copy()))
    if len(stack.hist) > tau + 1:
        stack.hist.pop(0)
    if len(stack.hist) > tau:
        f_then, _ = stack.hist[0]
        pred = np.einsum("nid,nd->ni", stack.W, f_then)
        e_x = visual - pred
        stack.err = np.linalg.norm(e_x, axis=1) / np.sqrt(cfg.n_inputs)
        if stack.learn:
            norm = np.einsum("nd,nd->n", f_then, f_then) + 1e-8
            stack.W += (cfg.eta_pred / norm)[:, None, None] * np.einsum(
                "ni,nd->nid", e_x, f_then)

    stack.conf = 0.98 * stack.conf + 0.02 * np.exp(-stack.err)
    stack.novelty = np.clip(view_change / (np.median(view_change[np.isfinite(view_change)])
                                          + 1e-9), 0.0, 1.0) if stack.prev_visual is not None \
        else np.ones(n)

    # -- 7. publish: events, each naming the entity it is about -------------
    stack._events = []
    pub = stack.publication()
    for i, ent in enumerate(stack.selves):
        for k, name in enumerate(PUBLISHED):
            stack._events.append(Event(source=f"tower{i}", kind=name,
                                       value=float(pub[i, k]), at=stack.t,
                                       entity=ent.id))

    stack.prev_visual = visual.copy()
    stack.prev_anchor_obs = anchor_obs.copy()
    stack.prev_frame = stack.frame.copy()
    stack.prev_proprio = proprio.copy()
    stack.prev_observable = observable.copy()
    stack.t += 1

    return {"t": stack.t,
            "frac_observable": float(observable.mean()),
            "prediction_error": float(stack.err.mean()),
            "confidence": float(stack.conf.mean()),
            "uncertainty": float(stack.uncertainty.mean()),
            "n_events": len(stack._events)}


register(Layer(
    name="tower",
    horizon=TowerConfig().horizon,
    inputs_from="neuron",
    floor=Floor(
        beats="trivial_memory",
        margin=0.05,
        why="`raw_input` is the wrong floor here and would produce a false pass: on an "
            "observable world the frame already contains most of the answer. The discriminating "
            "floor is the cheapest possible memory -- store one value once, never update it -- "
            "which measured 2.26 cells in the tunnel world and beat all three frozen "
            "architectures. Gate CGE-A-01.",
    ),
    build=lambda seed=0, **kw: build_stack(TowerConfig(seed=seed, **kw)),
    step=lambda s, u: step(s, u[0], u[1]),
    readout=lambda s: s.publication(),
    describe=lambda s: s.describe(),
))
