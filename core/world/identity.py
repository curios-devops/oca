"""World v3 — a world where object identity is computationally *necessary*.

Every failure so far has one cause: the mesh never built an object representation,
because nothing ever made one worth paying for. Predicting your own patch is solved more
cheaply by a local motion field than by objects, and the architecture is being rational
in refusing to pay for the expensive latent.

Weighting rare events more heavily does not fix that. Make occlusions a hundred times
more important and local trajectory extrapolation still solves them. Rarity was never the
problem. The question that matters is: **what prediction is literally impossible without
object identity?**

This world answers it. Objects come in two visually distinct kinds:

* a **passer** crosses the occluder band and exits the far side;
* a **bouncer** reverses inside the band and exits the side it entered.

The kind is obvious in the open — the two differ in brightness — and completely invisible
while occluded, because the band renders as flat grey. So the exit side is determined
entirely by *which object went in*, and there is no motion-field solution: at the moment
an object disappears, its velocity is identical in both cases. A model that tracks only
local motion is at chance forever. A model that remembers what entered is at ceiling.

That yields a much sharper measurement than pixel MSE: decode the hidden object's kind
from the model's state during occlusion. Chance is 50%, a motion field is 50%, an object
representation approaches 100%. It fails loudly, which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gridworld import CIRCLE
from .physics import PhysicsConfig, PhysicsWorld

PASSER, BOUNCER = 0, 1


@dataclass
class IdentityConfig(PhysicsConfig):
    """Physics config plus the identity rule. Gravity is off by default.

    Vertical motion is a distraction here: it adds variance to the exit *height* that has
    nothing to do with identity, and the question is purely which *side* an object comes
    out of. Turning gravity off makes the measurement binary and clean.
    """

    gravity: float = 0.0
    n_objects: int = 4
    hspeed_range: tuple[float, float] = (0.55, 0.85)
    vspeed_init: float = 0.35
    occluder_rect: tuple[int, int, int, int] = (18, 0, 28, 64)
    bright_by_kind: tuple[float, float] = (0.55, 1.0)
    """Appearance of a passer and a bouncer. The only cue to kind, and it is visible
    only while the object is outside the band."""

    regime_period: int = 10 ** 9        # no regime flips; identity is the only variable


@dataclass
class IdentityWorld(PhysicsWorld):
    """PhysicsWorld whose objects reverse or pass through according to hidden kind."""

    cfg: IdentityConfig = field(default_factory=IdentityConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        cfg = self.cfg
        n = cfg.n_objects
        # alternate kinds rather than sampling them: with only a handful of objects a
        # random draw regularly yields all-one-kind, and a decode of a constant is
        # undefined. Which object is which is still randomised.
        self.kind = self.rng.permutation(np.arange(n) % 2)
        self.bright = np.array([cfg.bright_by_kind[k] for k in self.kind])
        self.shape = np.full(n, CIRCLE)   # identical shape: brightness is the only cue
        self._turned = np.zeros(n, dtype=bool)
        self._entry_x = np.zeros(n)

    # ---------------------------------------------------------------- dynamics

    def step(self) -> None:
        cfg = self.cfg
        x0, _, w, _ = cfg.occluder_rect
        mid = x0 + w * 0.5

        for i in range(cfg.n_objects):
            hidden = self.is_fully_occluded(i)
            if hidden and not self._turned[i]:
                # a bouncer reverses once, at the midpoint of the band, so both kinds
                # look identical at the moment of disappearance
                crossed = ((self.vel[i, 0] > 0 and self.pos[i, 0] >= mid)
                           or (self.vel[i, 0] < 0 and self.pos[i, 0] <= mid))
                if crossed:
                    if self.kind[i] == BOUNCER:
                        self.vel[i, 0] *= -1.0
                    self._turned[i] = True
            elif not hidden:
                self._turned[i] = False

        super().step()

    # ------------------------------------------------------------------ probes

    def kinds(self) -> np.ndarray:
        return self.kind.copy()

    def expected_exit_side(self, i: int) -> int:
        """-1 if the object should leave to the left, +1 to the right.

        Ground truth for the headline metric, computed from the entry direction and the
        kind. A model with no memory of what entered cannot do better than chance here.
        """
        entry_dir = np.sign(self._entry_dir[i]) if hasattr(self, "_entry_dir") else 0
        if entry_dir == 0:
            entry_dir = np.sign(self.vel[i, 0])
        return int(entry_dir * (-1 if self.kind[i] == BOUNCER else 1))

    def state_snapshot(self) -> dict:
        snap = super().state_snapshot()
        snap["kind"] = self.kind.copy()
        return snap


def make_identity_world(seed: int = 0, cfg: IdentityConfig | None = None) -> IdentityWorld:
    base = cfg if cfg is not None else IdentityConfig()
    return IdentityWorld(IdentityConfig(**{**base.__dict__, "seed": seed}))
