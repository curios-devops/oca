"""World v2 — a world whose optimal predictor is not linear.

World v1 was rigid shapes translating at constant velocity. A 7x7x2 linear filter
represents translation of anything smaller than its window *exactly*, so the
Bayes-optimal predictor there was very nearly linear, and no architecture — nor any
ablation of one — could demonstrate value above it. Every v1 null result is consistent
with that single fact.

This world is built to remove that ceiling. Four properties, each chosen against a
specific v1 failure:

1. **Gravity and elastic bounces.** A bounce reverses velocity at a surface, which a
   linear filter cannot represent at any window size. This is the headroom.
2. **A full-height occluder band.** Objects are hidden for 12-20 ticks and, because
   gravity keeps acting, they usually *bounce while invisible*. Predicting where one
   re-emerges therefore requires simulating dynamics through the occlusion rather than
   extrapolating a straight line — which is what object permanence actually means.
3. **Discrete gravity regimes**, flipped on a slow schedule and cued by the border
   brightness. Two regimes give a bistable unit something real to be bistable about.
4. **Per-object identity** (brightness and shape) that has to survive occlusion, so
   "which object is this" is a question the world actually asks.

Everything is deterministic given the seed. Difficulty comes from non-linearity and
long-range dependency, never from noise: irreducible noise would cap every model
equally and destroy the gate's power to discriminate between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gridworld import CIRCLE, SQUARE, TRIANGLE


@dataclass
class PhysicsConfig:
    size: int = 64
    n_objects: int = 3
    radius_range: tuple[float, float] = (3.0, 4.0)
    hspeed_range: tuple[float, float] = (0.5, 0.9)
    vspeed_init: float = 0.5

    gravity: float = 0.35
    restitution: float = 1.0          # exactly elastic: no settling, no energy drain
    max_vspeed: float = 2.2
    """Terminal vertical speed. Capping *vertical* speed only is deliberate: a cap on
    total speed lets gravity crowd out the horizontal component, and objects then stop
    crossing the occluder at all. Together with gravity this sets the bounce period
    (2v/g ~ 16 ticks) and amplitude (v^2/2g ~ 11 px) to be comparable to the occlusion
    duration, which is what makes a bounce happen while the object is invisible."""

    regime_period: int = 360          # ticks between gravity flips
    border: int = 4
    border_bright: tuple[float, float] = (0.12, 0.72)

    occluder: bool = True
    occluder_rect: tuple[int, int, int, int] = (18, 0, 28, 64)
    """Wide enough that an object is *fully* hidden for a useful stretch: the hidden
    window is the band width minus the object's own extent, so a narrow band gives
    occlusions only a few ticks long no matter how slowly objects move."""
    occluder_value: float = 0.30

    edge_softness: float = 1.0
    seed: int = 0


@dataclass
class PhysicsWorld:
    """Same interface as GridWorld, so Sensors and the rollout path are unchanged."""

    cfg: PhysicsConfig = field(default_factory=PhysicsConfig)

    def __post_init__(self) -> None:
        cfg = self.cfg
        self.rng = np.random.default_rng(cfg.seed)
        n, s = cfg.n_objects, cfg.size
        b = cfg.border

        self.radius = self.rng.uniform(*cfg.radius_range, size=n)
        lo, hi = b + self.radius.max() + 1, s - b - self.radius.max() - 1
        self.pos = np.stack([
            self.rng.uniform(lo, hi, size=n),
            self.rng.uniform(lo, hi, size=n),
        ], axis=1)
        hs = self.rng.uniform(*cfg.hspeed_range, size=n) * self.rng.choice([-1.0, 1.0], n)
        vs = self.rng.uniform(-cfg.vspeed_init, cfg.vspeed_init, size=n)
        self.vel = np.stack([hs, vs], axis=1)

        # distinct, evenly spaced identities so binding is a well-posed question
        self.bright = np.linspace(0.55, 1.0, n)
        self.shape = np.array([SQUARE, CIRCLE, TRIANGLE] * (n // 3 + 1))[:n]

        self.t = 0
        self.contact = np.zeros(16, dtype=np.float32)

        yy, xx = np.mgrid[0:s, 0:s]
        self._xx = xx.astype(np.float32)
        self._yy = yy.astype(np.float32)

    # ------------------------------------------------------------------ regime

    @property
    def regime(self) -> int:
        """+1 when gravity pulls down (+y), -1 when it pulls up."""
        return 1 if (self.t // self.cfg.regime_period) % 2 == 0 else -1

    # ----------------------------------------------------------------- physics

    def step(self) -> None:
        cfg = self.cfg
        s, b = cfg.size, cfg.border
        self.contact[:] = 0.0

        self.vel[:, 1] += cfg.gravity * self.regime
        self.pos += self.vel

        for k in range(2):
            lo = b + self.radius
            hi = s - b - self.radius
            under = self.pos[:, k] < lo
            over = self.pos[:, k] > hi
            # mirror reflection, so a bounce lands where it physically should
            self.pos[under, k] = 2 * lo[under] - self.pos[under, k]
            self.pos[over, k] = 2 * hi[over] - self.pos[over, k]
            hit = under | over
            self.vel[hit, k] *= -cfg.restitution
            self._register_contact(hit)

        n = cfg.n_objects
        for i in range(n):
            for j in range(i + 1, n):
                delta = self.pos[j] - self.pos[i]
                dist = float(np.linalg.norm(delta))
                overlap = self.radius[i] + self.radius[j] - dist
                if dist < 1e-6 or overlap <= 0:
                    continue
                normal = delta / dist
                self.pos[i] -= normal * overlap * 0.5
                self.pos[j] += normal * overlap * 0.5
                rel = self.vel[i] - self.vel[j]
                along = float(rel @ normal)
                if along > 0:
                    self.vel[i] -= along * normal
                    self.vel[j] += along * normal
                mask = np.zeros(n, dtype=bool)
                mask[[i, j]] = True
                self._register_contact(mask)

        np.clip(self.vel[:, 1], -cfg.max_vspeed, cfg.max_vspeed, out=self.vel[:, 1])

        np.clip(self.pos, 0.0, s, out=self.pos)
        self.t += 1

    def _register_contact(self, mask: np.ndarray) -> None:
        if not mask.any():
            return
        cell = np.clip((self.pos[mask] / self.cfg.size * 4).astype(int), 0, 3)
        for cx, cy in cell:
            self.contact[cy * 4 + cx] = 1.0

    # --------------------------------------------------------------- rendering

    def _mask(self, i: int) -> np.ndarray:
        dx = self._xx - self.pos[i, 0]
        dy = self._yy - self.pos[i, 1]
        r = self.radius[i]
        shape = int(self.shape[i])
        if shape == CIRCLE:
            sdf = np.sqrt(dx * dx + dy * dy) - r
        elif shape == SQUARE:
            sdf = np.maximum(np.abs(dx), np.abs(dy)) - r * 0.85
        else:
            h = r * 1.3
            sdf = np.maximum.reduce([
                dy - h * 0.5,
                -dy - dx * 1.732 - h * 0.5,
                -dy + dx * 1.732 - h * 0.5,
            ])
        return np.clip(0.5 - sdf / self.cfg.edge_softness, 0.0, 1.0)

    def render(self) -> np.ndarray:
        cfg = self.cfg
        img = np.zeros((cfg.size, cfg.size), dtype=np.float32)
        for i in range(cfg.n_objects):
            img = np.maximum(img, self._mask(i) * self.bright[i])
        if cfg.occluder:
            x0, y0, w, h = cfg.occluder_rect
            img[y0:y0 + h, x0:x0 + w] = cfg.occluder_value

        # border carries the regime cue; drawn last so it is never occluded
        b = cfg.border
        val = cfg.border_bright[0] if self.regime > 0 else cfg.border_bright[1]
        img[:b, :] = val
        img[-b:, :] = val
        img[:, :b] = val
        img[:, -b:] = val
        return img

    # ------------------------------------------------------------------ probes

    def is_occluded(self, i: int) -> bool:
        if not self.cfg.occluder:
            return False
        x0, y0, w, h = self.cfg.occluder_rect
        x, y = self.pos[i]
        return bool(x0 <= x < x0 + w and y0 <= y < y0 + h)

    def is_fully_occluded(self, i: int) -> bool:
        if not self.cfg.occluder:
            return False
        x0, y0, w, h = self.cfg.occluder_rect
        x, y = self.pos[i]
        r = self.radius[i] * 1.35
        return bool(x0 + r <= x <= x0 + w - r and y0 - r <= y <= y0 + h + r)

    def occluded_objects(self) -> list[int]:
        return [i for i in range(self.cfg.n_objects) if self.is_occluded(i)]

    def perturb(self, i: int, rng: np.random.Generator, kind: str = "flip_y") -> None:
        if kind == "flip_y":
            self.vel[i, 1] *= -1.0
        elif kind == "reverse":
            self.vel[i] *= -1.0
        elif kind == "halt":
            self.vel[i] *= 0.05
        else:
            raise ValueError(f"unknown perturbation {kind!r}")

    def state_snapshot(self) -> dict:
        return {
            "t": self.t,
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "regime": self.regime,
            "occluded": self.occluded_objects(),
            "fully_occluded": [self.is_fully_occluded(i)
                               for i in range(self.cfg.n_objects)],
            "bright": self.bright.copy(),
        }


def make_physics_world(seed: int = 0, cfg: PhysicsConfig | None = None) -> PhysicsWorld:
    base = cfg if cfg is not None else PhysicsConfig()
    return PhysicsWorld(PhysicsConfig(**{**base.__dict__, "seed": seed}))
