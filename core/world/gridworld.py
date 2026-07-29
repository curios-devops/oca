"""A tiny 2-D world: objects, motion, collisions, occlusion.

Deliberately small and deterministic. No labels, no rewards, no supervision -- the
only thing that leaves this module is a luminance image and a contact map.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SQUARE, CIRCLE, TRIANGLE = 0, 1, 2


@dataclass
class WorldConfig:
    size: int = 64
    n_objects: int = 3
    radius_range: tuple[float, float] = (3.5, 5.5)
    speed_range: tuple[float, float] = (0.4, 1.1)
    occluder: bool = True
    occluder_rect: tuple[int, int, int, int] = (26, 8, 12, 48)  # x0, y0, w, h
    occluder_value: float = 0.35
    edge_softness: float = 1.0
    seed: int = 0


@dataclass
class GridWorld:
    """Objects bouncing in a box, optionally passing behind a fixed occluder."""

    cfg: WorldConfig = field(default_factory=WorldConfig)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.cfg.seed)
        n, s = self.cfg.n_objects, self.cfg.size
        self.pos = self.rng.uniform(0.2 * s, 0.8 * s, size=(n, 2))
        ang = self.rng.uniform(0, 2 * np.pi, size=n)
        spd = self.rng.uniform(*self.cfg.speed_range, size=n)
        self.vel = np.stack([np.cos(ang), np.sin(ang)], axis=1) * spd[:, None]
        self.radius = self.rng.uniform(*self.cfg.radius_range, size=n)
        self.shape = self.rng.integers(0, 3, size=n)
        self.bright = self.rng.uniform(0.65, 1.0, size=n)
        self.t = 0
        self.contact = np.zeros(16, dtype=np.float32)  # 4x4 coarse contact map

        yy, xx = np.mgrid[0:s, 0:s]
        self._xx = xx.astype(np.float32)
        self._yy = yy.astype(np.float32)

    # ------------------------------------------------------------------ physics

    def step(self) -> None:
        cfg = self.cfg
        s = cfg.size
        self.contact[:] = 0.0

        self.pos += self.vel

        # walls: reflect, and register a contact
        for k in range(2):
            low = self.pos[:, k] < self.radius
            high = self.pos[:, k] > s - self.radius
            hit = low | high
            self.pos[low, k] = self.radius[low]
            self.pos[high, k] = s - self.radius[high]
            self.vel[hit, k] *= -1.0
            self._register_contact(hit)

        # pairwise elastic collisions, equal masses
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
                if along > 0:  # equal masses: exchange the normal component
                    self.vel[i] -= along * normal
                    self.vel[j] += along * normal
                mask = np.zeros(n, dtype=bool)
                mask[[i, j]] = True
                self._register_contact(mask)

        self.t += 1

    def _register_contact(self, mask: np.ndarray) -> None:
        if not mask.any():
            return
        cell = np.clip((self.pos[mask] / self.cfg.size * 4).astype(int), 0, 3)
        for cx, cy in cell:
            self.contact[cy * 4 + cx] = 1.0

    # ---------------------------------------------------------------- rendering

    def _mask(self, i: int) -> np.ndarray:
        """Soft-edged filled shape mask for object i."""
        dx = self._xx - self.pos[i, 0]
        dy = self._yy - self.pos[i, 1]
        r = self.radius[i]
        shape = int(self.shape[i])
        if shape == CIRCLE:
            sdf = np.sqrt(dx * dx + dy * dy) - r
        elif shape == SQUARE:
            sdf = np.maximum(np.abs(dx), np.abs(dy)) - r * 0.85
        else:  # triangle: intersection of three half-planes, pointing up
            h = r * 1.3
            sdf = np.maximum.reduce(
                [
                    dy - h * 0.5,
                    -dy - dx * 1.732 - h * 0.5,
                    -dy + dx * 1.732 - h * 0.5,
                ]
            )
        return np.clip(0.5 - sdf / self.cfg.edge_softness, 0.0, 1.0)

    def render(self) -> np.ndarray:
        """(size, size) float32 luminance in [0, 1]. Occluder is painted last."""
        img = np.zeros((self.cfg.size, self.cfg.size), dtype=np.float32)
        for i in range(self.cfg.n_objects):
            m = self._mask(i)
            img = np.maximum(img, m * self.bright[i])
        if self.cfg.occluder:
            x0, y0, w, h = self.cfg.occluder_rect
            img[y0 : y0 + h, x0 : x0 + w] = self.cfg.occluder_value
        return img

    # ------------------------------------------------------------------- probes

    def is_occluded(self, i: int) -> bool:
        """True when object i's centre lies inside the occluder rectangle."""
        if not self.cfg.occluder:
            return False
        x0, y0, w, h = self.cfg.occluder_rect
        x, y = self.pos[i]
        return bool(x0 <= x < x0 + w and y0 <= y < y0 + h)

    def is_fully_occluded(self, i: int) -> bool:
        """True when object i is entirely inside the occluder, so no pixel of it shows.

        E2 needs this rather than `is_occluded`: an object whose centre is inside the
        rectangle but whose edge still pokes out is not hidden, and scoring it as hidden
        would let the mesh pass the permanence test by simply tracking the visible sliver.
        """
        if not self.cfg.occluder:
            return False
        x0, y0, w, h = self.cfg.occluder_rect
        x, y = self.pos[i]
        r = self.radius[i] * 1.35          # generous: covers the triangle's extent
        return bool(x0 + r <= x <= x0 + w - r and y0 + r <= y <= y0 + h - r)

    def occluded_objects(self) -> list[int]:
        return [i for i in range(self.cfg.n_objects) if self.is_occluded(i)]

    def perturb(self, i: int, rng: np.random.Generator, kind: str = "reverse") -> None:
        """Change an object's motion. Used by E2 to break velocity extrapolation."""
        if kind == "reverse":
            self.vel[i] *= -1.0
        elif kind == "flip_y":
            # Vertical flip only. Horizontal motion is untouched, so the object stays
            # hidden for exactly as long and emerges on the same side at the same tick
            # -- only its height is wrong. That keeps occlusion duration and emergence
            # timing matched across conditions, so an E2 asymmetry cannot be an artefact
            # of perturbed trials simply being shorter.
            self.vel[i, 1] *= -1.0
        elif kind == "rotate":
            th = rng.uniform(0.6, 1.2) * rng.choice([-1.0, 1.0])
            c, s = np.cos(th), np.sin(th)
            self.vel[i] = np.array([c * self.vel[i, 0] - s * self.vel[i, 1],
                                    s * self.vel[i, 0] + c * self.vel[i, 1]])
        elif kind == "halt":
            self.vel[i] *= 0.05
        else:
            raise ValueError(f"unknown perturbation {kind!r}")

    def state_snapshot(self) -> dict:
        return {
            "t": self.t,
            "pos": self.pos.copy(),
            "vel": self.vel.copy(),
            "occluded": self.occluded_objects(),
        }
