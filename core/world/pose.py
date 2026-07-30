"""Level 5 of the world curriculum — the pose world.

Every other world in this project is **pose-free**: objects translate, they never rotate, they
have no sides, and they look the same from everywhere. That is why matching pixels has always
been a complete solution here, and why `CGE-A-00` — *beat your own input* — has never once been
asked in a setting where the input could not simply answer.

This world exists to ask it fairly. An object is a rigid arrangement of identical blobs; the
same object appears at eight orientations; **two of those orientations are never shown during
training.** A system that memorised appearances scores at chance on them. One that holds an
object-centred model does not.

Three design decisions, each of which the world is void without.

**The blobs are identical across every kind.** Same radius, same brightness, same count. So a
bag-of-features probe — how many blobs, what do they look like — cannot tell the kinds apart at
all. **Arrangement is the only signal**, which is what makes this a test of structure rather
than of texture.

**Kinds 0 and 1 share their angles and their radii, permuted.** Same three distances from
centre, same three angles, paired differently. A radial histogram cannot separate them either.
That pair is the world's hardest control and the reason it is here.

**The shapes have no rotational symmetry.** An equilateral arrangement would make pose 0 and
pose 2 *identical*, which would silently leak held-out poses into training. Every shape is
irregular on purpose.

The sensor is a **fovea**: a window smaller than the canvas, moving over it, so any single frame
is a fragment. Integrating fragments across movement is the problem the Thousand Brains
proposal is about, and the efference copy — the same somatic channel the maze uses — is the only
signal saying how far the fovea moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# (angle in degrees, radius as a fraction of object_radius), in the object's own frame.
#
# Kinds 0 and 1 are the confusable pair: identical angle multiset {0, 100, 215}, identical
# radius multiset {1.00, 0.85, 0.62}, paired differently. Anything that ignores which radius
# goes with which angle -- a feature bag, a radial histogram, a blob count -- is at chance
# between them by construction.
SHAPES: tuple[tuple[tuple[float, float], ...], ...] = (
    ((0.0, 1.00), (100.0, 0.62), (215.0, 0.85)),
    ((0.0, 0.62), (100.0, 0.85), (215.0, 1.00)),
    ((0.0, 1.00), (55.0, 0.85), (150.0, 0.62)),
    ((0.0, 0.70), (175.0, 0.70), (300.0, 1.00)),
)

ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))    # up, down, left, right, hold


@dataclass
class PoseConfig:
    size: int = 64
    """What `render()` returns — the fovea's view, at the size every other world here uses."""

    canvas: int = 96
    """The object canvas the fovea moves within. Larger than `size`, so a frame is a fragment."""

    n_poses: int = 6
    """Orientations, evenly spaced. **Six at 60 degrees, chosen by the world's own validity
    checks and not by taste.** The first configuration used eight at 45 degrees and the world
    was VOID: nearest-template scored 0.37 on held-out poses against a chance of 0.25, because
    a foveal fragment at 45 degrees still matches a memorised fragment at the pose next door.
    At 60 degrees it falls to 0.20. Four poses at 90 degrees is worse again (0.66), so this is
    not monotone in coarseness and had to be measured."""

    held_out: tuple[int, ...] = (4,)
    """Pose indices **never shown during training**. Scoring happens only on these, and the
    whole specification rests on the raw control being at chance here."""

    object_radius: float = 26.0
    blob_radius: float = 7.0
    blob_softness: float = 2.0
    """Soft edges rather than hard discs: a hard disc aliases under rotation, and the aliasing
    pattern itself would become a pose cue."""

    episode_len: int = 40
    """Ticks before a new (kind, pose) is drawn. Long enough for a fovea to cross the object."""

    fovea_step: float = 5.0
    seed: int = 0
    n_objects: int = 1
    """One object at a time, on purpose. Which object a probe is being asked about is the
    assignment problem, and it is not what this world is for."""


@dataclass
class PoseWorld:
    cfg: PoseConfig = field(default_factory=PoseConfig)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.cfg.seed)
        self.t = 0
        self.contact = np.zeros(16, dtype=np.float32)
        self.last_action = len(ACTIONS) - 1
        self._new_episode(train_only=False)

    # ------------------------------------------------------------------ episode

    def _new_episode(self, train_only: bool | None = None) -> None:
        """Draw a kind and a pose. `train_only` restricts to poses the world may teach from."""
        cfg = self.cfg
        self.kind = int(self.rng.integers(0, len(SHAPES)))
        allowed = [p for p in range(cfg.n_poses)
                   if train_only is not True or p not in cfg.held_out]
        self.pose = int(self.rng.choice(allowed))
        self.episode_t = 0
        # start the fovea somewhere random over the object, so a frame's fragment is not a
        # function of the pose alone
        span = cfg.canvas - cfg.size
        self.fovea = self.rng.uniform(0, span, size=2)
        self._canvas = self._draw()

    def _draw(self) -> np.ndarray:
        """Render the object onto the canvas at its current kind and pose."""
        cfg = self.cfg
        c = cfg.canvas / 2.0
        theta = 2.0 * np.pi * self.pose / cfg.n_poses
        yy, xx = np.mgrid[0:cfg.canvas, 0:cfg.canvas].astype(np.float32)
        img = np.zeros((cfg.canvas, cfg.canvas), dtype=np.float32)
        for ang_deg, rad_frac in SHAPES[self.kind]:
            a = np.deg2rad(ang_deg) + theta
            r = rad_frac * cfg.object_radius
            by, bx = c + r * np.sin(a), c + r * np.cos(a)
            d = np.sqrt((yy - by) ** 2 + (xx - bx) ** 2)
            # identical blob for every kind and every position: appearance carries no identity
            img = np.maximum(img, 1.0 / (1.0 + np.exp((d - cfg.blob_radius)
                                                      / cfg.blob_softness)))
        return np.clip(img, 0.0, 1.0)

    # ------------------------------------------------------------------ dynamics

    def step(self, action: int | None = None) -> None:
        """Move the fovea. With no action given the world drives it as a random walk, which is
        how the passive worlds here behave and keeps this a prediction problem rather than a
        control one."""
        cfg = self.cfg
        if action is None:
            action = int(self.rng.integers(0, len(ACTIONS)))
        dy, dx = ACTIONS[action]
        span = cfg.canvas - cfg.size
        self.fovea = np.clip(self.fovea + np.array([dy, dx]) * cfg.fovea_step, 0, span)
        self.last_action = int(action)
        self._set_contact()

        self.t += 1
        self.episode_t += 1
        if self.episode_t >= cfg.episode_len:
            self._new_episode(train_only=None)

    def efference(self) -> np.ndarray:
        """One-hot of the action just taken. The **only** signal saying how far the fovea moved
        -- nothing in the image says it, because the object does not move."""
        e = np.zeros(len(ACTIONS), dtype=np.float32)
        e[self.last_action] = 1.0
        return e

    def _set_contact(self) -> None:
        self.contact[:] = 0.0
        self.contact[:len(ACTIONS)] = self.efference()

    # ------------------------------------------------------------------ sensing

    def render(self) -> np.ndarray:
        """The fovea's view: a `size x size` window of the canvas. A fragment, never the whole."""
        y, x = int(round(self.fovea[0])), int(round(self.fovea[1]))
        s = self.cfg.size
        return self._canvas[y:y + s, x:x + s].astype(np.float32)

    def full_object(self) -> np.ndarray:
        """The whole canvas. **Ground truth** -- the organism never receives this. It exists so
        a control can be given the thing the organism has to reconstruct, and thereby say how
        much of it was reconstructed."""
        return self._canvas.copy()

    # ------------------------------------------------------------------ ground truth

    def is_held_out(self) -> bool:
        return self.pose in self.cfg.held_out

    def state_snapshot(self) -> dict:
        """Full simulator state. Never observed; used only to score what was learned."""
        return {"t": self.t, "kind": self.kind, "pose": self.pose,
                "fovea": self.fovea.copy(), "held_out": self.is_held_out(),
                "episode_t": self.episode_t}


def make_pose_world(seed: int = 0, cfg: PoseConfig | None = None) -> PoseWorld:
    return PoseWorld(cfg or PoseConfig(seed=seed))
