"""Sensory transduction: world image -> per-unit input patches.

Two modalities. Vision is retinotopic: the 64 visual units tile an 8x8 block of the
mesh lattice in the same arrangement as their patches tile the retina, so spatial
neighbours in the world are spatial neighbours in the mesh. Touch is a coarse contact
map split across 4 somatic units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gridworld import GridWorld

RETINA = 32          # retina is RETINA x RETINA after downsampling
PATCH = 4            # each visual unit sees a PATCH x PATCH tile
P = PATCH * PATCH    # = 16, the sensory width p in SPEC_RPDU
N_VISUAL = (RETINA // PATCH) ** 2   # 64
N_SOMATIC = 4
N_SENSORY = N_VISUAL + N_SOMATIC    # 68


@dataclass
class Sensors:
    """Stateless transducer. Shapes are fixed at import time by the constants above."""

    def retina(self, img: np.ndarray) -> np.ndarray:
        """Downsample the world image to RETINA x RETINA by block mean."""
        s = img.shape[0]
        f = s // RETINA
        if f == 1:
            return img.astype(np.float32)
        return img.reshape(RETINA, f, RETINA, f).mean(axis=(1, 3)).astype(np.float32)

    def to_patches(self, ret: np.ndarray) -> np.ndarray:
        """(RETINA, RETINA) -> (N_VISUAL, P), row-major over patch grid."""
        g = RETINA // PATCH
        return (
            ret.reshape(g, PATCH, g, PATCH)
            .transpose(0, 2, 1, 3)
            .reshape(N_VISUAL, P)
            .astype(np.float32)
        )

    def from_patches(self, patches: np.ndarray) -> np.ndarray:
        """Inverse of to_patches -- used to score whole-frame prediction error."""
        g = RETINA // PATCH
        return (
            np.asarray(patches)
            .reshape(g, g, PATCH, PATCH)
            .transpose(0, 2, 1, 3)
            .reshape(RETINA, RETINA)
        )

    def somatic(self, contact: np.ndarray) -> np.ndarray:
        """(16,) contact map -> (N_SOMATIC, P); unit k sees only its own quadrant."""
        out = np.zeros((N_SOMATIC, P), dtype=np.float32)
        cm = contact.reshape(4, 4)
        for k in range(N_SOMATIC):
            qy, qx = divmod(k, 2)
            m = np.zeros((4, 4), dtype=np.float32)
            m[qy * 2 : qy * 2 + 2, qx * 2 : qx * 2 + 2] = 1.0
            out[k] = (cm * m).reshape(-1)
        return out

    def observe(self, world: GridWorld) -> tuple[np.ndarray, np.ndarray]:
        """Returns (sensory (N_SENSORY, P), retina (RETINA, RETINA))."""
        ret = self.retina(world.render())
        vis = self.to_patches(ret)
        som = self.somatic(world.contact)
        return np.concatenate([vis, som], axis=0), ret

    # -- placement of sensory units on the mesh lattice -----------------------

    @staticmethod
    def lattice_slots(side: int) -> np.ndarray:
        """Mesh indices for the N_SENSORY sensory units on a `side` x `side` lattice.

        Visual units occupy a contiguous 8x8 block in the same row-major order as
        their retinal patches, preserving retinotopy under local connectivity.
        Somatic units sit just below that block.
        """
        g = RETINA // PATCH  # 8
        if side < g + 1:
            raise ValueError(f"lattice side {side} too small for an {g}x{g} visual block")
        idx = [(r * side + c) for r in range(g) for c in range(g)]
        idx += [(g * side + c) for c in range(N_SOMATIC)]
        return np.array(idx, dtype=np.int64)
