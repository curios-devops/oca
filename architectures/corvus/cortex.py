"""The sensory boundary: raw sensory input in, per-tower patches out, predicted frames back.

Kept separate from the layers on purpose. A tower should not know that its input came from an eye,
and the routing from a sensory surface to towers is a fact about this particular body rather than
about the architecture. Swapping the world means rewriting this file and nothing else.

Two routing decisions, and the second is a correction of a measured mistake.

**Vision is retinotopic.** Each tower takes a contiguous block of retinal patches, so towers that
neighbour each other on the tower grid see neighbouring parts of the image. That is what makes
grid adjacency a spatial neighbourhood rather than an arbitrary one.

**Proprioception is broadcast, not routed.** Every tower receives the efference copy. Heron routed
it to exactly one node of seventeen, which left sixteen towers unable to distinguish "I moved"
from "the world moved" -- the distinction the entire sensorimotor story rests on, and plausibly
part of why it scored worse than chance at knowing where it was while blind. Proprioception is not
localised in the visual field, so it is not routed like something that is.
"""

from __future__ import annotations

import numpy as np

from core.world.sensors import N_SOMATIC, N_VISUAL, P, RETINA

from . import cluster as L2
from . import tower as L1

PATCH_SIDE = int(round(N_VISUAL ** 0.5))          # 8 patches across the retina
TOWER_SIDE = PATCH_SIDE // 2                      # 4 towers across, each taking 2x2
N_VISUAL_TOWERS = TOWER_SIDE * TOWER_SIDE         # 16
N_TOWERS = N_VISUAL_TOWERS + 1                    # + one for the somatic surface
INPUTS_PER_TOWER = 4 * P
N_PROPRIO = N_SOMATIC * P


def _routing(tower_side: int = TOWER_SIDE):
    block = PATCH_SIDE // tower_side
    node = np.zeros(N_VISUAL, dtype=np.int64)
    slot = np.zeros(N_VISUAL, dtype=np.int64)
    for p in range(N_VISUAL):
        pr, pc = divmod(p, PATCH_SIDE)
        node[p] = (pr // block) * tower_side + (pc // block)
        slot[p] = (pr % block) * block + (pc % block)
    return node, slot, block


PATCH_TOWER, PATCH_SLOT, BLOCK = _routing()


def split_sensory(sensory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N_SENSORY, P) -> (visual per tower, proprioception).

    The somatic block is the proprioceptive channel: in the maze worlds it carries the efference
    copy of the agent's own action, and in the passive worlds it carries a contact map. Either way
    it is information about the body rather than about a place, so it goes to every tower.
    """
    s = np.asarray(sensory, dtype=np.float64)
    visual = np.zeros((N_TOWERS, INPUTS_PER_TOWER))
    for p in range(N_VISUAL):
        o = PATCH_SLOT[p] * P
        visual[PATCH_TOWER[p], o:o + P] = s[p]
    proprio = s[N_VISUAL:N_VISUAL + N_SOMATIC].ravel()
    return visual, proprio


def towers_to_patches(tower_patches: np.ndarray) -> np.ndarray:
    out = np.zeros((N_VISUAL, P))
    for p in range(N_VISUAL):
        o = PATCH_SLOT[p] * P
        out[p] = tower_patches[PATCH_TOWER[p], o:o + P]
    return out


class Cortex:
    """Layers 0-2 wired together behind one entry point.

    Holds the layers rather than inheriting from them, so each remains independently replaceable
    -- which is the point of the contract. The cluster layer reads the towers' *published* state
    and their events, never their internals.
    """

    def __init__(self, seed: int = 0, use_entities: bool = True,
                 cluster_mode: str = "pass_through", **kw):
        tower_kw = {k: v for k, v in kw.items() if k in L1.TowerConfig.__dataclass_fields__}
        self.towers = L1.build_stack(L1.TowerConfig(
            seed=seed, n_towers=N_TOWERS, n_inputs=INPUTS_PER_TOWER,
            n_proprio=N_PROPRIO, use_entities=use_entities, **tower_kw))
        self.clusters = L2.build_stack(L2.ClusterConfig(seed=seed, mode=cluster_mode),
                                      n_towers=N_TOWERS)
        self._learn = True

    # the gates set `state.learn`; propagate it to every layer
    @property
    def learn(self) -> bool:
        return self._learn

    @learn.setter
    def learn(self, value: bool) -> None:
        self._learn = bool(value)
        self.towers.learn = bool(value)
        self.towers.neurons.learn = bool(value)
        self.clusters.learn = bool(value)

    @property
    def cfg(self):
        return self.towers.cfg

    @property
    def horizons(self) -> tuple[int, ...]:
        return (1, 4, 16)

    @property
    def n_units(self) -> int:
        return self.towers.n_units

    @property
    def h(self) -> np.ndarray:
        return self.towers.h

    @property
    def coalition(self) -> np.ndarray:
        return self.towers.coalition

    def n_params(self) -> int:
        return self.towers.n_params() + self.clusters.n_params()

    def describe(self) -> dict:
        d = {"layers": ["neuron", "tower", "cluster"], "n_params": self.n_params(),
             "n_units": self.n_units, "readout_dim": int(self.h.shape[1])}
        d.update({f"tower.{k}": v for k, v in self.towers.describe().items()})
        d.update({f"cluster.{k}": v for k, v in self.clusters.describe().items()})
        return d


def build_cortex(seed: int = 0, **kw) -> Cortex:
    return Cortex(seed=seed, **kw)


def tick(cortex: Cortex, sensory: np.ndarray) -> dict:
    """One tick from raw sensory input. The only entry point a world needs."""
    visual, proprio = split_sensory(sensory)
    out = L1.step(cortex.towers, visual, proprio)
    out.update(L2.step(cortex.clusters, cortex.towers.publication(),
                       cortex.towers.events()))
    return out


def predicted_retina(cortex: Cortex, tau: int, sensors) -> np.ndarray:
    """Assemble each tower's forecast of its own patch into one predicted frame.

    Each tower forecasts only the part of the surface it can see; the stitching is the reader's
    job, which is why this lives at the boundary and not in the layer.
    """
    t = cortex.towers
    pred = np.einsum("nid,nd->ni", t.W, t.entity_state())
    return sensors.from_patches(towers_to_patches(pred)).astype(np.float32)
