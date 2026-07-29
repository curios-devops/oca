"""The sensory boundary: retinal patches in, node patches out, predicted frames back.

Kept separate from `node.py` on purpose. A Dynamic Cortical Node should not know that its
input came from an eye, and the routing from retina to nodes is a fact about this
particular body rather than about the level. Swapping the world means rewriting this file
and nothing else.

The routing is retinotopic, and that is not decoration. Each node takes a 2x2 block of
retinal patches, so nodes that are neighbours on the node grid see neighbouring parts of
the image — which is what makes `_grid_adjacency` in `node.py` a spatial neighbourhood
rather than an arbitrary one, and what makes the neighbourhood model's job learnable at
all.
"""

from __future__ import annotations

import numpy as np

from core.world.sensors import N_SOMATIC, N_VISUAL, P, RETINA
from .contract import DCNLevel, register_dcn
from .node import NodeConfig, NodeStack, build_stack, predicted_patches
from .node import step as node_step

PATCH_SIDE = int(round(N_VISUAL ** 0.5))          # 8 patches across the retina
NODE_SIDE = PATCH_SIDE // 2                       # default: 4 nodes across, each taking 2x2
N_VISUAL_NODES = NODE_SIDE * NODE_SIDE            # 16
N_NODES = N_VISUAL_NODES + 1                      # + one somatic node
INPUTS_PER_NODE = 4 * P                           # four patches of P values each


def layout(node_side: int = NODE_SIDE) -> dict:
    """How many nodes there are, how wide each one's input is, and where each patch goes.

    Node resolution is a parameter and not a constant, because it is the one property of
    this level that has nothing to do with any of its mechanisms and can still decide a
    result. A 4x4 cortex sees the world through sixteen windows; asked for a 7x7 map of
    maze cells it will lose to raw pixels no matter how good its dynamics are, and reading
    that as evidence about the dynamics would be an error about resolution dressed up as a
    finding. Anything that depends on this is therefore run at more than one value.
    """
    block = PATCH_SIDE // node_side
    node = np.zeros(N_VISUAL, dtype=np.int64)
    slot = np.zeros(N_VISUAL, dtype=np.int64)
    for p in range(N_VISUAL):
        pr, pc = divmod(p, PATCH_SIDE)
        node[p] = (pr // block) * node_side + (pc // block)
        slot[p] = (pr % block) * block + (pc % block)
    return {"node_side": node_side, "block": block,
            "n_visual_nodes": node_side * node_side,
            "n_nodes": node_side * node_side + 1,
            "inputs_per_node": max(block * block * P, N_SOMATIC * P),
            "patch_node": node, "patch_slot": slot}


_DEFAULT = layout()
PATCH_NODE, PATCH_SLOT = _DEFAULT["patch_node"], _DEFAULT["patch_slot"]


def sensory_to_nodes(sensory: np.ndarray, lay: dict | None = None) -> np.ndarray:
    """(N_SENSORY, P) -> (n_nodes, inputs_per_node)."""
    lay = lay or _DEFAULT
    s = np.asarray(sensory, dtype=np.float64)
    out = np.zeros((lay["n_nodes"], lay["inputs_per_node"]))
    pn, ps = lay["patch_node"], lay["patch_slot"]
    for p in range(N_VISUAL):
        o = ps[p] * P
        out[pn[p], o:o + P] = s[p]
    som = s[N_VISUAL:N_VISUAL + N_SOMATIC].ravel()
    out[lay["n_visual_nodes"], :som.size] = som
    return out


def nodes_to_patches(node_patches: np.ndarray, lay: dict | None = None) -> np.ndarray:
    """Inverse routing for the visual half: (n_nodes, I) -> (N_VISUAL, P)."""
    lay = lay or _DEFAULT
    out = np.zeros((N_VISUAL, P))
    pn, ps = lay["patch_node"], lay["patch_slot"]
    for p in range(N_VISUAL):
        o = ps[p] * P
        out[p] = node_patches[pn[p], o:o + P]
    return out


def default_config(node_side: int = NODE_SIDE, **kw) -> NodeConfig:
    lay = layout(node_side)
    return NodeConfig(n_nodes=lay["n_nodes"], node_side=node_side,
                      n_inputs=lay["inputs_per_node"], **kw)


def build_cortex(seed: int = 0, node_side: int = NODE_SIDE, **kw) -> NodeStack:
    stack = build_stack(default_config(node_side, seed=seed, **kw))
    stack.layout = layout(node_side)
    return stack


def tick(stack: NodeStack, sensory: np.ndarray) -> dict:
    """One tick from raw sensory input. The only entry point a world needs."""
    return node_step(stack, sensory_to_nodes(sensory, getattr(stack, "layout", None)))


def predicted_retina(stack: NodeStack, tau: int, sensors) -> np.ndarray:
    """Assemble the nodes' individual patch forecasts into one predicted frame.

    Each node forecasts only the part of the image it can see, and no node is asked to
    know how its patch sits in the frame. The stitching is done here, by the reader.
    """
    patches = nodes_to_patches(predicted_patches(stack, tau),
                               getattr(stack, "layout", None))
    return sensors.from_patches(patches).astype(np.float32)


register_dcn(DCNLevel(
    name="node",
    horizon=NodeConfig().horizon,
    inputs_from="neuron",
    build=build_cortex,
    step=tick,
    readout=lambda s: s.publication(),
    describe=lambda s: s.describe(),
))
