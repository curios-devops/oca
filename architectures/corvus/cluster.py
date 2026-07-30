"""Layer 2 — Tower Cluster.

A collection of cooperative towers, responsible for **local coordination** before higher-level
integration. Summarising its members is permitted and not required.

That distinction is the whole content of this layer's design, and it comes from a decision taken
against four measurements. The draft said "local consensus", and *consensus already assumes
compression*. Aggregation has now failed four times with four operators: the legacy assembly
turned a 0.69x member state into a 6.56x workspace, and Heron's mean-pooling **control** beat both
of its relational operators (0.598 against 0.612 sketched and 0.629 exact), with a crossed
ablation putting all six cells within 5%.

So the floor here is `pass_through`. A cluster must beat the concatenation of its own members
before it is permitted to summarise them, and until it does, the compliant behaviour is to pass
them through unchanged. The burden of proof sits on the compression step, which is where four
failures say it belongs.

What the layer *does* do, unconditionally, is coordinate: it reads its members' published events,
tracks which of them currently agree, and -- the part that only exists because `Entity` is a
cross-cutting primitive -- **relates entities across tower boundaries without re-solving
identity.** If entities lived inside a tower, this layer would have to re-establish that a
referent crossing a boundary is the same referent, which is the aggregation problem again in its
hardest form. It does not, because the entity reference travels in the event.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .contract import Floor, Layer, register
from .primitives import Event


@dataclass(frozen=True)
class ClusterConfig:
    n_clusters: int = 4
    towers_per_cluster: int = 5
    d_state: int = 16
    horizon: int = 64
    """Strictly greater than the tower's 16. A layer that does not integrate over longer than the
    layer below it is a relabelling, and `check_stack` enforces it."""

    mode: str = "pass_through"
    """`pass_through` (the compliance floor: members unchanged) or `summarise` (a linear summary,
    which must beat pass-through to be used). Default is the floor, deliberately."""

    agreement_tau: float = 0.02
    seed: int = 0

    def variant(self, **changes) -> "ClusterConfig":
        return replace(self, **changes)


@dataclass
class ClusterStack:
    cfg: ClusterConfig
    rng: np.random.Generator
    membership: np.ndarray            # (n_clusters, towers_per_cluster) tower indices
    agreement: np.ndarray             # (n_clusters,) running fraction of members agreeing
    summary: np.ndarray               # (n_clusters, d_state) only used in `summarise` mode
    Ws: np.ndarray                    # (n_clusters, d_state, member_width) summary projection
    relations: dict = field(default_factory=dict)
    t: int = 0
    learn: bool = True
    _passthrough: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_units(self) -> int:
        return self.cfg.n_clusters

    def readout(self) -> np.ndarray:
        """Pass-through by default: the members' own published states, unchanged.

        Returning its members is not a placeholder. It is the compliance floor, and a cluster is
        only allowed to return something narrower once it has beaten this.
        """
        if self.cfg.mode == "pass_through":
            return self._passthrough
        return self.summary

    def n_params(self) -> int:
        return int(self.Ws.size) if self.cfg.mode == "summarise" else 0

    def describe(self) -> dict:
        return {"n_clusters": self.cfg.n_clusters, "mode": self.cfg.mode,
                "horizon": self.cfg.horizon, "n_params": self.n_params(),
                "n_cross_tower_relations": len(self.relations),
                "mechanism": "coordination; compression optional and off by default"}


def build_stack(cfg: ClusterConfig | None = None, n_towers: int = 17) -> ClusterStack:
    cfg = cfg or ClusterConfig()
    rng = np.random.default_rng(cfg.seed)
    k, m = cfg.n_clusters, cfg.towers_per_cluster
    idx = np.array([[(c * m + j) % n_towers for j in range(m)] for c in range(k)])
    return ClusterStack(
        cfg=cfg, rng=rng, membership=idx,
        agreement=np.zeros(k),
        summary=np.zeros((k, cfg.d_state)),
        Ws=np.zeros((k, cfg.d_state, 1)),
    )


def step(stack: ClusterStack, tower_publication: np.ndarray,
         events: list[Event] | None = None) -> dict:
    """One tick. Reads its members' published states and their events -- never their internals."""
    cfg = stack.cfg
    pub = np.asarray(tower_publication, dtype=np.float64)
    members = pub[stack.membership]                       # (k, m, width)
    k, m, width = members.shape

    # pass-through: the members, concatenated. The floor, and the default.
    stack._passthrough = members.reshape(k, m * width)

    # coordination, which happens either way: who currently agrees with whom
    spread = members.std(axis=1).mean(axis=1)             # (k,)
    stack.agreement = 1.0 - np.clip(spread / (np.median(spread) + 1e-9), 0.0, 1.0)

    # cross-tower identity, for free, because the entity reference rides in the event
    if events:
        for e in events:
            if e.entity is not None:
                stack.relations.setdefault(e.entity, set()).add(e.source)

    if cfg.mode == "summarise":
        if stack.Ws.shape[2] != m * width:
            stack.Ws = stack.rng.normal(0, 1 / np.sqrt(m * width),
                                        (k, cfg.d_state, m * width))
        stack.summary = np.einsum("kds,ks->kd", stack.Ws, stack._passthrough)

    stack.t += 1
    return {"t": stack.t, "agreement": float(stack.agreement.mean()),
            "n_relations": len(stack.relations)}


register(Layer(
    name="cluster",
    horizon=ClusterConfig().horizon,
    inputs_from="tower",
    floor=Floor(
        beats="pass_through",
        margin=0.05,
        why="Aggregation has failed four times with four operators, and Heron's mean-pooling "
            "control beat both of its relational operators. A cluster must beat the "
            "concatenation of its own members before it is permitted to summarise them; the "
            "burden of proof belongs on the compression step. Gate CGE-B-03.",
    ),
    build=lambda seed=0, **kw: build_stack(ClusterConfig(seed=seed, **kw)),
    step=lambda s, u: step(s, u[0], u[1] if len(u) > 1 else None),
    readout=lambda s: s.readout(),
    describe=lambda s: s.describe(),
))
