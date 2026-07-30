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

## Membership: earned, not positional

Coordination has a floor of its own now, and getting one forced a second question the layer had
never been asked: **who belongs in a cluster?**

The original answer was `(c * m + j) % n_towers` -- fixed, contiguous, positional, and chosen
without a shred of evidence. Combined with retinotopic tower routing, that hard-codes *proximity
is grouping* into two places at once.

A review of the cortical-column literature landed on the one claim that contradicts it and that
we had never tested: **connectivity matters more than proximity.** Distant columns can belong to
the same functional module while neighbours do unrelated work. So membership is now a swept
variable with three settings -- `proximity` (what we had), `connectivity` (towers that actually
co-vary), `random` (the null) -- and the coordination floor requires connectivity to beat both.

**This is deliberately not Swift.** Swift made every unit a Stuart-Landau oscillator and grouped
by phase alignment; its coalitions bound objects better than anything since (+0.124) and carried
no content at all (exactly 1.00x persistence). What is borrowed here is the *principle* --
grouping earned by function rather than fixed by position -- and not the substrate that cost
918,832 parameters to obtain it. Membership is re-derived from published states. Nothing
oscillates.
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

    membership: str = "connectivity"
    """`connectivity` (towers that co-vary), `proximity` (contiguous, what we had), or `random`
    (the null). The last two are the coordination floor's two controls, and the default is the
    mechanism because it is the one under test."""

    affinity_tau: float = 0.01
    """How fast the co-variation estimate follows. Slow on purpose: an assembly that re-forms
    every tick is noise, not a module."""

    reform_every: int = 200
    """Ticks between re-derivations. Assemblies form, hold and dissolve -- but at the cluster's
    own timescale (horizon 64), not the tower's."""

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
    n_towers: int = 17
    seeds: np.ndarray | None = None   # (n_clusters,) the tower each cluster is grown from
    affinity: np.ndarray | None = None  # (n_towers, n_towers) running co-variation
    n_reforms: int = 0
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
                "membership": self.cfg.membership, "n_reforms": self.n_reforms,
                "horizon": self.cfg.horizon, "n_params": self.n_params(),
                "n_cross_tower_relations": len(self.relations),
                "mechanism": "coordination; compression optional and off by default"}


def _proximity(seeds: np.ndarray, m: int, n_towers: int) -> np.ndarray:
    """Contiguous neighbours of each seed. The original rule, kept as a named control."""
    return np.array([[(s + j) % n_towers for j in range(m)] for s in seeds])


def _connectivity(seeds: np.ndarray, m: int, affinity: np.ndarray) -> np.ndarray:
    """Each seed plus the `m-1` towers it co-varies with most, wherever they sit.

    Same seeds as the proximity rule, so the *only* difference between the two conditions is
    which towers get pulled in. Anything else would confound the comparison the gate exists for.
    """
    rows = []
    for s in seeds:
        a = affinity[s].copy()
        a[s] = -np.inf                                  # the seed is already a member
        rows.append([s, *np.argsort(a)[::-1][:m - 1]])
    return np.array(rows)


def _random(seeds: np.ndarray, m: int, n_towers: int, rng) -> np.ndarray:
    """The null: same seeds, members drawn at random. If connectivity cannot beat this, the
    affinity estimate is measuring nothing."""
    rows = []
    for s in seeds:
        others = rng.choice([i for i in range(n_towers) if i != s], size=m - 1, replace=False)
        rows.append([s, *others])
    return np.array(rows)


def build_stack(cfg: ClusterConfig | None = None, n_towers: int = 17) -> ClusterStack:
    cfg = cfg or ClusterConfig()
    rng = np.random.default_rng(cfg.seed)
    k, m = cfg.n_clusters, cfg.towers_per_cluster
    seeds = np.array([(c * m) % n_towers for c in range(k)])
    return ClusterStack(
        cfg=cfg, rng=rng,
        membership=_proximity(seeds, m, n_towers),   # until affinity has anything in it
        agreement=np.zeros(k),
        summary=np.zeros((k, cfg.d_state)),
        Ws=np.zeros((k, cfg.d_state, 1)),
        n_towers=n_towers, seeds=seeds,
        affinity=np.zeros((n_towers, n_towers)),
    )


def reform(stack: ClusterStack) -> None:
    """Re-derive membership under the configured rule. Called on a schedule, not every tick."""
    cfg, m = stack.cfg, stack.cfg.towers_per_cluster
    if cfg.membership == "connectivity":
        stack.membership = _connectivity(stack.seeds, m, stack.affinity)
    elif cfg.membership == "random":
        stack.membership = _random(stack.seeds, m, stack.n_towers, stack.rng)
    else:
        stack.membership = _proximity(stack.seeds, m, stack.n_towers)
    stack.n_reforms += 1


def step(stack: ClusterStack, tower_publication: np.ndarray,
         events: list[Event] | None = None) -> dict:
    """One tick. Reads its members' published states and their events -- never their internals."""
    cfg = stack.cfg
    pub = np.asarray(tower_publication, dtype=np.float64)

    # who co-varies with whom, from published states only -- never a tower's internals.
    # Cosine between mean-removed publications, so a tower that is simply louder than another
    # does not read as related to it.
    dev = pub - pub.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(dev, axis=1) + 1e-9
    stack.affinity += cfg.affinity_tau * ((dev @ dev.T) / np.outer(norm, norm) - stack.affinity)
    if stack.t > 0 and stack.t % cfg.reform_every == 0:
        reform(stack)

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
            "n_relations": len(stack.relations), "n_reforms": stack.n_reforms}


register(Layer(
    name="cluster",
    horizon=ClusterConfig().horizon,
    inputs_from="tower",
    floor=(
        # --- coordination: the job that runs on every tick, in both modes ---------------
        Floor(
            job="coordination", always_on=True,
            beats="independent_towers", margin=0.05,
            why="The null. Grouping towers is only worth doing if a tower's cluster-mates tell "
                "you something the tower does not already tell you about itself. If they do "
                "not, coordination is decoration -- and until this floor existed the cluster "
                "ran on every tick with nothing to beat, which is precisely how Heron's node "
                "layer stayed compliant while being worse than its own neurons. Gate CGE-B-05.",
        ),
        Floor(
            job="coordination", always_on=True,
            beats="fixed_proximity_membership", margin=0.05,
            why="Membership was hard-coded as (c*m + j) %% n_towers -- contiguous, positional, "
                "and never measured -- while tower routing is retinotopic, so 'proximity is "
                "grouping' is assumed in two places at once. The cortical-column literature's "
                "one firm claim is the opposite: connectivity matters more than proximity. "
                "This floor makes that claim falsifiable at the cost of one gate instead of a "
                "new architecture. Gate CGE-B-05.",
        ),
        # --- compression: optional, off by default --------------------------------------
        Floor(
            job="compression", always_on=False,
            beats="pass_through", margin=0.05,
            why="Aggregation has failed five times with five operators, and Heron's "
                "mean-pooling control beat both of its relational operators. A cluster must "
                "beat the concatenation of its own members before it is permitted to summarise "
                "them; the burden of proof belongs on the compression step. Measured at "
                "-0.004 +/- 0.006, so this job stays off. Gate CGE-B-03.",
        ),
    ),
    build=lambda seed=0, **kw: build_stack(ClusterConfig(seed=seed, **kw)),
    step=lambda s, u: step(s, u[0], u[1] if len(u) > 1 else None),
    readout=lambda s: s.readout(),
    describe=lambda s: s.describe(),
))
