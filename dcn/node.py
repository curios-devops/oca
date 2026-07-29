"""Level 2 — the Dynamic Cortical Node.

The first cognitive unit. A DCN does not represent a concept; it represents a **dynamic
hypothesis**, and it publishes only its own state — prediction error, confidence, phase
spectrum, activation, novelty, energy. Never its contents.

Six parts, each with a stated function and a gate that can kill it:

* **Working State** — what is happening now.
* **Knowledge Model** — what has been consolidated.
* **Prediction Engine** — predicts its own future state and its own sensory patch, at a
  declared horizon.
* **Neighbourhood Model** — what nearby nodes are expected to publish. Not their weights.
* **Oscillation Engine** — a resonance spectrum, not one frequency.
* **State Adapter** — takes the global state vector and changes *how the node thinks*,
  never *what it knows*.

Four constraints are inherited from measurement rather than from taste, and each one cost
the legacy line a full experimental cycle to learn:

**Aggregation must be relational.** Averaging members destroyed the predictive code every
time it was tried: a mean never beat persistence, while pairwise co-activation over the
same members reached 0.79x and was the only operator that improved as the world got
harder. Four members reporting "edge moving right", "left", "right", "left" average to
nothing, though the population plainly contains two coherent hypotheses. The working state
here is a random bilinear sketch of member co-activation — a linear sketch of the full
outer product, so "42 active while 77 is silent" stays distinguishable from the reverse.
`aggregation="mean"` restores the operator that failed, as the kill control.

**A low-pass filter cannot integrate.** The legacy shared workspace was `w <- (1-a)w + a*u`
and was provably unable to hold a quantity across a gap — verified numerically at 6.17
against a chance of 6.14. The working state is therefore an echo-state reservoir the
members *perturb*, not a running mean they overwrite: spectral radius below one gives a
fading memory of the whole trajectory rather than of the last frame.

**Every level is scored at the timescale it integrates over.** No function of a legacy mesh
state beat "assume no change" one tick ahead; at 64 ticks a 44% gain was there for the
taking. This level declares `horizon` and is scored there.

**Phase is a clock.** Level 1 measured it directly: adding the rotor into the transmitted
value cost 19x in reconstruction. Here the resonance spectrum sets *when* nodes are
receptive to one another; it never enters what they publish.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .neuron import NeuronConfig, build_population
from .neuron import step as neuron_step

PUB_SCALARS = ("prediction_error", "confidence", "activation", "novelty", "energy")
"""Everything a node is allowed to say about itself, besides its phase spectrum.

The design's boldest claim is that this is enough — that a node publishes its state and
nothing else, and structure emerges from the interaction. Gate L2-4 gives a reader only
these and scores the task, which is the only way to find out.
"""


@dataclass(frozen=True)
class NodeConfig:
    """Every number for level 2. `use_*` and `aggregation` are the ablations."""

    n_nodes: int = 17
    node_side: int = 4
    """Nodes tile the retina on a `node_side` x `node_side` grid, so grid adjacency *is*
    spatial adjacency. Any node past the grid (the somatic one) attaches to the last."""

    n_neurons: int = 64
    """Per node. The design says hundreds to thousands; 64 is what runs a 20k-tick maze in
    seconds, and the count is a gate parameter rather than a claim."""

    n_inputs: int = 64
    """Sensory width per node: a 2x2 block of retinal patches, 4 x 16."""

    d_work: int = 64
    n_concepts: int = 16
    horizon: int = 16
    """Declared, not assumed, and enforced by the level contract. This is the timescale the
    node integrates over and the only one at which its prediction gate means anything."""

    # -- working state ------------------------------------------------------
    aggregation: str = "relational"
    """How member states become a node state. Three operators, all producing exactly
    `d_work` numbers from the same members, so the only difference is the operator:

    * `relational` — a random bilinear sketch of the member outer product. Cheap, sees
      every member, and is lossy: it compresses N^2 products into `d_work` numbers.
    * `pairwise` — the exact upper triangle of co-activation over as many members as fit
      in `d_work` without compression. This is the operator the legacy line actually
      measured at 0.79x, reproduced here rather than approximated. It trades coverage for
      exactness: it sees relations between a dozen members instead of a lossy shadow of
      relations between all of them.
    * `mean` — a linear projection of the member mean. The operator that was measured to
      destroy the predictive code, kept as the kill control.

    Having both `relational` and `pairwise` is the point. Without the exact version, a
    negative result cannot distinguish "relations do not help here" from "my sketch of
    them lost the information", and those call for opposite responses."""

    use_reservoir: bool = True
    """With the reservoir off, the working state is `w <- (1-leak)w + leak*drive`: exactly
    the legacy shared workspace, which was provably unable to hold a quantity across a gap.

    This exists to separate two things the legacy result had tangled together. Mean pooling
    was measured to destroy the predictive code -- but it was always measured *inside* a
    low-pass workspace, so "the mean is the problem" and "the filter is the problem" were
    never distinguished. Crossing `aggregation` with `use_reservoir` separates them, and
    the answer changes what carries over."""

    spectral_radius: float = 0.95
    """Below one, so the reservoir forgets its initial condition but keeps a fading memory
    of the trajectory. At or above one it drifts, which is the third dynamics this project
    has had to rule out on paper rather than by experiment."""

    input_gain: float = 0.8
    leak: float = 0.03
    """How much of the reservoir state is replaced per tick. Small, because the node is the
    slow layer: that is what lets it hold something its neurons cannot.

    Set from a structural measurement, not from a score. At the first value tried (0.25)
    the node's state decorrelated *faster* than its own members' -- 0.88 against 0.91 at
    the declared horizon -- so it was not the slow layer at all, and a level that does not
    integrate over a longer window than the one below it is a relabelling rather than a
    level. The value here is the largest that reverses that ordering at every lag measured.
    `tests/test_dcn_node.py` pins the ordering so it cannot silently break again; the
    gates were re-run afterwards, and the numbers they gave are reported either way."""

    # -- knowledge model ----------------------------------------------------
    use_knowledge: bool = True
    eta_k: float = 0.02
    recruit_novelty: float = 0.55
    """Above this, the working state matches nothing known and a stale slot is recruited to
    it. This is the mechanism axiom 1 was missing: "memory lives in the DCN" is a location
    until something in the node actually consolidates, and this is that something."""

    # -- prediction engine --------------------------------------------------
    eta_pred: float = 0.05
    conf_scale: float = 1.0

    # -- neighbourhood model ------------------------------------------------
    use_neighbourhood: bool = True
    eta_nb: float = 0.02
    coupling: float = 0.10
    """Kuramoto coupling toward neighbours, scaled per node by how well it predicts them.
    A node that cannot anticipate its neighbourhood does not synchronise with it."""

    # -- oscillation engine -------------------------------------------------
    use_oscillation: bool = True
    bands: tuple[float, ...] = (0.90, 0.55, 0.18, 0.05)
    """The resonance spectrum, in radians per tick: roughly gamma, beta, alpha, theta. One
    node participates in several conversations at different timescales at once, which is
    the whole reason this is a spectrum and not a frequency."""

    mu: float = 0.25

    # -- state adapter ------------------------------------------------------
    use_state_adapter: bool = True
    global_tau: float = 200.0
    adapt_gain: float = 0.5

    seed: int = 0
    horizons: tuple[int, ...] = (1, 4, 16)
    """Scored at all three by the shared benchmark, so this level is comparable with the
    legacy line on identical code paths. `horizon` above is the one it is built for."""

    def variant(self, **changes) -> "NodeConfig":
        return replace(self, **changes)

    def label(self) -> str:
        base = NodeConfig(seed=self.seed)
        diffs = [f"{k}={getattr(self, k)}" for k in self.__dataclass_fields__
                 if getattr(self, k) != getattr(base, k)]
        return "full" if not diffs else ",".join(diffs)

    @property
    def n_bands(self) -> int:
        return len(self.bands)

    @property
    def d_feature(self) -> int:
        """Working state, plus the active concept weighted by confidence, plus a bias."""
        return self.d_work + self.n_concepts + 1

    @property
    def d_publication(self) -> int:
        return len(PUB_SCALARS) + 2 * self.n_bands


@dataclass
class NodeStack:
    """All the nodes of one cortex, stored as stacked arrays rather than objects."""

    cfg: NodeConfig
    rng: np.random.Generator
    neurons: object                      # one grouped level-1 population

    # working state
    W_res: np.ndarray                    # (D, D) shared reservoir, fixed
    P1: np.ndarray                       # (D, N) sketch projections, fixed
    P2: np.ndarray
    W_mean: np.ndarray                   # (D, N) the mean-pooling control's projection
    pair_sel: np.ndarray                 # (m,) members used by exact pairwise
    pair_i: np.ndarray                   # (D,) upper-triangle row indices
    pair_j: np.ndarray                   # (D,) upper-triangle column indices
    work: np.ndarray                     # (n, D)

    # knowledge model
    K: np.ndarray                        # (n, C, D)
    k_age: np.ndarray                    # (n, C) ticks since last won
    k_win: np.ndarray                    # (n,)
    novelty: np.ndarray                  # (n,)

    # prediction engine
    A: np.ndarray                        # (n, D, F) own future working state
    B: np.ndarray                        # (n, I, F) own future sensory patch
    err: np.ndarray                      # (n,)
    conf: np.ndarray                     # (n,)

    # neighbourhood model
    M: np.ndarray                        # (n, P, F) expected neighbour publication
    nb_err: np.ndarray                   # (n,)
    neighbours: np.ndarray               # (n, n) adjacency, row-normalised

    # oscillation engine
    z: np.ndarray                        # (n, B, 2)

    # state adapter
    g: np.ndarray                        # (3,) global state vector

    hist_f: list = field(default_factory=list, repr=False)
    hist_work: list = field(default_factory=list, repr=False)
    hist_x: list = field(default_factory=list, repr=False)

    layout: dict | None = None
    """Set by `cortex.build_cortex`: how the sensory field is divided between nodes. The
    node level neither reads it nor needs it -- it is carried here so that the sensory
    boundary can route consistently for a stack it did not build."""

    t: int = 0
    learn: bool = True

    # -------------------------------------------------------------- readout

    @property
    def n_units(self) -> int:
        return self.cfg.n_nodes

    @property
    def phase(self) -> np.ndarray:
        """(n, B) phase per node per band."""
        return np.arctan2(self.z[..., 1], self.z[..., 0])

    def publication(self) -> np.ndarray:
        """(n, P) — the only thing a node is allowed to say. See `PUB_SCALARS`.

        Deliberately built from scalars and phases alone: no working state, no concept
        vector, no member activity. If the design is right, this is enough for the level
        above; gate L2-4 is what decides whether it is.
        """
        act = np.abs(self.member_state()).mean(axis=1)
        energy = self._node_energy()
        scal = np.stack([self.err, self.conf, act, self.novelty, energy], axis=1)
        ph = self.phase
        return np.concatenate([scal, np.cos(ph), np.sin(ph)], axis=1)

    def member_state(self) -> np.ndarray:
        """(n, N) what each node's neurons last published — not their internal activation.

        Axiom 5: the node reads its neurons through the level-1 interface, which is the
        zero-order hold of their emissions. Reaching for `pop.a` instead would be the
        convenient violation the axiom exists to prevent, and would also be a lie about
        what a real node could see.
        """
        return self.neurons.last_sent.reshape(self.cfg.n_nodes, self.cfg.n_neurons)

    def _node_energy(self) -> np.ndarray:
        e = self.neurons.energy.reshape(self.cfg.n_nodes, self.cfg.n_neurons).mean(axis=1)
        return e / max(self.t, 1)

    @property
    def h(self) -> np.ndarray:
        """(n, D + P) the level-3 readout: working state plus publication.

        Named `h` so this level meets the shared benchmark on the same code path as the
        legacy meshes. Everything above still only *acts* on `publication()`.
        """
        return np.concatenate([self.work, self.publication()], axis=1)

    @property
    def coalition(self) -> np.ndarray:
        """Which concept each node is currently expressing — the level-2 analogue.

        Not synchrony. The legacy line measured coalitions-by-synchrony at exactly 1.00x
        persistence, so grouping by phase is not carried across. Nodes expressing the same
        consolidated concept are the grouping this level actually claims.
        """
        return self.k_win.copy()

    def n_params(self) -> int:
        return int(self.neurons.w.size + self.K.size + self.A.size + self.B.size
                   + self.M.size + self.W_res.size + self.P1.size + self.P2.size)

    def describe(self) -> dict:
        return {"n_nodes": self.cfg.n_nodes, "n_neurons_total": self.neurons.n_neurons,
                "n_params": self.n_params(), "horizon": self.cfg.horizon,
                "aggregation": self.cfg.aggregation}


def build_stack(cfg: NodeConfig | None = None) -> NodeStack:
    cfg = cfg or NodeConfig()
    rng = np.random.default_rng(cfg.seed)
    n, N, D, C = cfg.n_nodes, cfg.n_neurons, cfg.d_work, cfg.n_concepts
    F, I, P = cfg.d_feature, cfg.n_inputs, cfg.d_publication

    neurons = build_population(
        NeuronConfig(seed=cfg.seed, n_neurons=N, n_inputs=I,
                     use_oscillation=cfg.use_oscillation),
        n_groups=n)

    W = rng.normal(0, 1, (D, D))
    W *= cfg.spectral_radius / max(abs(np.linalg.eigvals(W)).max(), 1e-9)

    # sign-random projections: a linear sketch of the member outer product, so the
    # relations between members survive the compression instead of being summed away
    P1 = rng.choice([-1.0, 1.0], (D, N)) / np.sqrt(N)
    P2 = rng.choice([-1.0, 1.0], (D, N)) / np.sqrt(N)

    z = rng.normal(0, 0.2, (n, cfg.n_bands, 2))

    # exact pairwise: the fewest members whose distinct pairs fill d_work, so this
    # operator is given the same output width as the other two and no more
    m = int(np.ceil((1 + np.sqrt(1 + 8 * D)) / 2))
    m = min(m, N)
    pair_sel = rng.choice(N, size=m, replace=False)
    iu = np.triu_indices(m, k=1)

    return NodeStack(
        cfg=cfg, rng=rng, neurons=neurons,
        W_res=W, P1=P1, P2=P2,
        W_mean=rng.normal(0, 1 / np.sqrt(N), (D, N)),
        pair_sel=pair_sel, pair_i=iu[0][:D], pair_j=iu[1][:D],
        work=np.zeros((n, D)),
        K=rng.normal(0, 0.1, (n, C, D)),
        k_age=np.zeros((n, C)),
        k_win=np.zeros(n, dtype=np.int64),
        novelty=np.ones(n),
        A=np.zeros((n, D, F)),
        B=np.zeros((n, I, F)),
        err=np.ones(n),
        conf=np.zeros(n),
        M=np.zeros((n, P, F)),
        nb_err=np.ones(n),
        neighbours=_grid_adjacency(n, cfg.node_side),
        z=z,
        g=np.zeros(3),
    )


def _grid_adjacency(n: int, side: int | None = None) -> np.ndarray:
    """Four-neighbour adjacency over a `side` x `side` grid of nodes, row-normalised.

    Nodes tile the retina retinotopically, so grid adjacency is spatial adjacency. Any
    node past the grid (the somatic one) attaches to its predecessor rather than floating
    free with an empty neighbourhood.
    """
    side = side or int(np.ceil(np.sqrt(n)))
    adj = np.zeros((n, n))
    for i in range(n):
        r, c = divmod(i, side)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            j = rr * side + cc
            if 0 <= rr < side and 0 <= cc < side and 0 <= j < n:
                adj[i, j] = 1.0
    for i in range(n):                      # never leave a node with no neighbourhood
        if adj[i].sum() == 0:
            adj[i, (i - 1) % n] = 1.0
    return adj / adj.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- tick


def step(stack: NodeStack, x: np.ndarray) -> dict:
    """One tick. `x` is (n_nodes, n_inputs): each node's own sensory patch.

    The order matters and follows the fractal rule — state, prediction, error, resonance,
    synchronisation, learning — so that every quantity a node publishes is about the tick
    it has just finished, not the one it is about to start.
    """
    cfg = stack.cfg
    n, D, C = cfg.n_nodes, cfg.d_work, cfg.n_concepts
    x = np.asarray(x, dtype=np.float64).reshape(n, cfg.n_inputs)

    # -- 1. neurons integrate and publish (level 1, through its own interface) --
    neuron_step(stack.neurons, x)
    e = stack.member_state()                                  # (n, N)

    # -- 2. working state: relational aggregation into a perturbed reservoir --
    if cfg.aggregation == "relational":
        # a linear sketch of the member outer product: component k of u*v is
        # sum_ij P1[k,i] P2[k,j] e_i e_j, so who is active *with* whom survives
        u = e @ stack.P1.T
        v = e @ stack.P2.T
        drive = np.tanh(u * v * cfg.input_gain)
    elif cfg.aggregation == "pairwise":
        # exactly what the legacy line measured: co-activation between named members, no
        # sketch in between. "42 active while 77 is silent" is a different number from the
        # reverse, and here it is that number rather than a projection of it.
        sub = e[:, stack.pair_sel]
        prod = sub[:, stack.pair_i] * sub[:, stack.pair_j]
        drive = np.tanh(prod * cfg.input_gain)
        if drive.shape[1] < D:                    # pad if d_work is not a triangular size
            drive = np.pad(drive, ((0, 0), (0, D - drive.shape[1])))
    elif cfg.aggregation == "mean":
        # the kill control: the operator that destroyed the predictive code every time
        drive = np.tanh((e @ stack.W_mean.T) * cfg.input_gain)
    else:
        raise ValueError(f"unknown aggregation {cfg.aggregation!r}")

    # perturb, do not overwrite. `leak` small keeps the node slower than its members.
    if cfg.use_reservoir:
        stack.work = ((1 - cfg.leak) * stack.work
                      + cfg.leak * np.tanh(stack.work @ stack.W_res.T + drive))
    else:
        stack.work = (1 - cfg.leak) * stack.work + cfg.leak * drive

    # -- 3. knowledge model: recognise, or consolidate something new -------
    if cfg.use_knowledge:
        wn = stack.work / (np.linalg.norm(stack.work, axis=1, keepdims=True) + 1e-9)
        Kn = stack.K / (np.linalg.norm(stack.K, axis=2, keepdims=True) + 1e-9)
        sim = np.einsum("ncd,nd->nc", Kn, wn)                 # (n, C)
        win = sim.argmax(axis=1)
        best = sim[np.arange(n), win]
        stack.novelty = np.clip(1.0 - best, 0.0, 1.0)

        stack.k_age += 1.0
        stack.k_age[np.arange(n), win] = 0.0
        stack.k_win = win

        if stack.learn:
            eta = cfg.eta_k * (1.0 + cfg.adapt_gain * stack.g[0]
                               if cfg.use_state_adapter else 1.0)
            # recognise: pull the winning prototype toward what is happening now
            stack.K[np.arange(n), win] += eta * (stack.work - stack.K[np.arange(n), win])
            # consolidate: nothing known matches, so recruit the stalest slot outright.
            # Without this the prototypes all converge on the average of everything, which
            # is mean pooling wearing a different hat.
            fresh = stack.novelty > cfg.recruit_novelty
            if fresh.any():
                idx = np.where(fresh)[0]
                slot = stack.k_age[idx].argmax(axis=1)
                stack.K[idx, slot] = stack.work[idx]
                stack.k_age[idx, slot] = 0.0
    else:
        stack.k_win = np.zeros(n, dtype=np.int64)
        stack.novelty = np.zeros(n)

    # -- 4. features: state, conditioned on what the node believes it is seeing --
    onehot = np.zeros((n, C))
    onehot[np.arange(n), stack.k_win] = stack.conf
    f = np.concatenate([stack.work, onehot, np.ones((n, 1))], axis=1)   # (n, F)

    # -- 5. prediction engine: score the prediction made `horizon` ticks ago --
    tau = cfg.horizon
    stack.hist_f.append(f)
    stack.hist_work.append(stack.work.copy())
    stack.hist_x.append(x.copy())
    if len(stack.hist_f) > tau + 1:
        stack.hist_f.pop(0)
        stack.hist_work.pop(0)
        stack.hist_x.pop(0)

    if len(stack.hist_f) > tau:
        f_then = stack.hist_f[0]                              # (n, F), tau ticks ago
        pred_w = np.einsum("ndf,nf->nd", stack.A, f_then)
        pred_x = np.einsum("nif,nf->ni", stack.B, f_then)
        e_w = stack.work - pred_w
        e_x = x - pred_x
        stack.err = np.linalg.norm(e_w, axis=1) / np.sqrt(D)

        if stack.learn:
            norm = np.einsum("nf,nf->n", f_then, f_then) + 1e-8
            stack.A += (cfg.eta_pred / norm)[:, None, None] * np.einsum(
                "nd,nf->ndf", e_w, f_then)
            stack.B += (cfg.eta_pred / norm)[:, None, None] * np.einsum(
                "ni,nf->nif", e_x, f_then)

    # confidence is the inverse of how surprised the node has been lately, not of how
    # surprised it is right now: one good tick is not evidence of a good hypothesis
    stack.conf = 0.98 * stack.conf + 0.02 * np.exp(-stack.err / cfg.conf_scale)

    # -- 6. neighbourhood model: what nearby nodes are expected to publish ---
    pub = stack.publication()                                 # (n, P)
    if cfg.use_neighbourhood:
        expected = np.einsum("npf,nf->np", stack.M, f)
        actual = stack.neighbours @ pub                       # mean of the neighbourhood
        e_nb = actual - expected
        stack.nb_err = np.linalg.norm(e_nb, axis=1) / np.sqrt(pub.shape[1])
        if stack.learn:
            norm = np.einsum("nf,nf->n", f, f) + 1e-8
            stack.M += (cfg.eta_nb / norm)[:, None, None] * np.einsum(
                "np,nf->npf", e_nb, f)

    # -- 7. oscillation engine: a spectrum, coupled where prediction is good --
    if cfg.use_oscillation:
        _resonate(stack)

    # -- 8. state adapter: the global wave changes how the node thinks -------
    if cfg.use_state_adapter:
        obs = np.array([stack.novelty.mean(), stack.err.mean(), stack.nb_err.mean()])
        stack.g += (obs - stack.g) / cfg.global_tau

    stack.t += 1
    return {
        "t": stack.t,
        "prediction_error": float(stack.err.mean()),
        "confidence": float(stack.conf.mean()),
        "novelty": float(stack.novelty.mean()),
        "neighbourhood_error": float(stack.nb_err.mean()),
        "event_rate": float(stack.neurons._last_event.mean()),
        "n_concepts_used": int(len(np.unique(stack.k_win))),
    }


def _resonate(stack: NodeStack) -> None:
    """Advance the resonance spectrum and pull each node toward its neighbourhood.

    Coupling is scaled by how well a node predicts its neighbours: anticipating them is
    what earns the right to fall into step with them, and a node that cannot stays free.
    That is the operational content of "waves coordinate" — otherwise coupling is a
    constant and synchrony says nothing about what is being coordinated.
    """
    cfg = stack.cfg
    n, Bnd = cfg.n_nodes, cfg.n_bands
    omega = np.asarray(cfg.bands)[None, :]                    # (1, B)

    r2 = np.einsum("nbk,nbk->nb", stack.z, stack.z)
    rot = np.stack([-omega * stack.z[..., 1], omega * stack.z[..., 0]], axis=-1)
    stack.z = stack.z + rot + ((cfg.mu - r2)[..., None]) * stack.z * 0.1

    if cfg.coupling > 0:
        ph = stack.phase                                      # (n, B)
        # mean neighbour phase per band, as a vector so it averages correctly on the circle
        nb_c = stack.neighbours @ np.cos(ph)
        nb_s = stack.neighbours @ np.sin(ph)
        pull = np.arctan2(nb_s, nb_c) - ph
        gain = cfg.coupling * np.exp(-stack.nb_err)[:, None]  # (n, 1)
        dphi = gain * np.sin(pull)
        c, s = np.cos(dphi), np.sin(dphi)
        zx, zy = stack.z[..., 0].copy(), stack.z[..., 1].copy()
        stack.z[..., 0] = c * zx - s * zy
        stack.z[..., 1] = s * zx + c * zy


# --------------------------------------------------------------- prediction out


def predicted_patches(stack: NodeStack, tau: int | None = None) -> np.ndarray:
    """(n_nodes, n_inputs) — each node's forecast of its own sensory patch.

    A node forecasts only what it can see. Assembling those forecasts into a frame is the
    reader's job, not the node's, which is why this returns patches rather than an image.
    """
    n, C = stack.cfg.n_nodes, stack.cfg.n_concepts
    onehot = np.zeros((n, C))
    onehot[np.arange(n), stack.k_win] = stack.conf
    f = np.concatenate([stack.work, onehot, np.ones((n, 1))], axis=1)
    pred = np.einsum("nif,nf->ni", stack.B, f)
    if tau is not None and tau != stack.cfg.horizon:
        # the engine is trained at one horizon; a shorter one is interpolated toward the
        # present rather than pretended to be a separate model
        w = min(tau / stack.cfg.horizon, 1.0)
        pred = w * pred + (1 - w) * stack.hist_x[-1] if stack.hist_x else pred
    return pred
