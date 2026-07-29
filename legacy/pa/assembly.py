"""The Predictive Assembly (SPEC_PA) — the first level above the RPDU.

The spec's claim is that memory, prediction and identity are *collective*: the object
appears in a shared field that many units perturb, not inside any one of them. That is a
direct answer to what this project measured, where every null traced to one cause -- the
mesh never represented objects -- and the assembly is the missing aggregator.

Four parts, as specified:

* **Local synchroniser** — which members currently agree. Read from the mesh's own
  coalition labels rather than recomputed, so v2's phase and vote machinery is what
  decides membership.
* **Shared workspace** — one slow field per assembly. Every member contributes; nothing
  is stored in any member. The slowness is the whole point: `alpha` is far below any
  unit's integration rate, so the assembly can hold what its members have already
  forgotten. An assembly running at member speed would have nothing to add.
* **Coalition manager** — hold while the assembly predicts better than its members do
  alone, otherwise dissolve and re-form.
* **Executive interface** — five scalars out, and nothing else.

This layer is **read-only over the mesh**. It observes state and never writes back, which
is deliberate: if it works, the mesh only ever lacked an aggregator, and if it fails
specifically for want of top-down influence, that failure is what would justify a v3
rather than an assumption made in advance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from legacy.v1.plasticity import normaliser


@dataclass(frozen=True)
class AssemblyConfig:
    n_assemblies: int = 4
    workspace_dim: int = 32

    aggregation: str = "pairwise"
    """How members contribute to the field: "pairwise", "mean", or "projection".

    Settled by measurement rather than by reading. Holding the mesh, the members, the
    target and the protocol fixed and varying only this operator, predicting members 64
    ticks ahead relative to persistence (3 seeds, lower is better):

        operator                1 object    2 objects   2 obj + occluder
        full mesh (ceiling)     0.60        0.84        0.92
        pairwise                0.73        0.81        0.79
        members, uncompressed   0.62        0.86        0.90
        synchrony graph         0.85        1.00        1.00
        reservoir               0.92        0.95        0.94
        attention-weighted      3.14        0.99        1.00
        mean                    37.6        0.99        0.99

    Three things follow. A mean never predicts at all -- it sits at persistence in the
    multi-object worlds and is numerically unstable in the single-object one, and
    attention weighting does not rescue it. Pairwise co-activation is the only operator
    that improves as the world gets harder, and from two objects onward it beats even the
    full mesh state. And it can beat that "ceiling" precisely because the ceiling was a
    *linear* one: a linear readout cannot form products of member activities, so handing
    it those products exposes structure that was present in the mesh but unreadable.

    The information was never missing. It was quadratic in member activity, and every
    averaging operator is linear.

    The spec says members perturb a shared field, but not how. Averaging them is the
    obvious reading and it is the one that fails, in two independent ways. Probing: the
    same 36 units carry object position at an error of 6.4 when kept apart and 44.4 once
    meaned, against a chance of 7.8. Predicting: an assembly fed the mean cannot beat
    "assume nothing changed", because what changes depends on *which* member is active and
    the mean has just discarded that.

    "projection" keeps members apart and learns a linear map from the concatenated member
    states. Same information budget as the mesh gives it; the difference is only whether
    the aggregation throws away identity before the field sees it."""

    horizon: int = 64
    """How far ahead the assembly predicts its members, in ticks.

    Not a free parameter -- it has to match the layer's own timescale, and that is
    measurable. Asking a slow field for a 1-tick prediction pits it against persistence,
    which is unbeatable there: a best-case offline fit from the *entire* mesh state scores
    1.32x persistence at tau=1 and 1.09x at tau=16, so no implementation of this layer
    could have won, however good. At tau=64 the same fit reaches 0.56x -- a 44% gain that
    is actually available. At 128 it fades again to 0.91x.

    That is the general principle the recursive hierarchy in SPEC_PA implies without
    saying: each level predicts at the scale it integrates over. The RPDU predicts 1 to 16
    ticks ahead because that is its timescale; an assembly that integrates over hundreds
    has no business being scored on the next frame."""

    alpha: float = 0.04
    """How strongly members drive the field. Sits well below the units' own rates (v2
    draws those log-uniformly from 0.04-0.25) so the assembly is a slower layer rather
    than a wider one."""

    retention: tuple[float, float] = (0.90, 0.9995)
    """Per-dimension retention of the field, log-spaced across this range.

    Decoupled from `alpha` deliberately, and this is the correction that made the layer
    work at all. The first version used `w <- (1-alpha)w + alpha*c`, which ties retention
    to 1-alpha = 0.96 and is a first-order low-pass whose fixed point is the *mean* of its
    input. A low-pass can never represent a running sum, so the workspace tracked the
    long-run average position and sat at chance -- measured 7.64 against a chance of 7.79.

    Verified directly on a 1-D walk: at retention 0.96 the best possible linear readout of
    the field gives position error 6.17 against a chance of 6.14, while at retention 1.0 it
    gives 0.00, because that is an accumulator. The same class of mistake as v1's gradient
    flow -- a dynamics chosen without checking it can express the target function.

    A *spread* rather than one value, so the field spans timescales: dimensions near 1.0
    accumulate displacement, lower ones track recent state. Nothing here builds a path
    integrator; it gives the field the capacity to be one and lets learning use it."""

    w_max: float = 8.0
    """Soft saturation bound on the field. Large enough to leave the integrating
    range linear, small enough that nothing can run away."""

    decorrelate: bool = False
    """Normalise each input feature by its own running scale before the delta rule.

    Off by default because it was tried and it *hurt*: 3.75x persistence against 2.30x
    without it, over three seeds. Kept as a flag because the negative result is worth
    preserving -- it rules out feature scaling as the explanation for the online/offline
    gap, and it is a distinct thing from the decorrelation that the offline pipeline
    actually performs. Scaling each feature independently does not orthogonalise them,
    and on low-variance noisy features it amplifies the noise."""

    norm_rate: float = 0.001
    """Rate of the running statistics used to normalise the readout. Slower than the
    signal, faster than the drift."""

    mean_rate: float = 0.002
    """Rate of the running mean subtracted before integration. Must be slower than
    the signal of interest and faster than the run, or it either removes what we
    want to keep or fails to stop the ramp."""

    eta: float = 0.02
    weight_decay: float = 1e-5
    novelty_ema: float = 0.01

    manager: bool = True
    manager_window: int = 200
    manager_patience: int = 3
    """Consecutive windows an assembly may predict worse than its own members before it
    dissolves. One bad window is noise; three is a verdict."""

    dynamic_membership: bool = True
    seed: int = 0


@dataclass
class Assembly:
    """State for a population of assemblies over one mesh."""

    cfg: AssemblyConfig
    members: np.ndarray            # (A, N) boolean membership mask
    W: np.ndarray                  # (A, k, d) or (A, k, m*d) -> workspace contribution
    member_idx: np.ndarray         # (A, m) member unit indices, for projection
    P: np.ndarray                  # (A, d, k) workspace -> predicted aggregate readout
    w: np.ndarray                  # (A, k) the shared field
    retention: np.ndarray          # (k,) per-dimension retention
    pooled_mean: np.ndarray        # (A, d) slow mean, removed before integrating
    in_var: np.ndarray | None      # running scale of the aggregation input
    w_mean: np.ndarray             # (A, k) running mean of the field
    w_var: np.ndarray              # (A, k) running variance of the field
    err_ema: np.ndarray            # (A,) assembly prediction error
    member_err_ema: np.ndarray     # (A,) members' own error, for the manager
    novelty: np.ndarray
    strikes: np.ndarray
    rng: np.random.Generator
    t: int = 0
    _last_pred: np.ndarray | None = field(default=None, repr=False)
    _horizon_ago: np.ndarray | None = field(default=None, repr=False)
    hist: list = field(default_factory=list, repr=False)
    pooled_hist: list = field(default_factory=list, repr=False)
    n_dissolved: int = 0

    @property
    def n_assemblies(self) -> int:
        return self.members.shape[0]


def build_assembly(mesh, cfg: AssemblyConfig | None = None) -> Assembly:
    cfg = cfg or AssemblyConfig()
    rng = np.random.default_rng(cfg.seed)
    n, d = mesh.h.shape
    a, k = cfg.n_assemblies, cfg.workspace_dim

    members = _spatial_partition(n, a, int(np.sqrt(n)))
    member_idx = np.stack([np.flatnonzero(members[i])[:members.sum(1).min()]
                           for i in range(a)])
    m = member_idx.shape[1]
    in_dim = {"mean": d, "pairwise": m * (m - 1) // 2}.get(cfg.aggregation, d * m)
    lo, hi = cfg.retention
    # log-spaced in (1 - retention), so the spread is even in time constant
    retention = 1.0 - np.geomspace(1.0 - lo, 1.0 - hi, k)
    return Assembly(
        cfg=cfg,
        members=members,
        retention=retention,
        W=rng.normal(0, 1 / np.sqrt(in_dim), (a, k, in_dim)),
        member_idx=member_idx,
        P=rng.normal(0, 0.1 / np.sqrt(k), (a, d, k)),
        w=np.zeros((a, k)),
        pooled_mean=np.zeros((a, in_dim)),
        in_var=np.ones((a, in_dim)),
        w_mean=np.zeros((a, k)),
        w_var=np.ones((a, k)),
        err_ema=np.ones(a),
        member_err_ema=np.ones(a),
        novelty=np.ones(a),
        strikes=np.zeros(a, dtype=int),
        rng=rng,
    )


def _spatial_partition(n_units: int, n_assemblies: int, side: int) -> np.ndarray:
    """Contiguous blocks of the lattice, so an assembly starts with a coherent patch.

    Membership becomes coalition-driven once the mesh is running; this is only the
    starting condition, and starting from a spatial patch means the workspace is
    measurable before dynamic membership adds a second moving part.
    """
    members = np.zeros((n_assemblies, n_units), dtype=bool)
    per_side = int(np.ceil(np.sqrt(n_assemblies)))
    block = int(np.ceil(side / per_side))
    for i in range(n_units):
        r, c = divmod(i, side)
        a = min((r // block) * per_side + (c // block), n_assemblies - 1)
        members[a, i] = True
    return members


def step_assembly(asm: Assembly, mesh, learn: bool = True) -> dict:
    """One assembly tick. Call after the mesh tick; never modifies the mesh."""
    cfg = asm.cfg
    h = mesh.h                                  # (N, d) readout
    conf = getattr(mesh, "conf", np.ones(len(h)))
    a, k = asm.n_assemblies, cfg.workspace_dim

    # -- 1. members contribute to the shared field -------------------------
    weight = asm.members * conf[None, :]        # (A, N)
    total = weight.sum(axis=1, keepdims=True) + 1e-8
    pooled = (weight @ h) / total               # (A, d) confidence-weighted member mean

    # Integrate *fluctuations*, not the raw pooled state. Any constant component of the
    # input drives a near-unity accumulator into an unbounded ramp, so the field never
    # revisits the same value for the same situation and a time-split probe extrapolates
    # off the end of its training range -- measured: position error 145 cells against a
    # chance of 7.8. Subtracting a slow running mean first is the standard fix, and the
    # same one biological path integrators need: high-pass, then integrate.
    if cfg.aggregation == "mean":
        raw_in = pooled
    elif cfg.aggregation == "pairwise":
        # co-activation between members: who is active *while* who else is. A mean cannot
        # tell "42 active, 77 silent" from the reverse; this can, and that distinction is
        # where the prediction lives.
        act = np.linalg.norm(h[asm.member_idx], axis=2)        # (A, m) per-member activity
        outer = act[:, :, None] * act[:, None, :]
        iu = np.triu_indices(act.shape[1], k=1)
        raw_in = outer[:, iu[0], iu[1]]
    else:
        # members kept apart: (A, m*d), so the field can still tell who said what
        raw_in = h[asm.member_idx].reshape(a, -1)
    if asm.pooled_mean.shape != raw_in.shape:
        asm.pooled_mean = np.zeros_like(raw_in)
    asm.pooled_mean = ((1 - cfg.mean_rate) * asm.pooled_mean + cfg.mean_rate * raw_in)
    drive = raw_in - asm.pooled_mean
    if cfg.decorrelate:
        if asm.in_var.shape != drive.shape:
            asm.in_var = np.ones_like(drive)
        asm.in_var = (1 - cfg.norm_rate) * asm.in_var + cfg.norm_rate * drive ** 2
        drive = drive / np.sqrt(asm.in_var + 1e-8)
    contrib = np.einsum("akd,ad->ak", asm.W, drive)

    # Soft saturation rather than a hard clip. A near-unity accumulator driven by a
    # not-quite-zero-mean input ramps without bound, and a hard clip only converts the
    # ramp into a rail. Bounded integration keeps the linear regime for genuine
    # integration while making runaway impossible -- which is also why real integrators
    # have bounded firing rates.
    raw = asm.retention * asm.w + cfg.alpha * contrib
    asm.w = cfg.w_max * np.tanh(raw / cfg.w_max)

    # Track the field's own running statistics. The *state* may drift; what anything
    # downstream reads must not, or a probe held out in time is extrapolating rather
    # than decoding -- measured: workspace drift of 0.55 sd between halves against
    # pooled's 0.07, and a position error of 53 where a leak-prone random split gave 12.
    asm.w_mean = (1 - cfg.norm_rate) * asm.w_mean + cfg.norm_rate * asm.w
    asm.w_var = ((1 - cfg.norm_rate) * asm.w_var
                 + cfg.norm_rate * (asm.w - asm.w_mean) ** 2)

    # -- 2. the assembly predicts its members' state one horizon ahead -----
    # Residual, as the mesh's sensory head is: member state barely moves tick to tick, so
    # an absolute prediction is competing with "no change" and loses. Predicting the
    # residual leaves only what actually changes, which is the part the field could know.
    pred = pooled + np.einsum("adk,ak->ad", asm.P, asm.w)
    asm.hist.append((pred, asm.w.copy()))

    err = None
    if len(asm.hist) > cfg.horizon:
        old_pred, old_w = asm.hist.pop(0)
        err = pooled - old_pred                 # (A, d), scored a horizon later
        e2 = np.einsum("ad,ad->a", err, err)
        asm.err_ema = (1 - cfg.novelty_ema) * asm.err_ema + cfg.novelty_ema * e2
        asm.novelty = e2 / (asm.err_ema + 1e-8)

        # The manager's reference has to be in the same units as the assembly's own
        # error, or it can never fire. Mesh `surprise` is on a different scale entirely
        # (measured: assembly 0.005 against member surprise of order 1, so no assembly
        # ever dissolved). Persistence -- predict that the members stay where they are --
        # is the honest like-for-like baseline.
        if asm._horizon_ago is not None:
            pe = pooled - asm._horizon_ago
            p2 = np.einsum("ad,ad->a", pe, pe)
            asm.member_err_ema = ((1 - cfg.novelty_ema) * asm.member_err_ema
                                  + cfg.novelty_ema * p2)

        if learn:
            # credit goes to the field as it was when the prediction was made
            scale = cfg.eta * normaliser(old_w)          # (A,)
            asm.P *= 1.0 - cfg.weight_decay
            asm.P += scale[:, None, None] * err[:, :, None] * old_w[:, None, :]
            # the contribution map is trained the same way, through the field
            back = np.einsum("adk,ad->ak", asm.P, err)
            wscale = cfg.eta * cfg.alpha * normaliser(drive)
            asm.W *= 1.0 - cfg.weight_decay
            asm.W += wscale[:, None, None] * back[:, :, None] * drive[:, None, :]

    asm.pooled_hist.append(pooled.copy())
    if len(asm.pooled_hist) > cfg.horizon:
        asm._horizon_ago = asm.pooled_hist.pop(0)
    asm._last_pred = pred
    asm.t += 1

    dissolved = 0
    if cfg.manager and asm.t % cfg.manager_window == 0:
        dissolved = _run_manager(asm, mesh)

    return {"t": asm.t, "err": float(asm.err_ema.mean()),
            "novelty": float(asm.novelty.mean()), "dissolved": dissolved}


def _run_manager(asm: Assembly, mesh) -> int:
    """Hold assemblies that beat their own members; dissolve and re-form the rest.

    "Should we stay together?" made measurable. The comparison is against the members'
    own prediction error, so an assembly has to justify itself against the units it is
    made of rather than against nothing.
    """
    cfg = asm.cfg
    worse = asm.err_ema > asm.member_err_ema
    asm.strikes = np.where(worse, asm.strikes + 1, 0)
    doomed = np.flatnonzero(asm.strikes >= cfg.manager_patience)
    if doomed.size == 0:
        return 0

    coalition = getattr(mesh, "coalition", None)
    for a in doomed:
        asm.members[a] = _reform(asm, mesh, coalition, a)
        asm.w[a] = 0.0
        asm.strikes[a] = 0
        asm.err_ema[a] = asm.member_err_ema[a]
        asm.n_dissolved += 1
    return int(doomed.size)


def _reform(asm: Assembly, mesh, coalition, a: int) -> np.ndarray:
    """Re-seed a dissolved assembly around a currently coherent group of units."""
    n = asm.members.shape[1]
    size = max(int(asm.members[a].sum()), 8)
    if coalition is not None and asm.cfg.dynamic_membership:
        labels, counts = np.unique(coalition, return_counts=True)
        big = labels[counts >= 2]
        if big.size:
            pick = big[asm.rng.integers(big.size)]
            idx = np.flatnonzero(coalition == pick)
            if idx.size >= 4:
                out = np.zeros(n, dtype=bool)
                out[idx[:size]] = True
                return out
    out = np.zeros(n, dtype=bool)
    out[asm.rng.choice(n, size=min(size, n), replace=False)] = True
    return out


def workspace(asm: Assembly) -> np.ndarray:
    """(A, k) the field as anything downstream should read it: gain-controlled.

    The raw field is allowed to drift -- that is what integration looks like. What is
    exposed is normalised against the field's own running statistics, so the same
    situation reads the same way whether it happens early or late in a run. Without this
    the representation is unmeasurable rather than uninformative, which are very
    different failures and were confused in the first version.
    """
    return (asm.w - asm.w_mean) / np.sqrt(asm.w_var + 1e-8)


def export(asm: Assembly, mesh, goal: np.ndarray | None = None) -> np.ndarray:
    """The executive interface: (A, 5) and nothing else.

    prediction / confidence / novelty / energy / goal relevance. The spec is emphatic
    that this is all an assembly exports -- no embeddings -- and gate PA4 tests whether
    that bottleneck really is sufficient, which is its boldest claim.
    """
    a = asm.n_assemblies
    pred = asm._last_pred if asm._last_pred is not None else np.zeros((a, mesh.h.shape[1]))
    prediction = np.linalg.norm(pred, axis=1)
    confidence = 1.0 / (1.0 + asm.err_ema)
    novelty = asm.novelty
    energy = np.linalg.norm(workspace(asm), axis=1)
    ws = workspace(asm)
    relevance = (ws @ goal if goal is not None and goal.shape[-1] == ws.shape[1]
                 else np.zeros(a))
    return np.stack([prediction, confidence, novelty, energy, relevance], axis=1)


def pooled_members(asm: Assembly, mesh) -> np.ndarray:
    """(A, d) plain mean of member readouts -- the control PA1 must beat.

    If the workspace never improves on this, "the object emerges in the shared field" is
    unsupported and the assembly is an aggregation convenience.
    """
    counts = asm.members.sum(axis=1, keepdims=True) + 1e-8
    return (asm.members @ mesh.h) / counts
