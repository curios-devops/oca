"""Configuration and state for architecture v2."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from core.world.sensors import N_SENSORY, P


@dataclass(frozen=True)
class Config2:
    """Every v2 number in one place. `use_*` fields are the ablations.

    Defaults are the full model. Setting `use_oscillation=False` drives `mu` negative,
    which collapses every rotor to a fixed point and recovers v1's essential limitation
    inside the v2 code path -- so "oscillation matters" is a measurement rather than an
    assumption.
    """

    # -- sizes ---------------------------------------------------------------
    lattice_side: int = 24
    d: int = 32
    """Rotor-space width; the content readout is the d//2 amplitudes, so this is twice
    v1's state width to keep the readout comparable at 16 dimensions."""
    q: int = 8
    horizons: tuple[int, ...] = (1, 4, 16)
    n_local: int = 32
    n_long: int = 8
    n_exec_links: int = 1
    n_exec_units: int = 20

    # -- oscillator dynamics -------------------------------------------------
    mu_init: float = 0.15              # limit-cycle radius is sqrt(mu)
    """Kept small on purpose: at mu=0.5 the intrinsic oscillation is the same scale as
    the sensory drive and swamps it, and the state becomes a clock rather than a
    representation."""
    omega_range: tuple[float, float] = (0.03, 1.2)
    """Per-rotor natural frequencies, log-uniform. A spread matters: identical
    frequencies synchronise trivially and tell us nothing, while a spread means
    phase-locking has to be produced by coupling and by shared input."""

    eta_range: tuple[float, float] = (0.04, 0.25)
    """Per-unit integration rates, log-uniform -- the timescale hierarchy. Slow units
    integrate over long windows and can bridge an occlusion; with one shared time
    constant no hierarchy of abstraction can emerge."""

    sub_steps: int = 2
    state_noise: float = 0.005
    r_max: float = 4.0
    sensory_gain: float = 3.0

    # -- the non-local objective --------------------------------------------
    mask_prob: float = 0.25
    """Probability a sensory unit's input is zeroed on a given tick while it is still
    scored against the true patch. This is the change aimed squarely at v1's root cause:
    local prediction there was *locally satisfiable*, so nothing ever forced a unit to
    represent anything beyond its own patch. A masked unit can only succeed by using its
    neighbours, which is what gives the protocol a job and coalitions something to be."""

    # -- protocol ------------------------------------------------------------
    drive_norm: bool = True
    """Divide incoming message drive by sqrt(degree). Without it, summing 41 links
    produces coupling strong enough to lock the entire population into one
    synchronous blob -- measured: 92% of units in a single coalition carrying zero
    object information, which is global sync, not binding."""

    # -- voting (increment 2) ------------------------------------------------
    use_voting: bool = True
    """Thousand Brains' missing piece. v1 detected coalitions and, even had it found
    any, they did nothing -- membership nudged a gain by 1.2x. Increment 1 made
    coalitions *possible* but they carried no object information, because nothing in the
    architecture required agreement to be about anything. A vote is a hypothesis a unit
    holds, reconciles with its neighbours, and *feeds to its own prediction heads*, so
    agreeing with the right neighbours is worth something and agreeing with the wrong
    ones costs."""

    vote_dim: int = 8
    vote_rate: float = 0.25            # how fast a vote moves per tick
    vote_self: float = 0.8
    """Mix between a unit's own proposal and its neighbours' consensus. Pure consensus
    collapses to one global vote; pure self-proposal never agrees with anyone. Binding
    lives in between, which is why this is an explicit knob."""
    eta_vote: float = 0.02
    vote_agree: float = 1.0
    vote_task: float = 3.0
    """Relative weight of the two pressures on a vote. Agreement alone has exactly
    one stable outcome -- every unit voting identically -- which is what happens
    at vote_task=0: measured vote similarity 0.999 and a single coalition of the
    whole population. Task pressure is what makes units that see different things
    want different votes."""
    vote_sharpen: float = 6.0
    vote_theta: float = 0.9            # cosine similarity for a vote coalition

    # -- location signal (increment 2) --------------------------------------
    use_freq_modulation: bool = True
    freq_mod: float = 0.5
    """Scales each unit's rotor frequency by its own novelty, so phase advances faster
    where the world is changing faster and phase becomes a running integral of local
    motion -- the Thousand Brains location signal, built from a purely local quantity.

    Note this is the *only* practical route by which anything task-related reaches
    phase: the readout is rotor amplitude, and rotation preserves amplitude exactly in
    the continuous flow (a first-order Euler step leaves an O(eta^2) residue), so
    prediction error has essentially no gradient with respect to frequency. Phase is
    shaped by coupling and by input, not by the heads."""

    coupling: float = 1.0
    """Overall scale on message drive. Coupled oscillators have two regimes: below a
    critical coupling they stay incoherent, above it the whole population locks into
    one phase. Neither is useful -- binding needs the middle, where groups lock
    selectively. This is the knob that finds it."""

    phase_gate: float = 1.0
    """Strength of phase gating: gamma is multiplied by ((1+cos dphi)/2)**phase_gate.
    Implements the design document's "when synchronised, information flows; otherwise
    almost disconnected" as a real gate, where v1 only nudged a gain by 1.2x."""

    msg_width: int = 4
    novelty_ema: float = 0.01
    credit_ema: float = 0.005
    rewire_every: int = 200
    rewire_prune: int = 2
    sample_temp: float = 0.3
    new_link_gain: float = 0.1

    # -- learning ------------------------------------------------------------
    eta_head: float = 0.01
    eta_sigma: float = 1e-3
    eta_link: float = 1e-3
    weight_decay: float = 1e-5
    beta: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.5)

    # -- coalitions ----------------------------------------------------------
    coalition_every: int = 10
    coalition_window: int = 32
    plv_theta: float = 0.30
    """Threshold on sustained in-phase agreement. Coalitions are clusters of units that
    stay aligned, where v1 correlated raw state -- meaningless between units that have
    both settled to fixed points. At 0.6 almost nothing groups; at 0.3 roughly twenty
    moderate coalitions form instead of one global blob."""
    label_prop_rounds: int = 5

    # -- ablations -----------------------------------------------------------
    use_oscillation: bool = True
    use_phase_gate: bool = True
    use_masking: bool = True
    use_timescales: bool = True
    use_long_range: bool = True
    use_rewiring: bool = True
    use_uncertainty: bool = True
    use_plasticity: bool = True

    seed: int = 0

    # -- derived -------------------------------------------------------------
    @property
    def n_units(self) -> int:
        return self.lattice_side**2

    @property
    def m(self) -> int:
        return self.d // 2

    @property
    def readout(self) -> int:
        """Width the prediction heads consume: amplitudes plus the vote."""
        return self.m + (self.vote_dim if self.use_voting else 0)

    @property
    def degree(self) -> int:
        return self.n_local + (self.n_long if self.use_long_range else 0) + self.n_exec_links

    @property
    def n_horizons(self) -> int:
        return len(self.horizons)

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    def variant(self, **changes) -> "Config2":
        return replace(self, **changes)

    def label(self) -> str:
        base = Config2(seed=self.seed)
        diffs = [f"{k}={getattr(self, k)}" for k in self.__dataclass_fields__
                 if getattr(self, k) != getattr(base, k)]
        return "full" if not diffs else ",".join(diffs)


@dataclass
class MeshState2:
    cfg: Config2
    rng: np.random.Generator

    z: np.ndarray                      # (N, m, 2) rotor state
    omega: np.ndarray                  # (N, m)
    mu: np.ndarray                     # (N, m)
    eta: np.ndarray                    # (N,)

    src: np.ndarray                    # (N, D)
    w: np.ndarray                      # (N, D)
    a: np.ndarray                      # (N, D, d)
    u_proj: np.ndarray                 # (N, D, m)
    credit: np.ndarray                 # (N, D)
    is_long: np.ndarray                # (N, D)

    P: np.ndarray                      # (N_SENSORY, H, P, m + vote_dim)
    M: np.ndarray                      # (N, H, D, m + vote_dim)
    C: np.ndarray                      # (N, H, q, m + vote_dim)
    v: np.ndarray                      # (N, m + vote_dim)
    v0: np.ndarray                     # (N,)
    kappa: np.ndarray                  # (N,)
    S_enc: np.ndarray                  # (N_SENSORY, d, P)  injects into rotor space
    sensory_idx: np.ndarray
    R_proj: np.ndarray                 # (q, d)

    surprise: np.ndarray
    surprise_ema: np.ndarray
    novelty: np.ndarray
    sigma_hat: np.ndarray
    conf: np.ndarray
    coalition: np.ndarray

    y: np.ndarray = field(default=None)      # (N, vote_dim) the vote
    V: np.ndarray = field(default=None)      # (N, vote_dim, m) proposal map
    msg_e: np.ndarray = field(default=None)
    msg_c: np.ndarray = field(default=None)
    msg_n: np.ndarray = field(default=None)
    msg_i: np.ndarray = field(default=None)

    hist_z: deque = field(default_factory=deque)
    hist_h: deque = field(default_factory=deque)
    hist_pred: dict = field(default_factory=dict)
    g: np.ndarray = field(default=None)

    t: int = 0
    n_clipped: int = 0

    @property
    def n_units(self) -> int:
        return self.z.shape[0]

    @property
    def h(self) -> np.ndarray:
        """(N, m + vote_dim) what the prediction heads read: content *and* the vote.

        Concatenating the vote is what gives voting a job. If the heads could not see
        it, a vote would be another quantity that is measured and never used -- exactly
        the mistake v1 made with coalitions.
        """
        if self.y is None or not self.cfg.use_voting:
            return self.h_amp
        return np.concatenate([self.h_amp, self.y], axis=1)

    @property
    def h_amp(self) -> np.ndarray:
        """(N, m) content readout: rotor amplitudes, centred on the limit-cycle radius.

        Centring matters more than it looks. Raw amplitudes are strictly positive and
        clustered near sqrt(mu), so as regressor inputs they are nearly constant plus a
        small deviation -- badly conditioned for the linear heads and for any probe.
        Subtracting the resting radius makes the readout a zero-centred *deviation from
        rest*, which is also the more meaningful quantity: it is what the input did to
        the unit, rather than the unit's idling level.
        """
        r = np.sqrt(np.einsum("nmk,nmk->nm", self.z, self.z))
        return r - np.sqrt(np.maximum(self.mu, 0.0))

    @property
    def z_flat(self) -> np.ndarray:
        """(N, d) raw rotor coordinates, for drive assembly only."""
        return self.z.reshape(self.z.shape[0], -1)

    def n_params(self) -> int:
        return int(sum(x.size for x in (self.omega, self.mu, self.P, self.M, self.C,
                                        self.v, self.v0, self.a, self.u_proj, self.w,
                                        self.kappa, self.S_enc)))


def sensory_width() -> int:
    return P


def n_sensory() -> int:
    return N_SENSORY
