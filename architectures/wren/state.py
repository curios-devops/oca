"""Configuration and mesh-wide state arrays.

The whole mesh lives in a handful of (N, ...) numpy arrays. There is no per-unit Python
object anywhere: a tick is a fixed sequence of vectorized operations over these arrays,
which is what keeps 576 units at ~1 kHz on a laptop CPU.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from core.world.sensors import N_SENSORY, P


@dataclass(frozen=True)
class Config:
    """Every number in SPEC_RPDU and SPEC_PMP, in one place.

    The `use_*` flags at the bottom are the E3 ablations. They all default to the full
    model; each one turns off exactly one claim from the design documents.
    """

    # -- sizes ---------------------------------------------------------------
    lattice_side: int = 24            # N = side^2 mesh units
    d: int = 16                       # latent width
    r: int = 3                        # landscape rank
    q: int = 8                        # global context width
    horizons: tuple[int, ...] = (1, 4, 16)
    n_local: int = 32
    n_long: int = 8
    n_exec_links: int = 1
    n_exec_units: int = 20

    # -- landscape dynamics --------------------------------------------------
    alpha: float = 1.0
    eta_h: float = 0.15
    relax_steps: int = 2
    state_noise: float = 0.01
    h_max: float = 10.0
    osc_period: int = 40
    osc_gain: float = 0.15
    """Oscillator amplitude. At 1.0 the shared clock is the single largest term in the
    drive and it swamps the retina at the very units whose job is to see, which caps
    how well any head can predict. The oscillator is meant to be modulatory -- a phase
    reference distant units can align to -- not the dominant input."""

    sensory_gain: float = 3.0
    """Gain on the retina drive into sensory units. Cortical layer 4 is dominated by
    its afferent input; without this the mesh's own chatter drowns out the world."""

    residual_sensory: bool = True
    """Predict the *change* from the current patch: s_hat = s(t) + P h.

    A sensory unit physically has its own afferent input, so making it re-derive that
    input from a lossy 16-d state before it can say anything about the future is a
    plumbing handicap, not a scientific one -- and the same handicap is why the GRU
    baseline gets a residual output too. With this on, copy-last is the mesh's
    zero-initialisation and every improvement over it is attributable to prediction."""

    # -- learning ------------------------------------------------------------
    eta_head: float = 3e-3
    eta_sigma: float = 1e-3
    eta_landscape: float = 1e-3
    eta_link: float = 1e-3
    weight_decay: float = 1e-5
    u_max_fro: float = 3.0
    beta: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.5)  # sens, msg, self, glob

    # -- protocol ------------------------------------------------------------
    novelty_ema: float = 0.01
    credit_ema: float = 0.005
    rewire_every: int = 200
    rewire_prune: int = 2
    sample_temp: float = 0.3
    new_link_gain: float = 0.1

    # -- coalitions ----------------------------------------------------------
    coalition_every: int = 10
    coalition_window: int = 32
    coalition_theta: float = 0.6
    coalition_boost: float = 0.2
    label_prop_rounds: int = 5

    # -- ablation switches ---------------------------------------------------
    use_uncertainty: bool = True
    use_rewiring: bool = True
    use_long_range: bool = True
    use_landscape: bool = True
    use_coalition_feedback: bool = True
    use_oscillator: bool = True
    use_plasticity: bool = True
    msg_width: int = 4                # 4 = full; 1 = expectation only
    instant_delivery: bool = False

    seed: int = 0

    # -- derived -------------------------------------------------------------
    @property
    def n_units(self) -> int:
        return self.lattice_side**2

    @property
    def degree(self) -> int:
        n_long = self.n_long if self.use_long_range else 0
        return self.n_local + n_long + self.n_exec_links

    @property
    def n_horizons(self) -> int:
        return len(self.horizons)

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    @property
    def flow_kind(self) -> str:
        return "energy" if self.use_landscape else "linear"

    def variant(self, **changes) -> "Config":
        """A copy with fields overridden -- how ablations are constructed."""
        return replace(self, **changes)

    def label(self) -> str:
        base = Config(seed=self.seed)
        diffs = [
            f"{k}={getattr(self, k)}"
            for k in self.__dataclass_fields__
            if getattr(self, k) != getattr(base, k)
        ]
        return "full" if not diffs else ",".join(diffs)


@dataclass
class MeshState:
    """All mutable state. Constructed by `build_mesh` in mesh.py."""

    cfg: Config
    rng: np.random.Generator

    # dynamical state
    h: np.ndarray                     # (N, d)
    U: np.ndarray                     # (N, d, r)   landscape basis
    b: np.ndarray                     # (N, d)      resting tilt
    omega: np.ndarray                 # (N, d)      oscillator injection direction

    # topology, indexed by (receiver, slot)
    src: np.ndarray                   # (N, D) int   source unit of each in-link
    w: np.ndarray                     # (N, D)       link gain
    a: np.ndarray                     # (N, D, d)    receiver-owned read vector
    u_proj: np.ndarray                # (N, D, d)    sender-owned channel projector
    credit: np.ndarray                # (N, D)       running usefulness
    is_long: np.ndarray               # (N, D) bool  which slots are rewireable

    # heads
    P: np.ndarray                     # (N_SENSORY, H, P, d)
    M: np.ndarray                     # (N, H, D, d)
    C: np.ndarray                     # (N, H, q, d)
    v: np.ndarray                     # (N, d)
    v0: np.ndarray                    # (N,)
    kappa: np.ndarray                 # (N,)        importance bid scale

    # sensory wiring
    sensory_idx: np.ndarray           # (N_SENSORY,) mesh indices of sensory units
    S_enc: np.ndarray                 # (N_SENSORY, d, P) retina -> drive encoder

    # executive
    R_proj: np.ndarray                # (q, d) fixed random summary projection

    # running statistics
    surprise: np.ndarray              # (N,)  S_i
    surprise_ema: np.ndarray          # (N,)  mu_i
    novelty: np.ndarray               # (N,)  n_i
    sigma_hat: np.ndarray             # (N,)  predicted surprise
    conf: np.ndarray                  # (N,)  1/(1+sigma_hat)
    coalition: np.ndarray             # (N,)  current label

    # message buffer (previous tick's outgoing messages, read this tick)
    msg_e: np.ndarray = field(default=None)   # (N, D)
    msg_c: np.ndarray = field(default=None)
    msg_n: np.ndarray = field(default=None)
    msg_i: np.ndarray = field(default=None)

    # history for delayed comparison
    hist_h: deque = field(default_factory=deque)
    hist_pred: dict = field(default_factory=dict)
    hist_state: deque = field(default_factory=deque)
    g: np.ndarray = field(default=None)       # (q,) current global context

    t: int = 0
    n_clipped: int = 0

    @property
    def n_units(self) -> int:
        return self.h.shape[0]

    def n_params(self) -> int:
        return int(
            sum(
                x.size
                for x in (self.U, self.b, self.P, self.M, self.C, self.v, self.v0,
                          self.a, self.u_proj, self.w, self.kappa, self.S_enc)
            )
        )


def sensory_width() -> int:
    return P


def n_sensory() -> int:
    return N_SENSORY
