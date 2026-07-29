"""Level 1 — the Dynamic Neuron.

Five responsibilities from the design: integrate local signals, hold a small internal
state, learn weights, oscillate on its own, and **emit only when its state changes
significantly**.

That last one is what makes this a different object from a summer, and it is the one the
gates are built around. A neuron that emits every tick is a wire; a neuron that emits only
on significant change is doing lossy predictive coding of its own trajectory, and the
interesting question is the exchange rate — how much precision does silence cost?

Two decisions carried across from the legacy line as *constraints*, because each cost a
full experimental cycle there:

**Oscillation is a limit cycle, not a fixed point.** A gradient flow has
`dE/dt = -||grad E||^2 <= 0` and therefore no periodic orbits. v1's units could not
oscillate, which is why they could not synchronise and could not carry a moving quantity.
The phase here follows the Stuart-Landau normal form, whose amplitude settles while its
phase keeps advancing, so "the neuron oscillates on its own" is a property of the
mathematics rather than a hope.

**Emission is compared against its own control.** Any subsampling of a smooth signal looks
good. The gate is not "does send-on-delta reconstruct well" but "does it reconstruct better
than periodic or random sampling *at the same event rate*". Without that control the
mechanism cannot be distinguished from the rate.

And one constraint the gates added, which is worth more than either because it was
measured here rather than inherited: **phase gates when a neuron speaks, never what it
says.** The first wiring added the rotor into the activation and the battery caught it
immediately -- 17x worse reconstruction while emitting less, because the event budget went
on transmitting the neuron's own rhythm. Axiom 3 says phase is a clock, not content; the
rotor now scales the emission threshold and never touches the transmitted value.

Axiom 2 says neurons are simple compute units holding no knowledge worth naming, so this
one does not predict the world. It encodes its own activation and transmits it sparsely.
Knowledge is the DCN's job, one level up.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .contract import DCNLevel, register_dcn


@dataclass(frozen=True)
class NeuronConfig:
    """Every number for level 1. `use_*` fields are the ablations."""

    n_inputs: int = 16
    n_neurons: int = 64

    # -- membrane -----------------------------------------------------------
    tau: tuple[float, float] = (3.0, 40.0)
    """Per-neuron membrane time constants, log-spaced. A spread rather than one value:
    the legacy line established that a population with a single time constant cannot form
    any hierarchy of abstraction, and it costs nothing to avoid that here."""

    leak_floor: float = 1e-3

    # -- local oscillation --------------------------------------------------
    use_oscillation: bool = True
    omega: tuple[float, float] = (0.05, 0.8)
    """Per-neuron natural frequencies, log-spaced, in radians per tick."""
    mu: float = 0.25
    """Limit-cycle radius squared. The amplitude settles at sqrt(mu) while the phase keeps
    advancing -- the property a gradient flow cannot have."""

    gate_depth: float = 0.6
    """How strongly phase gates *when* a neuron may speak. Not how strongly it modulates
    what the neuron says -- that was the first wiring and the battery rejected it.

    Adding the rotor into the activation made reconstruction 17x worse while emitting
    *less*, because the neuron spent its event budget transmitting its own rhythm. That is
    axiom 3 failing in the only way it can fail in code: phase was being used as content.
    Here the rotor never touches the transmitted value. It scales the emission threshold,
    so a neuron speaks readily at its permissive phase and needs a larger change at its
    refractory phase. Depth 0 recovers a pure send-on-delta neuron, which is the ablation.

    For one neuron in isolation this can only cost precision -- a gate can delay an
    emission, never improve it. It is not built to pay off here. It pays off when many
    neurons share a channel, which is a level-2 measurement, and level 2 measures it."""

    osc_into_activation: float = 0.0
    """The rejected wiring, kept as a named ablation rather than deleted.

    At any value above zero the rotor is added into the activation -- phase used as content
    -- which is what the first version did. Keeping it runnable means the result that
    rejected it stays reproducible instead of surviving only as a paragraph, and a future
    change that quietly reintroduces the mixing is caught by the same number."""

    # -- emission -----------------------------------------------------------
    theta: float = 0.10
    """Emission threshold. A neuron speaks when its activation has drifted this far from
    what it last told anyone -- send-on-delta. Raising it trades precision for silence,
    and the rate-distortion curve that traces is the level-1 result."""

    use_adaptive_theta: bool = True
    theta_target_rate: float = 0.15
    theta_adapt: float = 0.01
    """Adapt the threshold toward a target event rate, so a neuron in a quiet region does
    not fall silent forever and one in a noisy region does not saturate the channel."""

    # -- plasticity ---------------------------------------------------------
    use_plasticity: bool = True
    eta: float = 0.02
    weight_decay: float = 1e-5

    # -- energy -------------------------------------------------------------
    e_idle: float = 0.01
    e_tick: float = 0.05
    e_event: float = 1.0
    """Relative costs. Emission dominates on purpose: in cortex, transmission is far more
    expensive than maintenance, and that asymmetry is the whole reason to stay silent."""

    seed: int = 0

    def variant(self, **changes) -> "NeuronConfig":
        return replace(self, **changes)

    def label(self) -> str:
        base = NeuronConfig(seed=self.seed)
        diffs = [f"{k}={getattr(self, k)}" for k in self.__dataclass_fields__
                 if getattr(self, k) != getattr(base, k)]
        return "full" if not diffs else ",".join(diffs)


@dataclass
class NeuronPopulation:
    """A population of Dynamic Neurons, stored as arrays rather than objects."""

    cfg: NeuronConfig
    rng: np.random.Generator

    w: np.ndarray            # (N, n_inputs) learned weights
    v: np.ndarray            # (N,) internal potential
    z: np.ndarray            # (N, 2) local phase rotor
    a: np.ndarray            # (N,) activation
    last_sent: np.ndarray    # (N,) what each neuron last told the world
    theta: np.ndarray        # (N,) per-neuron emission threshold
    tau: np.ndarray          # (N,)
    omega: np.ndarray        # (N,)
    energy: np.ndarray       # (N,) accumulated cost
    rate_ema: np.ndarray     # (N,) running event rate

    group: np.ndarray | None = None
    """Which node each neuron belongs to, or None for a single flat population.

    Level 2 runs many small populations at once. Looping over them in Python and calling
    `step` per node costs an order of magnitude more than one grouped call, and a level
    that is expensive to run is a level that gets tested less. The grouping lives here
    rather than in level 2 because it is a fact about how neurons are laid out, and axiom 5
    says level 2 talks to this interface rather than reaching inside it."""

    t: int = 0
    n_events: int = 0
    learn: bool = True
    _last_event: np.ndarray | None = field(default=None, repr=False)
    _last_theta_eff: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_neurons(self) -> int:
        return len(self.v)

    @property
    def phase(self) -> np.ndarray:
        return np.arctan2(self.z[:, 1], self.z[:, 0])

    @property
    def amplitude(self) -> np.ndarray:
        return np.linalg.norm(self.z, axis=1)

    def n_params(self) -> int:
        return int(self.w.size + self.tau.size + self.omega.size + self.theta.size)


def build_population(cfg: NeuronConfig | None = None,
                     n_groups: int = 1) -> NeuronPopulation:
    """`n_groups > 1` lays out `n_groups` populations of `cfg.n_neurons` in one array."""
    cfg = cfg or NeuronConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_neurons * n_groups
    group = (np.repeat(np.arange(n_groups), cfg.n_neurons) if n_groups > 1 else None)

    def logu(lo, hi, size):
        return np.exp(rng.uniform(np.log(lo), np.log(hi), size))

    z = rng.normal(0, 0.2, (n, 2))
    return NeuronPopulation(
        cfg=cfg, rng=rng,
        w=rng.normal(0, 1 / np.sqrt(cfg.n_inputs), (n, cfg.n_inputs)),
        v=np.zeros(n),
        z=z,
        a=np.zeros(n),
        last_sent=np.zeros(n),
        theta=np.full(n, cfg.theta),
        tau=logu(*cfg.tau, n),
        omega=(logu(*cfg.omega, n) * rng.choice([-1.0, 1.0], n)
               if cfg.use_oscillation else np.zeros(n)),
        energy=np.zeros(n),
        rate_ema=np.full(n, cfg.theta_target_rate),
        group=group,
    )


def step(pop: NeuronPopulation, x: np.ndarray) -> dict:
    """One tick.

    `x` is either the (n_inputs,) local signal the whole population integrates, or, when
    the population is grouped, a (n_groups, n_inputs) array in which each neuron reads
    only its own group's row.
    """
    cfg = pop.cfg
    n = pop.n_neurons

    # -- 1. integrate local signals ----------------------------------------
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        drive = pop.w @ x                                # (N,)
        x_local = x
    else:
        x_local = x[pop.group]                           # (N, n_inputs)
        drive = np.einsum("ni,ni->n", pop.w, x_local)
    pop.v += (drive - pop.v) / pop.tau                   # leaky integration

    # -- 2. oscillate ------------------------------------------------------
    if cfg.use_oscillation:
        r2 = np.einsum("nk,nk->n", pop.z, pop.z)
        rot = np.stack([-pop.omega * pop.z[:, 1], pop.omega * pop.z[:, 0]], axis=1)
        # Stuart-Landau: amplitude settles at sqrt(mu), phase keeps turning
        pop.z = pop.z + rot + ((cfg.mu - r2)[:, None]) * pop.z * 0.1

    # -- 3. activation -- content only, no rhythm in it --------------------
    # The rotor is deliberately absent from this line. Axiom 3 says phase is a clock, not
    # content; putting the rhythm here is what made the first wiring fail its own gate.
    # `osc_into_activation` restores that wiring, and exists only to keep the ablation
    # that rejected it runnable.
    if cfg.use_oscillation and cfg.osc_into_activation:
        pop.a = np.tanh(pop.v + cfg.osc_into_activation * pop.z[:, 0])
    else:
        pop.a = np.tanh(pop.v)

    # -- 4. emit on significant change, when phase permits -----------------
    delta = pop.a - pop.last_sent
    if cfg.use_oscillation and cfg.gate_depth > 0.0:
        # permissive at phase 0, refractory half a cycle later. The gate scales the
        # threshold; it never scales the value, so silence can delay information but
        # cannot corrupt it.
        openness = 0.5 * (1.0 + np.cos(pop.phase))            # (N,) in [0, 1]
        theta_eff = pop.theta * (1.0 + cfg.gate_depth * (1.0 - openness))
    else:
        theta_eff = pop.theta
    pop._last_theta_eff = theta_eff
    fired = np.abs(delta) > theta_eff
    pop.last_sent = np.where(fired, pop.a, pop.last_sent)
    pop._last_event = fired
    pop.n_events += int(fired.sum())

    rate = fired.astype(float)
    pop.rate_ema = (1 - cfg.theta_adapt) * pop.rate_ema + cfg.theta_adapt * rate
    if cfg.use_adaptive_theta:
        # raise the threshold where a neuron is talking too much, lower it where it has
        # gone quiet, so no neuron either saturates the channel or drops off it entirely
        pop.theta *= 1.0 + cfg.theta_adapt * np.sign(pop.rate_ema - cfg.theta_target_rate)
        np.clip(pop.theta, 1e-3, 2.0, out=pop.theta)

    # -- 5. learn ----------------------------------------------------------
    if cfg.use_plasticity and pop.learn:
        # A neuron that had to speak was wrong about itself: its own activation moved
        # further than it had told anyone. Push the weights to reduce that surprise. The
        # rule is local -- it uses only this neuron's own error and its own input.
        err = np.where(fired, delta, 0.0)
        pop.w *= 1.0 - cfg.weight_decay
        if x_local.ndim == 1:
            norm = float(x_local @ x_local) + 1e-8
            pop.w += (cfg.eta / norm) * np.outer(err, x_local)
        else:
            norm = np.einsum("ni,ni->n", x_local, x_local) + 1e-8
            pop.w += (cfg.eta * err / norm)[:, None] * x_local

    # -- 6. energy ---------------------------------------------------------
    pop.energy += cfg.e_tick + cfg.e_idle * np.abs(pop.a) + cfg.e_event * fired
    pop.t += 1

    return {
        "t": pop.t,
        "event_rate": float(fired.mean()),
        "activation": float(np.abs(pop.a).mean()),
        "energy": float(pop.energy.mean()),
        "theta": float(pop.theta.mean()),
    }


def received(pop: NeuronPopulation) -> np.ndarray:
    """What a downstream reader sees: the last emitted value, held between events.

    Zero-order hold is the honest reconstruction. A reader that only receives events has
    nothing else to go on between them, so this is the signal the rest of the system
    actually has -- not the internal activation, which nobody downstream can see.
    """
    return pop.last_sent.copy()


def dense(pop: NeuronPopulation) -> np.ndarray:
    """The internal activation -- what would be transmitted if it never stayed silent."""
    return pop.a.copy()


register_dcn(DCNLevel(
    name="neuron",
    # One tick, and stated rather than defaulted. Send-on-delta *is* a one-tick
    # self-prediction: "nothing has changed enough to be worth saying" is a forecast, and
    # the zero-order hold downstream is what consumes it. Anything longer would be a claim
    # this level does not make -- knowledge is the node's job.
    horizon=1,
    inputs_from=None,
    build=lambda seed=0, **kw: build_population(NeuronConfig(seed=seed, **kw)),
    step=step,
    readout=received,
    describe=lambda p: {"n_neurons": p.n_neurons, "n_params": p.n_params()},
))
