"""Layer 0 — Synthetic Neuron.

The smallest computational unit: integrate a local signal, adapt locally, and **emit events only
on significant change**.

This is the one layer that arrives with a clean positive result behind it, so it is deliberately
close to Heron's Layer 0 rather than reinvented. Two things carry across as *constraints*, both
because they were measured here rather than because they sound right:

**Sparse event communication works.** Emitting on significant change beat the best matched-rate
control by 92.9%, with the largest margin in the sparse regime, and the policy transferred to a
frozen architecture's own trace. It is a requirement at this layer, not an efficiency
preference.

**A population spans time constants.** A population sharing one integration window cannot support
any hierarchy of abstraction above it. It costs nothing to avoid.

And one thing deliberately does *not* carry across. Heron put a phase rotor here and it never
paid: -10% for a lone unit, +0.9% with the sign flipping under channel contention, <0.5% on
ablation. Under the Q1 decision temporal coordination is an optional service whose mechanism is
unspecified, so **this layer has no oscillator at all.** If a Corvus layer later wants temporal
coordination it declares it, implements it however it likes, and shows it beats its own ablation.
Not carrying it here is not a claim that timing is useless; it is a refusal to build a mechanism
that has never earned its place.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .contract import Floor, Layer, register
from .primitives import Event


@dataclass(frozen=True)
class NeuronConfig:
    n_inputs: int = 16
    n_neurons: int = 64

    tau: tuple[float, float] = (3.0, 40.0)
    """Per-neuron integration windows, log-spaced. A spread, not a value."""

    theta: float = 0.10
    use_adaptive_theta: bool = True
    theta_target_rate: float = 0.15
    theta_adapt: float = 0.01

    use_plasticity: bool = True
    eta: float = 0.02
    weight_decay: float = 1e-5

    e_tick: float = 0.05
    e_event: float = 1.0
    """Emission dominates on purpose: transmission is the expensive operation, and that
    asymmetry is the entire reason to stay silent."""

    seed: int = 0

    def variant(self, **changes) -> "NeuronConfig":
        return replace(self, **changes)


@dataclass
class Population:
    cfg: NeuronConfig
    rng: np.random.Generator
    w: np.ndarray
    v: np.ndarray
    a: np.ndarray
    last_sent: np.ndarray
    theta: np.ndarray
    tau: np.ndarray
    energy: np.ndarray
    rate_ema: np.ndarray
    group: np.ndarray | None = None
    t: int = 0
    n_events: int = 0
    learn: bool = True
    _fired: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_neurons(self) -> int:
        return len(self.v)

    def n_params(self) -> int:
        return int(self.w.size + self.tau.size + self.theta.size)

    def received(self) -> np.ndarray:
        """What a reader actually has: the last value it was told, held between events."""
        return self.last_sent.copy()

    def events(self, source: str = "L0") -> list[Event]:
        """This tick's emissions, as events.

        Level 0 events carry no entity reference: a neuron reporting its own activation is not
        talking about a thing in the world. Attribution starts at Layer 1, which is the layer
        that can observe a referent and therefore the only one allowed to name one.
        """
        if self._fired is None:
            return []
        return [Event(source=source, kind="activation", value=float(self.a[i]), at=self.t)
                for i in np.flatnonzero(self._fired)]


def build(cfg: NeuronConfig | None = None, n_groups: int = 1) -> Population:
    cfg = cfg or NeuronConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_neurons * n_groups
    return Population(
        cfg=cfg, rng=rng,
        w=rng.normal(0, 1 / np.sqrt(cfg.n_inputs), (n, cfg.n_inputs)),
        v=np.zeros(n), a=np.zeros(n), last_sent=np.zeros(n),
        theta=np.full(n, cfg.theta),
        tau=np.exp(rng.uniform(np.log(cfg.tau[0]), np.log(cfg.tau[1]), n)),
        energy=np.zeros(n), rate_ema=np.full(n, cfg.theta_target_rate),
        group=(np.repeat(np.arange(n_groups), cfg.n_neurons) if n_groups > 1 else None),
    )


def step(pop: Population, x: np.ndarray) -> dict:
    """One tick. `x` is (n_inputs,), or (n_groups, n_inputs) when the population is grouped."""
    cfg = pop.cfg
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 1:
        drive, x_local = pop.w @ x, x
    else:
        x_local = x[pop.group]
        drive = np.einsum("ni,ni->n", pop.w, x_local)

    pop.v += (drive - pop.v) / pop.tau
    pop.a = np.tanh(pop.v)

    delta = pop.a - pop.last_sent
    fired = np.abs(delta) > pop.theta
    pop.last_sent = np.where(fired, pop.a, pop.last_sent)
    pop._fired = fired
    pop.n_events += int(fired.sum())

    pop.rate_ema = ((1 - cfg.theta_adapt) * pop.rate_ema
                    + cfg.theta_adapt * fired.astype(float))
    if cfg.use_adaptive_theta:
        pop.theta *= 1.0 + cfg.theta_adapt * np.sign(pop.rate_ema - cfg.theta_target_rate)
        np.clip(pop.theta, 1e-3, 2.0, out=pop.theta)

    if cfg.use_plasticity and pop.learn:
        # A neuron that had to speak was wrong about itself. Local: its own error, its own input.
        err = np.where(fired, delta, 0.0)
        pop.w *= 1.0 - cfg.weight_decay
        if x_local.ndim == 1:
            pop.w += (cfg.eta / (float(x_local @ x_local) + 1e-8)) * np.outer(err, x_local)
        else:
            norm = np.einsum("ni,ni->n", x_local, x_local) + 1e-8
            pop.w += (cfg.eta * err / norm)[:, None] * x_local

    pop.energy += cfg.e_tick + cfg.e_event * fired
    pop.t += 1
    return {"t": pop.t, "event_rate": float(fired.mean()),
            "energy": float(pop.energy.mean() / max(pop.t, 1))}


register(Layer(
    name="neuron",
    horizon=1,
    inputs_from=None,
    floor=Floor(
        beats="periodic_and_random_sampling",
        margin=0.05,
        why="Any subsampling of a smooth signal reconstructs reasonably, so the question is "
            "never 'does send-on-delta work' but 'does it beat the same number of emissions "
            "spent another way'. Without a matched-rate control the mechanism cannot be told "
            "apart from the rate. Gate CGE-A-02.",
    ),
    build=lambda seed=0, **kw: build(NeuronConfig(seed=seed, **kw)),
    step=step,
    readout=lambda p: p.received(),
    describe=lambda p: {"n_neurons": p.n_neurons, "n_params": p.n_params(),
                        "mechanism": "leaky integration, send-on-delta, no oscillator"},
))
