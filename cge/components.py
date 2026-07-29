"""Phase 1 — component benchmarks, one battery per level of abstraction.

Built like an aircraft: you do not fly the whole thing to find out whether a subsystem
works. Each level gets its own gates and its own results file, so a change that helps one
layer and hurts another is visible immediately instead of being averaged into a single
score.

Lives in `bench/` rather than in `dcn/` on purpose. Gates belong to the testbed, not to an
architecture — that is what keeps "the same test" meaning the same thing when a new level
or a new architecture arrives.

Level 1 measures **precision and efficiency**, and the two only mean something together. A
neuron that emits every tick is maximally precise and is a wire; one that never emits is
maximally efficient and is a rock. What the design claims is that emitting *on significant
change* buys a better exchange rate than the same number of emissions spent at random —
so every precision number here is reported against its event rate, and against a sampling
control at that identical rate.
"""

from __future__ import annotations

import numpy as np


def nrmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Reconstruction error, normalised by the signal's own scale.

    Normalised so it is comparable across neurons and configurations: 0 is perfect, and
    1.0 is what you get by predicting the mean, i.e. transmitting nothing at all.
    """
    denom = truth.std(axis=0).mean() + 1e-12
    return float(np.sqrt(((estimate - truth) ** 2).mean()) / denom)


def zero_order_hold(values: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Reconstruct a signal from the samples an event policy chose to transmit.

    What a downstream reader actually has: the last value it was told, held until it is
    told another. Any event policy can be scored this way, which is what makes the
    comparison between policies fair.
    """
    out = np.zeros_like(values)
    held = np.zeros(values.shape[1])
    for t in range(len(values)):
        held = np.where(events[t], values[t], held)
        out[t] = held
    return out


# ------------------------------------------------------------- event policies


def policy_periodic(values: np.ndarray, rate: float, rng=None) -> np.ndarray:
    """Emit on a fixed schedule at the given rate. The strongest naive control."""
    n_t, n = values.shape
    period = max(int(round(1.0 / max(rate, 1e-6))), 1)
    ev = np.zeros((n_t, n), dtype=bool)
    # stagger the phase per neuron, otherwise every neuron speaks on the same tick and
    # the control is handicapped by a burst pattern no sane scheme would use
    for i in range(n):
        ev[(np.arange(n_t) + i) % period == 0, i] = True
    return ev


def policy_random(values: np.ndarray, rate: float, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    return rng.random(values.shape) < rate


def policy_send_on_delta(values: np.ndarray, rate: float, rng=None) -> np.ndarray:
    """The design's policy, with the threshold solved for so the rate matches exactly.

    Solving for the threshold is what makes the comparison honest: otherwise send-on-delta
    is being compared to controls at a different budget, and the rate rather than the
    policy explains any difference.
    """
    lo, hi = 1e-5, 5.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        ev = _delta_events(values, mid)
        if ev.mean() > rate:
            lo = mid
        else:
            hi = mid
    return _delta_events(values, 0.5 * (lo + hi))


def _delta_events(values: np.ndarray, theta: float) -> np.ndarray:
    n_t, n = values.shape
    ev = np.zeros((n_t, n), dtype=bool)
    held = np.zeros(n)
    for t in range(n_t):
        fire = np.abs(values[t] - held) > theta
        held = np.where(fire, values[t], held)
        ev[t] = fire
    return ev


POLICIES = {
    "send_on_delta": policy_send_on_delta,
    "periodic": policy_periodic,
    "random": policy_random,
}


# ------------------------------------------------------------------- gates


def gate_rate_distortion(values: np.ndarray, rates=(0.05, 0.1, 0.2, 0.4, 0.8),
                         seed: int = 0) -> dict:
    """Precision against efficiency, for each event policy at matched rates.

    The level-1 result. If send-on-delta does not beat periodic and random sampling at the
    *same* rate, then emitting on significant change is not doing anything that a cheaper
    schedule would not, and the design's claim for it is unsupported.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for name, policy in POLICIES.items():
        curve = {}
        for rate in rates:
            ev = policy(values, rate, rng)
            recon = zero_order_hold(values, ev)
            curve[f"{rate:g}"] = {"nrmse": nrmse(recon, values),
                                  "actual_rate": float(ev.mean())}
        out[name] = curve
    return out


def gate_oscillation(pop, step_fn, n: int = 400) -> dict:
    """Does the neuron sustain a rhythm with no input at all?

    The design says a neuron oscillates on its own rather than waiting to be synchronised.
    That is only true if the dynamics can express it, which is exactly what the legacy line
    got wrong three times, so it is checked directly rather than assumed.

    **Read from the rotor, not from the activation.** Under axiom 3 the rhythm is a clock,
    so it is deliberately absent from the transmitted value -- with no input the activation
    is flat and *should* be. Scoring "still moving" on the activation would therefore fail a
    correctly wired neuron and pass an incorrectly wired one, which is precisely backwards.
    The activation's stillness is reported alongside as `activation_activity`: under the
    right wiring it is ~0, and a large value means content and clock are mixed again.
    """
    zero = np.zeros(pop.cfg.n_inputs)
    for _ in range(200):
        step_fn(pop, zero)
    phases, amps, acts, rotor = [], [], [], []
    for _ in range(n):
        step_fn(pop, zero)
        phases.append(pop.phase.copy())
        amps.append(pop.amplitude.copy())
        acts.append(pop.a.copy())
        rotor.append(pop.z.copy())
    phases, amps = np.array(phases), np.array(amps)
    acts, rotor = np.array(acts), np.array(rotor)

    advance = np.abs(np.diff(np.unwrap(phases, axis=0), axis=0)).mean()
    rotor_activity = float(np.abs(np.diff(rotor, axis=0)).mean())
    act_activity = float(np.abs(np.diff(acts, axis=0)).mean())
    return {
        "phase_advance_per_tick": float(advance),
        "amplitude_mean": float(amps.mean()),
        "amplitude_drift": float(np.abs(amps[-50:].mean(0) - amps[:50].mean(0)).mean()),
        "still_moving": rotor_activity > 1e-4,
        "activity": rotor_activity,
        "activation_activity": act_activity,
        "clock_is_not_content": act_activity < 1e-6,
    }


def gate_noise_robustness(run_fn, noise_levels=(0.0, 0.1, 0.3), seed: int = 0) -> dict:
    """How precision degrades as the input is corrupted."""
    out = {}
    for s in noise_levels:
        values, stats = run_fn(noise=s, seed=seed)
        ev = _delta_events(values, float(np.median(stats["theta"])))
        out[f"{s:g}"] = {
            "nrmse": nrmse(zero_order_hold(values, ev), values),
            "event_rate": float(ev.mean()),
        }
    return out


def gate_energy(stats: dict) -> dict:
    """Cost per tick and per unit of precision retained."""
    return {
        "energy_per_neuron_per_tick": stats["energy_per_tick"],
        "energy_per_event": stats["energy_per_event"],
        "event_rate": stats["event_rate"],
    }


def summarise_rate_distortion(rd: dict) -> dict:
    """Area under the rate-distortion curve, lower is better, plus the pairwise margins."""
    out = {}
    for name, curve in rd.items():
        rates = np.array([float(k) for k in curve])
        errs = np.array([curve[k]["nrmse"] for k in curve])
        order = np.argsort(rates)
        out[name] = float(np.trapezoid(errs[order], rates[order]))
    best_control = min(out[k] for k in ("periodic", "random"))
    out["delta_vs_best_control"] = 1.0 - out["send_on_delta"] / max(best_control, 1e-12)
    return out
