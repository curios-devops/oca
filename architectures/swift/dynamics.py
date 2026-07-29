"""Oscillatory unit dynamics — the core of architecture v2.

v1's unit descended an energy landscape. That was a faithful reading of the design
document's "a rolling ball always falls into one valley", and it is exactly why v1
failed twice over. A gradient flow has `dE/dt = -||grad E||^2 <= 0`, so it has no
periodic orbits: it cannot oscillate, so units cannot synchronise and coalitions were
mechanically impossible; and it cannot hold a moving quantity, so nothing could be
carried across an occlusion. Measured on v1: with drive frozen a unit's step size decayed
by 4.2e-4 and halted, and 74% of state power sat at periods over 50 ticks.

Here the state is a set of 2-D rotors in the Stuart-Landau normal form (the normal form
of a Hopf bifurcation), per rotor k:

    r^2 = x^2 + y^2
    x' = -omega*y + (mu - r^2)*x + drive_x
    y' = +omega*x + (mu - r^2)*y + drive_y

With `mu > 0` this has a stable *limit cycle* at radius sqrt(mu) rather than a stable
point. Three consequences, each targeting a specific v1 failure:

* **It never stops.** Amplitude settles, phase keeps advancing, so the unit holds a
  quantity that persists with no input to sustain it.
* **It can synchronise.** Coupled oscillator populations phase-lock generically; this is
  what gives coalitions a mechanism to exist at all.
* **Phase integrates.** A rotor's phase is the running integral of its frequency, so
  modulating frequency by observed motion turns phase into a location signal — the
  Thousand Brains idea, and the same fix arrived at from the other direction.

Setting `mu <= 0` collapses the limit cycle back to a fixed point at the origin, which
makes "oscillation" itself an ablation rather than an assumption.
"""

from __future__ import annotations

import numpy as np


def as_rotors(h: np.ndarray) -> np.ndarray:
    """(N, d) -> (N, d//2, 2). Adjacent state dimensions are paired into a rotor."""
    n, d = h.shape
    return h.reshape(n, d // 2, 2)


def as_flat(z: np.ndarray) -> np.ndarray:
    """(N, m, 2) -> (N, 2m)."""
    return z.reshape(z.shape[0], -1)


def radius(z: np.ndarray) -> np.ndarray:
    """(N, m) rotor amplitudes.

    This is the unit's *content* channel, and every head reads it rather than the raw
    rotor coordinates. The reason is not cosmetic. A rotor's (x, y) sweeps a full circle
    every 2*pi/omega ticks, so the same represented value produces wildly different
    coordinates depending on when you look, and any linear readout of them is chasing a
    moving target -- measured directly: reading raw coordinates made position decoding
    twice as bad as v1. Amplitude is phase-invariant, so content stays legible while
    phase is left free to do the binding.

    This is the standard division of labour in the oscillatory-binding literature: rate
    (here, amplitude) carries what, phase carries what-goes-with-what.
    """
    return np.sqrt(np.einsum("nmk,nmk->nm", z, z))


def phase(z: np.ndarray) -> np.ndarray:
    """(N, m) rotor phases in radians."""
    return np.arctan2(z[..., 1], z[..., 0])


def flow(z: np.ndarray, omega: np.ndarray, mu: np.ndarray, drive: np.ndarray
         ) -> np.ndarray:
    """Stuart-Landau vector field. All args broadcast over (N, m[, 2])."""
    r2 = np.einsum("nmk,nmk->nm", z, z)
    rot = np.stack([-omega * z[..., 1], omega * z[..., 0]], axis=-1)
    contract = ((mu - r2)[..., None]) * z
    return rot + contract + drive


def step(z: np.ndarray, omega: np.ndarray, mu: np.ndarray, drive: np.ndarray,
         *, eta: np.ndarray | float = 0.15, sub_steps: int = 2,
         noise: float = 0.0, rng: np.random.Generator | None = None,
         r_max: float = 4.0) -> tuple[np.ndarray, int]:
    """Integrate the flow. `eta` may be per-unit, giving a spread of timescales.

    Returns (z, n_clipped). A per-unit `eta` is how the timescale hierarchy is built:
    slow units integrate over long windows and can bridge an occlusion, fast units track
    detail, and with a single shared time constant no hierarchy of abstraction can
    emerge at all.
    """
    e = np.asarray(eta)
    if e.ndim == 1:
        e = e[:, None, None]
    for _ in range(sub_steps):
        z = z + e * flow(z, omega, mu, drive)
        if noise and rng is not None:
            z = z + noise * rng.standard_normal(z.shape)
    r = radius(z)
    over = r > r_max
    n_clipped = int(over.sum())
    if n_clipped:
        z = np.where(over[..., None], z * (r_max / np.maximum(r, 1e-12))[..., None], z)
    return z, n_clipped


def rollout(z: np.ndarray, omega: np.ndarray, mu: np.ndarray, drive: np.ndarray,
            horizons: tuple[int, ...], *, eta=0.15, sub_steps: int = 2,
            r_max: float = 4.0) -> dict[int, np.ndarray]:
    """Self-prediction: roll the unit's own dynamics forward with drive frozen.

    As in v1 this *is* the self-prediction head rather than a separate weight matrix, so
    the dynamics stay load-bearing. Unlike v1 the rollout does not collapse to a fixed
    point, so predictions at different horizons genuinely differ.
    """
    out, cur, done = {}, z, 0
    for tau in sorted(horizons):
        for _ in range(tau - done):
            cur, _ = step(cur, omega, mu, drive, eta=eta, sub_steps=sub_steps,
                          r_max=r_max)
        done = tau
        out[tau] = cur.copy()
    return out


def phase_alignment(z: np.ndarray, src: np.ndarray) -> np.ndarray:
    """(N, D) mean cos(phase difference) between each unit and each of its sources.

    +1 when a pair is exactly in phase across all rotors, -1 in antiphase. This is the
    quantity phase-gated communication is built on, and the natural measure of the
    "synchronise transiently into coalitions" operation the design document asks for.
    """
    p = phase(z)                       # (N, m)
    return np.cos(p[:, None, :] - p[src]).mean(axis=2)
