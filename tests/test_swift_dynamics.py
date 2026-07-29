"""Tests for the v2 oscillator dynamics.

The first three are the whole point of the rewrite. v1 provably could not satisfy any of
them: `test_v1_could_not_do_this` pins that contrast down so the change is not just
asserted in a docstring.
"""

import numpy as np
import pytest

from architectures.swift import dynamics as dyn


def _z(n=6, m=8, seed=0, scale=0.3):
    rng = np.random.default_rng(seed)
    return rng.normal(0, scale, (n, m, 2))


def test_amplitude_settles_onto_the_limit_cycle():
    z = _z()
    mu = np.full((6, 8), 0.6)
    omega = np.full((6, 8), 0.2)
    drive = np.zeros_like(z)
    for _ in range(600):
        z, _ = dyn.step(z, omega, mu, drive, eta=0.05, sub_steps=1)
    r = dyn.radius(z)
    assert np.allclose(r, np.sqrt(0.6), atol=1e-3), f"radii {r.min():.3f}-{r.max():.3f}"


def test_phase_keeps_advancing_forever():
    """The property a gradient flow cannot have: motion that never stops."""
    z = _z(n=1, m=1)
    mu, omega = np.full((1, 1), 0.5), np.full((1, 1), 0.25)
    drive = np.zeros_like(z)
    for _ in range(400):                       # reach the limit cycle first
        z, _ = dyn.step(z, omega, mu, drive, eta=0.05, sub_steps=1)

    steps = []
    for _ in range(400):
        prev = z.copy()
        z, _ = dyn.step(z, omega, mu, drive, eta=0.05, sub_steps=1)
        steps.append(np.linalg.norm(z - prev))
    steps = np.array(steps)
    assert steps.min() > 1e-3, "the unit stopped moving"
    assert steps[-1] / steps[0] > 0.5, "motion is decaying toward a fixed point"


def test_v1_could_not_do_this():
    """Same measurement on v1's gradient flow: it halts. This is the contrast."""
    from architectures.wren import landscape as L

    rng = np.random.default_rng(0)
    h = rng.normal(0, 0.5, (4, 16))
    U = rng.normal(0, 0.25, (4, 16, 3))
    b = np.zeros((4, 16))
    for _ in range(400):
        h, _ = L.relax(h, U, b, eta=0.05, steps=1, noise=0.0)
    first = np.linalg.norm(L.flow(h, U, b), axis=1).mean()
    for _ in range(2000):
        h, _ = L.relax(h, U, b, eta=0.05, steps=1, noise=0.0)
    last = np.linalg.norm(L.flow(h, U, b), axis=1).mean()
    assert last < first * 1e-3, "v1's gradient flow was expected to halt"


def test_coupled_units_synchronise():
    """Coalitions need synchronisation to be possible at all."""
    n, m = 12, 4
    rng = np.random.default_rng(1)
    z = rng.normal(0, 0.5, (n, m, 2))
    mu = np.full((n, m), 0.5)
    omega = np.full((n, m), 0.3) + rng.normal(0, 0.01, (n, m))

    src = np.array([[(i + 1) % n, (i - 1) % n] for i in range(n)])
    spread = []
    for t in range(3000):
        mean_nb = z[src].mean(axis=1)
        drive = 0.25 * (mean_nb - z)          # diffusive phase coupling
        z, _ = dyn.step(z, omega, mu, drive, eta=0.05, sub_steps=1)
        if t % 100 == 0:
            p = dyn.phase(z)[:, 0]
            spread.append(float(np.abs(np.angle(np.exp(1j * (p - p.mean())))).mean()))

    assert spread[-1] < spread[0] * 0.5, f"phases did not converge: {spread[0]:.3f} -> {spread[-1]:.3f}"
    assert dyn.radius(z).min() > 0.1, "units collapsed instead of synchronising"


def test_mu_below_zero_collapses_to_a_fixed_point():
    """Oscillation is an ablation, not an assumption."""
    z = _z(seed=2)
    mu = np.full((6, 8), -0.3)
    omega = np.full((6, 8), 0.2)
    for _ in range(2000):
        z, _ = dyn.step(z, omega, mu, np.zeros_like(z), eta=0.05, sub_steps=1)
    assert dyn.radius(z).max() < 1e-3, "a subcritical unit should decay to the origin"


def test_rollout_horizons_actually_differ():
    """v1's rollout collapsed onto a fixed point, making multi-horizon prediction moot."""
    z = _z(seed=3)
    mu, omega = np.full((6, 8), 0.5), np.full((6, 8), 0.3)
    out = dyn.rollout(z, omega, mu, np.zeros_like(z), (1, 4, 16), eta=0.1)
    d4 = np.linalg.norm(out[4] - out[1])
    d16 = np.linalg.norm(out[16] - out[1])
    assert d16 > d4 > 0
    assert d16 > 0.1 * np.linalg.norm(out[1]), "horizons are nearly identical"


def test_phase_alignment_bounds_and_meaning():
    z = np.zeros((3, 2, 2))
    z[:, :, 0] = 1.0                            # every unit at phase 0
    src = np.array([[1, 2], [0, 2], [0, 1]])
    assert np.allclose(dyn.phase_alignment(z, src), 1.0)

    z[2, :, 0] = -1.0                           # unit 2 in antiphase
    al = dyn.phase_alignment(z, src)
    assert al[0, 1] == pytest.approx(-1.0)
    assert al[0, 0] == pytest.approx(1.0)


def test_step_is_deterministic_and_clips():
    z = _z(seed=5)
    mu, omega = np.full((6, 8), 0.5), np.full((6, 8), 0.2)
    a, _ = dyn.step(z.copy(), omega, mu, np.zeros_like(z))
    b, _ = dyn.step(z.copy(), omega, mu, np.zeros_like(z))
    assert np.array_equal(a, b)

    big = np.full((2, 2, 2), 50.0)
    _, n_clip = dyn.step(big, np.zeros((2, 2)), np.zeros((2, 2)),
                         np.zeros_like(big), eta=0.0, r_max=4.0)
    assert n_clip == 4


def test_per_unit_timescales_change_convergence_speed():
    z = np.tile(np.array([[[0.05, 0.0]]]), (2, 4, 1))
    mu, omega = np.full((2, 4), 0.5), np.zeros((2, 4))
    eta = np.array([0.01, 0.2])
    for _ in range(60):
        z, _ = dyn.step(z, omega, mu, np.zeros_like(z), eta=eta, sub_steps=1)
    r = dyn.radius(z)
    assert r[1].mean() > r[0].mean() * 1.5, "the fast unit should approach the cycle sooner"
