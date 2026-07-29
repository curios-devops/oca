"""Gate for build step 3: the landscape must be a real gradient flow with real valleys.

If these fail, everything downstream is a recurrent net wearing a physics costume.
"""

import numpy as np
import pytest

from legacy.v1.landscape import energy, fixed_points_1d, flow, relax, rollout


def _setup(n=8, d=6, r=2, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.normal(0, 0.5, (n, d))
    U = rng.normal(0, 1 / np.sqrt(d), (n, d, r))
    b = rng.normal(0, 0.1, (n, d))
    return h, U, b


def test_flow_descends_the_energy():
    h, U, b = _setup()
    e0 = energy(h, U, b)
    for _ in range(50):
        h, _ = relax(h, U, b, eta=0.02, steps=1, noise=0.0)
        e1 = energy(h, U, b)
        assert np.all(e1 <= e0 + 1e-9), "gradient flow increased the energy"
        e0 = e1


def test_frozen_drive_converges_to_a_fixed_point():
    h, U, b = _setup(seed=3)
    for _ in range(4000):
        h, _ = relax(h, U, b, eta=0.05, steps=1, noise=0.0)
    resid = np.linalg.norm(flow(h, U, b), axis=1)
    assert np.all(resid < 1e-6), f"largest residual {resid.max():.2e}"


def test_rank_one_landscape_is_bistable_with_hysteresis():
    d = 4
    u = np.zeros((1, d, 1))
    u[0, 0, 0] = 1.2                     # single direction, lam = 1.44
    b = np.zeros((1, d))

    roots = fixed_points_1d(u[0, :, 0])
    valley = roots[-1]
    assert valley == pytest.approx(1.2, rel=1e-9)

    # settle from either side -> the two valleys, not one shared answer
    up, _ = relax(np.full((1, d), 0.3) * np.eye(d)[0], u, b, eta=0.05, steps=400, noise=0.0)
    dn, _ = relax(-np.full((1, d), 0.3) * np.eye(d)[0], u, b, eta=0.05, steps=400, noise=0.0)
    assert up[0, 0] == pytest.approx(valley, abs=1e-4)
    assert dn[0, 0] == pytest.approx(-valley, abs=1e-4)

    # hysteresis: a tilt too small to cross the ridge does not flip the state
    small_tilt = np.zeros((1, d))
    small_tilt[0, 0] = -0.2
    held, _ = relax(up.copy(), u, small_tilt, eta=0.05, steps=400, noise=0.0)
    assert held[0, 0] > 0.5, "a sub-threshold tilt flipped the hypothesis"

    # a large enough tilt does flip it -- the valley is annihilated
    big_tilt = np.zeros((1, d))
    big_tilt[0, 0] = -1.5
    flipped, _ = relax(up.copy(), u, big_tilt, eta=0.05, steps=800, noise=0.0)
    assert flipped[0, 0] < -0.5, "a super-threshold tilt failed to flip the hypothesis"


def test_rollout_matches_repeated_relaxation():
    h, U, b = _setup(seed=7)
    out = rollout(h, U, b, (1, 4, 16), eta=0.05, steps=1)
    ref = h.copy()
    for step in range(1, 17):
        ref, _ = relax(ref, U, b, eta=0.05, steps=1, noise=0.0)
        if step in (1, 4, 16):
            assert np.allclose(out[step], ref, atol=1e-12)


def test_linear_ablation_stays_bounded():
    h, U, b = _setup(seed=11)
    h = h * 50.0
    for _ in range(500):
        h, _ = relax(h, U, b, eta=0.1, steps=1, noise=0.0, kind="linear")
    assert np.all(np.isfinite(h))
    assert np.abs(h).max() < 10.0


def test_clip_counter_reports_runaway():
    h = np.full((3, 4), 100.0)
    U = np.zeros((3, 4, 1))
    b = np.zeros((3, 4))
    _, n_clip = relax(h, U, b, eta=0.0, steps=1, noise=0.0, h_max=10.0)
    assert n_clip == 3
