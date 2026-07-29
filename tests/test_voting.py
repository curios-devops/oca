"""Tests for increment 2: voting and the location signal.

The first two pin down the failure mode that actually happened during development --
agreement pressure alone collapses the whole population onto one vote -- so a future
change that reintroduces it fails loudly instead of quietly producing one global
"coalition" that means nothing.
"""

import numpy as np
import pytest

from core.data import rollout
from core.world.physics import make_physics_world
from legacy.v2 import Config2, build_mesh2, tick2
from legacy.v2 import protocol as proto


def _run(cfg, n=800, seed=1):
    m = build_mesh2(cfg)
    _, sen, _ = rollout(n, seed=seed, world_factory=make_physics_world)
    for t in range(n):
        tick2(m, sen[t])
    return m


def test_votes_stay_on_the_unit_sphere():
    m = _run(Config2(lattice_side=12, seed=0), n=400)
    norms = np.linalg.norm(m.y, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), "votes drifted off the sphere"
    assert np.all(np.isfinite(m.V))


def test_agreement_alone_collapses_to_one_global_vote():
    """The documented failure mode: consensus with no task pressure has one attractor."""
    cfg = Config2(lattice_side=12, seed=0, vote_task=0.0, vote_sharpen=1.0,
                  vote_self=0.2)
    m = _run(cfg, n=800)
    sim = m.vote_pct["p50"]
    assert sim > 0.9, f"expected collapse to a single vote, got median similarity {sim:.3f}"


def test_task_pressure_and_sharpening_prevent_collapse():
    m = _run(Config2(lattice_side=12, seed=0), n=800)
    sim = m.vote_pct["p50"]
    assert sim < 0.5, f"votes collapsed despite task pressure (median sim {sim:.3f})"
    sizes = np.bincount(m.vote_coalition)
    assert sizes.max() < m.n_units * 0.5, "one coalition swallowed the population"


def test_vote_coalitions_are_not_all_singletons():
    m = _run(Config2(lattice_side=12, seed=0), n=800)
    sizes = np.bincount(m.vote_coalition)
    assert (sizes > 1).sum() >= 2, "no groups formed at all"


def test_vote_update_is_a_weighted_consensus():
    """A unit surrounded by identical votes must move toward that vote."""
    cfg = Config2(lattice_side=12, seed=0, vote_self=0.0, vote_rate=1.0,
                  vote_sharpen=1.0)
    m = build_mesh2(cfg)
    target = np.zeros(cfg.vote_dim)
    target[0] = 1.0
    m.y[:] = target
    m.y[0] = -target
    gamma = np.ones((m.n_units, cfg.degree))
    y = proto.vote_update(m, gamma)
    assert y[0] @ target > 0.9, "a unit ignored a unanimous neighbourhood"


def test_disabling_voting_shrinks_the_readout():
    on, off = Config2(), Config2(use_voting=False)
    assert on.readout == on.m + on.vote_dim
    assert off.readout == off.m
    m = _run(Config2(lattice_side=12, seed=0, use_voting=False), n=200)
    assert m.h.shape[1] == off.m


def test_frequency_modulation_tracks_novelty():
    """Phase should advance faster where the world is changing faster."""
    cfg = Config2(lattice_side=12, seed=0)
    m = build_mesh2(cfg)
    m.novelty = np.concatenate([np.full(m.n_units // 2, 0.2),
                                np.full(m.n_units - m.n_units // 2, 3.0)])
    eff = m.omega * (1.0 + cfg.freq_mod * np.tanh(m.novelty - 1.0)[:, None])
    slow = np.abs(eff[: m.n_units // 2]).mean()
    fast = np.abs(eff[m.n_units // 2:]).mean()
    assert fast > slow * 1.2, f"novelty did not speed up phase ({slow:.3f} vs {fast:.3f})"


def test_frequency_modulation_barely_touches_the_amplitude_readout():
    """Why the location signal has to come from input rather than from the heads.

    Rotation is amplitude-preserving, so in the continuous flow the readout the heads
    consume is *exactly* invariant to frequency and no head error could ever shape it. A
    first-order Euler step leaves an O(eta^2) residue, so the invariance is very good but
    not perfect -- the gradient is not identically zero, it is negligible. The test pins
    the size of that residue: a nine-fold change in frequency moves the state a lot and
    the readout by well under a percent.
    """
    from legacy.v2 import dynamics as dyn

    rng = np.random.default_rng(0)
    z = rng.normal(0, 0.5, (5, 4, 2))
    mu = np.full((5, 4), 0.5)
    drive = np.zeros_like(z)
    a, _ = dyn.step(z.copy(), np.full((5, 4), 0.1), mu, drive, eta=0.05, sub_steps=1)
    b, _ = dyn.step(z.copy(), np.full((5, 4), 0.9), mu, drive, eta=0.05, sub_steps=1)

    def ratio(eta):
        a, _ = dyn.step(z.copy(), np.full((5, 4), 0.1), mu, drive, eta=eta, sub_steps=1)
        b, _ = dyn.step(z.copy(), np.full((5, 4), 0.9), mu, drive, eta=eta, sub_steps=1)
        state = np.abs(a - b).mean()
        read = np.abs(dyn.radius(a) - dyn.radius(b)).mean()
        assert state > 1e-4, "frequency should change the state"
        return read / state

    coarse, fine = ratio(0.05), ratio(0.01)
    assert coarse < 0.05, f"readout is too sensitive to frequency ({coarse:.3f})"
    # the decisive part: the residue shrinks with the step, so it is an artefact of
    # discretisation rather than a real dependence the heads could exploit
    assert fine < 0.3 * coarse, (
        f"residue did not shrink with step size ({coarse:.4f} -> {fine:.4f}), so it is "
        "not just an integration artefact")


def test_voting_ablations_run():
    for v in (dict(use_voting=False), dict(vote_task=0.0), dict(vote_sharpen=1.0),
              dict(use_freq_modulation=False)):
        m = _run(Config2(lattice_side=12, seed=0).variant(**v), n=200)
        assert np.all(np.isfinite(m.z))
