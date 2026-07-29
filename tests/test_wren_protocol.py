"""Protocol and learning-rule tests.

The planted-coalition test matters most. E4 reports that coalitions essentially never
form in the trained mesh, and that claim is only worth anything if the detector is
demonstrably able to find a coalition that is really there.
"""

import numpy as np
import pytest

from core.data import rollout
from architectures.wren.mesh import build_mesh, tick
from architectures.wren.plasticity import normaliser, update_credit
from architectures.wren.protocol import assemble_messages, detect_coalitions, receiver_gate, softplus
from architectures.wren.state import Config

SMALL = dict(lattice_side=12, seed=0)


def test_softplus_is_positive_and_smooth():
    x = np.linspace(-20, 20, 100)
    y = softplus(x)
    assert np.all(y > 0) and np.all(np.isfinite(y))
    assert np.all(np.diff(y) >= 0)


def test_message_shapes_and_narrow_width():
    cfg = Config(**SMALL)
    m = build_mesh(cfg)
    h1 = np.random.default_rng(0).normal(size=(m.n_units, cfg.d))
    full = assemble_messages(m, h1)
    for k in "ecni":
        assert full[k].shape == (m.n_units, cfg.degree)

    narrow = build_mesh(Config(**SMALL).variant(msg_width=1))
    out = assemble_messages(narrow, h1)
    assert np.allclose(out["c"], 1.0)
    assert np.allclose(out["n"], 0.0)
    assert np.allclose(out["i"], 0.0)
    assert not np.allclose(out["e"], 0.0), "the expectation must still carry information"


def test_detector_finds_a_planted_coalition():
    """Two groups with perfectly correlated internal states must be recovered."""
    cfg = Config(**SMALL).variant(use_coalition_feedback=False)
    m = build_mesh(cfg)
    N, d, W = m.n_units, cfg.d, cfg.coalition_window
    rng = np.random.default_rng(0)

    # every unit is noise, except two groups that share a common trajectory
    window = rng.normal(size=(W, N, d)) * 0.05
    group_a = m.src[0][:6]
    group_b = m.src[50][:6]
    for g in (group_a, group_b):
        signal = rng.normal(size=(W, d))
        for u in g:
            window[:, u, :] = signal + rng.normal(size=(W, d)) * 0.01

    gamma = np.ones((N, cfg.degree))
    labels = detect_coalitions(m, window, gamma)

    for g in (group_a, group_b):
        planted = labels[g]
        assert len(set(planted.tolist())) == 1, f"planted group split: {planted}"
    assert m.coh_pct["max"] > cfg.coalition_theta

    # and the noise units must not all be swept into one blob
    sizes = np.bincount(labels)
    assert sizes.max() < N * 0.5, "detector collapsed everything into one coalition"


def test_detector_reports_nothing_when_states_are_independent():
    cfg = Config(**SMALL)
    m = build_mesh(cfg)
    rng = np.random.default_rng(1)
    window = rng.normal(size=(cfg.coalition_window, m.n_units, cfg.d))
    labels = detect_coalitions(m, window, np.ones((m.n_units, cfg.degree)))
    _, counts = np.unique(labels, return_counts=True)
    assert counts.max() <= 2, "found structure in pure noise"


def test_coalition_boost_only_applies_within_a_coalition():
    cfg = Config(**SMALL)
    m = build_mesh(cfg)
    msg = {"e": np.ones((m.n_units, cfg.degree)), "c": np.ones((m.n_units, cfg.degree)),
           "n": np.zeros((m.n_units, cfg.degree)), "i": np.zeros((m.n_units, cfg.degree))}
    # With every unit its own label the only boosted links are the handful of
    # self-links, where a unit is trivially in its own coalition.
    m.coalition = np.arange(m.n_units)
    solo = receiver_gate(m, msg)
    self_link = np.arange(m.n_units)[:, None] == m.src
    assert np.allclose(solo, np.where(self_link, 1.0 + cfg.coalition_boost, 1.0))

    m.coalition = np.zeros(m.n_units, dtype=int)
    joined = receiver_gate(m, msg)
    assert np.allclose(joined, 1.0 + cfg.coalition_boost)


def test_credit_rewards_predictable_links_that_reduce_surprise():
    cfg = Config(**SMALL)
    m = build_mesh(cfg)
    N, D = m.n_units, cfg.degree
    msg_e = np.ones((N, D))
    pred = np.ones((N, D))
    pred[:, 1] = -5.0                       # slot 1 is unpredictable noise
    d_surprise = np.full(N, 1.0)            # surprise fell everywhere
    m.credit[:] = 0.0
    for _ in range(50):
        update_credit(m, msg_e, pred, d_surprise)
    assert m.credit[:, 0].mean() > m.credit[:, 1].mean(), "noisy link out-earned a useful one"

    # a link that is predictable but coincides with no change earns nothing
    m2 = build_mesh(cfg)
    m2.credit[:] = 0.0
    for _ in range(50):
        update_credit(m2, msg_e, pred, np.zeros(N))
    assert np.allclose(m2.credit, 0.0, atol=1e-9)


def test_normaliser_is_scale_free():
    h = np.array([[3.0, 4.0], [30.0, 40.0]])
    n = normaliser(h, eps=0.0)
    assert n[0] == pytest.approx(1 / 25)
    assert n[1] == pytest.approx(1 / 2500)


def test_rewiring_changes_only_long_range_slots():
    cfg = Config(**SMALL).variant(rewire_every=50)
    m = build_mesh(cfg)
    _, sen, _ = rollout(200, seed=4)
    before = m.src.copy()
    for t in range(200):
        tick(m, sen[t])
    changed = before != m.src
    assert changed.any(), "rewiring never fired"
    assert not changed[~m.is_long].any(), "a local or executive link was rewired"
