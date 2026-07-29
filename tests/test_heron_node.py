"""Level 2 — what a Dynamic Cortical Node must be true of, regardless of how well it scores.

These are invariants, not results. The gates in `bench/nodes.py` decide whether the level
is any good; these decide whether it is the thing it says it is. Several of them pin
properties that the legacy line got wrong and only discovered after a full experimental
cycle, which is exactly the kind of thing a test is cheaper than.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from architectures.heron.contract import LEVELS
from architectures.heron.cortex import (INPUTS_PER_NODE, N_NODES, build_cortex, layout,
                        nodes_to_patches, sensory_to_nodes, tick)
from architectures.heron.node import PUB_SCALARS, NodeConfig, build_stack
from architectures.heron.node import step as node_step


def _drive(stack, n=200, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, scale, (n, stack.cfg.n_nodes, stack.cfg.n_inputs))
    out = []
    for t in range(n):
        node_step(stack, x[t])
        out.append(stack.work.copy())
    return np.array(out)


# ------------------------------------------------------------------ structure


def test_publication_is_only_scalars_and_phase():
    """The design's boldest claim is structural: a node publishes state, never contents.

    If the workspace or a concept vector ever leaks into the publication, gate L2-4 stops
    testing anything -- it would be comparing the full state against a copy of itself.
    """
    stack = build_cortex(seed=0)
    _drive(stack, 50)
    pub = stack.publication()
    assert pub.shape == (stack.cfg.n_nodes,
                         len(PUB_SCALARS) + 2 * stack.cfg.n_bands)
    assert pub.shape[1] < stack.cfg.d_work, "the mouthpiece is not narrower than the state"
    assert np.isfinite(pub).all()


def test_every_aggregation_produces_the_same_width():
    """Three operators, one output width -- otherwise the comparison is about capacity."""
    for agg in ("relational", "pairwise", "mean"):
        stack = build_cortex(seed=0, aggregation=agg)
        _drive(stack, 30)
        assert stack.work.shape == (stack.cfg.n_nodes, stack.cfg.d_work), agg
        assert np.isfinite(stack.work).all(), agg


def test_a_node_reads_its_neurons_through_the_level_one_interface():
    """Axiom 5. The node sees what its neurons published, not what they are thinking."""
    stack = build_cortex(seed=0)
    _drive(stack, 40)
    members = stack.member_state()
    held = stack.neurons.last_sent.reshape(members.shape)
    assert np.array_equal(members, held)
    # and it is genuinely different from the internal activation, or the interface is
    # decorative
    dense = stack.neurons.a.reshape(members.shape)
    assert not np.allclose(members, dense)


def test_level_is_registered_with_a_declared_horizon():
    assert "node" in LEVELS
    lvl = LEVELS["node"]
    assert lvl.horizon >= 1
    assert lvl.inputs_from == "neuron", "a level may only speak to the one below it"


# ------------------------------------------------------------------- dynamics


def _autocorr(X, lag):
    a, b = X[:-lag], X[lag:]
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0)) + 1e-12
    return float(np.median((a * b).sum(0) / den))


def test_the_node_is_slower_than_its_neurons():
    """The point of the level. If it is not slower it is not a level, it is a relabelling.

    This test exists because the first configuration failed it: the node decorrelated
    *faster* than its own members, 0.88 against 0.91 at the declared horizon, and nothing
    else in the battery would have noticed.

    Driven by a real sensory stream, not by noise. White noise decorrelates in a couple of
    ticks, so under noise every layer looks slow relative to its input and the comparison
    that matters -- node against its own members -- is swamped. Level 1's gates use a real
    stream for the same reason.
    """
    from core.data import rollout
    from core.world.physics import make_physics_world

    stack = build_cortex(seed=0)
    _, sen, _ = rollout(1600, seed=0, world_factory=make_physics_world)
    work, members = [], []
    for t in range(1500):
        tick(stack, sen[t])
        work.append(stack.work.ravel().copy())
        members.append(stack.member_state().ravel().copy())
    work, members = np.array(work)[300:], np.array(members)[300:]

    for lag in (4, stack.cfg.horizon, 64):
        w, m = _autocorr(work, lag), _autocorr(members, lag)
        assert w > m, (f"at lag {lag} the node ({w:.3f}) forgets faster than its "
                       f"members ({m:.3f}): it is not the slow layer")


def test_reservoir_and_lowpass_are_actually_different():
    """The ablation has to bite, or the crossed comparison in the L2 battery is empty."""
    a = build_cortex(seed=0)
    b = build_cortex(seed=0, use_reservoir=False)
    wa, wb = _drive(a, 200), _drive(b, 200)
    assert not np.allclose(wa[-1], wb[-1])


def test_the_reservoir_does_not_drift():
    """The third dynamics this project has had to rule out: unbounded accumulators drift.

    Spectral radius below one is what makes that a property of the mathematics rather than
    a hope, so it is checked rather than trusted.
    """
    stack = build_cortex(seed=0)
    work = _drive(stack, 1200, scale=2.0)
    early = np.abs(work[100:200]).mean()
    late = np.abs(work[-100:]).mean()
    assert late < 3 * early + 1e-6, f"working state is growing: {early:.3f} -> {late:.3f}"
    assert np.isfinite(work).all()


def test_knowledge_model_recruits_rather_than_averaging_everything():
    """Axiom 1 needs a mechanism. Recruitment is it, so it has to actually happen.

    Without recruitment every prototype converges on the mean of everything, which is mean
    pooling wearing a different hat -- and the whole level would then be the thing its own
    kill control tests against.
    """
    stack = build_cortex(seed=0)
    _drive(stack, 400, seed=1)
    used = set()
    for _ in range(200):
        _drive(stack, 1, seed=np.random.randint(1 << 20))
        used.update(stack.k_win.tolist())
    assert len(used) >= 3, f"only {len(used)} concepts ever won: the model is degenerate"
    norms = np.linalg.norm(stack.K, axis=2)
    assert (norms > 1e-6).all(), "a prototype collapsed to zero"


def test_frozen_learning_stops_all_learning():
    stack = build_cortex(seed=0)
    _drive(stack, 100)
    stack.learn = False
    K0, A0, M0 = stack.K.copy(), stack.A.copy(), stack.M.copy()
    _drive(stack, 100, seed=7)
    assert np.array_equal(stack.K, K0)
    assert np.array_equal(stack.A, A0)
    assert np.array_equal(stack.M, M0)


# ------------------------------------------------------- the sensory boundary


def test_routing_is_lossless_and_retinotopic():
    from core.world.sensors import N_VISUAL, P
    rng = np.random.default_rng(0)
    sensory = rng.normal(0, 1, (N_VISUAL + 4, P))
    back = nodes_to_patches(sensory_to_nodes(sensory))
    assert np.allclose(back, sensory[:N_VISUAL]), "a patch was lost or misplaced"


@pytest.mark.parametrize("node_side", [2, 4, 8])
def test_node_resolution_is_a_parameter(node_side):
    """Resolution decides results and has nothing to do with the mechanisms, so anything
    that depends on it is run at more than one value -- which requires it to work."""
    lay = layout(node_side)
    stack = build_cortex(seed=0, node_side=node_side)
    assert stack.cfg.n_nodes == lay["n_nodes"]
    from core.world.sensors import N_SENSORY, P
    rng = np.random.default_rng(0)
    for _ in range(20):
        tick(stack, rng.normal(0, 1, (N_SENSORY, P)))
    assert np.isfinite(stack.work).all()
    assert stack.h.shape[0] == lay["n_nodes"]


def test_coalitions_are_concepts_not_synchrony():
    """What this level claims binding is. The legacy alternative measured 1.00x
    persistence, so it does not carry across and the difference should be visible."""
    stack = build_cortex(seed=0)
    _drive(stack, 100)
    lab = stack.coalition
    assert lab.shape == (stack.cfg.n_nodes,)
    assert np.array_equal(lab, stack.k_win)
