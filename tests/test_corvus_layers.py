"""Corvus layers 0 and 2 — the mechanisms the R1 freeze condition rests on.

Numbers live in `logs/`; these pin the *mechanisms*, so that a refactor cannot quietly change
what `experiments/corvus_l0.py` and `experiments/corvus_l2.py` are measuring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from architectures.corvus import cluster as L2
from architectures.corvus import neuron as L0
from architectures.corvus.contract import LAYERS
from cge.components import gate_rate_distortion, summarise_rate_distortion
from core.probes import whiten


# ------------------------------------------------------------------ layer 0

def test_layer_0_has_no_oscillator():
    """The Q1 decision, as an executable fact.

    Heron's rotor never paid (-10% alone, +0.9% under contention with the sign flipping) and
    Corvus deliberately carries no oscillator at any level. If one is ever added it must be
    declared and must beat its own ablation -- not appear because it sounded right.
    """
    fields = set(L0.NeuronConfig.__dataclass_fields__)
    assert not (fields & {"use_oscillation", "gate_depth", "omega", "osc_into_activation"})
    assert "no oscillator" in LAYERS["neuron"].describe(L0.build())["mechanism"]


def test_send_on_delta_beats_its_own_rate_on_corvus_traces():
    """CGE-A-02, in miniature, on the activations Corvus's own dynamics produce.

    The point is that Heron's +92.9% is *not* inherited: this layer is that layer minus the
    rotor, and a result measured on a predecessor does not transfer (rule R1).
    """
    cfg = L0.NeuronConfig(seed=0, n_neurons=32)
    pop = L0.build(cfg)
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 0.1, (900, cfg.n_inputs)), axis=0)
    dense = []
    for t in range(len(x)):
        L0.step(pop, x[t])
        dense.append(pop.a.copy())

    rd = summarise_rate_distortion(gate_rate_distortion(np.array(dense), seed=0))
    assert rd["delta_vs_best_control"] > LAYERS["neuron"].floor.margin


def test_a_neuron_holds_the_last_value_it_sent():
    """What a reader has between events is the held value, not the live one. If those were the
    same the entire send-on-delta comparison would be measuring nothing."""
    pop = L0.build(L0.NeuronConfig(seed=0, n_neurons=8))
    for _ in range(50):
        L0.step(pop, np.ones(pop.cfg.n_inputs) * 0.4)
    silent = ~pop._fired
    assert silent.any()
    assert np.all(pop.received()[silent] != pop.a[silent])


# ------------------------------------------------------------------ layer 2

def _stack_and_pub(mode: str, n_towers: int = 17, width: int = 12):
    stack = L2.build_stack(L2.ClusterConfig(seed=0, mode=mode), n_towers=n_towers)
    rng = np.random.default_rng(1)
    return stack, rng.normal(0, 1, (n_towers, width))


def test_pass_through_returns_its_members_unchanged():
    """The compliance floor is literal: the members, concatenated, not a copy of some summary."""
    stack, pub = _stack_and_pub("pass_through")
    L2.step(stack, pub)
    out = stack.readout()
    k, m = stack.cfg.n_clusters, stack.cfg.towers_per_cluster
    assert out.shape == (k, m * pub.shape[1])
    assert np.allclose(out[0], pub[stack.membership[0]].ravel())
    assert stack.n_params() == 0


def test_the_summary_is_a_deterministic_linear_map_of_the_pass_through():
    """Why `experiments/corvus_l2.py` takes both readouts from one run.

    The cluster does not feed back into the towers, so the two readouts describe the same tick.
    Running twice would inject seed noise into a comparison that has none.
    """
    stack, pub = _stack_and_pub("summarise")
    L2.step(stack, pub)
    expected = np.einsum("kds,ks->kd", stack.Ws, stack._passthrough)
    assert np.allclose(stack.summary, expected)

    L2.step(stack, pub)                       # same input, same output: no hidden state
    assert np.allclose(stack.summary, expected)


def test_layer_2_cannot_clear_its_own_floor_in_its_default_mode():
    """**A recorded structural defect, not a passing test of a working thing.**

    Layer 2's floor is `beats="pass_through"` with a positive margin, and its default mode
    *is* pass-through. A tie is not a win, so in the default mode the floor is unreachable by
    construction -- which means rule R1 ("freeze when every layer clears its own floor") can
    never be satisfied by this stack however good the other layers are.

    Measured separately: in `summarise` mode the compression scores -0.004 +/- 0.006 against
    pass-through over three seeds, so switching modes does not rescue it either.

    This test exists so the defect cannot be forgotten. It must be deleted, not "fixed", when
    the layer's obligations are redeclared.
    """
    floor = LAYERS["cluster"].floor
    assert floor.beats == "pass_through"
    assert floor.margin > 0
    assert L2.ClusterConfig().mode == "pass_through"


def test_matched_capacity_actually_reduces_the_wider_representation():
    """The guard on the L2 gate. An unmatched probe measures width, not representation -- the
    error that made relational aggregation look like a result until mean pooling matched it."""
    rng = np.random.default_rng(0)
    wide = rng.normal(0, 1, (400, 240))
    project = whiten(wide, cut=240, n_components=64)
    assert project(wide).shape == (400, 64)


def test_coordination_happens_in_both_modes():
    """Coordination is unconditional: it is the part of this layer that always runs, in either
    mode. That it currently has no floor of its own is the open question, not an accident."""
    for mode in ("pass_through", "summarise"):
        stack, pub = _stack_and_pub(mode)
        L2.step(stack, pub)
        assert stack.agreement.shape == (stack.cfg.n_clusters,)
        assert np.all(np.isfinite(stack.agreement))
