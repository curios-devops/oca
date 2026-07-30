"""Corvus Layer 0, and the retired Layer 2.

Numbers live in `logs/`; these pin the *mechanisms*, so a refactor cannot quietly change what
`experiments/corvus_l0.py` and the two `corvus_l2*` experiments are measuring.

**Layer 2 is retired and these tests stay.** It was removed from the live stack because both of
its jobs measured as nulls, not because the code was wrong, and a retired layer's numbers have to
stay reproducible or the reason it was retired becomes a claim instead of a measurement. The
contract rules it produced -- one floor per job, an always-on job must have one -- outlived it and
are tested here against the declaration it left behind.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from architectures.corvus.retired import cluster as L2
from architectures.corvus import neuron as L0
from architectures.corvus.contract import LAYERS, Floor, Layer
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


# --------------------------------------------------- layer 2 (retired, kept reproducible)

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


def test_the_always_on_job_has_a_floor():
    """Q7 option B, as an executable rule. Replaces the test that recorded the defect.

    Layer 2 does two jobs. Compression is optional and off by default; coordination runs on
    every tick in both modes and used to have nothing to beat. A layer whose unconditional work
    is unmeasured is compliant and useless, which is exactly what Heron's node layer was.
    """
    jobs = {f.job: f for f in L2.RETIRED_LAYER.floors}
    assert set(jobs) == {"coordination", "compression"}
    assert jobs["coordination"].always_on
    assert not jobs["compression"].always_on
    assert L2.RETIRED_LAYER.floor.always_on, "the primary floor must govern the always-on job"


def test_a_layer_with_floors_only_over_optional_jobs_is_rejected():
    """The construction-time guard. This is the defect that blocked the OCA v4 freeze, and it is
    now impossible to declare rather than merely documented."""
    with pytest.raises(ValueError, match="optional jobs"):
        Layer(name="decorative", horizon=8, inputs_from=None,
              floor=Floor(job="compression", always_on=False, beats="pass_through"),
              build=lambda **kw: None, step=lambda s, u: {}, readout=lambda s: None)


def test_an_always_on_floor_may_not_name_its_own_job_as_the_baseline():
    """`beats="pass_through"` on a layer whose mode *is* pass-through is a tie, and a tie is
    never a win. That unreachable floor is what Q7 was opened about."""
    with pytest.raises(ValueError, match="tie and never a win"):
        Floor(job="pass_through", beats="pass_through", always_on=True)


# ------------------------------------------------------ membership: earned, not positional

def test_the_three_membership_rules_are_selectable_and_differ():
    """Connectivity, proximity and random must actually pick different towers, or `CGE-B-10`
    compared a rule with itself."""
    rng = np.random.default_rng(0)
    n_towers, width = 17, 12
    affinities = rng.normal(0, 1, (n_towers, n_towers))
    affinities = (affinities + affinities.T) / 2

    picks = {}
    for rule in ("proximity", "connectivity", "random"):
        stack = L2.build_stack(L2.ClusterConfig(seed=0, membership=rule), n_towers=n_towers)
        stack.affinity = affinities
        L2.reform(stack)
        picks[rule] = stack.membership
        assert stack.membership.shape == (stack.cfg.n_clusters, stack.cfg.towers_per_cluster)

    assert not np.array_equal(picks["connectivity"], picks["proximity"])
    assert not np.array_equal(picks["random"], picks["proximity"])


def test_every_rule_grows_its_clusters_from_the_same_seeds():
    """The rules must differ in *which towers they pull in* and in nothing else. Different
    seeds would confound the one comparison the gate exists for."""
    seeds = {}
    for rule in ("proximity", "connectivity", "random"):
        stack = L2.build_stack(L2.ClusterConfig(seed=0, membership=rule), n_towers=17)
        stack.affinity = np.random.default_rng(1).normal(0, 1, (17, 17))
        L2.reform(stack)
        seeds[rule] = stack.membership[:, 0]
    assert np.array_equal(seeds["proximity"], seeds["connectivity"])
    assert np.array_equal(seeds["proximity"], seeds["random"])


def test_a_tower_is_never_its_own_cluster_mate():
    """A seed pulled in as its own neighbour would inflate every connectivity score by handing
    the probe the target's own state twice."""
    stack = L2.build_stack(L2.ClusterConfig(seed=0, membership="connectivity"), n_towers=17)
    stack.affinity = np.eye(17) * 10.0        # every tower most similar to itself
    L2.reform(stack)
    for row in stack.membership:
        assert len(set(row)) == len(row)


def test_membership_is_re_derived_on_a_schedule_not_every_tick():
    """An assembly that re-forms every tick is noise, not a module."""
    stack, pub = _stack_and_pub("pass_through")
    for _ in range(L2.ClusterConfig().reform_every + 5):
        L2.step(stack, pub)
    assert stack.n_reforms == 1


def test_affinity_reads_published_state_only():
    """Coordination may read what its members publish and never their internals -- the property
    that lets a layer be replaced without rewriting the one above it."""
    stack, pub = _stack_and_pub("pass_through")
    L2.step(stack, pub)
    assert stack.affinity.shape == (stack.n_towers, stack.n_towers)
    assert np.all(np.isfinite(stack.affinity))
    assert np.any(stack.affinity != 0)


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
