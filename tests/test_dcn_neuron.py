"""Level 1 gates as tests, so a change that breaks the neuron fails immediately."""

import numpy as np
import pytest

from bench.components import (_delta_events, gate_rate_distortion, nrmse,
                              summarise_rate_distortion, zero_order_hold)
from legacy.dcn.neuron import NeuronConfig, build_population, dense, received, step


def _run(cfg=None, ticks=1500, seed=0):
    cfg = cfg or NeuronConfig(n_neurons=16, seed=0)
    pop = build_population(cfg)
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 0.15, (ticks, cfg.n_inputs)), axis=0)
    x = np.tanh(x)
    d, s, e = [], [], []
    for t in range(ticks):
        step(pop, x[t])
        d.append(dense(pop)); s.append(received(pop)); e.append(pop._last_event.copy())
    return pop, np.array(d), np.array(s), np.array(e)


def test_population_runs_and_stays_finite():
    pop, d, s, e = _run()
    for name in ("w", "v", "z", "a", "theta", "energy"):
        assert np.all(np.isfinite(getattr(pop, name))), f"{name} went non-finite"


def test_a_neuron_is_silent_most_of_the_time():
    """The property that makes it not a wire."""
    pop, d, s, e = _run()
    assert 0.01 < e.mean() < 0.6, f"event rate {e.mean():.3f} is not sparse"


def test_send_on_delta_beats_its_own_rate():
    """The level-1 claim: emitting on change beats spending the same budget elsewhere.

    Without this control the mechanism cannot be told apart from simply emitting less.
    """
    _, d, _, _ = _run(ticks=2000)
    rd = gate_rate_distortion(d, rates=(0.05, 0.1, 0.2, 0.4))
    s = summarise_rate_distortion(rd)
    assert s["delta_vs_best_control"] > 0.05, (
        f"send-on-delta only {s['delta_vs_best_control']*100:.1f}% better than control")


def _rotor_motion(pop, zero, n):
    moved = []
    for _ in range(n):
        prev = pop.z.copy()
        step(pop, zero)
        moved.append(np.abs(pop.z - prev).mean())
    return moved


def test_neuron_oscillates_with_no_input():
    """A gradient flow provably cannot do this; legacy v1 learned it the hard way.

    Read from the rotor, not from the activation. Under the corrected wiring the rhythm is
    deliberately absent from the transmitted value, so a neuron that is oscillating
    correctly has a *flat* activation with no input -- the previous version of this test
    asserted the opposite and passed only because content and clock were mixed.
    """
    cfg = NeuronConfig(n_neurons=8, seed=0)
    pop = build_population(cfg)
    zero = np.zeros(cfg.n_inputs)
    for _ in range(300):
        step(pop, zero)
    moved = _rotor_motion(pop, zero, 200)
    assert min(moved) > 1e-6, "the neuron settled to a fixed point"
    assert np.mean(moved[-50:]) > 0.3 * np.mean(moved[:50]), "the rhythm is dying out"


def test_the_clock_never_enters_the_content():
    """Axiom 3, pinned. Phase gates *when* a neuron speaks, never *what* it says.

    The first wiring added the rotor into the activation and cost 19x in reconstruction
    while emitting more. This is the assertion that stops that returning by convenience:
    with no input the activation must be exactly still, however fast the rotor is turning.
    """
    cfg = NeuronConfig(n_neurons=8, seed=0)
    pop = build_population(cfg)
    zero = np.zeros(cfg.n_inputs)
    for _ in range(300):
        step(pop, zero)
    acts = []
    for _ in range(100):
        prev = pop.a.copy()
        step(pop, zero)
        acts.append(np.abs(pop.a - prev).mean())
    assert max(acts) < 1e-9, "the rhythm is leaking into the transmitted value"

    # and the rejected wiring, kept runnable, must still show the leak it was rejected for
    leaky = build_population(cfg.variant(osc_into_activation=0.35))
    for _ in range(300):
        step(leaky, zero)
    moved = [np.abs(leaky.a - p).mean()
             for p in [leaky.a.copy()] for _ in [0]]
    prev = leaky.a.copy()
    step(leaky, zero)
    assert np.abs(leaky.a - prev).mean() > 1e-4, (
        "the phase-as-content ablation no longer reproduces the failure it documents")


def test_phase_gate_only_raises_the_threshold():
    """The gate may delay an emission; within a tick it may never alter the value sent.

    Pinned with plasticity off, and that is the whole subtlety. With learning on the two
    populations *do* diverge after a few ticks, because a neuron that stayed silent had no
    error to learn from -- the gate changes what is learned, which changes what is computed
    later. That is intended and is not a leak of clock into content. The invariant that
    actually matters is the one-tick one: the activation is computed before the gate is
    consulted and does not depend on it.
    """
    # adaptive thresholds off too: a gated neuron fires less, so its rate controller
    # lowers its base threshold to compensate, and the two bases stop being comparable
    cfg = NeuronConfig(n_neurons=16, seed=0, use_plasticity=False,
                       use_adaptive_theta=False)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (200, cfg.n_inputs))

    gated = build_population(cfg)
    plain = build_population(cfg.variant(gate_depth=0.0))
    for t in range(200):
        step(gated, x[t])
        step(plain, x[t])
        assert np.allclose(gated.a, plain.a), (
            "phase changed the activation, not just the emission threshold")
    # and the gate only ever makes a neuron harder to hear, never easier
    assert (gated._last_theta_eff >= plain._last_theta_eff - 1e-12).all()


def test_oscillation_ablation_silences_the_rhythm():
    cfg = NeuronConfig(n_neurons=8, seed=0, use_oscillation=False, gate_depth=0.0)
    pop = build_population(cfg)
    zero = np.zeros(cfg.n_inputs)
    for _ in range(300):
        step(pop, zero)
    assert np.mean(_rotor_motion(pop, zero, 50)) < 1e-9, (
        "ablating oscillation should leave the rotor still")


def test_zero_order_hold_is_what_a_reader_actually_sees():
    values = np.array([[0.0], [1.0], [2.0], [3.0]])
    events = np.array([[True], [False], [True], [False]])
    held = zero_order_hold(values, events)
    assert held.ravel().tolist() == [0.0, 0.0, 2.0, 2.0]


def test_threshold_solver_hits_the_requested_rate():
    rng = np.random.default_rng(0)
    v = np.tanh(np.cumsum(rng.normal(0, 0.1, (1200, 8)), axis=0))
    for target in (0.1, 0.3):
        from bench.components import policy_send_on_delta
        ev = policy_send_on_delta(v, target)
        assert abs(ev.mean() - target) < 0.05, f"rate {ev.mean():.3f} vs target {target}"


def test_energy_grows_with_emissions():
    quiet, _, _, eq = _run(NeuronConfig(n_neurons=16, seed=0, theta=0.5,
                                        use_adaptive_theta=False))
    loud, _, _, el = _run(NeuronConfig(n_neurons=16, seed=0, theta=0.01,
                                       use_adaptive_theta=False))
    assert el.mean() > eq.mean(), "a lower threshold should emit more"
    assert loud.energy.mean() > quiet.energy.mean(), "more emissions should cost more"


def test_dcn_still_imports_nothing_from_legacy():
    import ast, pathlib
    for p in (pathlib.Path(__file__).resolve().parents[1] / "dcn").rglob("*.py"):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                    else [node.module] if isinstance(node, ast.ImportFrom) and node.module
                    else [])
            for m in mods:
                assert m.split(".")[0] != "legacy", f"{p.name} imports {m}"
