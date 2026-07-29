"""Invariants for the v2 mesh.

Mirrors tests/test_mesh.py so the two architectures are held to the same standard --
determinism, stability, frozen-plasticity, and above all the locality guarantee, which
is the claim the whole project rests on and must survive the rewrite.
"""

import numpy as np
import pytest

from core.data import rollout
from architectures.wren.topology import lattice_distance, local_offsets
from core.world import Sensors
from core.world.physics import make_physics_world
from architectures.swift import Config2, build_mesh2, predicted_retina2, tick2

SMALL = dict(lattice_side=12, seed=0)


def _run(cfg, n=200, seed=1):
    m = build_mesh2(cfg)
    _, sen, _ = rollout(n, seed=seed, world_factory=make_physics_world)
    diags = [tick2(m, sen[t]) for t in range(n)]
    return m, diags


def test_runs_without_nan_or_clipping():
    m, _ = _run(Config2(**SMALL), n=600)
    for name in ("z", "a", "w", "P", "M", "C", "v", "S_enc", "u_proj"):
        assert np.all(np.isfinite(getattr(m, name))), f"{name} went non-finite"
    assert m.n_clipped == 0, f"amplitude clip bound {m.n_clipped} times"


def test_state_keeps_oscillating_rather_than_settling():
    """The property v1 could not have. Measured on the live mesh, not in isolation."""
    m, _ = _run(Config2(**SMALL), n=400)
    _, sen, _ = rollout(120, seed=7, world_factory=make_physics_world)
    steps = []
    for t in range(120):
        prev = m.z.copy()
        tick2(m, sen[t])
        steps.append(float(np.abs(m.z - prev).mean()))
    assert min(steps) > 1e-4, "the mesh stopped moving"
    assert np.mean(steps[-30:]) > 0.3 * np.mean(steps[:30]), "motion is dying out"


def test_same_seed_is_bit_identical():
    a, da = _run(Config2(**SMALL))
    b, db = _run(Config2(**SMALL))
    assert np.array_equal(a.z, b.z)
    assert np.array_equal(a.P, b.P)
    assert da == db


def test_frozen_plasticity_leaves_parameters_untouched():
    cfg = Config2(**SMALL)
    m = build_mesh2(cfg)
    _, sen, _ = rollout(80, seed=2, world_factory=make_physics_world)
    for t in range(40):
        tick2(m, sen[t])
    m.learn = False
    snap = {k: getattr(m, k).copy() for k in ("P", "M", "C", "a", "w", "v", "S_enc")}
    for t in range(40, 80):
        tick2(m, sen[t])
    for k, v in snap.items():
        assert np.array_equal(getattr(m, k), v), f"{k} changed with plasticity frozen"


def test_learning_is_local():
    """No error signal crosses a unit boundary -- the claim the project rests on."""
    cfg = Config2(lattice_side=16, seed=0, use_long_range=False, n_exec_links=0,
                  use_masking=False, use_rewiring=False)
    _, sen, _ = rollout(20, seed=3, world_factory=make_physics_world)

    def run(perturb):
        m = build_mesh2(cfg)
        for t in range(4):
            s = sen[t].copy()
            if perturb:
                s[0] += 5.0
            tick2(m, s)
        return m

    a, b = run(False), run(True)
    changed = np.flatnonzero(~np.isclose(a.w, b.w).all(axis=1))
    if changed.size == 0:
        return
    side = cfg.lattice_side
    source = Sensors.lattice_slots(side)[0]
    reach = 4 * max(int(np.max(np.abs(local_offsets(cfg.n_local)))), 1)
    dist = lattice_distance(np.full(changed.size, source), changed, side)
    assert np.all(dist <= reach), f"unit at distance {dist.max()} changed, reach {reach}"


def test_masking_actually_hides_input_but_still_scores_it():
    cfg = Config2(**SMALL).variant(mask_prob=1.0)
    m, diags = _run(cfg, n=40)
    assert all(d["masked"] > 0 for d in diags), "masking never fired"
    assert np.all(np.isfinite(m.P))

    off = Config2(**SMALL).variant(use_masking=False)
    m2, diags2 = _run(off, n=40)
    assert all(d["masked"] == 0 for d in diags2)


def test_oscillation_ablation_recovers_fixed_points():
    """use_oscillation=False must reproduce v1's essential limitation inside v2.

    A driven subcritical mesh is not near the origin -- input pushes it around. The
    property that distinguishes it is that it does not *self-sustain*: cut the input and
    it decays, where the full model keeps going. That is exactly v1's failure mode.
    """
    from architectures.swift import dynamics as dyn
    from core.world.sensors import N_SENSORY, P as PATCH_W

    blank = np.zeros((N_SENSORY, PATCH_W))
    decay = {}
    for flag in (False, True):
        cfg = Config2(**SMALL).variant(use_oscillation=flag)
        m, _ = _run(cfg, n=300)
        assert np.all(m.mu < 0) if not flag else np.all(m.mu > 0)
        start = float(dyn.radius(m.z).mean())
        for _ in range(400):                       # free-run on no input at all
            tick2(m, blank)
        decay[flag] = float(dyn.radius(m.z).mean()) / max(start, 1e-9)

    assert decay[False] < 0.2, f"subcritical mesh failed to decay ({decay[False]:.3f})"
    assert decay[True] > 0.5, f"full mesh died in silence ({decay[True]:.3f})"


def test_all_ablations_construct_and_run():
    for v in (dict(use_phase_gate=False), dict(use_masking=False),
              dict(use_timescales=False), dict(use_long_range=False),
              dict(use_oscillation=False), dict(msg_width=1),
              dict(use_uncertainty=False), dict(drive_norm=False),
              dict(coupling=0.2)):
        m, _ = _run(Config2(**SMALL).variant(**v), n=60)
        assert np.all(np.isfinite(m.z)), f"ablation {v} produced non-finite state"


def test_predicted_retina_shape():
    m, _ = _run(Config2(**SMALL), n=80)
    sensors = Sensors()
    for tau in m.cfg.horizons:
        f = predicted_retina2(m, tau, sensors)
        assert f.shape == (32, 32) and np.all(np.isfinite(f))


def test_config_label_names_only_differences():
    assert Config2().label() == "full"
    assert Config2(msg_width=1).label() == "msg_width=1"
