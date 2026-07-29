"""Invariants from SPEC_RPDU section 7, plus the locality guarantee.

The locality test is the important one: the central claim of both design documents is
that learning uses only local prediction errors. A test that pins that down is what
stops a global gradient from creeping in later and quietly doing the real work.
"""

import numpy as np
import pytest

from core.data import rollout
from architectures.wren.mesh import build_mesh, predicted_retina, tick
from architectures.wren.state import Config
from architectures.wren.topology import lattice_distance, local_offsets
from core.world import Sensors

SMALL = dict(lattice_side=12, seed=0)


def _run(cfg, n=120, seed=1):
    m = build_mesh(cfg)
    _, sen, _ = rollout(n, seed=seed)
    diags = [tick(m, sen[t]) for t in range(n)]
    return m, diags


def test_runs_without_nan_or_clipping():
    cfg = Config(**SMALL)
    m, _ = _run(cfg, n=300)
    for name in ("h", "U", "b", "P", "M", "C", "a", "w", "v", "S_enc"):
        arr = getattr(m, name)
        assert np.all(np.isfinite(arr)), f"{name} went non-finite"
    assert m.n_clipped == 0, f"safety clip bound {m.n_clipped} times"


def test_same_seed_is_bit_identical():
    a, da = _run(Config(**SMALL))
    b, db = _run(Config(**SMALL))
    assert np.array_equal(a.h, b.h)
    assert np.array_equal(a.U, b.U)
    assert da == db


def test_different_seed_diverges():
    a, _ = _run(Config(**SMALL))
    b, _ = _run(Config(lattice_side=12, seed=1))
    assert not np.allclose(a.h, b.h)


def test_learning_reduces_surprise():
    cfg = Config(**SMALL)
    m, diags = _run(cfg, n=1500)
    early = np.mean([d["surprise"] for d in diags[100:300]])
    late = np.mean([d["surprise"] for d in diags[-200:]])
    assert late < early, f"surprise rose: {early:.3f} -> {late:.3f}"


def test_frozen_plasticity_leaves_parameters_untouched():
    cfg = Config(**SMALL)
    m = build_mesh(cfg)
    _, sen, _ = rollout(60, seed=2)
    for t in range(30):
        tick(m, sen[t])
    m.learn = False
    snap = {k: getattr(m, k).copy() for k in ("U", "b", "P", "M", "C", "a", "w", "v", "S_enc")}
    for t in range(30, 60):
        tick(m, sen[t])
    for k, v in snap.items():
        assert np.array_equal(getattr(m, k), v), f"{k} changed with plasticity frozen"


def test_learning_is_local_a_distant_unit_is_unaffected():
    """Perturbing one unit's input must not change parameters of units it cannot reach.

    Information moves one hop per tick, so after `k` ticks nothing beyond lattice
    distance k (via local links) plus the long-range shortcuts may have changed. Run
    with long-range links and the executive off so reachability is purely the lattice.
    """
    cfg = Config(lattice_side=16, seed=0, use_long_range=False, n_exec_links=0,
                 use_oscillator=False, use_coalition_feedback=False)
    _, sen, _ = rollout(20, seed=3)

    def run(perturb):
        m = build_mesh(cfg)
        for t in range(4):
            s = sen[t].copy()
            if perturb:
                s[0] += 5.0          # blast the first visual unit's patch
            tick(m, s)
        return m

    a, b = run(False), run(True)
    changed = np.flatnonzero(~np.isclose(a.b, b.b).all(axis=1))

    side = cfg.lattice_side
    source = Sensors.lattice_slots(side)[0]
    radius = max(int(np.max(np.abs(local_offsets(cfg.n_local)))), 1)
    reach = 4 * radius                       # 4 ticks, `radius` hops of lattice each
    dist = lattice_distance(np.full(changed.size, source), changed, side)
    assert np.all(dist <= reach), (
        f"units at distance {dist.max()} changed but only {reach} is reachable"
    )


def test_predicted_retina_shape_and_finiteness():
    cfg = Config(**SMALL)
    m, _ = _run(cfg, n=80)
    sensors = Sensors()
    for tau in cfg.horizons:
        frame = predicted_retina(m, tau, sensors)
        assert frame.shape == (32, 32)
        assert np.all(np.isfinite(frame))


def test_ablations_all_construct_and_run():
    variants = [
        dict(use_uncertainty=False),
        dict(use_rewiring=False),
        dict(use_long_range=False),
        dict(use_landscape=False),
        dict(use_coalition_feedback=False),
        dict(use_oscillator=False),
        dict(msg_width=1),
        dict(instant_delivery=True),
    ]
    for v in variants:
        cfg = Config(**SMALL).variant(**v)
        m, _ = _run(cfg, n=60)
        assert np.all(np.isfinite(m.h)), f"ablation {v} produced non-finite state"


def test_config_label_names_only_the_differences():
    assert Config().label() == "full"
    assert Config(msg_width=1).label() == "msg_width=1"
