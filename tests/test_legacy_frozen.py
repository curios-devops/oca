"""Legacy is frozen. This is what enforces it.

`legacy/` holds the RPDU line — v1's gradient-flow mesh, v2's oscillator mesh, and the
Predictive Assembly. It is a historical reference and a benchmark opponent, not a codebase
under development. Nothing in it should change again.

Freezing matters more than it sounds. The value of everything in `docs/RPDU/RESULTS.md` rests
on those numbers being reproducible, and the whole point of starting a new architecture on
a blank page is that the old one stays available as an unambiguous baseline. A legacy that
quietly drifts is worse than no baseline at all, because comparisons against it keep
looking valid.

These tests are deliberately cheap so they run on every commit.
"""

import numpy as np
import pytest

from core.data import rollout
from core.world.physics import make_physics_world


def test_v1_mesh_is_bit_reproducible():
    """The same seed must give the same mesh, forever."""
    from legacy.v1.mesh import build_mesh, tick
    from legacy.v1.state import Config

    def run():
        m = build_mesh(Config(lattice_side=12, seed=0, eta_head=0.01))
        _, sen, _ = rollout(150, seed=1, world_factory=make_physics_world)
        for t in range(150):
            tick(m, sen[t])
        return m

    a, b = run(), run()
    assert np.array_equal(a.h, b.h)
    assert np.array_equal(a.P, b.P)


def test_v1_published_shape_is_unchanged():
    """Guards the interface RESULTS.md and every gate depend on."""
    from legacy.v1.mesh import build_mesh
    from legacy.v1.state import Config

    m = build_mesh(Config(lattice_side=24, seed=0, eta_head=0.08))
    assert m.n_units == 576
    assert m.h.shape == (576, 16)
    assert m.src.shape[1] == 41, "degree changed; the published parameter count moves with it"
    assert m.n_params() == 2_250_944, "the 2.25M figure in RESULTS.md is load-bearing"


def test_v2_published_shape_is_unchanged():
    from legacy.v2 import Config2, build_mesh2

    m = build_mesh2(Config2(lattice_side=12, seed=0, eta_head=0.01))
    assert m.h.shape == (144, 24), "readout width feeds every probe comparison"
    assert m.z.shape == (144, 16, 2), "rotor layout is what makes v2 v2"


def test_the_two_architectures_remain_distinguishable():
    """v1 halts, v2 does not. The one structural difference the rewrite was for."""
    from legacy.v1 import landscape as L
    from legacy.v2 import dynamics as dyn

    rng = np.random.default_rng(0)
    h = rng.normal(0, 0.5, (4, 16))
    U = rng.normal(0, 0.25, (4, 16, 3))
    b = np.zeros((4, 16))
    for _ in range(500):
        h, _ = L.relax(h, U, b, eta=0.05, steps=1, noise=0.0)
    assert np.linalg.norm(L.flow(h, U, b), axis=1).mean() < 1e-3, "v1 should reach a fixed point"

    z = rng.normal(0, 0.5, (4, 8, 2))
    mu, omega = np.full((4, 8), 0.5), np.full((4, 8), 0.3)
    for _ in range(500):
        z, _ = dyn.step(z, omega, mu, np.zeros_like(z), eta=0.05, sub_steps=1)
    steps = []
    for _ in range(50):
        prev = z.copy()
        z, _ = dyn.step(z, omega, mu, np.zeros_like(z), eta=0.05, sub_steps=1)
        steps.append(np.linalg.norm(z - prev))
    assert min(steps) > 1e-3, "v2 should still be moving"


def test_assembly_stays_read_only_over_the_mesh():
    """The property that made the PA results interpretable."""
    from legacy.pa import build_assembly, step_assembly
    from legacy.v2 import Config2, build_mesh2, tick2

    mesh = build_mesh2(Config2(lattice_side=12, seed=0, eta_head=0.01))
    asm = build_assembly(mesh)
    _, sen, _ = rollout(60, seed=1, world_factory=make_physics_world)
    for t in range(60):
        tick2(mesh, sen[t])
    snap = {k: getattr(mesh, k).copy() for k in ("z", "P", "M", "a", "w", "y")}
    for _ in range(40):
        step_assembly(asm, mesh)
    for k, v in snap.items():
        assert np.array_equal(getattr(mesh, k), v), f"the assembly modified mesh.{k}"


def test_every_legacy_version_still_answers_the_benchmark_contract():
    """A frozen architecture is only useful if it can still be scored against a new one."""
    from bench.registry import available, get

    for name in ("v1", "v2"):
        assert name in available()
        v = get(name)
        state = v.new(seed=0, side=12)
        _, sen, _ = rollout(20, seed=1, world_factory=make_physics_world)
        for t in range(20):
            v.tick(state, sen[t])
        assert v.readout(state).ndim == 2
        assert v.coalitions(state) is not None
        assert "n_params" in v.describe(state)
