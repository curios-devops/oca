import numpy as np
import pytest

from core.data import rollout
from core.world import RETINA, GridWorld, Sensors, WorldConfig
from core.world.sensors import N_SENSORY, P


def test_objects_stay_inside_and_render_is_bounded():
    w = GridWorld(WorldConfig(seed=3))
    for _ in range(2000):
        w.step()
        assert np.all(w.pos >= -1e-6)
        assert np.all(w.pos <= w.cfg.size + 1e-6)
    img = w.render()
    assert img.shape == (64, 64)
    assert 0.0 <= img.min() and img.max() <= 1.0


def test_collisions_conserve_kinetic_energy():
    cfg = WorldConfig(seed=5, occluder=False)
    w = GridWorld(cfg)
    ke0 = float((w.vel**2).sum())
    for _ in range(1000):
        w.step()
    ke1 = float((w.vel**2).sum())
    # walls flip a sign and equal-mass elastic collisions exchange the normal
    # component, so neither may inject or drain energy
    assert ke1 == pytest.approx(ke0, rel=1e-9)


def test_occluder_actually_occludes():
    cfg = WorldConfig(seed=1)
    w = GridWorld(cfg)
    x0, y0, wd, ht = cfg.occluder_rect
    w.pos[0] = [x0 + wd / 2, y0 + ht / 2]
    w.bright[0] = 1.0
    img = w.render()
    patch = img[y0 : y0 + ht, x0 : x0 + wd]
    assert np.allclose(patch, cfg.occluder_value)
    assert w.is_occluded(0)


def test_patch_roundtrip_is_exact():
    s = Sensors()
    ret = np.random.default_rng(0).random((RETINA, RETINA)).astype(np.float32)
    assert np.allclose(s.from_patches(s.to_patches(ret)), ret)


def test_rollout_shapes_and_determinism():
    r1, s1, snap1 = rollout(50, seed=7)
    r2, s2, _ = rollout(50, seed=7)
    assert r1.shape == (50, RETINA, RETINA)
    assert s1.shape == (50, N_SENSORY, P)
    assert np.array_equal(r1, r2) and np.array_equal(s1, s2)
    assert len(snap1) == 50
    # a different seed must give a different world
    r3, _, _ = rollout(50, seed=8)
    assert not np.array_equal(r1, r3)


def test_lattice_slots_are_unique_and_retinotopic():
    idx = Sensors.lattice_slots(24)
    assert idx.size == N_SENSORY
    assert np.unique(idx).size == N_SENSORY
    # first two visual units are horizontally adjacent on the lattice
    assert idx[1] - idx[0] == 1


def test_perturb_changes_motion():
    w = GridWorld(WorldConfig(seed=2))
    rng = np.random.default_rng(0)
    v0 = w.vel[0].copy()
    w.perturb(0, rng, "reverse")
    assert np.allclose(w.vel[0], -v0)
