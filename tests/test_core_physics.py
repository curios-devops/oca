"""World v2 tests.

The important ones are the last three: they check the properties the world was built to
have. If bounces do not happen during occlusion, or the regime cue is invisible, or
occlusions are too short, the world cannot test what it is meant to test.
"""

import numpy as np
import pytest

from core.data import rollout
from core.world import RETINA
from core.world.physics import PhysicsConfig, PhysicsWorld, make_physics_world


def _run(n=3000, seed=0, **kw):
    w = PhysicsWorld(PhysicsConfig(seed=seed, **kw))
    snaps = []
    for _ in range(n):
        w.step()
        snaps.append(w.state_snapshot())
    return w, snaps


def test_objects_stay_inside_and_render_is_bounded():
    w, snaps = _run(3000)
    for s in snaps:
        assert np.all(s["pos"] >= 0.0) and np.all(s["pos"] <= w.cfg.size)
    img = w.render()
    assert img.shape == (64, 64)
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert np.all(np.isfinite(img))


def test_motion_neither_dies_nor_explodes():
    """Elastic bounces plus gravity must keep objects moving indefinitely."""
    w, snaps = _run(5000)
    late = np.stack([s["vel"] for s in snaps[-500:]])
    speed = np.linalg.norm(late, axis=2)
    assert speed.mean() > 0.3, f"objects settled (mean speed {speed.mean():.3f})"
    assert np.abs(late[:, :, 1]).max() <= w.cfg.max_vspeed + 1e-6


def test_gravity_regime_flips_and_is_visible_in_the_frame():
    cfg = PhysicsConfig(seed=0, regime_period=50)
    w = PhysicsWorld(cfg)
    seen = {}
    for _ in range(200):
        seen[w.regime] = w.render()[0, :].mean()   # top border row
        w.step()
    assert set(seen) == {1, -1}, "gravity regime never flipped"
    assert abs(seen[1] - seen[-1]) > 0.3, "the regime cue is not visible in the frame"


def test_gravity_actually_accelerates_objects():
    w = PhysicsWorld(PhysicsConfig(seed=1, n_objects=1, occluder=False))
    w.pos[0] = [32.0, 20.0]
    w.vel[0] = [0.0, 0.0]
    v0 = w.vel[0, 1]
    for _ in range(5):
        w.step()
    assert w.vel[0, 1] > v0 + 0.2, "gravity is not acting"


def test_occlusions_are_long_enough_to_require_memory():
    """Objects must be hidden for longer than the longest prediction horizon (16)."""
    w = PhysicsWorld(PhysicsConfig(seed=2))
    hidden_since, durations = {}, []
    for t in range(20000):
        for i in range(w.cfg.n_objects):
            now = w.is_fully_occluded(i)
            if now and i not in hidden_since:
                hidden_since[i] = t
            elif not now and i in hidden_since:
                durations.append(t - hidden_since.pop(i))
        w.step()
    assert len(durations) > 50, f"only {len(durations)} occlusion events"
    med = float(np.median(durations))
    assert 10 <= med <= 40, f"median occlusion {med} ticks is outside the useful band"


def test_objects_usually_bounce_while_hidden():
    """The property that makes this a permanence test rather than an extrapolation test.

    If an object travelled in a straight line while invisible, predicting its
    re-emergence would need no world model at all -- a ruler would do.
    """
    w = PhysicsWorld(PhysicsConfig(seed=3))
    hidden_since, vy_at_entry, bounced, total = {}, {}, 0, 0
    for t in range(20000):
        for i in range(w.cfg.n_objects):
            now = w.is_fully_occluded(i)
            if now and i not in hidden_since:
                hidden_since[i] = t
                vy_at_entry[i] = w.vel[i, 1]
            elif not now and i in hidden_since:
                hidden_since.pop(i)
                total += 1
                # a sign flip in vertical velocity means a bounce happened out of sight
                if np.sign(w.vel[i, 1]) != np.sign(vy_at_entry.pop(i)):
                    bounced += 1
        w.step()
    frac = bounced / max(total, 1)
    assert frac > 0.3, f"only {frac:.2f} of occlusions contained a bounce"


def test_rollout_is_deterministic_and_seed_sensitive():
    r1, s1, _ = rollout(200, seed=5, world_factory=make_physics_world)
    r2, s2, _ = rollout(200, seed=5, world_factory=make_physics_world)
    assert np.array_equal(r1, r2) and np.array_equal(s1, s2)
    assert r1.shape == (200, RETINA, RETINA)
    r3, _, _ = rollout(200, seed=6, world_factory=make_physics_world)
    assert not np.array_equal(r1, r3)


def test_occluder_hides_objects_completely():
    cfg = PhysicsConfig(seed=1)
    w = PhysicsWorld(cfg)
    x0, y0, wd, ht = cfg.occluder_rect
    w.pos[0] = [x0 + wd / 2, 32.0]
    w.radius[0] = 3.0
    img = w.render()
    band = img[8:56, x0:x0 + wd]
    assert np.allclose(band, cfg.occluder_value)
    assert w.is_fully_occluded(0)
