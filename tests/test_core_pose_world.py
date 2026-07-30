"""The pose world's structural guarantees.

The *statistical* checks live in `experiments/validate_pose_world.py` and are what decide
whether the world is valid at all. These pin the properties that make those checks meaningful,
so a later edit cannot quietly remove one and leave the validation passing for the wrong reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.world import Sensors
from core.world.pose import ACTIONS, SHAPES, PoseConfig, PoseWorld


def test_every_kind_has_the_same_blobs():
    """Blob count and appearance are identical across kinds, so a feature bag is at chance by
    construction and **arrangement is the only signal**. This is what makes the world a test of
    structure rather than of texture."""
    counts = {len(s) for s in SHAPES}
    assert counts == {3}, "all kinds must have the same number of blobs"

    cfg = PoseConfig()
    total = []
    for k in range(len(SHAPES)):
        w = PoseWorld(PoseConfig(seed=0))
        w.kind, w.pose = k, 0
        w._canvas = w._draw()
        total.append(float(w._canvas.sum()))
    # same ink on the canvas for every kind, to within the overlap of blobs at different radii
    assert max(total) / min(total) < 1.25, f"kinds differ in total mass: {total}"


def test_the_confusable_pair_shares_its_angles_and_its_radii():
    """Kinds 0 and 1 are the world's hardest control: identical angle multiset, identical radius
    multiset, paired differently. A radial histogram and a feature bag are both at chance
    between them, so only the angle-to-radius *pairing* separates them."""
    a, b = SHAPES[0], SHAPES[1]
    assert sorted(x for x, _ in a) == sorted(x for x, _ in b)
    assert sorted(r for _, r in a) == sorted(r for _, r in b)
    assert a != b, "the pair must actually differ in the pairing"


def test_no_shape_maps_onto_itself_under_a_pose_rotation():
    """A rotationally symmetric shape would make a held-out pose *identical* to a trained one
    and leak the answer without any probe noticing."""
    cfg = PoseConfig()
    for k in range(len(SHAPES)):
        imgs = []
        for p in range(cfg.n_poses):
            w = PoseWorld(PoseConfig(seed=0))
            w.kind, w.pose = k, p
            imgs.append(w._draw())
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                d = np.abs(imgs[i] - imgs[j]).mean()
                assert d > 0.005, f"kind {k}: poses {i} and {j} are near-identical ({d:.4f})"


def test_a_frame_is_a_fragment_not_the_whole_object():
    """The sensor is a fovea. If a frame showed the whole canvas there would be nothing to
    integrate across movement, and the world would not pose the problem it exists for."""
    cfg = PoseConfig()
    assert cfg.size < cfg.canvas
    w = PoseWorld(cfg)
    assert w.render().shape == (cfg.size, cfg.size)
    assert w.full_object().shape == (cfg.canvas, cfg.canvas)


def test_moving_the_fovea_changes_the_view_and_not_the_object():
    """The object is rigid and stationary; only the sensor moves. So nothing in the image says
    how far the fovea travelled -- that is what makes the efference copy load-bearing."""
    w = PoseWorld(PoseConfig(seed=0))
    before_obj = w.full_object().copy()
    before_view = w.render().copy()
    for _ in range(6):
        w.step(action=3)                      # right
    assert np.array_equal(w.full_object(), before_obj), "the object moved"
    assert not np.array_equal(w.render(), before_view), "the view did not"


def test_the_efference_copy_is_delivered_every_tick():
    w = PoseWorld(PoseConfig(seed=0))
    for a in range(len(ACTIONS)):
        w.step(action=a)
        e = w.efference()
        assert e.sum() == 1.0 and e[a] == 1.0
        assert np.allclose(w.contact[:len(ACTIONS)], e)


def test_held_out_poses_are_a_strict_subset_and_non_empty():
    cfg = PoseConfig()
    assert cfg.held_out, "with no held-out pose the world measures nothing"
    assert set(cfg.held_out) < set(range(cfg.n_poses))


def test_training_mode_never_draws_a_held_out_pose():
    """`train_only=True` is what the gates will use to fit. If it ever emitted a held-out pose,
    the entire generalisation claim would be contaminated and nothing would report it."""
    w = PoseWorld(PoseConfig(seed=0))
    for _ in range(300):
        w._new_episode(train_only=True)
        assert w.pose not in w.cfg.held_out


def test_ground_truth_is_reachable_and_not_in_the_sensory_stream():
    """The gap between the simulator's state and what the sensors deliver *is* the cognitive
    problem. Kind and pose must be available to score against and absent from what is sensed."""
    w = PoseWorld(PoseConfig(seed=0))
    snap = w.state_snapshot()
    assert {"kind", "pose", "held_out", "fovea"} <= set(snap)

    sensory, _ = Sensors().observe(w)
    assert np.isfinite(sensory).all()
    # the somatic block carries the efference copy and nothing else; no label rides along
    assert float(np.abs(w.contact[len(ACTIONS):]).sum()) == 0.0


def test_the_world_is_deterministic_given_a_seed():
    def trace():
        w = PoseWorld(PoseConfig(seed=7))
        out = []
        for _ in range(50):
            w.step()
            out.append((w.kind, w.pose, float(w.render().sum())))
        return out
    assert trace() == trace()


@pytest.mark.parametrize("n_poses,held", [(6, (4,)), (8, (5, 7))])
def test_alternative_pose_schedules_still_construct(n_poses, held):
    """The schedule is a swept parameter -- eight poses at 45 degrees made the world VOID and
    six at 60 did not -- so alternatives must remain constructible for that sweep to be
    repeatable."""
    w = PoseWorld(PoseConfig(seed=0, n_poses=n_poses, held_out=held))
    for _ in range(20):
        w.step()
    assert w.render().shape == (w.cfg.size, w.cfg.size)
