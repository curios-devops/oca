"""Tests for the tunnel world, the shared probe, and the Predictive Assembly.

Several of these encode mistakes that were actually made and measured during this work,
so a future change that reintroduces one fails loudly instead of quietly producing a
number that looks plausible.
"""

import numpy as np
import pytest

from core.probes import decode_error, whiten
from core.world import Sensors
from core.world.maze import MazeConfig, MazeWorld
from architectures.swift import Config2, build_mesh2, tick2
from architectures.swift.pa import AssemblyConfig, build_assembly, export, pooled_members, step_assembly


# ------------------------------------------------------------------- tunnels


def test_inside_a_tunnel_every_view_is_identical():
    """The property that makes the pixel control chance-level by construction."""
    w = MazeWorld(MazeConfig(seed=0, tunnels=True))
    views = []
    for _ in range(4000):
        if w.in_tunnel():
            views.append(w.local_view().copy())
        w.step()
    assert len(views) > 200, "the agent barely entered a tunnel"
    V = np.stack(views)
    assert np.all(V == V[0]), "tunnel views differ, so pixels leak position"


def test_agent_still_moves_and_still_gets_its_efference_copy_while_blind():
    w = MazeWorld(MazeConfig(seed=0, tunnels=True))
    moved, effs = 0, []
    prev = tuple(w.pos)
    for _ in range(3000):
        w.step()
        if w.in_tunnel():
            if tuple(w.pos) != prev:
                moved += 1
            effs.append(w.efference().copy())
        prev = tuple(w.pos)
    assert moved > 100, "the agent froze inside tunnels"
    E = np.stack(effs)
    assert E[:, :4].sum(axis=1).min() == 1.0, "efference copy is not one-hot"
    assert E.std(axis=0)[:4].max() > 0, "the action signal is constant while blind"


def test_tunnel_entry_is_always_recorded_while_inside():
    """The frozen-at-entry baseline has nothing to freeze without this."""
    w = MazeWorld(MazeConfig(seed=0, tunnels=True))
    for _ in range(4000):
        assert not (w.in_tunnel() and w._tunnel_entry is None)
        w.step()


def test_tunnels_are_off_by_default():
    assert MazeWorld(MazeConfig(seed=0)).tunnel.sum() == 0


# --------------------------------------------------------------------- probe


def test_probe_survives_wildly_different_feature_scales():
    """An unstandardised ridge returned errors *above chance* on the workspace."""
    rng = np.random.default_rng(0)
    n = 600
    signal = np.cumsum(rng.normal(0, 1, (n, 2)), axis=0)
    X = np.c_[signal * 1e-7, signal * 1e5, rng.normal(0, 1, (n, 20))]
    err = decode_error(X, signal).mean()
    chance = np.linalg.norm(signal[:360].mean(0) - signal[360:], axis=1).mean()
    assert err < 0.5 * chance, f"probe failed on scale-mixed features ({err:.2f} vs {chance:.2f})"


def test_pca_is_taken_on_centred_not_whitened_data():
    """Whitening first ranks numerical noise as signal.

    With one real direction and many near-constant ones, per-dimension whitening inflates
    every constant to unit variance and the SVD can no longer tell them apart.
    """
    rng = np.random.default_rng(1)
    n = 400
    real = rng.normal(0, 1, (n, 1))
    X = np.c_[real, rng.normal(0, 1e-9, (n, 40))]
    reduce = whiten(X, cut=240, n_components=2)
    Z = reduce(X)
    corr = max(abs(np.corrcoef(Z[:, i], real[:, 0])[0, 1]) for i in range(Z.shape[1]))
    assert corr > 0.9, "the real direction was not among the leading components"


def test_matched_capacity_makes_wide_and_narrow_comparable():
    rng = np.random.default_rng(2)
    n = 500
    y = np.cumsum(rng.normal(0, 1, (n, 2)), axis=0)
    wide = np.c_[y, rng.normal(0, 1, (n, 900))]
    err_free = decode_error(wide, y).mean()
    err_matched = decode_error(wide, y, n_components=32).mean()
    assert err_matched < err_free, "matching capacity should curb overfitting on a wide input"


# ------------------------------------------------------------------ assembly


def _mesh_and_assembly(cfg=None, ticks=400):
    from core.data import rollout
    from core.world.physics import make_physics_world
    mesh = build_mesh2(Config2(lattice_side=12, seed=0, eta_head=0.01))
    asm = build_assembly(mesh, cfg or AssemblyConfig())
    _, sen, _ = rollout(ticks, seed=1, world_factory=make_physics_world)
    for t in range(ticks):
        tick2(mesh, sen[t])
        step_assembly(asm, mesh)
    return mesh, asm


def test_assembly_runs_and_stays_finite():
    mesh, asm = _mesh_and_assembly()
    for name in ("w", "W", "P", "pooled_mean"):
        assert np.all(np.isfinite(getattr(asm, name))), f"{name} went non-finite"
    assert np.abs(asm.w).max() <= 50.0 + 1e-6


def test_assembly_never_writes_to_the_mesh():
    """The layer is read-only, and that is what makes a failure here interpretable."""
    from core.data import rollout
    from core.world.physics import make_physics_world
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


def test_workspace_spans_a_range_of_timescales():
    _, asm = _mesh_and_assembly(ticks=100)
    tau = 1.0 / (1.0 - asm.retention)
    assert tau.min() < 20 and tau.max() > 100, f"time constants {tau.min():.0f}-{tau.max():.0f}"
    assert np.all(np.diff(asm.retention) > 0), "retention should be ordered"


def test_a_pure_lowpass_cannot_represent_a_running_sum():
    """Why retention is decoupled from alpha. The first version tied them and sat at chance."""
    rng = np.random.default_rng(0)
    steps = rng.choice([-1.0, 1.0], size=400)
    truth = np.cumsum(steps)
    chance = np.abs(truth - truth.mean()).mean()

    def filtered(retention, alpha=0.04):
        w, out = 0.0, []
        for s in steps:
            w = retention * w + alpha * s
            out.append(w)
        est = np.array(out)
        return np.abs(np.polyval(np.polyfit(est, truth, 1), est) - truth).mean()

    assert filtered(0.96) > 0.9 * chance, "a low-pass should be no better than chance here"
    assert filtered(1.0) < 0.01 * chance, "a perfect accumulator should be exact"


def test_export_is_exactly_five_scalars():
    mesh, asm = _mesh_and_assembly(ticks=100)
    e = export(asm, mesh)
    assert e.shape == (asm.n_assemblies, 5)
    assert np.all(np.isfinite(e))


def test_pooled_members_matches_membership():
    mesh, asm = _mesh_and_assembly(ticks=100)
    p = pooled_members(asm, mesh)
    expect = mesh.h[asm.members[0]].mean(axis=0)
    assert np.allclose(p[0], expect)


def test_manager_reference_is_in_matched_units():
    """Comparing assembly error to mesh `surprise` meant it could never fire."""
    _, asm = _mesh_and_assembly(AssemblyConfig(manager_window=50), ticks=600)
    ratio = asm.err_ema.mean() / (asm.member_err_ema.mean() + 1e-12)
    assert 1e-3 < ratio < 1e3, f"the two errors are on incomparable scales (ratio {ratio:.2g})"


def test_assembly_predicts_at_its_own_horizon():
    """A slow layer scored on a 1-tick prediction is competing with persistence.

    Measured: a best-case offline fit from the *entire* mesh state scores 1.32x
    persistence at tau=1 and 0.69x at tau=64, so the horizon is not a free parameter --
    below it there is nothing for any implementation to win.
    """
    cfg = AssemblyConfig(horizon=8, manager=False)
    mesh, asm = _mesh_and_assembly(cfg, ticks=200)
    assert len(asm.hist) <= cfg.horizon + 1, "prediction queue is not horizon-bounded"
    assert asm._horizon_ago is not None, "no horizon-delayed reference was kept"


def test_aggregation_modes_both_run_and_differ():
    from core.data import rollout
    from core.world.physics import make_physics_world
    fields = {}
    for agg in ("mean", "projection"):
        mesh = build_mesh2(Config2(lattice_side=12, seed=0, eta_head=0.01))
        asm = build_assembly(mesh, AssemblyConfig(aggregation=agg, manager=False))
        _, sen, _ = rollout(300, seed=1, world_factory=make_physics_world)
        for t in range(300):
            tick2(mesh, sen[t])
            step_assembly(asm, mesh)
        assert np.all(np.isfinite(asm.w))
        fields[agg] = asm.w.copy()
    assert not np.allclose(fields["mean"], fields["projection"])


def test_workspace_readout_is_gain_controlled():
    """The raw field may drift; what anything downstream reads must not."""
    from architectures.swift.pa import workspace
    mesh, asm = _mesh_and_assembly(ticks=900)
    ws = workspace(asm)
    assert np.all(np.isfinite(ws))
    assert np.abs(ws).max() < 1e3, "normalised readout should not blow up"


def test_pairwise_aggregation_preserves_what_a_mean_destroys():
    """The distinction a mean cannot make, and the reason it is the default.

    Two members active and two silent averages to the same thing as the reverse pairing.
    Co-activation does not, and that difference is where the prediction was hiding.
    """
    a = np.array([1.0, 1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 1.0, 1.0])
    assert np.isclose(a.mean(), b.mean()), "the example needs equal means"
    iu = np.triu_indices(4, k=1)
    pa, pb = np.outer(a, a)[iu], np.outer(b, b)[iu]
    assert not np.allclose(pa, pb), "co-activation should tell these apart"


def test_pairwise_is_the_default_and_sized_correctly():
    from architectures.swift.pa import AssemblyConfig, build_assembly
    assert AssemblyConfig().aggregation == "pairwise"
    mesh, asm = _mesh_and_assembly(AssemblyConfig(manager=False), ticks=120)
    m = asm.member_idx.shape[1]
    assert asm.W.shape[2] == m * (m - 1) // 2, "input width should be the upper triangle"
    assert np.all(np.isfinite(asm.w))
