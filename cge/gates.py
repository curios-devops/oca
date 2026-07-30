"""Every gate behind one signature, so versions are scored identically.

Each gate takes a `MeshVersion` and returns a dict with a headline number and the control
that decides whether the headline means anything. Gates reuse the experiment code rather
than reimplementing it -- a second implementation of a probe is a second thing that can
disagree with the paper.

Every gate here reports `higher_is_better` and a `control` field. A gate whose control has
failed returns its numbers with `valid: False` so the scorecard can grey it out rather
than quietly averaging nonsense into a summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from core.data import train_test_streams
from core.metrics import frame_mse
from core.world import Sensors
from core.world.physics import make_physics_world


# ----------------------------------------------------------------- prediction


def gate_prediction(version, seed=0, side=12, train=6000, test=1200, **kw):
    """Held-out frame MSE against copy-last, on world v2. Lower is better."""
    data = train_test_streams(train, test, world_factory=make_physics_world)
    tr, te = data["train"], data["test"]
    state = version.new(seed=seed, side=side)
    sensors = Sensors()

    for t in range(len(tr["sensory"])):
        version.tick(state, tr["sensory"][t])

    state.learn = False
    # Corvus publishes its horizons on the stack rather than on a per-layer config, because a
    # layered architecture has one set of horizons and several configs. Ask the state first.
    horizons = getattr(state, "horizons", None) or state.cfg.horizons
    errs = {tau: [] for tau in horizons}
    frames = te["retina"]
    for t in range(len(te["sensory"])):
        version.tick(state, te["sensory"][t])
        for tau in horizons:
            if t + tau < len(frames):
                pred = _predicted_frame(version, state, tau, sensors)
                errs[tau].append(frame_mse(pred, frames[t + tau]))

    copy_last = {tau: float(np.mean([(frames[t] - frames[t + tau]) ** 2
                                     for t in range(len(frames) - tau)]))
                 for tau in horizons}
    out = {f"mse_t{tau}": float(np.mean(errs[tau])) for tau in horizons}
    out.update({f"vs_copylast_t{tau}": out[f"mse_t{tau}"] / copy_last[tau]
                for tau in horizons})
    out["headline"] = out[f"vs_copylast_t{max(horizons)}"]
    out["higher_is_better"] = False
    out["control"] = {"copy_last_t16": copy_last[max(horizons)]}
    out["valid"] = True
    return out


def _predicted_frame(version, state, tau, sensors):
    return version.predicted_frame(state, tau, sensors)


# ---------------------------------------------------------------------- maze


def gate_maze(version, seed=0, side=12, ticks=12000, warmup=2500, **kw):
    """Out-of-view wall decode, against the raw-pixel control. Higher is better."""
    from exp09_maze import RADIUS, visible_mask, wall_probe
    from core.world.maze import MazeConfig, MazeWorld

    world = MazeWorld(MazeConfig(seed=seed))
    sensors = Sensors()
    state = version.new(seed=0, side=side)
    learn_until = int(ticks * 0.8)

    H, R, W = [], [], []
    for t in range(ticks):
        s_now, ret = sensors.observe(world)
        state.learn = t < learn_until
        version.tick(state, s_now)
        if t > warmup:
            H.append(version.readout(state).ravel().copy())
            R.append(ret.ravel().copy())
            W.append(world.surrounding_walls(RADIUS))
        world.step()

    mask = visible_mask(MazeConfig().view)
    model = wall_probe(np.array(H), np.array(W), mask)
    retina = wall_probe(np.array(R), np.array(W), mask)
    return {
        "hidden_acc": model["hidden"],
        "visible_acc": model["visible"],
        "headline": model["hidden"] - retina["hidden"],
        "higher_is_better": True,
        "control": {"retina_hidden": retina["hidden"],
                    "retina_visible": retina["visible"]},
        # the probe is only meaningful if pixels can read what is actually on screen
        "valid": bool(retina["visible"] is not None and retina["visible"] > 0.8),
    }


# ------------------------------------------------------------------ identity


def gate_identity(version, seed=5, side=12, ticks=12000, warmup=2000, **kw):
    """Decode a hidden object's kind. Chance is 50%; higher is better.

    Collected through the adapter rather than through the experiment's own loop, because
    the experiment dispatches on a version name and a third architecture would have had to
    be special-cased into it. Each version says which of its units are local to the object;
    everything after that is identical code.
    """
    from exp08_identity import local_units, patch_pixels, probe_all
    from core.world.identity import IdentityConfig, IdentityWorld

    state = version.new(seed=0, side=side)
    world = IdentityWorld(IdentityConfig(seed=seed))
    sensors = Sensors()
    for _ in range(30):
        world.step()

    rows = []
    learn_until = int(ticks * 0.8)
    for t in range(ticks):
        s_now, ret = sensors.observe(world)
        state.learn = t < learn_until
        version.tick(state, s_now)
        if t > warmup:
            occ = [world.is_fully_occluded(i) for i in range(world.cfg.n_objects)]
            for i in range(world.cfg.n_objects):
                if occ[i] and sum(occ) != 1:
                    continue                    # ambiguous: more than one hidden
                tiles = local_units(world, i, side)[0]
                idx = version.object_units(state, world, i, side)
                rows.append({"h": version.readout(state)[idx].ravel().copy(),
                             "r": patch_pixels(ret, tiles),
                             "kind": int(world.kind[i]), "occ": bool(occ[i]), "obj": i})
        world.step()

    res = probe_all(rows)

    occ = res.get("occluded", {})
    ctrl_hidden = occ.get("raw_retina", {}).get("balanced_acc")
    ctrl_vis = res.get("visible", {}).get("raw_retina", {}).get("balanced_acc")
    model = occ.get("model_state", {}).get("balanced_acc")
    return {
        "hidden_acc": model,
        "visible_acc": res.get("visible", {}).get("model_state", {}).get("balanced_acc"),
        "headline": model,
        "higher_is_better": True,
        "control": {"retina_hidden": ctrl_hidden, "retina_visible": ctrl_vis},
        # valid only if the cue is readable when visible and leaks nothing when hidden
        "valid": bool(ctrl_vis and ctrl_vis > 0.65
                      and ctrl_hidden and abs(ctrl_hidden - 0.5) < 0.1),
    }


# ---------------------------------------------------------------- coalitions


def gate_coalitions(version, seed=3, side=12, ticks=8000, warmup=2000, **kw):
    """Do coalitions form, and do they carry object information above a shuffled null?"""
    from exp07_stage2 import normalised_mi
    from core.world.physics import PhysicsConfig, PhysicsWorld

    world = PhysicsWorld(PhysicsConfig(seed=seed, n_objects=3))
    sensors = Sensors()
    state = version.new(seed=0, side=side)
    for _ in range(30):
        world.step()

    fracs, mis, nulls = [], [], []
    for t in range(ticks):
        s_now, _ = sensors.observe(world)
        state.learn = t < int(ticks * 0.8)
        version.tick(state, s_now)
        if t > warmup and t % 50 == 0:
            lab = version.coalitions(state)
            if lab is None:
                continue
            sizes = np.bincount(lab)
            fracs.append(float((sizes[lab] > 1).mean()))
            # each version labels its *own* units, so a 17-node cortex and a 144-unit
            # mesh are asked the same question at their own granularity rather than the
            # finer one winning for being finer
            obj = version.unit_labels(world)
            seen = obj >= 0
            if seen.sum() >= 4 and len(set(obj[seen].tolist())) >= 2:
                sub = lab[:len(obj)][seen]
                mis.append(normalised_mi(sub, obj[seen]))
                rs = np.random.default_rng(t)
                nulls.append(np.mean([normalised_mi(rs.permutation(sub), obj[seen])
                                      for _ in range(20)]))
        world.step()

    excess = float(np.mean(mis) - np.mean(nulls)) if mis else 0.0
    sem = (float(np.std(np.array(mis) - np.array(nulls), ddof=1) / np.sqrt(len(mis)))
           if len(mis) > 1 else None)
    out = {
        "frac_in_coalition": float(np.mean(fracs)) if fracs else 0.0,
        "object_mi_excess": excess,
        "sem": sem,
        "headline": excess,
        "higher_is_better": True,
        "control": {"shuffled_null": float(np.mean(nulls)) if nulls else None,
                    "n_samples": len(mis)},
        "valid": len(mis) > 10,
    }
    if not out["valid"]:
        # This gate estimates MI *within* a frame, across units. That needs more labelled
        # units than there are objects, so an architecture with fewer units than a mesh
        # (17 nodes against 144) can never satisfy it -- three objects cannot label four
        # nodes. Reported as unmeasured rather than as zero, with the reason, and with a
        # pointer to the estimator that does work at this granularity: pooling across time
        # per unit, which is `cge.nodes.gate_concept_formation`.
        out["reason"] = (
            f"only {len(mis)} frames had enough labelled units for a within-frame MI; "
            "at this granularity use cge.nodes.gate_concept_formation, which pools "
            "across time per unit instead")
    return out


# ---------------------------------------------------------------- tunnel maze


def gate_tunnel(version, seed=0, side=12, ticks=14000, warmup=3000, **kw):
    """Position while blind, against remembering only where you went in. Higher is better.

    **The only gate in this battery where raw pixels cannot win.** Inside a covered corridor
    every frame is byte-identical, so the image carries exactly zero positional information
    and the pixel control is at chance by construction. Everywhere else in this repository
    the world is observable enough that the answer is already in the frame, and a
    representation can at best re-encode what the input already says — which is most of why
    nothing has beaten raw pixels yet.

    Here the answer is provably *not* in the frame. Position can only come from integrating
    your own moves since you entered, so this gate asks the question the rest of the battery
    cannot: does this architecture hold anything the current image does not contain?

    Three references bracket it. **Frozen-at-entry** is the bar — a model that merely stores
    "I went in at (r,c)" scores exactly this, so beating it is the definition of integrating.
    **Raw pixels** are the floor, at chance. A perfect dead-reckoner is the ceiling, at zero.
    """
    from core.probes import decode_error
    from core.world.maze import MazeConfig, MazeWorld

    world = MazeWorld(MazeConfig(seed=seed, tunnels=True))
    sensors = Sensors()
    state = version.new(seed=0, side=side)
    learn_until = int(ticks * 0.8)
    rows = []

    for t in range(ticks):
        s_now, ret = sensors.observe(world)
        state.learn = t < learn_until
        version.tick(state, s_now)
        if t > warmup and world.in_tunnel():
            rows.append({"h": version.readout(state).ravel().copy(),
                         "r": ret.ravel().copy(),
                         "pos": world.pos.astype(float).copy(),
                         "entry": np.array(world._tunnel_entry, dtype=float),
                         "steps": world.steps_in_tunnel()})
        world.step()

    if len(rows) < 200:
        return {"headline": None, "valid": False, "higher_is_better": True,
                "reason": f"only {len(rows)} in-tunnel frames"}

    Y = np.stack([r["pos"] for r in rows])
    entry = np.stack([r["entry"] for r in rows])
    cut = int(len(rows) * 0.6)

    # matched capacity: a 2304-dim mesh state and a 1024-dim frame are not comparable
    # decoders until they are given the same number of components
    def probe(key):
        return decode_error(np.stack([r[key] for r in rows]), Y, n_components=32)

    err_model, err_retina = probe("h"), probe("r")
    err_frozen = np.linalg.norm(entry[cut:] - Y[cut:], axis=1)
    err_chance = np.linalg.norm(Y[:cut].mean(0) - Y[cut:], axis=1)

    beats = 1.0 - float(err_model.mean()) / float(err_frozen.mean())
    return {
        "cells_error": float(err_model.mean()),
        "frozen_at_entry": float(err_frozen.mean()),
        "headline": beats,
        "higher_is_better": True,
        "control": {"retina": float(err_retina.mean()),
                    "chance": float(err_chance.mean()),
                    "n_frames": len(rows)},
        # the pixel control must be at chance, or the corridors are leaking position and
        # the whole point of this world is gone
        "valid": bool(abs(float(err_retina.mean()) - float(err_chance.mean()))
                      < 0.20 * float(err_chance.mean())),
    }


def gate_path_integration(version, seed=0, side=12, ticks=14000, warmup=3000, **kw):
    """Does the component integrate its own moves while blind? `CGE-A-09`.

    **This gate exists because `CGE-A-01` turned out not to be measurable**, and the difference
    between them is worth stating carefully because I got it wrong first.

    A-01 scored absolute position against `frozen-at-entry`, a baseline *handed the true entry
    coordinates*. An architecture has no such gift: it must encode where it is from a 5x5 view.
    Measured, that is not possible in a braided maze -- position decodes from the raw view at
    **4.92 cells while fully sighted**, against a chance of 8.58 and the baseline's 2.06. So no
    architecture could pass A-01 regardless of whether it persists, and all four failed it by
    similar margins for a reason that has nothing to do with persistence.

    This gate removes the anchor from the question. The target is **displacement since entering
    the tunnel**, not absolute position, so both sides are asked only about the part that can be
    inferred: how far have I moved since I last knew where I was.

    Controls, both at matched capacity:

    * `no_integration` -- predict zero displacement. This is what holding still achieves, and it
      is the honest analogue of frozen-at-entry now that the gift is removed.
    * the raw frame -- at chance by construction, since every in-tunnel frame is identical.
    """
    from core.probes import decode_error
    from core.world.maze import MazeConfig, MazeWorld

    world = MazeWorld(MazeConfig(seed=seed, tunnels=True))
    sensors = Sensors()
    state = version.new(seed=0, side=side)
    learn_until = int(ticks * 0.8)
    rows = []

    for t in range(ticks):
        s_now, ret = sensors.observe(world)
        state.learn = t < learn_until
        version.tick(state, s_now)
        if t > warmup and world.in_tunnel():
            rows.append({"h": version.readout(state).ravel().copy(),
                         "r": ret.ravel().copy(),
                         "disp": world.pos.astype(float) - np.array(world._tunnel_entry,
                                                                    dtype=float),
                         "steps": world.steps_in_tunnel()})
        world.step()

    if len(rows) < 200:
        return {"headline": None, "valid": False, "higher_is_better": True,
                "reason": f"only {len(rows)} in-tunnel frames"}

    Y = np.stack([r["disp"] for r in rows])
    cut = int(len(rows) * 0.6)

    def probe(key):
        return decode_error(np.stack([r[key] for r in rows]), Y, n_components=32)

    err_model = probe("h")
    err_retina = probe("r")
    # holding still: predict no displacement at all
    err_still = np.linalg.norm(Y[cut:], axis=1)

    beats = 1.0 - float(err_model.mean()) / max(float(err_still.mean()), 1e-9)
    return {
        "displacement_error": float(err_model.mean()),
        "no_integration": float(err_still.mean()),
        "headline": beats,
        "higher_is_better": True,
        "control": {"retina": float(err_retina.mean()),
                    "no_integration": float(err_still.mean()),
                    "n_frames": len(rows)},
        # the frame must carry no displacement information, or the corridors are leaking
        "valid": bool(float(err_retina.mean()) >= 0.85 * float(err_still.mean())),
    }


GATES = {
    "prediction": gate_prediction,
    "path_integration": gate_path_integration,
    "maze": gate_maze,
    "identity": gate_identity,
    "coalitions": gate_coalitions,
    "tunnel": gate_tunnel,
}
