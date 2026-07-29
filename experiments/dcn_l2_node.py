"""Level 2 gates — the Dynamic Cortical Node. Phase 1 (component) and Phase 2 (vs legacy).

Level 1 asked a channel question: what does silence cost? That is the wrong question for a
node. A node is the thing the design says *holds knowledge*, so level 2 asks whether it
forms concepts, whether compressing its members destroys what they carry, whether it
predicts at the timescale it claims, and whether five scalars and a phase spectrum are
really all it needs to say.

Phase 2 puts the frozen legacy meshes through the identical probe on the identical target.
That is the point of a shared benchmark and it is deliberately two-way: this architecture
is not assumed to be better, and anywhere v1 or v2 wins is a mechanism worth understanding
before it is discarded.

    python experiments/dcn_l2_node.py --ticks 16000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.components import _delta_events
from bench.nodes import (N_COMPONENTS, gate_channel_contention,
                         gate_concept_formation, gate_horizon, gate_publication,
                         gate_relational, kmeans_labels, predictive_gain,
                         retina_target)
from core.metrics import JsonlLogger
from core.world import Sensors
from core.world.physics import PhysicsConfig, PhysicsWorld
from legacy.dcn.cortex import NODE_SIDE, build_cortex, sensory_to_nodes, tick
from legacy.dcn.node import build_stack, step as node_step

TAUS = (1, 4, 16, 64)


def node_object_labels(world: PhysicsWorld, n_nodes: int) -> np.ndarray:
    """Which object is in each node's receptive field right now, or -1 for none.

    The nodes tile the image, so this is exactly "what is in front of you". A node whose
    patch is empty is labelled -1 and dropped from the concept gate rather than counted as
    a class of its own -- otherwise "nothing is here" dominates the mutual information and
    the gate rewards a node for noticing empty space.
    """
    s = world.cfg.size
    cell = s / NODE_SIDE
    lab = np.full(n_nodes, -1, dtype=np.int64)
    for i in range(world.cfg.n_objects):
        if world.is_fully_occluded(i):
            continue
        x, y = world.pos[i]
        nc, nr = int(np.clip(x // cell, 0, NODE_SIDE - 1)), int(np.clip(y // cell, 0, NODE_SIDE - 1))
        lab[nr * NODE_SIDE + nc] = i
    return lab


def collect(args, variants: dict) -> dict:
    """Drive every variant down one identical sensory stream.

    One stream, not one per variant. The neuron layer is seeded identically and does not
    depend on the node configuration, so every variant sees the *same* member states and
    the only thing that differs between them is the mechanism under test. Running them in
    separate loops would leave a difference in the world as an alternative explanation.
    """
    world = PhysicsWorld(PhysicsConfig(seed=args.seed, n_objects=3))
    sensors = Sensors()
    stacks = {name: build_cortex(seed=0, **kw) for name, kw in variants.items()}
    learn_until = int(args.ticks * 0.8)

    rec = {name: {"work": [], "pub": [], "h": [], "k": []} for name in stacks}
    retina, labels, patches, members = [], [], [], []

    for t in range(args.ticks):
        s_now, ret = sensors.observe(world)
        for name, st in stacks.items():
            st.learn = t < learn_until
            tick(st, s_now)
        if t > args.warmup:
            for name, st in stacks.items():
                r = rec[name]
                r["work"].append(st.work.ravel().copy())
                r["pub"].append(st.publication().ravel().copy())
                r["h"].append(st.h.ravel().copy())
                r["k"].append(st.k_win.copy())
            members.append(stacks["full"].member_state().ravel().copy())
            retina.append(ret.ravel().copy())
            labels.append(node_object_labels(world, stacks["full"].cfg.n_nodes))
            patches.append(sensory_to_nodes(s_now))
        world.step()

    out = {name: {k: np.array(v) for k, v in r.items()} for name, r in rec.items()}
    out["_retina"] = np.array(retina)
    out["_members"] = np.array(members)
    out["_labels"] = np.array(labels)
    out["_patches"] = np.array(patches)
    out["_stacks"] = stacks
    return out


def phase1(args) -> dict:
    # The first four cross aggregation with the workspace dynamics, because the legacy
    # result they descend from confounded the two: mean pooling was only ever measured
    # inside a low-pass filter, so it was never clear which one destroyed the code.
    variants = {
        "full": {},
        "pairwise_exact": {"aggregation": "pairwise"},
        "mean_pooling": {"aggregation": "mean"},
        "relational_lowpass": {"use_reservoir": False},
        "pairwise_lowpass": {"aggregation": "pairwise", "use_reservoir": False},
        "mean_lowpass": {"aggregation": "mean", "use_reservoir": False},
        "no_knowledge": {"use_knowledge": False},
        "no_neighbourhood": {"use_neighbourhood": False},
        "no_oscillation": {"use_oscillation": False, "coupling": 0.0},
        "no_state_adapter": {"use_state_adapter": False},
    }
    data = collect(args, variants)
    stack = data["_stacks"]["full"]
    cfg = stack.cfg
    tau = cfg.horizon
    cut = int(len(data["_retina"]) * 0.6)
    target = retina_target(data["_retina"], cut)

    res = {
        "config": cfg.label(),
        "n_nodes": cfg.n_nodes,
        "n_neurons_total": int(stack.neurons.n_neurons),
        "n_params": stack.n_params(),
        "declared_horizon": tau,
    }

    # -- L2-1 concept formation -------------------------------------------
    labels = data["_labels"]
    patch_clusters = np.stack(
        [kmeans_labels(data["_patches"][:, i, :], cfg.n_concepts, seed=i)
         for i in range(cfg.n_nodes)], axis=1)
    res["concept_formation"] = gate_concept_formation(
        data["full"]["k"], labels, patch_clusters, seed=args.seed)

    # -- L2-2 relational vs mean pooling (the kill gate) -------------------
    # Scored on two targets, because the legacy result this gate descends from used a
    # different one and that difference is the whole question. Legacy measured pairwise
    # against mean at predicting *the members' own future*, where relations between members
    # are the thing being predicted. If relations only pay off there and not on the world,
    # then "the predictive code is relational" is a statement about a mesh predicting
    # itself, and it does not carry across to a node predicting what is in front of it.
    member_pcs = retina_target(data["_members"], cut, N_COMPONENTS)
    res["relational"] = gate_relational(
        data["full"]["work"], data["mean_pooling"]["work"], target, tau)
    res["relational_member_target"] = gate_relational(
        data["full"]["work"], data["mean_pooling"]["work"], member_pcs, tau)
    # and the exact operator, on both targets, so a negative result cannot be blamed
    # on the sketch
    res["pairwise"] = gate_relational(
        data["pairwise_exact"]["work"], data["mean_pooling"]["work"], target, tau)
    res["pairwise_member_target"] = gate_relational(
        data["pairwise_exact"]["work"], data["mean_pooling"]["work"], member_pcs, tau)

    # -- L2-3 horizon, against the raw frame at the identical tau ----------
    res["horizon"] = gate_horizon(data["full"]["work"], target, TAUS, declared=tau,
                                  baseline=data["_retina"])

    # -- L2-4 the five-scalar bottleneck -----------------------------------
    res["publication"] = gate_publication(
        data["full"]["pub"], data["full"]["h"], target, tau)

    # -- L2-5 phase under channel contention -------------------------------
    res["contention"] = contention_gate(args)

    # -- ablations ---------------------------------------------------------
    res["ablations"] = {}
    for name in variants:
        g = predictive_gain(data[name]["work"], target, tau)
        cg = gate_concept_formation(data[name]["k"], labels, seed=args.seed)
        res["ablations"][name] = {
            "ratio_vs_persistence": g["ratio"],
            "concept_mi_excess": cg.get("excess_over_null"),
        }
    return res


def contention_gate(args) -> dict:
    """Gate L2-5: does phase staggering help when members share one channel?

    Two populations, identical apart from the phase gate, driven by the identical patch of
    the identical world. Their *offered* event rates are equalised by construction (both
    adapt to the same target), so anything left is scheduling.
    """
    from legacy.dcn.neuron import NeuronConfig, build_population, step as nstep

    world = PhysicsWorld(PhysicsConfig(seed=args.seed, n_objects=3))
    sensors = Sensors()
    base = NeuronConfig(seed=0, n_neurons=128, n_inputs=64)
    pops = {"gated": build_population(base),
            "ungated": build_population(base.variant(gate_depth=0.0))}
    dense, events = {k: [] for k in pops}, {k: [] for k in pops}

    for t in range(args.contention_ticks):
        s_now, _ = sensors.observe(world)
        x = sensory_to_nodes(s_now)[5]           # one node's patch, near the middle
        for name, p in pops.items():
            nstep(p, x)
            if t > 300:
                dense[name].append(p.a.copy())
                events[name].append(p._last_event.copy())
        world.step()

    d = np.array(dense["gated"])
    return gate_channel_contention(d, np.array(events["gated"]),
                                   np.array(events["ungated"]), seed=args.seed)


def phase2(args) -> dict:
    """The frozen legacy meshes, through the identical probe on the identical target.

    Deliberately symmetric. The question is not "is the new thing better" but "where does
    each win", because a mechanism that only the legacy line has is a mechanism worth
    lifting -- and the whole reason the benchmark lives outside both architectures is so
    that answer cannot be an artefact of the harness.
    """
    from legacy.v1.mesh import build_mesh, tick as tick1
    from legacy.v1.state import Config
    from legacy.v2 import Config2, build_mesh2, tick2

    world = PhysicsWorld(PhysicsConfig(seed=args.seed, n_objects=3))
    sensors = Sensors()
    learn_until = int(args.ticks * 0.8)

    models = {
        "rpdu_v1": (build_mesh(Config(lattice_side=args.side, seed=0, eta_head=0.01)),
                    tick1, lambda s: s.h.ravel().copy()),
        "rpdu_v2": (build_mesh2(Config2(lattice_side=args.side, seed=0, eta_head=0.01)),
                    tick2, lambda s: s.h.ravel().copy()),
        "dcn_v3": (build_cortex(seed=0), tick, lambda s: s.work.ravel().copy()),
    }
    rec = {k: [] for k in models}
    coal = {k: [] for k in models}
    retina, labels = [], []

    for t in range(args.ticks):
        s_now, ret = sensors.observe(world)
        for name, (state, step_fn, read) in models.items():
            state.learn = t < learn_until
            step_fn(state, s_now)
        if t > args.warmup:
            for name, (state, _, read) in models.items():
                rec[name].append(read(state))
                coal[name].append(np.asarray(state.coalition).copy())
            retina.append(ret.ravel().copy())
            labels.append(node_object_labels(world, models["dcn_v3"][0].cfg.n_nodes))
        world.step()

    retina = np.array(retina)
    cut = int(len(retina) * 0.6)
    target = retina_target(retina, cut)
    labels = np.array(labels)

    out = {}
    for name in models:
        X = np.array(rec[name])
        out[name] = {
            "dim": int(X.shape[1]),
            "n_params": int(models[name][0].n_params()),
            "horizon_curve": {str(t): predictive_gain(X, target, t)["ratio"]
                              for t in TAUS},
        }
    # the raw retina, as the floor: any model that cannot beat pixels is not adding a
    # representation, it is adding latency
    out["raw_pixels"] = {
        "dim": int(retina.shape[1]),
        "n_params": 0,
        "horizon_curve": {str(t): predictive_gain(retina, target, t)["ratio"]
                          for t in TAUS},
    }
    out["_grouping"] = _grouping_comparison(coal, labels, models, args.seed)
    return out


def _grouping_comparison(coal, labels, models, seed) -> dict:
    """What each architecture groups by, scored against the same world labels.

    v1 and v2 group units into coalitions by synchrony; the DCN groups nodes by which
    concept they are expressing. Both are claims about binding, so both are asked the same
    question -- does the grouping carry object identity above a shuffled null? -- rather
    than each being scored on the quantity that flatters it.
    """
    out = {}
    n_nodes = labels.shape[1]
    for name in models:
        C = np.array(coal[name])
        if C.ndim != 2:
            continue
        if name == "dcn_v3":
            out[name] = gate_concept_formation(C, labels, seed=seed)
        else:
            # a mesh coalition label per *unit*: take the label of the unit whose retinal
            # patch falls in each node's receptive field, so the two are asked about the
            # same region of the world
            idx = _mesh_units_for_nodes(C.shape[1], n_nodes)
            out[name] = gate_concept_formation(C[:, idx], labels, seed=seed)
    return out


def _mesh_units_for_nodes(n_units: int, n_nodes: int) -> np.ndarray:
    """One visual mesh unit per node region, so both are read at the same granularity."""
    from core.world.sensors import N_VISUAL
    side = int(round(N_VISUAL ** 0.5))
    per = side // NODE_SIDE
    idx = []
    for i in range(n_nodes):
        nr, nc = divmod(i, NODE_SIDE)
        p = min((nr * per) * side + (nc * per), n_units - 1)
        idx.append(p)
    return np.array(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=16000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--contention-ticks", type=int, default=4000)
    ap.add_argument("--budget", type=float, default=0.08)
    ap.add_argument("--side", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-phase2", action="store_true")
    ap.add_argument("--out", default="logs/dcn_l2.jsonl")
    args = ap.parse_args()

    print("Phase 1 -- component gates ...")
    res = {"phase1": phase1(args)}
    if not args.skip_phase2:
        print("Phase 2 -- versus the frozen legacy line ...")
        res["phase2"] = phase2(args)

    with JsonlLogger(args.out, meta=vars(args)) as log:
        log.log(kind="dcn_l2", **res)
    Path("logs/dcn_l2_summary.json").write_text(json.dumps(res, indent=2, default=float))
    _report(res)


def _report(res: dict) -> None:
    p1 = res["phase1"]
    print(f"\n{'='*72}\nLEVEL 2 -- DYNAMIC CORTICAL NODE\n{'='*72}")
    print(f"\n{p1['n_nodes']} nodes, {p1['n_neurons_total']} neurons, "
          f"{p1['n_params']:,} parameters, declared horizon {p1['declared_horizon']}\n")

    c = p1["concept_formation"]
    print("L2-1  CONCEPT FORMATION -- do the node's concepts track the world?")
    if not c.get("valid"):
        print(f"  UNMEASURED: {c.get('reason')}")
    else:
        print(f"  normalised MI            {c['mi']:.3f}  ({c['n_nodes_scored']} nodes)")
        print(f"  shuffled null            {c['shuffled_null']:.3f}")
        print(f"  excess over null         {c['excess_over_null']:+.3f}"
              + (f"  (sem {c['sem']:.3f})" if c.get("sem") else ""))
        if "raw_patch_clusters_mi" in c:
            print(f"  same k, clustered pixels {c['raw_patch_clusters_mi']:.3f}"
                  f"   -> node adds {c['excess_over_pixels']:+.3f}")
        ok = c["excess_over_null"] > 3 * (c.get("sem") or 1e9)
        print("  " + ("PASS -- concepts carry object identity beyond the null" if ok
                      else "FAIL -- indistinguishable from a shuffled labelling"))

    r = p1["relational"]
    print("\nL2-2  RELATIONAL vs MEAN POOLING  (the gate that can kill this level)")
    if not r.get("valid"):
        print("  UNMEASURED")
    else:
        print(f"  relational working state {r['relational_ratio']:.3f}x persistence")
        print(f"  mean pooling, same members, same capacity "
              f"{r['pooled_ratio']:.3f}x")
        print(f"  gain from keeping relations {r['gain_over_pooling']*100:+.1f}%")
        print("  " + ("PASS -- relations survive the compression"
                      if r["gain_over_pooling"] > 0.05 and r["beats_persistence"]
                      else "FAIL -- the operator is not what the legacy line measured"))

    rm = p1.get("relational_member_target", {})
    if rm.get("valid"):
        print("  same operators, but predicting the MEMBERS' own future instead of the "
              "world:")
        print(f"    relational {rm['relational_ratio']:.3f}x   "
              f"mean {rm['pooled_ratio']:.3f}x   "
              f"gain {rm['gain_over_pooling']*100:+.1f}%")
        print("    (this is the target the legacy result was measured on -- if relations "
              "pay\n     off here and not above, the finding was about a mesh predicting "
              "itself)")

    print("\n  the same three operators, exact rather than sketched, all at width d_work:")
    print(f"    {'operator':22s} {'world':>9s} {'members':>9s}")
    for lbl, key_w, key_m in (("relational (sketch)", "relational",
                               "relational_member_target"),
                              ("pairwise (exact)", "pairwise", "pairwise_member_target")):
        a, b = p1.get(key_w, {}), p1.get(key_m, {})
        if a.get("valid") and b.get("valid"):
            print(f"    {lbl:22s} {a['relational_ratio']:9.3f} "
                  f"{b['relational_ratio']:9.3f}")
    m = p1.get("relational", {})
    if m.get("valid"):
        print(f"    {'mean (control)':22s} {m['pooled_ratio']:9.3f} "
              f"{p1['relational_member_target']['pooled_ratio']:9.3f}")

    h = p1["horizon"]
    print("\nL2-3  HORIZON -- is the level scored where it actually works?")
    if h.get("valid"):
        print("  vs persistence  " + "  ".join(f"t={k}: {v:.3f}x" if v else f"t={k}: n/a"
                                               for k, v in h["curve"].items()))
        if h.get("curve_vs_pixels"):
            print("  vs raw pixels   " + "  ".join(
                f"t={k}: {v:.3f}x" for k, v in h["curve_vs_pixels"].items()))
            print("  (the second row is the one that means anything: every representation,"
                  "\n   pixels included, looks better the further out you ask it to predict)")
        print(f"  declared {h['declared_horizon']}, best {h['best_horizon']}  "
              + ("PASS -- the declaration holds" if h["declaration_holds"]
                 else "FAIL -- it wins somewhere it did not claim"))

    p = p1["publication"]
    print("\nL2-4  THE FIVE-SCALAR BOTTLENECK -- is publishing state really enough?")
    if p.get("valid"):
        print(f"  published scalars only   {p['published_only_ratio']:.3f}x persistence")
        print(f"  full internal state      {p['full_state_ratio']:.3f}x")
        print(f"  cost of the bottleneck   {p['cost_of_the_bottleneck']:.2f}x")
        print("  " + ("PASS -- the bottleneck is affordable"
                      if p["published_beats_persistence"]
                      and p["cost_of_the_bottleneck"] < 1.5
                      else "FAIL -- too much is lost at the mouthpiece"))

    ct = p1["contention"]
    print("\nL2-5  PHASE UNDER CHANNEL CONTENTION -- where oscillation should finally pay")
    print(f"  {'budget':>7s} {'dropped g/u':>14s} {'NRMSE gated':>12s} {'ungated':>9s} "
          f"{'gain':>7s}")
    for b, row in ct["curve"].items():
        g, u = row["phase_gated"], row["ungated"]
        print(f"  {b:>7s} {g['dropped_frac']:6.3f}/{u['dropped_frac']:<7.3f} "
              f"{g['nrmse']:12.3f} {u['nrmse']:9.3f} {row['nrmse_gain']*100:+6.1f}%")
    print(f"  mean gain across budgets {ct['mean_nrmse_gain']*100:+.1f}%, "
          f"best {ct['best_nrmse_gain']*100:+.1f}%")
    print("  " + ("PASS -- phase earns its place once members share a channel"
                  if ct["earns_its_place"]
                  else "FAIL -- no coordination benefit; axiom 3 has no payoff here"))

    print("\nABLATIONS (does each mechanism earn its place?)")
    print(f"  {'variant':18s} {'x persistence':>14s} {'concept MI excess':>18s}")
    for name, a in p1["ablations"].items():
        rr = f"{a['ratio_vs_persistence']:.3f}" if a["ratio_vs_persistence"] else "n/a"
        mm = (f"{a['concept_mi_excess']:+.3f}"
              if a["concept_mi_excess"] is not None else "n/a")
        print(f"  {name:18s} {rr:>14s} {mm:>18s}")

    if "phase2" not in res:
        return
    p2 = res["phase2"]
    print(f"\n{'='*72}\nPHASE 2 -- v1 vs v2 vs DCN, identical probe, identical target"
          f"\n{'='*72}\n")
    print("predicting the world tau ticks ahead, relative to assuming nothing changes")
    print("(below 1.000 beats persistence; the same ridge, the same 32 components)\n")
    print(f"  {'model':12s} {'dim':>6s} {'params':>10s} "
          + "".join(f"{'t='+str(t):>9s}" for t in TAUS))
    for name in ("raw_pixels", "rpdu_v1", "rpdu_v2", "dcn_v3"):
        if name not in p2:
            continue
        m = p2[name]
        cells = "".join(
            f"{m['horizon_curve'][str(t)]:9.3f}" if m["horizon_curve"].get(str(t))
            else f"{'n/a':>9s}" for t in TAUS)
        print(f"  {name:12s} {m['dim']:6d} {m['n_params']:10,d} {cells}")

    print("\ngrouping: does each architecture's own notion of binding carry object identity?")
    print(f"  {'model':12s} {'grouping':22s} {'MI':>7s} {'null':>7s} {'excess':>8s}")
    names = {"rpdu_v1": "coalitions (synchrony)", "rpdu_v2": "coalitions (synchrony)",
             "dcn_v3": "concepts (knowledge)"}
    for name, g in p2.get("_grouping", {}).items():
        if not g.get("valid"):
            print(f"  {name:12s} {names.get(name,''):22s}   unmeasured")
            continue
        print(f"  {name:12s} {names.get(name,''):22s} {g['mi']:7.3f} "
              f"{g['shuffled_null']:7.3f} {g['excess_over_null']:+8.3f}")


if __name__ == "__main__":
    main()
