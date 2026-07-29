"""Level 1 gates — the Dynamic Neuron. Phase 1 (component) and Phase 2 (versus legacy).

Phase 1 measures the two things the design asks of a neuron, **precision and efficiency**,
and they only mean anything together. Emitting every tick is perfectly precise and is a
wire. Never emitting is perfectly efficient and is a rock. The claim under test is that
emitting *on significant change* buys a better exchange rate than the same budget spent on
a schedule or at random — so every gate reports precision against event rate, with a
control at the identical rate.

Phase 2 compares against the frozen legacy line on the one axis where they are genuinely
comparable: a legacy unit publishes its state every tick, so its event rate is 1.0 by
construction. The question is what the Dynamic Neuron gives up to speak less often.

    python experiments/dcn_l1_neuron.py --ticks 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.components import (_delta_events, gate_energy, gate_noise_robustness,
                              gate_oscillation, gate_rate_distortion, nrmse,
                              summarise_rate_distortion, zero_order_hold)
from core.data import rollout
from core.metrics import JsonlLogger
from core.world.physics import make_physics_world
from legacy.dcn.neuron import NeuronConfig, build_population, step


def input_stream(ticks: int, seed: int, n_inputs: int, noise: float = 0.0):
    """A real sensory stream, not synthetic noise: one patch of the physics world.

    Level 1 should be tested on the same worlds every other level is, or its numbers say
    nothing about the system it is meant to be part of.
    """
    _, sen, _ = rollout(ticks + 50, seed=seed, world_factory=make_physics_world)
    x = sen[:, 0, :n_inputs].astype(np.float64)
    if noise:
        x = x + np.random.default_rng(seed).normal(0, noise, x.shape)
    return x


def run_population(cfg: NeuronConfig, ticks: int, seed: int, noise: float = 0.0):
    """Drive a population and record what it computes and what it transmits."""
    pop = build_population(cfg)
    x = input_stream(ticks, seed, cfg.n_inputs, noise)
    dense, sent, events, stats = [], [], [], []
    for t in range(ticks):
        d = step(pop, x[t])
        dense.append(pop.a.copy())
        sent.append(pop.last_sent.copy())
        events.append(pop._last_event.copy())
        stats.append(d)
    return (np.array(dense), np.array(sent), np.array(events), pop,
            {"theta": pop.theta.copy(),
             "energy_per_tick": float(pop.energy.mean() / ticks),
             "energy_per_event": float(pop.energy.sum() / max(pop.n_events, 1)),
             "event_rate": float(np.mean(events))})


def phase1(args) -> dict:
    cfg = NeuronConfig(seed=0, n_neurons=args.neurons)
    dense, sent, events, pop, stats = run_population(cfg, args.ticks, args.seed)

    res = {
        "config": cfg.label(),
        "n_neurons": cfg.n_neurons,
        "n_params": pop.n_params(),
        "live": {
            "event_rate": stats["event_rate"],
            "nrmse_as_run": nrmse(sent, dense),
            "energy_per_tick": stats["energy_per_tick"],
        },
        "rate_distortion": gate_rate_distortion(dense, seed=args.seed),
        "oscillation": gate_oscillation(build_population(cfg), step),
        "energy": gate_energy(stats),
    }
    res["rd_summary"] = summarise_rate_distortion(res["rate_distortion"])

    res["noise"] = gate_noise_robustness(
        lambda noise, seed: (run_population(cfg, args.ticks // 2, seed, noise)[0],
                             run_population(cfg, args.ticks // 2, seed, noise)[4]),
        seed=args.seed)

    # ablations: does each mechanism earn its place?
    res["ablations"] = {}
    for name, var in (("no_oscillation", dict(use_oscillation=False, gate_depth=0.0)),
                      ("no_phase_gate", dict(gate_depth=0.0)),
                      ("phase_as_content", dict(gate_depth=0.0, osc_into_activation=0.35)),
                      ("no_plasticity", dict(use_plasticity=False)),
                      ("fixed_threshold", dict(use_adaptive_theta=False))):
        d, s, e, _, st = run_population(cfg.variant(**var), args.ticks, args.seed)
        res["ablations"][name] = {"event_rate": st["event_rate"],
                                  "nrmse_as_run": nrmse(s, d),
                                  "energy_per_tick": st["energy_per_tick"]}
    return res


def phase2(args) -> dict:
    """Versus the frozen legacy line, on the axis where they are comparable.

    A legacy unit publishes its state every tick: event rate 1.0, reconstruction exact by
    definition. So the comparison is not "who is more accurate" -- that is settled in
    advance -- but what the Dynamic Neuron gives up in exchange for silence, and whether
    the exchange rate is favourable.
    """
    from legacy.v1.mesh import build_mesh, tick
    from legacy.v1.state import Config
    from legacy.v2 import Config2, build_mesh2, tick2

    _, sen, _ = rollout(args.ticks + 50, seed=args.seed,
                        world_factory=make_physics_world)
    out = {}

    for name, build, step_fn in (
            ("legacy_v1_unit", lambda: build_mesh(Config(lattice_side=12, seed=0,
                                                         eta_head=0.01)), tick),
            ("legacy_v2_unit", lambda: build_mesh2(Config2(lattice_side=12, seed=0,
                                                           eta_head=0.01)), tick2)):
        state = build()
        traces = []
        for t in range(args.ticks):
            step_fn(state, sen[t])
            traces.append(state.h[:args.neurons, 0].copy())
        traces = np.array(traces)
        out[name] = {
            "event_rate": 1.0,           # publishes every tick, by construction
            "nrmse": 0.0,
            "note": "publishes state every tick",
            "signal_std": float(traces.std()),
        }
        # what would this unit's own trace cost under the DN's emission policy?
        for target in (0.15, 0.30):
            ev = _delta_events(traces, _theta_for_rate(traces, target))
            out[name][f"if_gated_at_{target:g}"] = {
                "nrmse": nrmse(zero_order_hold(traces, ev), traces),
                "event_rate": float(ev.mean()),
            }

    cfg = NeuronConfig(seed=0, n_neurons=args.neurons)
    dense, sent, events, pop, stats = run_population(cfg, args.ticks, args.seed)
    out["dynamic_neuron"] = {
        "event_rate": stats["event_rate"],
        "nrmse": nrmse(sent, dense),
        "energy_per_tick": stats["energy_per_tick"],
    }
    return out


def _theta_for_rate(values, rate):
    lo, hi = 1e-6, 10.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _delta_events(values, mid).mean() > rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=8000)
    ap.add_argument("--neurons", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-phase2", action="store_true")
    ap.add_argument("--out", default="logs/dcn_l1.jsonl")
    args = ap.parse_args()

    print("Phase 1 -- component gates ...")
    p1 = phase1(args)
    res = {"phase1": p1}
    if not args.skip_phase2:
        print("Phase 2 -- versus legacy ...")
        res["phase2"] = phase2(args)

    with JsonlLogger(args.out, meta=vars(args)) as log:
        log.log(kind="dcn_l1", **{k: v for k, v in res.items()})
    Path("logs/dcn_l1_summary.json").write_text(json.dumps(res, indent=2))
    _report(res)


def _report(res: dict) -> None:
    p1 = res["phase1"]
    print(f"\n{'='*66}\nLEVEL 1 -- DYNAMIC NEURON\n{'='*66}")
    print(f"\n{p1['n_neurons']} neurons, {p1['n_params']} parameters\n")

    print("as run:")
    live = p1["live"]
    print(f"  event rate            {live['event_rate']:.3f} "
          f"(a legacy unit is 1.000)")
    print(f"  reconstruction NRMSE  {live['nrmse_as_run']:.3f} "
          f"(1.000 = transmitting nothing)")
    print(f"  energy / neuron / tick {live['energy_per_tick']:.3f}")

    print("\nPRECISION vs EFFICIENCY -- NRMSE at matched event rates")
    rd = p1["rate_distortion"]
    rates = list(next(iter(rd.values())))
    print(f"  {'policy':16s}" + "".join(f"{r:>9s}" for r in rates))
    for name, curve in rd.items():
        print(f"  {name:16s}" + "".join(f"{curve[r]['nrmse']:9.3f}" for r in rates))

    s = p1["rd_summary"]
    gain = s["delta_vs_best_control"]
    print(f"\n  area under the curve (lower is better): "
          + ", ".join(f"{k} {s[k]:.3f}" for k in ("send_on_delta", "periodic", "random")))
    print(f"  send-on-delta vs the best control at matched rate: {gain*100:+.1f}%")
    print("  " + ("PASS -- emitting on significant change beats spending the same "
                  "budget on a schedule" if gain > 0.05 else
                  "FAIL -- the policy is not beating its own rate; the mechanism is "
                  "not earning its place"))

    osc = p1["oscillation"]
    print(f"\nOSCILLATION (no input at all)")
    print(f"  phase advance / tick  {osc['phase_advance_per_tick']:.4f}")
    print(f"  amplitude             {osc['amplitude_mean']:.3f} "
          f"(drift {osc['amplitude_drift']:.4f})")
    print(f"  rotor still moving    {osc['still_moving']}  "
          + ("PASS -- sustains its own rhythm" if osc["still_moving"]
             else "FAIL -- settles to a fixed point, as legacy v1 did"))
    print(f"  activation activity   {osc['activation_activity']:.2e}  "
          + ("PASS -- the clock is not leaking into the content"
             if osc["clock_is_not_content"]
             else "FAIL -- the rhythm is in the transmitted value (axiom 3)"))

    print(f"\nNOISE ROBUSTNESS")
    for lvl, r in p1["noise"].items():
        print(f"  sigma {lvl:>4s}  NRMSE {r['nrmse']:.3f}  rate {r['event_rate']:.3f}")

    print(f"\nABLATIONS (does each mechanism earn its place?)")
    print(f"  {'variant':18s} {'rate':>7s} {'NRMSE':>8s} {'energy':>8s}")
    print(f"  {'full':18s} {live['event_rate']:7.3f} {live['nrmse_as_run']:8.3f} "
          f"{live['energy_per_tick']:8.3f}")
    for name, a in p1["ablations"].items():
        print(f"  {name:18s} {a['event_rate']:7.3f} {a['nrmse_as_run']:8.3f} "
              f"{a['energy_per_tick']:8.3f}")

    if "phase2" not in res:
        return
    print(f"\n{'='*66}\nPHASE 2 -- versus the frozen legacy line\n{'='*66}\n")
    p2 = res["phase2"]
    dn = p2["dynamic_neuron"]
    print(f"  {'unit':18s} {'event rate':>11s} {'NRMSE':>8s}")
    for name in ("legacy_v1_unit", "legacy_v2_unit"):
        print(f"  {name:18s} {p2[name]['event_rate']:11.3f} {p2[name]['nrmse']:8.3f}"
              f"   {p2[name]['note']}")
    print(f"  {'dynamic_neuron':18s} {dn['event_rate']:11.3f} {dn['nrmse']:8.3f}")
    print(f"\n  a legacy unit publishes {1/max(dn['event_rate'],1e-9):.1f}x more often.")
    print("  what a legacy unit's own trace would cost under the same emission policy:")
    for name in ("legacy_v1_unit", "legacy_v2_unit"):
        for k in ("if_gated_at_0.15", "if_gated_at_0.3"):
            g = p2[name][k]
            print(f"    {name:16s} at rate {g['event_rate']:.2f}  NRMSE {g['nrmse']:.3f}")


if __name__ == "__main__":
    main()
