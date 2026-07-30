"""Corvus Layer 0 — `CGE-A-02`, run against Corvus's own neuron rather than inherited.

Required by rule R1: *a result measured on a predecessor does not transfer.* Corvus's Layer 0
is Heron's Layer 0 **minus the phase rotor**, and Heron's +92.9% was measured with the rotor
present. Removing a component and keeping the number is exactly the substitution this project
has already made three times.

The gate itself does not change. `Floor(beats="periodic_and_random_sampling", margin=0.05)` is
declared in `architectures/corvus/neuron.py` and was written before this file existed. What is
measured is whether **send-on-delta still beats the same number of emissions spent on a
schedule or at random, on the activation traces Corvus's own dynamics produce.**

Two things are reported that the inherited number could not tell us:

1. **Does Corvus L0 pass on its own trace**, three seeds, margin 0.05.
2. **Did removing the rotor change the answer** — the same gate on Heron's traces, same
   streams, same seeds. If Corvus is materially worse, the rotor was doing something after all
   and the Q1 decision needs revisiting. If it is not, the +92.9% survives its own ablation and
   the rotor's removal cost nothing, which is what the decision assumed.

    python experiments/corvus_l0.py --ticks 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from architectures.corvus import neuron as corvus_l0
from architectures.corvus.contract import LAYERS
from cge.components import gate_rate_distortion, nrmse, summarise_rate_distortion
from core.data import rollout
from core.world.physics import make_physics_world


def input_stream(ticks: int, seed: int, n_inputs: int) -> np.ndarray:
    """The same real sensory stream Heron's L1 was measured on: one patch of the physics
    world. A component tested on synthetic noise says nothing about the system it is part of,
    and using a *different* stream here would confound the comparison this file exists for."""
    _, sen, _ = rollout(ticks + 50, seed=seed, world_factory=make_physics_world)
    return sen[:, 0, :n_inputs].astype(np.float64)


def run_corvus(ticks: int, seed: int, n_neurons: int) -> tuple[np.ndarray, dict]:
    cfg = corvus_l0.NeuronConfig(seed=0, n_neurons=n_neurons)
    pop = corvus_l0.build(cfg)
    x = input_stream(ticks, seed, cfg.n_inputs)
    dense, sent, fired = [], [], []
    for t in range(ticks):
        corvus_l0.step(pop, x[t])
        dense.append(pop.a.copy())
        sent.append(pop.last_sent.copy())
        fired.append(pop._fired.copy())
    dense, sent, fired = np.array(dense), np.array(sent), np.array(fired)
    return dense, {"event_rate": float(fired.mean()),
                   "nrmse_as_run": nrmse(sent, dense),
                   "energy_per_tick": float(pop.energy.mean() / ticks),
                   "n_params": pop.n_params()}


def run_heron(ticks: int, seed: int, n_neurons: int) -> tuple[np.ndarray, dict]:
    """The predecessor, with its rotor, on the identical stream."""
    from architectures.heron.neuron import NeuronConfig, build_population, step
    cfg = NeuronConfig(seed=0, n_neurons=n_neurons)
    pop = build_population(cfg)
    x = input_stream(ticks, seed, cfg.n_inputs)
    dense, sent, fired = [], [], []
    for t in range(ticks):
        step(pop, x[t])
        dense.append(pop.a.copy())
        sent.append(pop.last_sent.copy())
        fired.append(pop._last_event.copy())
    dense, sent, fired = np.array(dense), np.array(sent), np.array(fired)
    return dense, {"event_rate": float(fired.mean()),
                   "nrmse_as_run": nrmse(sent, dense),
                   "energy_per_tick": float(pop.energy.mean() / ticks)}


def measure(runner, ticks: int, seeds: list[int], n_neurons: int) -> dict:
    per_seed = []
    for s in seeds:
        dense, live = runner(ticks, s, n_neurons)
        rd = summarise_rate_distortion(gate_rate_distortion(dense, seed=s))
        per_seed.append({"seed": s, "delta_vs_best_control": rd["delta_vs_best_control"],
                         "auc_send_on_delta": rd["send_on_delta"],
                         "auc_periodic": rd["periodic"], "auc_random": rd["random"],
                         **live})
    d = np.array([r["delta_vs_best_control"] for r in per_seed])
    return {"per_seed": per_seed, "mean": float(d.mean()), "std": float(d.std(ddof=1)),
            "min": float(d.min())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=8000)
    ap.add_argument("--neurons", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    floor = LAYERS["neuron"].floor
    print(f"CGE-A-02 -- Corvus Layer 0, {args.ticks} ticks, seeds {args.seeds}")
    print(f"declared floor: beat {floor.beats} by {floor.margin:+.2f}\n")

    corvus = measure(run_corvus, args.ticks, args.seeds, args.neurons)
    heron = measure(run_heron, args.ticks, args.seeds, args.neurons)

    print("  vs the best matched-rate control (higher is better, 0 = no better than a schedule)")
    for name, r in (("Corvus L0 (no rotor)", corvus), ("Heron L0 (with rotor)", heron)):
        per = "  ".join(f"{p['delta_vs_best_control']:+.3f}" for p in r["per_seed"])
        print(f"    {name:22} {r['mean']:+.3f} +/- {r['std']:.3f}   [{per}]")

    passed = corvus["min"] > floor.margin
    verdict = "PASS" if passed else "FAIL"
    print(f"\n  worst seed {corvus['min']:+.3f} vs margin {floor.margin:+.3f}  =>  {verdict}")

    gap = corvus["mean"] - heron["mean"]
    print(f"\n  removing the rotor changed the result by {gap:+.3f} "
          f"({'no material cost' if abs(gap) < 0.05 else 'MATERIAL -- revisit Q1'})")

    rates = [p["event_rate"] for p in corvus["per_seed"]]
    print(f"  Corvus as-run event rate {np.mean(rates):.3f}, "
          f"{corvus['per_seed'][0]['n_params']} parameters")

    out = {"gate": "CGE-A-02", "layer": "neuron", "ticks": args.ticks,
           "floor": {"beats": floor.beats, "margin": floor.margin},
           "verdict": verdict, "corvus": corvus, "heron": heron,
           "rotor_removal_cost": gap}
    p = Path(__file__).resolve().parents[1] / "logs" / "corvus_l0.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p.relative_to(p.parents[1])}")


if __name__ == "__main__":
    main()
