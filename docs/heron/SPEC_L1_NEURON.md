# SPEC L1 — Dynamic Neuron (DN)

**Status: implemented, passes its gates.** Code in [`architectures/heron/neuron.py`](../../architectures/heron/neuron.py),
gates in [`cge/components.py`](../../cge/components.py), results in
[RESULTS_L1_NEURON.md](RESULTS_L1_NEURON.md).

A specification, not a description of the code: it states what this level must do, what it
must never do, what it publishes, and what measurement would show it failing. If the code
and this document disagree, one of them is a bug.

---

## Purpose

The universal primitive. Every Dynamic Neuron is identical — nothing about vision, nothing
about language, nothing about logic. It integrates what reaches it, keeps a small state,
learns locally, oscillates on its own, and **speaks only when it has something to say**.

That last property is what makes it a different object from a summer, and it is the one the
gates are built around.

## Responsibilities

| # | responsibility | mechanism |
|---|---|---|
| 1 | integrate local signals | leaky integration, per-neuron time constant |
| 2 | maintain a small internal state | membrane potential `v`, rotor `z` |
| 3 | learn weights | local delta rule on its own emission error |
| 4 | oscillate | Stuart–Landau limit cycle |
| 5 | emit only on significant change | send-on-delta against an adaptive threshold |

**Time constants are a spread, not a value** (log-uniform over `tau`). A population sharing
one time constant cannot form any hierarchy of abstraction, and a spread costs nothing.

## Interface

```
inputs   x : (n_inputs,)          the local signal, or (n_groups, n_inputs) when grouped
state    v, z, w, theta, last_sent, energy
publish  last_sent : (N,)         the value each neuron last transmitted
horizon  1 tick                   declared, mandatory, enforced by architectures/heron/contract.py
```

A neuron publishes **one number**, and only when that number has moved past its threshold.
A reader between events has the last value it was told and nothing else — zero-order hold
is the honest reconstruction, and it is what every gate scores against.

## Hard constraints

Each one is a constraint because breaking it was measured, not because it sounds right.

**C1 — The dynamics must be able to express what it is asked to do.** A gradient flow has
`dE/dt = -‖∇E‖² ≤ 0` and therefore no periodic orbits. Legacy v1 could not oscillate and
that was discovered three times by experiment. The rotor here follows the Stuart–Landau
normal form, whose amplitude settles while its phase keeps advancing, so "the neuron
oscillates on its own" is a property of the mathematics.

**C2 — Phase gates *when* a neuron speaks, never *what* it says.** Axiom 3. The first
wiring added the rotor into the activation and cost 19x in reconstruction while emitting
more, because the event budget went on transmitting the neuron's own rhythm. The rotor now
scales the emission threshold and never touches the transmitted value.
`osc_into_activation` keeps the rejected wiring runnable so the finding stays reproducible;
`tests/test_heron_neuron.py::test_the_clock_never_enters_the_content` pins it.

**C3 — A neuron holds no knowledge worth naming.** Axiom 2. It does not predict the world;
it encodes its own activation and transmits it sparsely. Knowledge is the node's job.

**C4 — Every emission claim is measured against a control at the identical event rate.**
Any subsampling of a smooth signal reconstructs reasonably. The claim is not "send-on-delta
works" but "send-on-delta beats periodic and random sampling at the same budget", and
without that control the mechanism cannot be told apart from the rate.

## Gates

| gate | question | pass condition | result |
|---|---|---|---|
| rate–distortion | does emitting on change beat the same budget spent otherwise? | >5% better than the best matched-rate control | **PASS, +92.9%** |
| oscillation | does the rotor sustain itself with no input? | phase advancing, amplitude stable | **PASS** |
| clock ≠ content | is the activation still with no input? | activation activity < 1e-6 | **PASS** |
| noise robustness | does precision degrade gracefully? | no cliff | PASS |
| ablations | does each mechanism earn its place? | removing it costs something | mixed — see below |

## What is known to be true, and what is known not to be

**Established.** Send-on-delta is 92.9% better than the best matched-rate control, with the
largest margin in the sparse regime. The rotor sustains a rhythm with zero drift. The
emission policy **transfers across architectures**: applied to a frozen legacy v1 unit's own
trace it gives NRMSE 0.123 at 15% of that unit's rate. That is the most portable thing this
level has produced.

**Known not to be true.** Phase gating does not help a lone neuron — it costs ~10%, and that
was predicted in the code before it was run, because a gate can delay an emission and can
never improve one. The claim was always that phase pays off in coordination.
[Level 2 tested that and it did not](RESULTS_L2_NODE.md#l2-5-phase-under-channel-contention--fail):
+0.9% with an inconsistent sign. **On the evidence so far, axiom 3 has no operational
payoff at any level.**

**Unmeasured.** Plasticity earns nothing here (0.657 against 0.672) — expected under axiom
2, and only worth revisiting if a level above gives the weights something to do.

## How to falsify this level

Show that a fixed-schedule emitter matches send-on-delta at the same rate on a natural
signal. That would mean the mechanism is doing nothing the rate does not already explain,
and the level reduces to a subsampler.
