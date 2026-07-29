# SPEC L0 — Synthetic Neuron

**OCA v4 · Layer 0 · status: specified, not implemented**

The smallest computational unit. Responsible for local computation, local adaptation, sparse
communication, and maintaining minimal internal state.

This is the only layer in v4 that arrives with a clean positive result behind it. Almost all of
it is a restatement of what Heron's Layer 1 measured, with two corrections the measurements
forced.

---

## Component contract

| field | specification |
|---|---|
| **Purpose** | Transform a local signal into a sparse event stream, adapting locally. |
| **Responsibilities** | Integrate local input; hold minimal state; adapt its own parameters from local error only; emit events on significant change. |
| **Internal State** | A potential (what it currently computes), a last-published value (what its readers believe), an emission threshold, and optionally a phase. Minimal by contract: anything that looks like knowledge belongs at Layer 1. |
| **Inputs** | A local signal vector. No global error signal, no gradient from above. |
| **Outputs** | Events: `(value, timestamp)` on change. Between events a reader has the last value it was told and nothing else. |
| **Operations** | `integrate` · `adapt` · `emit` |
| **Communication Model** | Event-driven, send-on-delta. Silence is a message: it means "nothing has changed enough to be worth saying". |
| **Lifecycle** | Created with its tower; lives as long as its tower; no dynamic recruitment at this layer. |
| **Dependencies** | The sensory surface, or the event streams of other Layer-0 units. |
| **Temporal Horizon** | **1 tick.** Send-on-delta *is* a one-tick self-prediction, and the zero-order hold downstream is what consumes it. |
| **Integration Window** | Must be **shorter** than its tower's, and a population must span a range of windows rather than sharing one. |
| **Floor** | Beats `periodic_and_random_sampling` **at a matched event rate**, by ≥ 5%. |
| **Failure Modes** | Threshold collapse (emits every tick — it is a wire); threshold saturation (never emits — it is a rock); rate runaway under noise. |
| **Scalability** | Cost is proportional to events, not to units. This is the layer where sparsity buys real capacity. |
| **Known Limitations** | Holds nothing worth calling knowledge. Cannot represent an object, an identity, or anything that persists. |

## Why that floor, and not `raw_input`

Any subsampling of a smooth signal reconstructs reasonably. So "does send-on-delta work" is not
a question — the question is whether it beats *the same number of emissions spent another way*.
Without a matched-rate control, the mechanism cannot be distinguished from the rate.

Measured on Heron: **+92.9%** against the best matched-rate control, largest margin in the
sparse regime. This layer's floor is met.

## ⚑ Challenged against the record

**Confirmed — sparse event communication.** The strongest positive result in the project. It
also **transferred across architectures**: applying the policy to a frozen Wren unit's own trace
gave NRMSE 0.123 at 15% of that unit's publishing rate. A mechanism that works on someone
else's signal is a mechanism, not a fit.

**Correction 1 — phase gates *when*, never *what*.** The first implementation added the local
rhythm into the transmitted value. It cost **19×** in reconstruction while emitting *more*,
because the unit spent its event budget transmitting its own oscillation. If Layer 0 has a
phase, it may scale the emission threshold and must never touch the emitted value. This is
cheap to state and was expensive to find.

**Correction 2 — a population must span time constants.** A population sharing a single
integration window cannot support any hierarchy of abstraction above it. Costs nothing to
avoid; specified as a requirement rather than left to an implementer.

**Unresolved — does phase belong here at all?** Phase-gated emission cost a lone unit ~10%,
which was predicted before the run: a gate can delay an emission and can never improve one. The
claim was always that it pays off in coordination, and at the one layer where that could be
tested it gave +0.9% with an inconsistent sign. See [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) Q1.
Until Q1 is resolved, phase is **optional** at this layer.

**Not evidenced — local plasticity.** Heron's Layer 1 plasticity earned nothing (0.657 against
0.672). Consistent with the unit holding no knowledge, and worth revisiting only when Layer 1
gives the weights something to do. Specified as required-to-exist, not required-to-help.
