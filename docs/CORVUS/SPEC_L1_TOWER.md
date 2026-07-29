# SPEC L1 — Cortical Tower

**OCA v4 · Layer 1 · status: specified, blocked on [Q2](OPEN_QUESTIONS.md)**

The primary cognitive building block. Responsible for local world modelling through perception
and action. Maintains prediction, local memory, reference frame, confidence and sensorimotor
state, and learns continuously through interaction.

**This is the layer where all three predecessors died.** Wren, Swift and Heron each had a layer
here with abundant internal state — 2304, 3456 and 1309 dimensions — and all three were beaten
by storing two numbers. The specification below is written around that fact.

---

## Component contract

| field | specification |
|---|---|
| **Purpose** | Maintain a local model of the part of the world this tower can act on and observe. |
| **Responsibilities** | Predict its own sensory input and its own future state; hold a reference frame; maintain persistent state about referents that are currently unobservable; report confidence; act, and learn from the consequence. |
| **Internal State** | Prediction state · **reference frame** · local memory · confidence · sensorimotor state (last action, expected consequence, observed consequence). |
| **Inputs** | Event streams from its Layer-0 units, through their published interface only. Its own efference copy. |
| **Outputs** | A narrow published state (see below) and actions. Never its internal contents. |
| **Operations** | `observe` · `predict` · `act` · `compare` · `update` · `retire` |
| **Communication Model** | Publishes state, sparsely. Does not transmit representations. |
| **Lifecycle** | Long-lived. Its reference frame and its entity records outlive any single observation. |
| **Dependencies** | Layer 0 only. |
| **Temporal Horizon** | Declared per implementation, **≥ 16 ticks**, and strictly greater than its units'. |
| **Integration Window** | **Measured, and must exceed its own units'.** Not asserted. |
| **Floor** | Beats `trivial_memory` — store one value once, never update it — by ≥ 5% on gate `P0`. |
| **Failure Modes** | State present but not persistent (all three predecessors); reference frame that tracks the observer rather than the world; confidence uncorrelated with error. |
| **Scalability** | Towers are the unit of parallelism. Nothing at this layer may require a global view. |
| **Known Limitations** | Models only what it can observe and act on. Cross-tower identity is Layer 2's problem — see [Q2](OPEN_QUESTIONS.md). |

## Why the floor is `trivial_memory` and not `raw_input`

`raw_input` is the wrong floor for this layer, and using it would produce a false pass.

On an observable world, the current frame contains most of the answer, so beating it is
near-impossible and failing to beat it means little. The discriminating floor is the *cheapest
possible memory*: one stored value, never updated. In the tunnel world — the only place where
the raw-input control is at chance by construction — that floor is 2.26 cells, and it beat every
architecture built here:

| entrant | error while blind |
|---|---|
| **`trivial_memory`** | **2.26** |
| Wren | 4.44 |
| Swift | 5.10 |
| Mirror (no state) | 7.02 = chance |
| Heron | 8.56 |

A tower that clears 2.26 has done something no previous architecture in this project could do.

## Published state

The tower publishes scalars and phases, not contents. Heron's version of this — five scalars
plus a phase spectrum instead of a 77-dimensional internal state — was its boldest claim and the
only gate it passed, at a cost of 1.50×.

The v4 minimum: **prediction error · confidence · novelty · activity · energy**, plus whatever
the resolution of [Q2](OPEN_QUESTIONS.md) requires for entity references.

Read the supporting evidence with its caveat: the bottleneck was cheap relative to a full state
that was itself worse than raw input. "Cheap to compress" is not "worth transmitting".

## ⚑ Challenged against the record

**The reference frame is the most promising element in v4.** It is the first thing in any
version of this architecture that is intrinsically about persistent, addressable structure
rather than about transforming a signal. It is also the element with no measurement behind it —
none of the three predecessors had one.

**"Local memory" is where the specification is weakest, and it is not a wording problem.** All
three frozen architectures maintained internal state and satisfied every clause about it. State
was never absent. What was absent is *persistence*: nothing in any of them survived its
referent becoming invisible in a way that could be read back.

The evidence rules out the obvious diagnosis. Wren's readout is its **entire** mesh state, with
nothing pooled or summarised, and it fails the same way as the heavily-compressed ones. So this
is not compression destroying state on the way up. **Persistent state is never formed.**

This spec therefore requires more than "maintains local memory": *state must be identifiable as
being about the same referent after that referent has been unobservable.* That requirement is
testable, and no previous contract in this project contained it.

**The sensorimotor loop is supported.** Four of five worlds here are passive; the one with a
closed action loop is where the best architecture did best (84.1% out-of-view decode against a
77.3% floor) and the only place any architecture beat the raw frame. Action belongs at this
layer, and this is the strongest structural claim v4 makes.

**Continuous learning is required but currently unearned.** Heron's tower-layer plasticity
changed its headline by less than 0.5%, and its knowledge model — the component whose entire job
was consolidation — scored **14× worse than k-means on the raw input patch**. Whatever this
layer does to consolidate, "cluster the current state" is measured to be the wrong operation.
One diagnosis worth testing: it clustered a slow reservoir state in which a moving object is a
small perturbation. Consolidating *prediction error* — what surprised this tower — is a
different mechanism for the same responsibility.

**Blocked.** This layer must not be implemented until [Q2](OPEN_QUESTIONS.md) is resolved. Every
element above could be built exactly as specified, and without an answer to Q2 the result would
lose to `remember(r, c)` for the fourth time.
