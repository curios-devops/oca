# OCA Architecture Specification

**Version 4 — Open Cognitive Architecture**
Reference implementation codename: **Corvus**

---

## Status of this document

This document defines the architecture. It does **not** define the benchmark suite.
Benchmarking, validation, Cognitive Gates, curricula, datasets, evaluation protocols and
experimental procedures belong to separate specifications and are referenced here only where a
contract depends on them.

Version 4 is written after three complete architectures — **Wren**, **Swift** and **Heron**
([register](../ARCHITECTURES.md)) — were built, benchmarked and frozen. All three satisfied
every architectural contract they were given. All three were beaten by their own sensory input
at predicting an observable world, and all three were beaten by a two-number memory at knowing
where they were while unable to see.

That is the central fact this version has to answer to. Sections marked **⚑ Evidence** record
where the architecture below is directly constrained by that record, and where it is in tension
with it. Sections marked **⚑ Open** are unresolved and are listed in
[OPEN_QUESTIONS.md](OPEN_QUESTIONS.md); they must be settled before the corresponding layer is
implemented.

## Purpose

OCA is an open cognitive architecture intended to support continual learning, adaptive
reasoning, long-term memory, sensorimotor intelligence and modular cognitive systems.

OCA is not a machine learning model. It is an architectural framework capable of hosting
multiple cognitive implementations.

The architecture intentionally separates **architecture**, **training**, **curriculum**,
**Cognitive Gates**, **benchmarks** and **runtime implementations**. Each evolves
independently under its own semantic version.

## Design philosophy

Modular · hierarchical · stateful · event-driven · sensorimotor · multi-modal · continual
learning · replaceable components · biology-inspired but engineering-first · implementation
agnostic.

No implementation strategy is assumed. Backpropagation, local learning, symbolic reasoning,
probabilistic inference, spiking computation, hybrid systems and future approaches are all
compatible if they satisfy the architectural contracts.

> **⚑ Evidence — implementation agnosticism is supported, and one data point is worth
> recording.** Local, backprop-free learning was measured competitive with a
> capacity-matched BPTT recurrent network on this project's worlds. That is evidence that the
> contract-first posture is not merely diplomatic: a non-gradient implementation was not
> handicapped. It is not evidence that gradients are unnecessary.

---

## Architectural layers

The architecture is organised into seven **architectural layers**. Layers describe
responsibility, not implementation.

### The layering invariant

Two rules govern every interaction in the system:

1. **Dependencies flow downward.** A layer may depend on the layer directly below it and on
   nothing above it. Lower layers never know higher-level semantics.
2. **Abstractions flow upward.** Each layer exposes a contract; higher layers never
   manipulate lower-layer implementation details directly.

### The floor invariant

Version 4 adds a third rule, and it is the one this version exists for.

3. **Every layer must declare what it has to beat.** A layer states, before it is
   implemented, the cheapest alternative that could discharge its responsibility — and the
   margin by which it must beat it. A layer that cannot beat its declared floor is not a
   layer, and the stack is not compliant.

> **⚑ Evidence — this is the requirement whose absence let three architectures fail
> unnoticed.** Rules 1 and 2 are standard layered-systems practice and both previous
> contracts enforced them. Rule 2 in particular contains a hidden assumption: that a layer
> produces a better abstraction than the one it received. Wren, Swift and Heron all
> respected rules 1 and 2 perfectly, and in all three the "abstraction" carried *less* about
> the world than the raw input did. No gate objected, because no contract asked.
>
> Three floors are already measured and named in the reference implementation
> (`corvus/contract.py`):
>
> | floor | what it is | what it beat |
> |---|---|---|
> | `raw_input` | the sensory frame itself | all three architectures, at every horizon |
> | `trivial_memory` | store one value once, never update | all three, at knowing where it is while blind (2.26 cells) |
> | `mean_pool` | a linear summary of the layer below | Heron's relational aggregation, on both targets |
>
> A floor chosen for being easy is worse than no floor, because it produces a pass. Each
> declaration therefore states *why* that floor is the right one.

### Layer 0 — Synthetic Neuron

Smallest computational unit. Responsible for local computation, local adaptation, sparse
communication and maintaining minimal internal state.

Full contract: [SPEC_L0_NEURON.md](SPEC_L0_NEURON.md).

> **⚑ Evidence — the only layer with a clean result behind it.** Sparse, event-driven
> communication is the strongest positive finding in the project: emitting on significant
> change beat the best control at a *matched event rate* by 92.9%, and the policy transferred
> to a frozen architecture's own trace. "Sparse communication" should be read as a hard
> requirement here, not an efficiency preference.
>
> One correction the record forces: **oscillation at this layer must gate *when* a unit
> communicates, never modulate *what* it communicates.** The first implementation added the
> local rhythm into the transmitted value and cost 19× in reconstruction while emitting more.

### Layer 1 — Cortical Tower

Primary cognitive building block. Responsible for local world modelling through perception and
action. Maintains prediction, local memory, reference frame, confidence and sensorimotor
state. Learns continuously through interaction.

Full contract: [SPEC_L1_TOWER.md](SPEC_L1_TOWER.md).

> **⚑ Evidence — "reference frame" is the most promising element in this version, and
> "local memory" is where all three predecessors died.** A reference frame is persistent,
> addressable state, which is precisely what the tunnel-maze result says is missing. Heron's
> equivalent layer had abundant internal state — 1309 dimensions — and lost to two numbers.
> Presence of state was never the problem. **Persistence is.** This layer's floor is therefore
> `trivial_memory`, not `raw_input`.

### Layer 2 — Tower Cluster

Collection of cooperative towers. Responsible for solving coherent local cognitive problems.
Provides local consensus before higher-level integration. Coordinates specialised tower
populations.

Full contract: [SPEC_L2_CLUSTER.md](SPEC_L2_CLUSTER.md).

> **⚑ Open — "consensus before integration" is the single most-failed operation in this
> project's history, and as written it is a mandatory responsibility.** The legacy
> Predictive Assembly took a mesh state that scored 0.69× persistence and produced a
> workspace that scored 6.56× — the aggregation destroyed the signal. Heron's node layer then
> lost to mean pooling at matched width, and every aggregation it tried lost to the raw frame.
>
> Consensus is a compression step, and compression is what has failed. See
> [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) Q3: does a cluster that provably adds nothing have
> to be permitted to be a pass-through?

### Layer 3 — Functional Region

Coordinates multiple clusters. Represents functional capabilities rather than anatomical
structures — for example vision, language, motor control, spatial reasoning, executive
processing, planning, social cognition.

The architecture does not require biological anatomical fidelity. Regional specialisation is
expected to be **observed rather than assigned**: a region is a region because the clusters
inside it demonstrably reduce each other's prediction error.

### Layer 4 — Cognitive Systems

Cross-regional cognitive services: working memory, semantic memory, episodic memory,
attention, planning, motivation, sleep, memory consolidation. These coordinate information
across multiple regions.

> **⚑ Open — five memory types are named and the project does not yet have one.** A
> two-number memory currently beats every architecture built here. Specifying five varieties
> of memory before one works is the mistake the build order exists to prevent. See
> [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) Q4: which single memory service is required for the
> first maturity level, and which are deferred?

### Layer 5 — Synthetic Cortex

Global cognitive coordinator: global planning, cross-region coordination, conflict resolution,
attention routing, global synchronisation, high-level reasoning.

**The cortex never manages individual neurons or towers directly.** This follows from the
layering invariant and is restated because it is the rule most often broken by convenience.

### Layer 6 — Synthetic Brain

Complete cognitive agent: goals, identity, long-term objectives, external memory interfaces,
tools, robotic interfaces, environment interaction, execution policies. The deployable system.

---

## External systems

Intentionally outside the architecture: training, curriculum, datasets, simulation, teacher
agents, human interaction, robotics environments, evaluation, benchmarking. These interact
with OCA but are not architectural layers.

## Component contract

Every architectural component must define:

| field | notes |
|---|---|
| Purpose | |
| Responsibilities | |
| Internal State | |
| Inputs | |
| Outputs | |
| Operations | |
| Communication Model | |
| Lifecycle | |
| Dependencies | |
| Failure Modes | |
| Scalability Considerations | |
| Known Limitations | |
| **Temporal Horizon** | **added in v4** — how far ahead this component commits to being scored |
| **Integration Window** | **added in v4** — the timescale over which its state actually decorrelates, measured |
| **Floor** | **added in v4** — what it must beat, by what margin, and why that is the right floor |

Components are specified through functional contracts rather than implementation details.

> **⚑ Evidence — the three added fields each correspond to a specific failure.**
>
> **Temporal Horizon.** No function of a legacy mesh state beat "assume no change" one tick
> ahead; at 64 ticks a 44% gain was available. A slow component scored on a fast prediction is
> being set an impossible test, and it took a full experimental cycle to notice.
>
> **Integration Window.** Heron's tower-equivalent layer decorrelated *faster* than the units
> feeding it — 0.88 against 0.91 at its own declared horizon. It was therefore a relabelling
> rather than a layer, and **not one gate in the battery detected it.** Declaring a horizon is
> not enough; the window has to be measured and has to be monotonic up the stack. This is
> enforced in `corvus/contract.py::check_stack`.
>
> **Floor.** See the floor invariant above.

## Communication model

Communication is hierarchical and sparse, supported at every scale: neuron, tower, cluster,
region, cognitive system, global cortex, global brain.

Communication favours **structured events over continuous dense state sharing**.
Synchronisation mechanisms are architecture-level services.

> **⚑ Evidence — supported, twice.** Event-driven emission beat matched-rate controls by
> 92.9% at Layer 0. And a deliberately narrow interface survived its test: Heron's towers
> published five scalars and a phase spectrum instead of their internal state, and a reader
> given only that came within 1.50× of a reader given everything. That was the boldest claim
> in the previous architecture and the only gate it passed. Read the result with its caveat:
> the bottleneck was cheap relative to a full state that was itself worse than raw input.

## Internal state

Every architectural component maintains internal state. **State is a first-class
architectural concept.** Stateless cognitive components are discouraged except as explicitly
defined utility modules.

> **⚑ Evidence — necessary and demonstrably not sufficient.** All three frozen architectures
> satisfy this clause completely, with 2304, 3456 and 1309-dimensional states respectively,
> and all three lose to storing two numbers once. A contract satisfied by every failure
> discriminates nothing.
>
> Version 4 therefore strengthens the clause: state must be **persistent and addressable
> across unobservability**, not merely present. A component's state must survive the target
> becoming invisible, and must be identifiable as being about the same thing afterwards. This
> is what the frozen line never had, and it is the subject of
> [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) Q2.

## Oscillation layer

OCA defines oscillatory coordination as an architectural service. **Oscillations are
synchronisation mechanisms. They are not memory containers.** They coordinate timing,
attention, synchronisation, communication windows, learning windows and offline
consolidation. Multiple temporal scales should be supported. The exact implementation is
intentionally unspecified.

> **⚑ Open — "oscillations are not memory containers" is confirmed; that they coordinate
> anything useful is not.** The negative half of the claim is well supported: a synchrony
> graph read from real rotor phases scored *exactly* 1.00× persistence, carrying no content
> whatsoever. Phase is a clock.
>
> The positive half has three measurements against it and none for it. Phase-gated emission
> cost a lone unit ~10%. Under channel contention — the setting where scheduling *should*
> pay — it gave +0.9% with the sign flipping across budgets. Removing the oscillation engine
> from Heron's tower layer changed its headline by less than 0.5%.
>
> Making oscillation a named architecture-level service spends the specification's complexity
> budget on the one mechanism with a perfect record of no payoff. See
> [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) Q1.

## Global cognitive state

The architecture defines a global cognitive state abstraction whose purpose is to coordinate
overall system behaviour. Possible modes include focused, exploratory, learning, planning,
dreaming, consolidating, recovery and idle. The architecture defines responsibilities rather
than implementation; future implementations may represent this state differently.

> **⚑ Evidence — eight modes are specified and the minimal version of this interface has
> been built and measured at zero effect.** Heron's State Adapter received a global state
> vector and modulated how each tower learned without touching what it knew — exactly this
> abstraction, minimally. Removing it changed the headline by 0.4%.
>
> That does not refute the abstraction; a global mode may only matter once there is
> consolidation and dreaming to switch between, and neither exists. It does mean the eight
> modes are unevidenced, and the specification should say so rather than presenting them as
> settled.

## Memory

Memory is distributed. The architecture distinguishes short-term, working, episodic, semantic
and procedural memory. No specific storage mechanism is required.

See the **⚑ Open** note under Layer 4.

## Sensorimotor loop

Perception and action are symmetric architectural concepts. Every cognitive layer may receive
observations, generate actions, receive feedback and adapt. The architecture is intentionally
grounded in interaction rather than passive observation.

> **⚑ Evidence — the strongest structural result in the project's favour.** Four of the five
> worlds built here are passive; the one with a closed action loop is the one where the best
> architecture performed best (84.1% out-of-view wall decode against a 77.3% floor) and the
> only world where any architecture beat the raw frame at all. Grounding in interaction is
> supported. It is also under-tested: a fully active benchmark does not yet exist.

## Extensibility

Every layer may have multiple implementations — different neuron models, tower models, memory
systems, synchronisation systems, planning systems. All implementations must satisfy the
architectural contract, **including the floor declaration**.

## Architecture validation

This specification intentionally does not define evaluation procedures. Validation is
performed through the independent **OCA Cognitive Gates Specification (CBS)**, which defines
functional validation, behavioural validation, robustness, scalability, efficiency, regression
testing and architectural comparison.

No architecture is OCA-compliant until it satisfies the required Cognitive Gates for its
maturity level.

> **⚑ Evidence — one gate must be mandatory at every maturity level.** The gates that
> discriminate are the ones set on worlds where the answer is provably absent from the current
> observation. Most of this project's gates were not: they ran on worlds observable enough
> that a representation could at best re-encode its input, which is most of why "nothing beats
> raw pixels" went unexplained for so long.
>
> The exception is the tunnel world, where every frame inside a corridor is byte-identical and
> the raw-input control sits at chance by construction — measured at 7.02 against a chance of
> 7.02. That gate (`P0`) is the project's standing open challenge and should be a compliance
> requirement rather than an optional extra.

## Non-goals

OCA does not attempt to replicate biological anatomy, model molecular neuroscience, optimise
for benchmark scores, depend on a specific learning algorithm, require transformer
architectures, require language as the primary cognitive modality, specify implementation
details, or constrain future cognitive architectures.

## Future specifications

Complementary and independently versioned: Cognitive Gates Specification · Training
Specification · Curriculum Specification · Communication Protocol Specification · Memory
Specification · Oscillation Specification · Reference Implementations · Engineering APIs.

## Closing statement

OCA defines a cognitive architecture, not a single model. Its objective is to establish stable
architectural contracts that let multiple cognitive implementations evolve, interoperate and
be evaluated through a common engineering framework.

Version 4 adds one commitment to that objective. **A contract that no implementation can fail
is not a contract.** Three architectures satisfied their contracts and were beaten by their
own input; the floor invariant exists so that the fourth cannot.
