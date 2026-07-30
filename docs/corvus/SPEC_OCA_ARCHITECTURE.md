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
with it. Sections marked **⚑ Decided** record an architectural decision taken against that record; the
reasoning is in [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) and the decisions in
[DECISIONS.md](DECISIONS.md).

**The design rule adopted before any of those decisions**, and the reason five of the six
*reduced* what this document commits to:

> Standardise invariants, not hypotheses. A concept may be standardised if its necessity is
> supported by evidence, or if its absence prevents interoperability. Mechanisms stay
> replaceable until evidence favours one.

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

## Architectural primitives

Two concepts belong to no layer and are manipulated by all of them, in the way `process`,
`address` and `message` belong to no layer of an operating system. They are specified before the
layers because every interface contract below depends on them.

### Entity — the unit of persistence

An addressable record of *a thing that continues to exist*.

```
id            immutable, unique, never reused
created_at    immutable
attributes    what is currently believed about it
uncertainty   how much that belief is trusted
last_seen     when its referent was last observable
observable    whether it is observable now
relations     references to other entities
```

**Its defining property is that it survives its referent becoming unobservable**, and that it
can afterwards be recognised as being about the same referent.

> **⚑ Decided — this is the only concept v4 adds, and the only one with a measurement behind
> it.** In the tunnel world, where the raw-input control is at chance by construction, storing
> two numbers beat every architecture ever built here. Wren's readout is its *entire*
> 2304-dimensional mesh state with nothing pooled, and it lost too — so persistence is not
> destroyed on the way up the stack, it is never formed.
>
> Memory answers *what happened*; identity answers *which thing happened*. Without identity,
> memory fragments. Object permanence appears in human development before language, before
> episodic memory and before planning, which argues for putting this low and making it shared.

**Ownership is layered even though the type is not.** A cross-cutting primitive that any layer
may create is the architectural equivalent of a global variable and would dissolve the layering
invariant below.

| operation | permitted to |
|---|---|
| **create** | only the layer that can observe the referent — Layer 1 |
| **retire** | only the creating layer |
| **read, reference, annotate, relate** | any layer |
| **write `id` or `created_at`** | nobody; immutable |

**The primitive is falsifiable, and its test already exists.** Any implementation claiming
entity persistence must pass gate `CGE-A-01`: beat frozen-at-entry, 2.26 cells, while blind.
Without that clause this would be the most attractive unfalsifiable idea in the document.

### Event — the unit of communication

Everything that crosses a component boundary is an event: **sparse, timestamped, and carrying an
optional entity reference.**

```
source        the component that emitted it
kind          what sort of claim it makes
value         the quantity
entity        which entity it is about, or None
at            when
```

Not a dense state vector, and not a bare scalar.

> **⚑ Decided — the draft specified seven layers and never said what flows between them.**
> ROS2 says messages, actor systems say messages, the internet says packets, Thousand Brains
> says sensorimotor object representations. Leaving it implicit makes every interface contract
> below unverifiable.
>
> Our evidence points at this answer from two directions. **Events beat dense state sharing:**
> send-on-delta emission beat the best matched-rate control by 92.9%, and the policy transferred
> to a frozen architecture's own trace. **A dense-vector carrier is measured to fail:** the one
> architecture that made a shared state vector its carrier destroyed the signal, 0.69× → 6.56×.
>
> The entity reference resolves the tension between them. A bare scalar is sparse but cannot say
> what it is about, which is exactly the Q2 failure; a latent vector can carry content but is the
> carrier that failed. An event referencing an entity is sparse **and** attributable, and it lets
> persistence cross a boundary without the layer above re-solving identity.

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
Provides **local coordination** before higher-level integration, and coordinates specialised
tower populations. **Summarising its members is permitted, not required.**

Full contract: [SPEC_L2_CLUSTER.md](SPEC_L2_CLUSTER.md).

> **⚑ Decided — coordination is the responsibility; compression is an optional algorithm.**
> The draft said "local consensus", and *consensus already assumes compression*. Aggregation has
> now failed four times with four operators: the legacy assembly turned a 0.69× state into a
> 6.56× workspace, and Heron's mean-pooling **control** beat both of its relational operators
> (0.598 against 0.612 and 0.629).
>
> So this layer's floor is `pass_through`: it must beat the concatenation of its own members
> before it is permitted to summarise them. That inverts the burden of proof onto the compression
> step, which is where four failures say it belongs.

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

> **⚑ Decided — one required service, four deferred.** Exactly one is required at the first
> maturity level: the **Persistent State Service**, which maintains state about a referent that
> is currently unobservable. Short-term, working, episodic, semantic and procedural memory are
> deferred to the Memory Specification and marked unevidenced.
>
> Five categories borrowed from psychology, in a project that does not yet have one working
> memory of any kind — a two-number memory beats everything built here. The classification can
> come after there is something to classify.

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
> Version 4 therefore strengthens the clause: state must be **persistent and addressable across
> unobservability**, not merely present. That requirement is what the `Entity` primitive exists
> to carry, and gate `CGE-A-01` is what checks it.

## Temporal Coordination Service

OCA defines temporal coordination as an **optional** architectural service. Its responsibility
is to decide *when* components communicate, attend, and learn: timing, attention, communication
windows, learning windows, offline consolidation, and multiple temporal scales.

**The responsibility is named; the mechanism is not.** Permitted implementations include
oscillatory phase, event scheduling, asynchronous message passing, token passing, priority
queues and temporal attention windows.

One clause is normative because it is measured: **whatever implements this service is a
synchronisation mechanism and not a memory container.** It coordinates when things happen; it
does not carry content.

An implementation that provides temporal coordination **and claims benefit from it** must show
it beats the same system with the service ablated, on some gate, by its declared margin.

> **⚑ Decided — the responsibility survives; oscillation specifically does not get to be
> mandatory.** The negative clause is well supported: a synchrony graph read from Swift's real
> rotor phases scored *exactly* 1.00× persistence, carrying no content.
>
> The positive claim has three measurements against it and none for it. Phase-gated emission
> cost a lone unit ~10%. Under channel contention — the setting where scheduling *should* pay,
> many senders and one wire — it gave +0.9% with the sign flipping across four budgets. Removing
> Heron's oscillation engine entirely changed its headline by less than 0.5%.
>
> None of that says temporal coordination is unnecessary. It says *that implementation* of it
> did not pay. Naming the responsibility and leaving the mechanism open is what the evidence
> supports, and it costs nothing to be right either way.

## Global cognitive state

The architecture defines a global cognitive state abstraction whose purpose is to coordinate
overall system behaviour. It permits one or more global operational modes and **intentionally
does not standardise the taxonomy of modes.**

Implementations may define any number of modes, provided **each mode demonstrably alters at
least one measurable system behaviour.** Names such as focused, exploratory, dreaming or
consolidating are illustrative, not normative.

> **⚑ Decided — keep the abstraction, drop the taxonomy.** Heron's State Adapter is the minimal
> version of exactly this interface — a global vector that changed *how* each tower learned
> without changing *what* it knew — and removing it altogether moved the headline by 0.4%.
>
> That does not refute the abstraction: a global mode plausibly needs consolidation and dreaming
> to switch between, and neither has been built. It does mean eight named modes are speculation.
> The pass condition above is much harder to argue with than a list.

## Memory

Memory is distributed. **One service is required** at the first maturity level:

**Persistent State Service** — maintains state about a referent that is currently unobservable,
keyed by `Entity`. Required. Its compliance test is gate `CGE-A-01`.

Short-term, working, episodic, semantic and procedural memory are **deferred** to the Memory
Specification and marked unevidenced. No specific storage mechanism is required for any of them.

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
