# OCA v4 — architectural decisions

Six questions, six decisions, dated. The analysis behind each is in
[OPEN_QUESTIONS.md](OPEN_QUESTIONS.md), kept unedited.

**The rule adopted before answering any of them**, and the reason five of the six went the way
they did:

> **Standardise invariants, not hypotheses.** The architecture may standardise a concept whose
> necessity is supported by evidence, or whose absence prevents interoperability. Mechanisms
> stay replaceable until evidence favours one.

That is what keeps OCA an engineering specification rather than a neuroscience hypothesis in
engineering clothing. Applied honestly it means exactly one of these questions produced a new
core concept, and the other five produced *less* commitment than the draft had.

---

## Q1 — Oscillation → **Temporal Coordination Service** · optional

**Decided:** the architecture defines a *Temporal Coordination Service*. Oscillation is one
permitted implementation among several — event scheduling, asynchronous message passing, token
passing, priority queues, temporal attention windows. Providing it is optional; providing it
*and claiming benefit* requires beating the same system with it ablated.

Better than the draft's "optional oscillation", because it names the responsibility rather than
the mechanism. Three measurements found no benefit from oscillation specifically (−10% for a
lone unit, +0.9% with the sign flipping under contention, <0.5% on ablation). None of them says
temporal coordination is unnecessary — they say *that* implementation of it did not pay. The
generalisation is what the evidence supports.

The clause that survives intact: **oscillations are synchronisation mechanisms, not memory
containers.** A synchrony graph read from Swift's real rotor phases scored exactly 1.00×
persistence. That half is measured.

## Q2 — Entity becomes a **cross-cutting architectural primitive**

**Decided:** `Entity` is an architectural primitive, not a layer and not a half-layer. Like
`process`, `thread`, `address` and `message` in an operating system, it belongs to no single
layer and every layer can manipulate it.

This is the only question where the evidence justified adding a core concept, and it is
overwhelming: in the one world where the raw-input control is at chance by construction, storing
two numbers beats every architecture ever built here. Wren's readout is its *entire* 2304-dim
mesh state with nothing pooled, and it still loses — so persistence is not being destroyed on
the way up the stack. It is never formed.

The distinction that makes it a primitive rather than a memory feature: **memory answers *what
happened*; identity answers *which thing happened*.** Without identity, memory fragments. In
developmental psychology object permanence appears before language, before episodic memory,
before planning — which is an argument for putting it low and making it shared, not for making
it a service somewhere near the top.

**Two refinements added on review, both closing holes in the primitive as proposed.**

**(i) Ownership is layered even though the type is not.** A cross-cutting primitive that any
layer may create is the architectural equivalent of a global variable, and it would quietly
dissolve the layering invariant — Layer 5 could mint something addressing what Layer 1 owns,
which is exactly what the invariant forbids. So:

| operation | who may |
|---|---|
| **create** an entity | only the layer that can observe its referent (Layer 1) |
| **retire** an entity | only its creating layer |
| **read, reference, annotate, relate** | any layer |
| **write core identity** (`id`, `created_at`) | nobody, ever — the fields are immutable |

Cross-cutting as a *type*, layered as an *authority*. Enforced in code, not by review.

**(ii) The primitive must be falsifiable, and its test already exists.** "We added an Entity
abstraction" is not a claim that can be checked. So the primitive's compliance test is literally
gate `CGE-A-01`: an implementation claiming entity persistence must beat frozen-at-entry — 2.26
cells — while blind. Without that clause this decision would be the most attractive
unfalsifiable idea in the document.

## Q3 — Clusters coordinate; compression is optional

**Decided:** the cluster's responsibility is renamed from *local consensus* to **local
coordination**, and its compliance floor is `pass_through`.

The rename matters more than it looks: *consensus already assumes compression*, and coordination
does not. Aggregation has now failed four times with four operators — the legacy assembly turned
a 0.69× state into a 6.56× workspace, and Heron's mean-pooling control beat both its relational
operators. A cluster must beat the concatenation of its own members before it is permitted to
summarise them. Responsibilities are mandatory; algorithms are not.

## Q4 — One required memory: the **Persistent State Service**

**Decided:** exactly one memory service is required at the first maturity level — persistent
state about a referent that is currently unobservable. Short-term, working, episodic, semantic
and procedural are deferred to the Memory Specification and marked unevidenced.

Five categories borrowed from psychology, in a project that does not yet have one working
memory of any kind. The classification can come after something exists to classify.

## Q5 — Global modes: abstraction normative, taxonomy illustrative

**Decided:** the architecture permits one or more global operational modes and **does not
standardise the taxonomy**. Any implementation providing modes must show at least one mode
demonstrably alters at least one measurable behaviour.

Heron's State Adapter is the minimal version of this interface and removing it moved the
headline by 0.4%. That does not refute the abstraction — a global mode plausibly needs
consolidation and dreaming to switch between, and neither exists — but eight named modes are
speculation, and the pass condition above is much harder to argue with than a list.

## Q6 — The information carrier is an **Event**, and its referent is an **Entity**

**Added on review and decided.** The draft specified seven layers and never said what flows
between them. ROS2 says messages, actor systems say messages, the internet says packets,
Thousand Brains says sensorimotor object representations. OCA said nothing, and every interface
contract depends on it.

**Decided:** the unit of communication is an **Event** — sparse, timestamped, and carrying an
optional entity reference. Not a dense state vector, and not a bare scalar.

This is not a free choice; it is the one place where our evidence points at a specific answer,
from two directions:

- **Events beat dense state sharing.** Send-on-delta emission beat the best matched-rate control
  by 92.9%, and the policy transferred to a frozen architecture's own trace. The strongest
  positive result in the project is specifically about event-driven communication.
- **A dense vector carrier is measured to fail.** The one architecture that made a shared state
  vector its carrier destroyed the signal: 0.69× → 6.56×.

And the entity reference is what resolves the tension between the two. A bare scalar is sparse
but cannot say *what it is about*, which is precisely the persistence failure in Q2; a latent
vector can carry content but is the carrier that failed. An event that references an entity is
sparse **and** attributable, and it lets persistence cross a layer boundary without the layer
above re-solving identity.

---

## What changed, in one table

| | draft v4 | decided |
|---|---|---|
| Oscillation Layer | required service | **Temporal Coordination Service**, optional, oscillation one implementation |
| entity persistence | absent | **`Entity`: cross-cutting primitive**, layered ownership, gated by CGE-A-01 |
| cluster responsibility | local consensus (mandatory) | **local coordination**; compression optional, floor `pass_through` |
| memory | five named systems | **one required**: Persistent State Service |
| global modes | eight named | abstraction normative, taxonomy illustrative, each mode must move a measurement |
| information carrier | unspecified | **`Event`**, sparse, referencing an `Entity` |

Five of six decisions *reduce* what the architecture commits to. One adds a concept, and it is
the one with a measurement behind it. That asymmetry is the design rule working.
