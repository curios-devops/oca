# EIS — Emergent Intelligence Specification

*Status: **charter only.** No level below is implemented. This document defines what would have
to be true for one to be, and deliberately stops there.*

---

## The separation

Two questions that are usually mixed, and OCA keeps apart permanently:

| | question | answered by |
|---|---|---|
| **Cognitive correctness** | Does each component correctly fulfil its functional contract? | [CGE](corvus/SPEC_CGE.md) — the Cognitive Gates |
| **Cognitive emergence** | Does the *interaction* between components produce capabilities no component has? | **EIS** — this document |

The CGE does not prove intelligence, and it was never meant to. It proves that an architecture
possesses certain fundamental cognitive capabilities — which is a different claim and a
weaker one.

An aircraft does not demonstrate it can cross the Atlantic by starting its engines. But if it
cannot take off, turn, hold stability and land, it will never cross it. **The gates are the
minimum necessary properties, not the destination.**

The discipline this buys: build components with explicit contracts → validate them with gates
that have controls able to kill them → integrate progressively → and only then look for
emergent capability, with an independent specification. Most architectures attempt the last
step first.

---

## Why this document is a charter and not yet a specification

Because the claims an EIS makes are the **easiest in the entire project to fake with a bad
measurement**, and this project's record on measurement is poor: three of its own gates turned
out to be artefacts — one gate whose baseline was handed the answer, one confounded by
persistence decay, one that starved half its population through a drop rule. Those were gates
about *position error* and *reconstruction*, quantities with units.

"Forms concepts without supervision" and "restructures its own knowledge" have no units. Written
today, against nothing, an EIS is aspirational prose that will later be quoted as a target and
then as a result. So it gets two hard requirements before any level of it may be implemented.

### Requirement 1 — emergence needs an operational definition with teeth

*"A behaviour no individual component implemented explicitly"* is too weak. No element of a
list implements the mean. Aggregation is not emergence, and this project has already mistaken
one for the other: relational aggregation was the strongest result here until mean pooling beat
it at matched width.

> **A capability is emergent when the integrated system beats (a) its best single component
> *and* (b) the best non-interacting ensemble of the same components at matched capacity.**

Control (b) is the whole definition. Without it, "emergence" means "we added parameters".

### Requirement 2 — every level ships with its anti-test

An **anti-test** is the cheapest system that passes the level *without* having the capability.
Naming it is part of specifying the level, not a later refinement.

> A level with no anti-test is recorded as `UNMEASURABLE`, never as `PROPOSED`. `PROPOSED` means
> "we know how we would measure this"; the levels below mostly do not qualify yet, and say so.

---

## The levels

Seven, escalating. The final column is the only one that decides whether a level is real.

| # | capability | anti-test — what passes it *without* the capability | status |
|---|---|---|---|
| 1 | **Learns something new without forgetting** | A model with enough spare capacity never forgets; interleaved replay passes without any architectural property. Anti-test must force capacity pressure and forbid replay. | `PROPOSED` — measurable today, and the closest to reachable |
| 2 | **Generalises spontaneously** | A linear probe on raw pixels generalises. Every "generalisation" result here must clear the input it was computed from — which is `CGE-A-00`, still unpassed by anything. | `PROPOSED` |
| 3 | **Forms concepts without supervision** | k-means on the raw input patch. It beat Heron's Layer 2 concept formation by **14×**. Any concept claim must beat clustering on pixels first. | `PROPOSED` — anti-test already exists and has already killed one claim |
| 4 | **Transfers knowledge between domains** | Shared low-level statistics between the two domains. Anti-test: a frozen random projection of domain A that transfers equally well. | `UNMEASURABLE` — no second domain exists; the suite is vision plus a coarse touch map plus an efference copy |
| 5 | **Plans over a long horizon** | A greedy policy plus a good world model looks like planning. Anti-test: greedy-with-lookahead-1 at matched compute. | `UNMEASURABLE` — no task here rewards planning |
| 6 | **Discovers strategies nobody programmed** | The environment's own structure. Anti-test: exhaustive enumeration of what the action space makes trivially available. | `UNMEASURABLE` — and the level most vulnerable to being narrated rather than measured |
| 7 | **Restructures its own knowledge to improve future decisions** | Ordinary continued learning. Anti-test: the same system with representation frozen and only the readout adapting. | `UNMEASURABLE` |

**Four of seven are `UNMEASURABLE` for want of a world, not for want of an architecture.** That
is the honest state, and it is a finding about the *benchmark*: the current battery is a single
modality plus proprioception. Levels 4–7 cannot be attempted without a second modality and a
task with a horizon. The CGE catalogue already records the same limitation from the other
direction — `cge.catalogue.audit()` reports no audio gate and no implemented cross-modal gate.

---

## The gate between CGE and EIS

**No EIS level may be attempted until `CGE-A-00` is passed.**

`CGE-A-00` — *beat your own input at predicting an observable world* — is failed by all four
architectures built here. Until an architecture's representation is worth more than the raw
frame it was computed from, "emergent capability" measured on that representation is measuring
the frame.

That is the standing open problem of this project, and it is also the entry condition for this
document.

---

## Relationship to the rest of the specification

- [`SPEC_CGE.md`](corvus/SPEC_CGE.md) — the gates, their classes, verdicts and controls.
- [`EVOLUTION_RULES.md`](EVOLUTION_RULES.md) — R3 applies here in full: an EIS level's control
  must be declared before the mechanism it will judge is built.
- [`architecture-history/`](../architecture-history/) — where levels that fail are recorded.

The order is not negotiable: correctness, then integration, then emergence. If continual
learning, transfer, hierarchical planning or spontaneous concept formation ever appear here, it
will not be because a language benchmark returned a high score. It will be because the system
began exhibiting properties that its own controls could not reproduce.
