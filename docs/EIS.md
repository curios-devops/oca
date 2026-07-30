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

Twelve capacities, ordered by cognitive complexity, each drawn from a protocol validated on humans
and animals. The ordering is close to a Piaget-plus-comparative-psychology ladder, and that is the
point: these are facultades, not tasks, and each one must be measurable by **more than one
challenge** or the challenge is what got measured.

The final column is the only one that decides whether a level is real. An **anti-test** is the
cheapest system that passes the level *without* the capability.

| # | capability | protocol | anti-test — what passes it *without* the capability | status |
|---|---|---|---|---|
| 0 | **Prediction** | forecast `x(t+τ)` | copy-last, and the raw frame at the same horizon. Persistence degrades fastest, so everything looks better at longer τ | `IMPLEMENTED` as `CGE-B-05` at component level |
| 1 | **Working memory** | delayed match-to-sample | a low-pass filter with a long enough time constant. The delay must exceed the system's measured state autocorrelation | `BUILDABLE NOW` — closest to reachable |
| 2 | **Object permanence** | Piaget hiding, then visible → invisible displacement → container swap | frozen-at-entry, and any baseline handed the answer. This mistake already cost us `CGE-A-01` | `BUILDABLE NOW` |
| 3 | **Identity / invariance** | same object rotated, occluded, re-lit, then a new instance | shared low-level statistics between the two views | `IMPLEMENTED` at level 1 only (chance, 0.51) |
| 4 | **Categorization** | train on members, test on an unseen member of the class | k-means on the raw input patch. It beat Heron's concept formation by **14×** | `BLOCKED` — no labelled classes with held-out instances |
| 5 | **Relational reasoning** | A>B, B>C ⇒ ? ; same/different | the absolute features of the objects, rather than the relation. Controls must hold features constant across relations | `BUILDABLE NOW` |
| 6 | **Cognitive flexibility** | reversal learning: red→food, then blue→food | a system with no memory relearns instantly and looks maximally flexible. Must be scored against retention, not speed alone | `BUILDABLE NOW` |
| 7 | **Spatial navigation** | maps, shortcuts, novel routes — not just solving | wall-following and other local policies solve many mazes with no map at all | partially — `spatial_memory` exists, shortcuts do not |
| 8 | **Path integration** | dead reckoning with no landmarks: where did you start? | predicting zero displacement, which is what holding still achieves | `IMPLEMENTED` as `CGE-A-09`. Corvus **+0.572** |
| 9 | **Planning** | Tower of Hanoi, Sokoban — greedy must fail | greedy with lookahead-1 at matched compute | `BLOCKED` — needs object manipulation; ours only moves |
| 10 | **Causal reasoning** | lever → food, then break the lever; Aesop's-fable variants | association without intervention. The gate must include an action the agent chooses | `BLOCKED` — needs an action space with consequences |
| 11 | **Social reasoning** | cooperation, competition, simplified theory of mind | **a model of the environment the other agent is reacting to.** That is what would be mistaken for theory of mind | `BLOCKED` — no second agent exists |

**Four are blocked for want of a world, not an architecture**, and the reason is the same each
time: our agent has a 5×5 view and four locomotion actions. Planning, causal reasoning and social
reasoning all need an action space with consequences beyond moving; categorization needs classes
with held-out members. That is a finding about the *benchmark*, and `cge.catalogue.audit()`
records the same limitation from the other side.

**Four are buildable now** — working memory, object permanence, cognitive flexibility, relational
reasoning — and that is the right next batch, in that order. Delayed match-to-sample is the
cheapest of them.

### A protocol validated on animals is not automatically valid here

Delayed match-to-sample means something for a crow because a crow could solve it many ways. Here
it can be passed by a filter with a long enough time constant. **Importing a famous paradigm does
not relieve it of needing its anti-test** — and the anti-test is the column above, not an
afterthought.

### Where the boundary with the CGE falls

The same capability can legitimately appear in both batteries, at different levels, and object
permanence is the clearest case: `CGE-A-09` asks whether a **layer's state** contains displacement
while its referent is unobservable; a Piaget hiding task asks whether the **system behaves** as
though the object still exists. Neither substitutes for the other.

The test for which battery a capability belongs to is one question: **can you ask a single layer
this?** Prediction, path integration, identity and object binding — yes, probe the layer's
readout, so they are CGE. Working memory, object permanence, categorization, relational reasoning,
flexibility, navigation, planning, causality and social reasoning — no. You cannot ask a neuron
whether it does reversal learning. Those are EIS.

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
