# SPEC L3 — Dynamic Resonance Cluster (DRC)

**Status: specified, deliberately not built.** Level 2 fails four of its five gates, and a
coordination layer over nodes that hold nothing worth coordinating is the aeroplane built to
find out whether the wing works. This document exists so that when level 2 passes, level 3
starts from a stated hypothesis and a gate rather than from a fresh conversation.

Everything below is proposal. Nothing here has been measured.

---

## Purpose

Not an anatomical region — a **temporary functional structure, defined by coherence rather
than position**. A DRC appears when many DCNs enter resonance, may last 30 ms, 500 ms or
several seconds, and then dissolves.

**A DRC does not compute. It coordinates.** Four functions:

| function | what it does |
|---|---|
| resolve conflicts | many nodes propose hypotheses; only the mutually compatible survive |
| synchronise phases | two nodes that must collaborate are put in phase |
| amplify | a consistent hypothesis gains amplitude; the rest are extinguished |
| create context | not by sending messages — by generating a **field** that modifies the nodes inside it |

## Interface

```
inputs    the publications of its member nodes, and nothing else   (axiom 5)
state     membership, coherence, field
output    a field: a modulation applied to member nodes, not a message
horizon   to be declared before implementation, and measured, not assumed
```

The output being a *field* rather than a *message* is the substantive claim. A field changes
the dynamics of everything inside it without addressing anyone; a message is a directed
transmission. If the implementation ends up sending per-node values, it has become a message
bus and the claim has quietly been abandoned.

## Constraints inherited from measurement

**C1 — Membership is coherence, not position.** If clusters end up being fixed spatial pools
they are regions, and this level has no content beyond level 2.

**C2 — The horizon is declared and measured.** Enforced by
[`architectures/heron/contract.py`](../../architectures/heron/contract.py). A cluster that lasts 500 ms and is scored one
tick ahead is being set an impossible test — that mistake cost the legacy line a full cycle.

**C3 — Coalitions-by-synchrony must be measured as a grouping, not only as a
representation.** The legacy measurement that synchrony scored exactly 1.00x persistence was
about synchrony as *something to predict from*. Level 2 then measured v2's synchrony
coalitions carrying fourteen times more object identity than knowledge-based concepts.
Those are different questions and this level depends on the second one.

**C4 — Averaging is not aggregation, and neither is any operator by default.** Level 2
retired the relational constraint; nothing has replaced it. Whatever this level uses to
combine node publications must be gated against mean pooling at matched width, because mean
pooling is currently the operator that keeps winning.

## Gates, to be written before code

Stated now so they cannot be chosen after the numbers are in.

| gate | question | control that can kill it |
|---|---|---|
| **L3-1** clusters form | do coherent groups appear and dissolve, rather than being static? | a fixed random partition of the same size |
| **L3-2** conflict resolution | do incompatible hypotheses actually get suppressed? | the same nodes with the field switched off |
| **L3-3** the field does something | does membership change what a node computes? | field applied with shuffled membership |
| **L3-4** binding | does cluster membership carry object identity? | shuffled null, **and v2's coalitions**, which is the number to beat |
| **L3-5** the cluster is slower than its nodes | does it integrate over a longer window? | its own members' autocorrelation at the declared horizon |

L3-4 is the one that matters. v2's coalitions set +0.124 object MI above a shuffled null
using a mechanism that was discarded from this architecture. A resonance cluster that binds
worse than that has not earned the level.

## Entry condition

**Do not start this level until level 2 passes L2-1 and L2-3** — until a node's state
carries something about the world that raw pixels do not, and it does so at the timescale it
declares. Coordinating representations that are worse than their own input cannot produce a
representation better than the input.
