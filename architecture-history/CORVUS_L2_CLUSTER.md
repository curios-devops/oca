# Corvus Layer 2 — Tower Cluster

*Retired 2026-07-30, build `corvus-v4.4`. The first **layer** in this graveyard rather than a
whole architecture. Code kept executable at
[`architectures/corvus/retired/cluster.py`](../architectures/corvus/retired/cluster.py).*

## Goals

The level above the tower: a group of cooperative towers providing **local coordination** before
higher-level integration, and — because `Entity` was made a cross-cutting primitive rather than
tower-owned — **relating entities across tower boundaries without re-solving identity.**

Summarising its members was permitted and not required. That was already a concession to
evidence: aggregation had failed four times before this layer was written, so its floor inverted
the burden of proof onto the compression step.

## What it did

Two jobs, and the distinction between them is the whole story.

| job | when it ran | what it did |
|---|---|---|
| **compression** | optional, **off** by default | a linear summary of the members' published states |
| **coordination** | **every tick, in both modes** | member agreement, plus a cross-tower entity relation graph built from the entity references riding in events |

Membership grew each cluster from a fixed seed tower. Three rules were implemented and swept:
`proximity` (contiguous — the original, and the only one for most of the layer's life),
`connectivity` (the towers a seed actually co-varies with, from published state only, re-derived
every 200 ticks), and `random`.

## Gates

**Failed both.**

| gate | floor | result |
|---|---|---|
| `CGE-B-03` compression | beat `pass_through` by 0.05 | **−0.004 ± 0.006** |
| `CGE-B-10` coordination | beat `independent_towers` by 0.05 | **−0.000 ± 0.001** |
| `CGE-B-10` coordination | beat `fixed_proximity_membership` by 0.05 | **+0.010 ± 0.001** |

Three seeds, matched capacity, at the layer's own declared 64-tick horizon.

## Why it was retired

**The coordination null is the reason, and it is unusually clean.** A tower's cluster-mates tell
it nothing about its own future that it does not already know: −0.000 with a standard deviation of
0.001. That holds **however the members are grouped** — all four membership rules land within
0.007 NRMSE of each other, and on two of three seeds the best rule is *no grouping at all*.

This layer was not underperforming. It was doing nothing measurable, on both of its jobs, and **a
level of abstraction that contributes nothing is a relabelling** — the same verdict, arrived at by
the same measurement, that retired Heron's node layer.

## Lessons learned

1. **A floor over an optional job is not a floor.** This layer declared `beats="pass_through"`
   while its default mode *was* pass-through. A tie is not a win, so the floor was unreachable in
   the compliance mode and failed in the other: neither state could pass, and rule R1 could never
   have been satisfied by the stack that contained it. This is now a construction-time error —
   `Floor` names its `job` and whether it is `always_on`, and a layer whose floors govern only
   optional jobs is rejected. **The contract had failed to enforce its own premise**, and this
   layer is what exposed it.
2. **The always-on job is the one that must be earned.** Coordination ran on every tick with
   nothing to beat, for the layer's entire life, while the job it *did* have a floor for was
   switched off. That is the Heron failure reproduced one level up — in the contract instead of
   the implementation.
3. **Grouping by connectivity beats grouping by position, and it does not matter.** The claim came
   from a cortical-column critique: distant columns can belong to the same functional module while
   neighbours do unrelated work. It held — connectivity beat the hard-coded positional rule on all
   three seeds, correctly signed and reproducible, with a membership overlap of 0.53 confirming
   the rules really did pick different towers — and it was **five times below the margin.** The
   hard-coded rule turned out to be the worst of the three, slightly below random.
4. **The cost of finding that out was one gate, not one architecture.** The same critique proposed
   rebuilding this layer as a synchrony-formed Dynamic Assembly. Making its central claim
   falsifiable first is what kept that from being built on a +0.010.

## Principle worth keeping vs implementation to leave buried

**Keep:** the contract rules it produced — one floor per job, and the always-on job must have one.
Those outlived the layer and are now enforced at construction. Also keep the *question* it never
got to answer: cross-tower entity relations were never targeted by any gate, so the one thing this
layer did that nothing else does remains unmeasured rather than refuted.

**Bury:** the cluster as a level of the stack. Corvus is a two-layer architecture and now says so.
Bringing a third layer back requires a job with a declared floor and a reason to expect it to be
beaten — not a slot in a diagram.
