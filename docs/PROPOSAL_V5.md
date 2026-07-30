# Proposal — OCA v5, and why L2 cannot be rebuilt on Corvus

*2026-07-30, written the day `corvus-v4.4-alpha` was frozen.*

**Answer to the question asked: a new generation, and the reason is at Layer 1, not Layer 2.**

The Thousand Brains critique is the strongest architectural proposal this project has received,
and it explains our own Layer 2 failure better than our explanation did. It also cannot be built
on Corvus, for a reason that is visible in three lines of our sensory boundary.

---

## Why voting cannot be added to Corvus

In Hawkins's theory each cortical column is a complete model: it learns **the whole object**, in
its own reference frame, and the next level is columns **voting** on which object it is. Voting
is meaningful because N columns hold N independent hypotheses *about the same thing*.

Corvus's towers cannot hold such a hypothesis. From `architectures/corvus/cortex.py`:

```
17 towers | each sees a 2x2 block of an 8x8 patch grid
=> each tower sees 6.2% of the retina, fixed and disjoint
```

**A Corvus tower never sees an object.** It sees a fixed sixteenth of the visual field, and the
same sixteenth forever. Ask seventeen towers to vote and you are not reconciling seventeen
opinions about one object — you are concatenating seventeen views of seventeen different places.

That is not a voting mechanism with a bad rule. **There is nothing to vote about**, and this is
almost certainly the real reason `CGE-B-10` returned −0.000 ± 0.001. We concluded "coordination
buys nothing." The more precise conclusion is **"coordination buys nothing between components
that have no shared referent"** — which is a much more useful negative, and the critique is what
supplies it.

**Hawkins's precondition is a property of the layer below.** Supplying it means every tower
receives the whole sensory surface through its own movable sensor, and maintains a reference
frame in which it can express *"the object is at pose X relative to me."* That is a different
Layer 1. Under R6 a design change is a new generation, not a build.

So: **OCA v5**, starting from a frozen, tagged, certified predecessor. That is the launch-vehicle
model working exactly as intended, not a setback.

## Why this proposal is not the five aggregation failures again

This distinction decides whether the proposal is worth anything, so it is worth stating flatly.

**Voting is selection. Aggregation is averaging.** Every one of our five nulls averaged: the
legacy workspace, Heron's two relational operators against a mean-pool control, Corvus's linear
summary, Corvus's cluster-mate pooling. Averaging N hypotheses about N *different* things
destroys them, which is what we measured five times.

Selecting among N hypotheses about **one** thing is a different operation with a different failure
mode, and **it has never been tested here.** The five nulls do not bear on it.

Two existing results support the proposal directly:

- **Swift's synchrony coalitions carry +0.124 object MI** — the best grouping evidence in the
  project. Consensus by voting would have to beat that number, and Swift is the control it must
  face.
- **Heron's five-scalar publication bottleneck costs only 1.50×.** That is direct evidence that a
  **narrow message protocol between independent modules is affordable** — which is the load-bearing
  assumption of a Cortical Messaging Protocol. This is the most transferable positive result we
  have, and it happens to be exactly what CMP needs.

## What would have to change, minimally

Named now so the scope cannot drift later.

| level | change | why |
|---|---|---|
| **L0 neuron** | **none** | passes at +0.917. It is the one component that has never needed changing |
| **L1 tower** | whole-surface input through a **movable** sensor; a reference frame holding object pose relative to self; each tower models complete objects | the precondition voting requires. This is the generation change |
| **L2 assembly + consensus** | towers exchange (object hypothesis, pose, confidence) over a narrow protocol; consensus by **selection**, never averaging | the actual proposal |
| **Global workspace** | **do not build** | the legacy Predictive Assembly *was* a shared workspace: 0.69× member state → 6.56× workspace. Already refuted here |

Four new levels were proposed (Assembly → Consensus → Workspace → Cognitive System). R5 permits
one, after the layer below clears its floor.

### L2's floor, declared before the layer exists

Two controls, because one is not enough:

- **`best_single_tower`** — the most confident tower's hypothesis, alone. **This is the control
  that defines voting.** If consensus cannot beat the best individual expert, it is not
  reconciling anything.
- **`mean_of_hypotheses`** — averaging instead of selecting. The control that has killed five
  layers, kept so a win cannot be "we pooled differently".

## The blocker nobody has named yet: we need a world, not just an architecture

**Thousand Brains is a theory about learning objects by moving a sensor over them.** Our agent
moves *through a maze*; it does not move a sensor *over an object*. Our worlds have three object
kinds, no rotation, no 3D pose, and no viewpoint change.

A tower that models "the whole object in its own reference frame" has, in our current worlds,
essentially nothing to model. **v5 needs a world with objects that can be viewed from more than
one pose**, and that is a larger piece of work than the architecture change.

This is the honest big-ticket item, and it should be built first — because if the world cannot
distinguish "recognises the object from a new viewpoint" from "matches pixels", then no amount of
voting will be measurable in it.

## Does this attack `CGE-A-00`? A prediction, recorded before building

`CGE-A-00` — beat your own input at predicting an observable world — is unpassed by four
architectures. Every mechanism we have built lost to raw pixels for the same structural reason:
**a tower's representation is a compressed slice of what the pixels already show.** Compression of
available information cannot beat the information.

Pose-invariant object identity is **not** in the raw frame. A system that recognises the same
object from a viewpoint it has not seen is computing something the pixels do not directly contain.
**Thousand Brains is therefore the first proposal here that is structurally aimed at A-00**, rather
than at rearranging what sits above it.

Stated as a falsifiable prediction, to be scored honestly afterwards:

| prediction | confidence |
|---|---|
| v5's L1, with movable sensing and reference frames, beats raw pixels on a **pose-varying** world | **35%** |
| It does not beat raw pixels on our current worlds, because there is no pose to be invariant to | **85%** |
| Voting beats `best_single_tower` once towers share a referent | 50% |
| Voting beats `mean_of_hypotheses` | **75%** — this is the one I would bet on, and it would retire "aggregation always fails" as too broad |
| Consensus beats Swift's +0.124 on binding | 30% |

**Kill criterion.** If voting cannot beat the single most confident tower, consensus is not
reconciling hypotheses and the level should be retired the way Corvus's L2 was — immediately, and
into the graveyard with its numbers.

## On the research programme

Five lines were proposed. My honest assessment of their value **to what we build next**:

| line | verdict |
|---|---|
| **Monty / CMP implementation** | **Do this one properly.** It is running code and a concrete protocol we can adopt or reject on its merits. Everything else is background |
| HTM: what survived into Thousand Brains | skim. Mostly superseded, and the superseding theory is the one above |
| Neuromorphic hardware (Loihi 2, SpiNNaker 2, TaiBai) | **not on the critical path.** It validates our L0 send-on-delta choice, which is pleasant and changes nothing. Our bottleneck is that nothing beats raw pixels, not that we lack a chip |
| Cell assemblies and engrams | skim. We have already measured assemblies twice (Swift, Corvus L2) |
| Connectomics above the column | **this is what told us there is no consensus**, which we have already acted on. Re-reading it will not produce the object it does not contain |

FinalSpark's organoid wetware is a genuinely different paradigm and genuinely irrelevant to a
numpy architecture that loses to its own input.

## Recommendation

1. **Do not rebuild L2 on Corvus.** Corvus is frozen at `v4.4-alpha` and stays that way.
2. **Open OCA v5.** Suggested codename **Jay** — Eurasian and scrub jays are the standard animal
   model for episodic-like *what–where–when* memory and future planning, which is where the EIS
   battery goes next.
3. **Build the world first.** Objects viewable from multiple poses. Without it, nothing above is
   measurable.
4. **Then v5's L1**, with its floor declared before it is written, and R5 holding: no L2 until L1
   clears it.
5. **Read Monty properly.** Skim the rest.

The critique's own framing is the argument for all of this: *cada nivel añade coordinación, no
centralización.* Coordination between components with no shared referent is what we just measured
at −0.000. Give them a shared referent first.
