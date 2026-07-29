# Dynamic Cortical Network — First Principles (v3, blank page)

*Working name: DCN. Alternatives on the table were Synthetic Brain v2 and Resonant Brain
Architecture (RBA). It is a directory name and one line in the benchmark registry — cheap
to change, and worth choosing deliberately before there is code to rename.*

---

## The rule this document exists to enforce

**Nothing enters by compatibility. Everything enters because it has a stated function.**

The RPDU line is frozen in `legacy/`. It is not a starting point, a base class, or a
default. Anything carried across has to be re-justified here against evidence, and named
as such. If we start adapting the old model piece by piece, we end up with a hybrid where
no behaviour can be attributed to any hypothesis, and the ability to run a decisive
experiment is the only real asset this project has.

Concretely, that means: DCN code imports from `core/` and from nothing in `legacy/`. A
test enforces it.

---

## Axioms

Written as claims that could be wrong, because each one should be individually falsifiable.

1. **Memory lives in the DCN.** Not in the neurons and not in the weights connecting them.
2. **Neurons are simple compute units.** They hold no knowledge worth naming.
3. **Waves coordinate synchronisation; they do not store information.** Phase is a clock, not content.
4. **Intelligence emerges from the interaction of memory, attention and resonance** — not from prediction alone.
5. **Higher structures never address individual neurons.** Each level speaks only to the level directly below it, through a stated interface.

### What each axiom already has going for it, or against it

Honest accounting, so the new architecture starts from evidence rather than from a fresh
slate that quietly repeats old mistakes.

**Axiom 3 is already supported.** A synchrony graph — who is phase-locked with whom, read
from real rotor phases — scored *exactly* 1.00x persistence in every multi-object world.
Phase carried no content. That is a direct measurement, and it means the axiom is starting
from a stronger position than the corresponding v1 assumption ever had.

**Axiom 4 is the deliberate break from v1.** v1 assumed structure emerges from prediction
alone, and that was refuted from four directions — most sharply in a world where object
identity was made computationally necessary and the mesh remained at chance on the identity
of an object *in plain view*. Adding memory, attention and resonance as first-class
alongside prediction is the response to that refutation, and it needs a mechanism, not just
a name.

**Axiom 1 needs a definition before it can be tested.** "Memory lives in the DCN" is
currently a location, not a mechanism. The open question is what a DCN *does* that makes it
the thing that remembers, and until that is stated the axiom cannot be falsified.

**Axiom 5 is the one that makes the whole thing testable.** A strict interface between
levels is what lets any level be replaced without breaking the rest, and it is what makes
the benchmark meaningful across architectures. It is also the axiom most easily violated by
convenience — reaching down one level for a quantity that is "right there" — so it should
be enforced in code rather than in intent.

---

## What carries over, and on what evidence

Three findings from the legacy line are strong enough to be design inputs. They are stated
here as constraints, not as inherited code.

**~~The predictive code is relational, not additive.~~ — RETIRED at level 2.** The claim
was: averaging members destroys what they carry, a mean never beats persistence, pairwise
co-activation over the same members reaches 0.79x, and therefore any aggregation in DCN
must preserve relations rather than summarise over them. It was called the single most
transferable result the old line produced.

It did not transfer. Measured at level 2 with all three operators at identical output
width, on identical members, inside identical dynamics, on both the world and the members'
own future — **mean pooling wins every cell** (0.598 against 0.612 for a bilinear sketch and
0.629 for the exact upper triangle). Both alternative explanations were tested and ruled
out: the exact operator loses too, so it is not the sketch; and crossing aggregation with
the workspace dynamics puts all six cells within 5%, so it is not the filter either.

The narrowest statement the evidence supports is: *mean pooling destroys the predictive code
in the legacy mesh, predicting legacy members*. That is a fact about one aggregator over one
population on one target, and it was generalised into a design constraint on the strength of
being the most interesting number available. See
[RESULTS_L2_NODE.md](RESULTS_L2_NODE.md) §L2-2.

**Every level must be scored at the timescale it integrates over.** No function of a mesh
state beat "assume no change" one tick ahead; at 64 ticks a 44% gain was available. A slow
level asked for a fast prediction is being set an impossible test. Every DCN level should
declare its horizon, and that horizon should be measured rather than assumed.

**A dynamics has to be checked against the function it must express, before it is built.**
Gradient flows cannot oscillate. Low-pass filters cannot integrate. Unbounded accumulators
drift. Each cost a full experimental cycle to discover empirically and each was provable on
paper in minutes.

## What explicitly does not carry over

The gradient-flow unit. The energy-landscape formulation. Four-scalar message passing —
untested rather than refuted, but it arrives with no evidence. Coalitions-by-synchrony as a
representation. Mean-pooled shared workspaces. Voting as a route to binding.

**Correction, from level 2.** Two entries on that list were wrong to be there.

*Coalitions-by-synchrony* was discarded because a synchrony graph scored exactly 1.00x
persistence — evidence about synchrony as a **representation to predict from**. Nothing on
that list was ever measured about synchrony as a **grouping**, and measured as a grouping it
beats everything this architecture has: v2's coalitions carry +0.124 object MI above a
shuffled null against the DCN's +0.009, at each architecture's own granularity. Fourteen
times more object binding, from the mechanism that was thrown away. It should be
transplanted in isolation and scored here.

*Mean-pooled shared workspaces* were discarded on the relational-code result above, which
has now been retired. Mean pooling is currently the best aggregation operator measured at
level 2. It goes back on the table as a control that keeps winning.

Both mistakes have the same shape: a measurement that answered one question was used to
close a different one. Worth watching for, because it is cheap to do and expensive to find.

---

## Ideas awaiting a mechanism

Named in the new design and not yet specified sharply enough to build or to test. Each
needs a sentence saying what it computes and what would show it working.

- **DCN as the functional unit** — what is inside one, and what makes it the locus of memory.
- **Hierarchical grouping** — how a level is composed, and what its interface exports.
- **Local waves and global waves** — if phase is a clock, what the two rhythms coordinate.
- **Sleep and consolidation** — what is replayed, what is pruned, and what improves as a result.
- **A cortex organised by modes** — what a mode is, and what decides which is active.

---

## How this will be judged

The benchmark is shared and architecture-agnostic. `core/` holds the worlds, sensors,
probes, metrics and baselines; `cge/` holds the registry and the gates; `tests/` holds
the battery. A DCN registers in `cge/registry.py` with four methods and is then scored
against v1 and v2 on identical code paths — and a future v3 inherits the whole battery for
free.

The gates as they stand: prediction against copy-last; out-of-view wall decode in the maze;
hidden-object identity; coalition/object binding against a shuffled null; path integration
in the tunnel maze, where the pixel control is at chance by construction.

Two standing rules, both learned the hard way. **Every probe reports its control**, and a
result whose control has failed is reported as unmeasured rather than as zero. And **new
gates are added to `core`/`bench`, never to an architecture**, so that adding a level or a
version never quietly changes what "the same test" means.
