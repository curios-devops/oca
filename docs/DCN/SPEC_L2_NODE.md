# SPEC L2 — Dynamic Cortical Node (DCN)

**Status: implemented, fails four of five gates.** Code in [`dcn/node.py`](../../legacy/dcn/node.py)
and [`dcn/cortex.py`](../../legacy/dcn/cortex.py), gates in [`bench/nodes.py`](../../bench/nodes.py),
results in [RESULTS_L2_NODE.md](RESULTS_L2_NODE.md).

The specification is kept as written rather than rewritten to match what happened, because
the gap between the two is the finding. Where the implementation was measured to fall short
of the spec, it says so inline.

---

## Purpose

The first cognitive unit. A node does not represent a concept — it represents a **dynamic
hypothesis**: what is happening in its part of the world right now, what it expects next,
and how sure it is.

Hundreds to thousands of neurons per node in the design; 64 in the current build, which is
what runs a 20,000-tick maze in seconds. The count is a parameter, not a claim.

## Components

| component | function | measured verdict |
|---|---|---|
| **Working State** | what is happening now | works as specified; the operator does not matter |
| **Knowledge Model** | consolidated knowledge | **fails** — 14x worse than k-means on the raw patch |
| **Prediction Engine** | predicts its own future state and its own sensory patch | works, but worse than raw pixels |
| **Neighbourhood Model** | what nearby nodes are expected to publish | no measurable effect |
| **Oscillation Engine** | a resonance spectrum, not one frequency | no measurable effect |
| **State Adapter** | the global wave changes *how* it thinks, never *what* it knows | no measurable effect |

Four of six change the headline by less than 0.5%. On this evidence they are decoration.

## Interface

```
inputs    x : (n_nodes, n_inputs)     each node's own sensory patch, retinotopic
reads     the level below through its published interface only  (axiom 5)
state     work, K, A, B, M, z
publish   5 scalars + a phase spectrum, and nothing else
horizon   16 ticks                     declared, mandatory
```

### The publication is the whole interface

```
prediction_error   confidence   activation   novelty   energy   +  cos/sin per band
```

**A DCN does not send information. It publishes only its state.** No working state, no
concept vector, no member activity. This is the design's boldest claim and the only gate
this level passes: a reader given the 13 published numbers per node does within 1.50x of a
reader given the full 77-dimensional internal state.

Read that with its caveat. The bottleneck is cheap *relative to a full state that is itself
worse than raw pixels* — it says the mouthpiece is not the problem, not that what comes
through it is worth having.

## Hard constraints

**C1 — The node must be slower than its neurons.** A level that integrates over a shorter
window than the one below it is a relabelling. This failed on the first build (0.882 against
its members' 0.907 at the declared horizon) and no gate in the battery would have caught it;
`leak` now enforces it and `tests/test_dcn_node.py` pins the ordering. **The fix changed no
conclusion**, which makes the other failures stronger, not weaker.

**C2 — The node reads its neurons through the level-1 interface.** Axiom 5: what they
published, held between events — never `pop.a`. Reaching for the internal activation is the
convenient violation the axiom exists to prevent, and would also be a lie about what a real
node can see.

**C3 — A low-pass filter cannot integrate.** The legacy shared workspace was
`w ← (1−α)w + α·u` and was provably unable to hold a quantity across a gap (6.17 against a
chance of 6.14). The working state is an echo-state reservoir that members *perturb*, with
spectral radius below one so it neither forgets in one tick nor drifts.

**C4 — ~~Aggregation must preserve relations between members.~~ RETIRED.** Carried over
from the legacy line's strongest result and measured false here: at identical width, on
identical members, in identical dynamics, **mean pooling wins on both targets** (0.598
against 0.612 sketched and 0.629 exact). Both escape routes were closed — the exact operator
loses too, and crossing aggregation with the workspace dynamics puts all six cells within
5%. See [FIRST_PRINCIPLES_DCN.md](FIRST_PRINCIPLES_DCN.md).

**C5 — Every level is scored at the timescale it integrates over, against a baseline at the
same timescale.** Scored against persistence alone, every representation — including raw
pixels — looks better the further out it predicts, because persistence degrades fastest.
That trend belongs to the baseline. The horizon gate divides by what raw pixels achieve at
the identical tau.

## Gates

| gate | question | result |
|---|---|---|
| **L2-1** concept formation | do consolidated concepts track the world, above a shuffled null *and* above k-means on the raw patch? | **FAIL** — +0.009 vs 0.139 for pixels |
| **L2-2** relational aggregation | does keeping relations beat summarising over them, at matched width? | **FAIL** — mean wins on both targets |
| **L2-3** horizon | does it add anything over raw pixels, at the tau it declares? | **FAIL** — 1.12x worse at tau=16 |
| **L2-4** publication bottleneck | are five scalars and a phase spectrum enough? | **PASS** — 1.50x |
| **L2-5** phase coordination | does phase staggering help when members share a channel? | **FAIL** — +0.9%, sign inconsistent |

## What is known to be true, and what is known not to be

**Established, and negative.** No architecture in this project beats raw pixels at
predicting the world, at any horizon — v1, v2 and DCN all lose, and this is the third
independent route to that result. The relational-aggregation constraint does not generalise.
Phase has bought nothing at either level.

**Established, and surprising.** v2's synchrony coalitions carry **fourteen times** more
object identity than this level's knowledge-based concepts (+0.124 against +0.009 above a
shuffled null, each at its own granularity). Coalitions-by-synchrony was discarded from the
new architecture on evidence about synchrony as a *representation*; as a *grouping* it is
the best object-binding signal any version here has produced.

**Ruled out as explanations for the failures.** Node resolution — four times finer made the
maze decode *worse*. The bilinear sketch — the exact operator loses too. The workspace
dynamics — the crossed ablation is flat. The timescale defect — fixing it changed nothing.

## How to falsify, and what to try next

This level is currently falsified by its own gates. Level 3 is **not** built, because
building a coordination layer over nodes that hold nothing worth coordinating is building
the aeroplane to find out whether the wing works.

Three things the evidence points at, in order:

1. **The knowledge model consolidates the wrong thing.** It clusters a slow reservoir state
   in which a moving object is a small perturbation. Consolidating *prediction error* — what
   surprised this node — is a different mechanism for the same axiom, testable against the
   same gate.
2. **Transplant v2's coalitions in isolation** and score them here. This is exactly what a
   shared, architecture-agnostic benchmark was built to allow.
3. **Attack "nothing beats raw pixels" directly.** Until some representation beats the
   input on some world, every architecture comparison is between things that are all losing.
