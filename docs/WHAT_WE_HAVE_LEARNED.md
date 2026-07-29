# What three architectures have taught us

Written after DCN level 2 failed four of five gates, and after the same result arrived for
the third time from a third direction. This is not a post-mortem for one level. It is the
empirical foundation the project now rests on, and it is more valuable than any of the
architectures that produced it.

Two things are separated throughout: **what was measured**, and **what we think it means**.
The first is durable and the second is not.

---

## The invariant

Across SB-1/RPDU v1, RPDU v2 and DCN v3 — different primitives, different dynamics,
different aggregation, different learning rules — one result never varied:

> **No learned representation beat the raw sensory input at predicting the world, at any
> horizon, on any world.**

That is no longer an implementation bug. It is a finding, and it should be the headline of
this project rather than a footnote in a results file.

## The correction that changes what it means

The obvious reading is *early compression destroys information* — hierarchy is fine,
bottlenecks are not. That reading is intuitive, it matches the history of hand-engineered
vision pipelines, and **the tunnel-maze measurement says it is wrong.**

The tunnel maze is the only world in this repository where the question is well posed. Inside
a covered corridor every frame is byte-identical, so the image carries exactly zero
positional information and the pixel control sits at chance *by construction* — measured at
7.02 cells against a chance of 7.02, to two decimals, across three seeds. Everywhere else,
the world is observable enough that the answer is already in the frame, and a representation
can at best re-encode what the input already says. **Most of "nothing beats raw pixels" was
an artefact of asking on worlds where nothing needed to be remembered.**

Here is the same question asked where it bites. Position error in maze cells while blind,
lower is better, three seeds, matched decoder and matched capacity:

| entrant | error | vs the bar |
|---|---|---|
| **frozen-at-entry** — store the coordinates you walked in at, once | **2.26** | — |
| RPDU v1 (2304-dim state, uncompressed) | 4.44 | −98% |
| RPDU v2 | 5.10 | −127% |
| raw pixels | 7.02 | −212% (chance, as it must be) |
| DCN v3 | 8.56 | −277% (worse than chance) |

**Every architecture loses to a two-number memory.** The newest loses to having no memory at
all — its state is actively misleading about position, which is what fitting noise looks
like.

And this kills the compression diagnosis. RPDU v1's readout is its **entire** mesh state,
2304 dimensions, nothing pooled and nothing summarised. It still loses to storing two
numbers. If compression were the problem, the uncompressed representation would not fail the
same way.

The narrower and better-supported statement:

> **The problem is not that persistent state is being destroyed on the way up. It is that
> persistent state is never formed.**

Everything these architectures do is a function of the recent past with a fading memory.
None of them has an object that continues to exist. Frozen-at-entry is not a clever baseline
— it is one `remember(r, c)`, and it beats two years of mechanism.

## The ledger

Confidence is stated so that it can be argued with.

| observation | confidence | where |
|---|---|---|
| No learned representation beats raw input on an **observable** world | **very high** | [L2 results](DCN/RESULTS_L2_NODE.md), [scorecard](../bench/run.py) |
| No architecture forms persistent state; a two-number memory beats all of them | **very high** | tunnel gate, 3 seeds, control exactly at chance |
| Local backprop-free learning is competitive with capacity-matched BPTT | high | [legacy results](RPDU/RESULTS.md) |
| Event-driven emission beats matched-rate controls, and the policy transfers across architectures | high | [L1 results](DCN/RESULTS_L1_NEURON.md) |
| Five-scalar publication survives its bottleneck | high | L2-4, 1.50x |
| Synchrony behaves as timing, not as content | high | 1.00x persistence as a representation |
| Synchrony **as a grouping** binds objects better than anything that replaced it | medium-high | +0.124 vs +0.009, one measurement |
| Pairwise relations carry signal *in the legacy mesh, predicting its own members* | medium | exp11; did **not** reproduce at DCN L2 |
| "The predictive code is relational" as a general design constraint | **retired** | mean pooling wins at matched width, both targets |
| Phase coordination pays off anywhere | **no evidence at any level** | −10% at L1, +0.9% inconsistent at L2 |
| Emergent object permanence | **refuted** | chance on an object in plain view |
| Coalitions as the substrate of thought | **refuted** | exactly 1.00x persistence |

Three of these were this project's own published claims, retracted by its own measurements.
That is the discipline working.

## The mistake we keep making

Both retractions this quarter have the same shape: **a measurement that answered one question
was used to close a different one.**

- A synchrony graph scored 1.00x as a *representation to predict from*. That was used to
  discard coalitions-by-synchrony as a *grouping*, which was never measured. Measured later,
  it beat its replacement fourteen-fold.
- Mean pooling destroyed the predictive code *in one mesh, predicting its own members*. That
  became "aggregation must preserve relations" — a constraint on a whole architecture. It is
  false there.

Both were cheap to make and expensive to find. The guard is to write down, next to every
result, the question it does **not** answer.

## The question to work on now

Not "what does level 3 look like". That question has an architecture baked into it, and this
project has now spent three architectures learning that its architectural instincts are not
the binding constraint.

> **What is the smallest computational primitive that beats a two-number memory at knowing
> where it is while blind?**

It mentions no node, no cortex, no resonance, no hierarchy, no wave. It has a control that
is already measured, a world where the control is provably at chance, and a bar that is
almost insultingly low. It is falsifiable in an afternoon.

This is now gate **P0** in `bench/gates.py` (`--gates tunnel`), and it is the project's
standing open challenge. Anything that clears it has done something none of the three
architectures here can do, and has earned the right to be built on.

## What this changes, concretely

**Level 3 is not started.** It was already blocked; it is now blocked for a better-stated
reason. A coordination layer over nodes that hold no persistent state cannot produce
persistent state.

**The benchmark was letting everything off the hook.** Gates on observable worlds cannot
distinguish a good representation from a re-encoding of the input, and most of ours are.
Worlds where the answer is provably absent from the frame — the tunnel maze, the occlusion
band, delayed cause-and-effect — are the only ones that discriminate, and they should
dominate the battery rather than sit at the end of it.

**Prediction may be the wrong objective.** Every gate here asks "what happens next", which is
answerable from the frame on an observable world. "What happens if I move the wall" is not,
and we already have an agent that acts. This is worth one honest experiment before it becomes
a belief.

**What is deliberately not concluded.** That hierarchy is wrong; that resonance is wrong;
that prediction is wrong. None of those has been tested — what has been tested is a
particular implementation of each, and the failures so far are consistent with a single much
simpler cause. Fix persistence first, then re-ask.

---

*Related: [SPEC_ARCHITECTURE.md](SPEC_ARCHITECTURE.md) for the whole picture,
[RESULTS_L2_NODE.md](DCN/RESULTS_L2_NODE.md) for the measurements this rests on,
[FIRST_PRINCIPLES_DCN.md](DCN/FIRST_PRINCIPLES_DCN.md) for which axioms have since been
corrected.*
