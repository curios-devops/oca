# Level 2 — Dynamic Cortical Node: results

Reported per level, as the methodology asks. Level 1 is in
[RESULTS_L1_NEURON.md](RESULTS_L1_NEURON.md); level 3 (DRC) is specified in
[Design.md](Design.md) and not built — level 2 has not earned it.

Run with `make dcn-l2`. 17 nodes, 1088 neurons, 293,485 parameters, declared horizon 16,
driven by the same physics world every other level and every legacy version is scored on.

**Headline: this level does not work yet.** Four of its five gates fail, it is the weakest
of the three architectures on the shared benchmark, and none of the three beats raw pixels.
Three of those failures are informative and one of them reverses the strongest finding the
legacy line produced. The details are below in the order they were measured.

---

## Before the gates: a structural defect the gates would not have caught

The design says the node is the slow layer — that is what lets it hold something its
neurons cannot. Measured, it was not:

| lag | node state | its own members |
|---|---|---|
| 4 | 0.987 | 0.980 |
| **16** (declared horizon) | **0.882** | **0.907** |
| 64 | 0.606 | 0.679 |

At the declared horizon the node was decorrelating *faster* than the neurons feeding it. A
level that integrates over a shorter window than the level below it is a relabelling, not a
level, and no gate in the battery would have reported it — every gate would simply have
returned a slightly worse number for reasons that looked like the mechanisms.

The cause was one parameter (`leak`, how much of the state is replaced per tick). It is now
set to the largest value that reverses the ordering at every lag, and
`tests/test_dcn_node.py` pins the ordering so it cannot silently break again.

**Every number below is from after that fix, and the fix changed no conclusion.** Concept
formation got *worse* (MI excess +0.043 → +0.009), prediction moved by 1%, and every
verdict stayed the same. That is worth more than if it had helped: the failures are not a
mistuned timescale.

---

## L2-1 Concept formation — FAIL

Does the knowledge model's active concept track what is actually in each node's receptive
field? Two controls: a shuffled null, and clustering the node's own raw input patch into
the same number of clusters.

| | normalised MI |
|---|---|
| node concepts | 0.010 |
| shuffled null | 0.001 |
| **excess over null** | **+0.009** (sem 0.004) |
| **same k, clustered raw pixels** | **0.139** |

The excess over the null is barely two standard errors, and the honest comparison is the
last row: **k-means on the raw input patch tracks the world fourteen times better than the
node's consolidated concepts.** Whatever the knowledge model is consolidating, it is not
what is in front of it.

The mechanism is not obviously broken — prototypes stay distinct, recruitment fires,
`no_knowledge` scores exactly 0.000 as it must. It competes on the working state, and the
working state is a slow reservoir summary in which a fast-moving object is a small
perturbation. Consolidating a slow state is the wrong operation for tracking a fast one.

## L2-2 Relational vs mean pooling — FAIL, and the sign is the finding

This is the gate that was meant to be able to kill the level, carried over from the
strongest result the legacy line produced: *a mean over members never beat persistence,
while pairwise co-activation over the same members reached 0.79x and was the only operator
that improved as the world got harder.* The DCN first-principles document turned that into
a constraint — "any aggregation must preserve relations between members, not summarise over
them".

Every operator below is built from the identical member states, at the identical output
width, inside the identical dynamics, and scored on the identical target. Ratios are
against persistence; lower is better.

| operator | predicting the world | predicting the members' own future |
|---|---|---|
| relational (random bilinear sketch) | 0.612 | 0.886 |
| pairwise (exact co-activation) | 0.629 | 0.854 |
| **mean (the control)** | **0.598** | **0.770** |

**Mean pooling wins on both targets.** The constraint does not hold here, and the ordering
is the opposite of the one it was written from.

Two alternative explanations were tested and both are ruled out:

- **"The sketch lost the information."** The exact upper-triangle operator — literally what
  the legacy line measured, not an approximation of it — also loses, on both targets.
- **"The workspace dynamics is doing the work, not the operator."** Crossing aggregation
  against the low-pass/reservoir choice puts all six cells within 5% of each other
  (0.598–0.648). Neither factor matters much, and neither matters in the claimed direction.

The narrowest statement the evidence supports: *mean pooling destroys the predictive code
in the legacy mesh, predicting legacy members.* It does not generalise to this level, these
members, or this target. The constraint in
[FIRST_PRINCIPLES_DCN.md](FIRST_PRINCIPLES_DCN.md) has been narrowed accordingly rather
than quietly dropped.

## L2-3 Horizon — FAIL

| tau | 1 | 4 | 16 (declared) | 64 |
|---|---|---|---|---|
| vs persistence | 4.849 | 0.785 | 0.612 | 0.561 |
| **vs raw pixels at the same tau** | **5.212** | **1.301** | **1.120** | **1.090** |

The second row is the one that means anything, and the first version of this gate did not
have it. Scored against persistence alone, every representation on the bench — including
the raw frame — looks better the further out it is asked to predict, because persistence
degrades faster than anything else. That trend belongs to the baseline. Dividing by what
raw pixels achieve at the identical tau removes it.

What is left says the node adds nothing at any horizon: it is 9–30% *worse* than pixels
everywhere, and worst at the horizon it declares. It also improves monotonically out to 64,
so even the shape of the curve does not support the declaration.

## L2-4 The five-scalar bottleneck — PASS

A node publishes prediction error, confidence, activation, novelty, energy and its phase
spectrum: 13 numbers per node, 221 in total, against 1309 for the full internal state.

| reader is given | ratio vs persistence |
|---|---|
| the published scalars only | 0.977 |
| the full internal state | 0.653 |
| **cost of the bottleneck** | **1.50x** |

The design's boldest claim survives its gate, and it is the only one that does. I had put
this at roughly 30% before running, on the strength of the legacy version of the same claim
failing. Read it with the caveat it deserves: the bottleneck is cheap *relative to a full
state that is itself worse than raw pixels*, so this says the mouthpiece is not the
bottleneck — not that what comes through it is worth having.

## L2-5 Phase under channel contention — FAIL

Level 1 found that phase gating costs a lone neuron precision and said so at the time: a
gate can delay an emission and can never improve it. The claim was always that phase pays
off in *coordination*, which needs more than one neuron. This is the first place that could
be tested rather than asserted — 128 members sharing one channel, swept over four budgets,
at matched offered rates.

| channel budget | dropped (gated / ungated) | NRMSE gain from phase |
|---|---|---|
| 0.02 | 0.884 / 0.893 | +4.4% |
| 0.04 | 0.833 / 0.845 | −0.1% |
| 0.08 | 0.725 / 0.745 | −3.2% |
| 0.16 | 0.558 / 0.584 | +2.4% |

Mean +0.9%, sign inconsistent. Phase staggering does reduce collisions slightly at every
budget, and that does not translate into precision. **On this evidence axiom 3 has no
operational payoff at either level so far** — phase is measurably a clock, and having a
clock has so far bought nothing.

## Ablations

Ratio against persistence at the declared horizon, and concept MI above the null:

| variant | x persistence | concept MI excess |
|---|---|---|
| full | 0.612 | +0.009 |
| pairwise_exact | 0.629 | +0.010 |
| mean_pooling | **0.598** | +0.010 |
| relational_lowpass | 0.648 | +0.007 |
| pairwise_lowpass | 0.621 | +0.022 |
| mean_lowpass | 0.608 | +0.020 |
| no_knowledge | 0.612 | 0.000 |
| no_neighbourhood | 0.612 | +0.009 |
| no_oscillation | 0.609 | +0.009 |
| no_state_adapter | 0.612 | +0.009 |

**Four of the six mechanisms change the headline by less than 0.5%.** The knowledge model,
the neighbourhood model, the oscillation engine and the state adapter are all, on this
measurement, decoration — removing any of them costs nothing. Only aggregation and the
workspace dynamics move the number at all, and the best cell is the control.

---

## Phase 2 — v1 vs v2 vs DCN, identical probe, identical target

The whole reason the benchmark lives in `core/` and `bench/` rather than inside an
architecture. Every model below met these gates through the same code path.

Predicting the world tau ticks ahead, relative to assuming nothing changes:

| model | dim | params | t=1 | t=4 | t=16 | t=64 |
|---|---|---|---|---|---|---|
| **raw pixels** | 1024 | **0** | **0.930** | **0.603** | **0.546** | **0.515** |
| RPDU v1 | 2304 | 614,960 | 3.411 | 0.686 | 0.568 | 0.533 |
| RPDU v2 | 3456 | 918,832 | 4.064 | 0.681 | 0.554 | 0.522 |
| DCN v3 | 1088 | 293,485 | 4.849 | 0.785 | 0.612 | 0.561 |

**No architecture beats the raw frame at any horizon.** This is the third time this project
has landed on that result from a different direction, and the first time all three
architectures have been in the same table when it happened. The DCN is last, though it is
also the cheapest by a factor of two to three — which is a real observation about efficiency
and not a defence.

And the finding that most directly answers "does the legacy line have something we are
missing":

| model | its own notion of binding | MI | null | excess |
|---|---|---|---|---|
| RPDU v1 | coalitions (synchrony) | 0.001 | 0.000 | +0.001 |
| **RPDU v2** | **coalitions (synchrony)** | **0.161** | **0.037** | **+0.124** |
| DCN v3 | concepts (knowledge) | 0.010 | 0.001 | +0.009 |

**v2's synchrony coalitions carry fourteen times more object identity than the DCN's
knowledge-based concepts.** Each architecture was asked about its own units at its own
granularity, so this is not a resolution artefact — and it was checked directly: giving the
DCN four times the node resolution made its maze decode *worse*, not better.

This contradicts a decision made on the blank page. `FIRST_PRINCIPLES_DCN.md` lists
"coalitions-by-synchrony as a representation" under what explicitly does not carry over, on
the evidence that a synchrony graph scored exactly 1.00x persistence. That evidence was
about synchrony as a *representation to predict from*. It said nothing about synchrony as a
*grouping*, and as a grouping it is the best object-binding signal any version in this
project has produced. Those are different claims and the first was used to discard the
second.

## The race

`make race` — all three in the same maze, same planner, same 900 steps, exits reached:

| | out-of-view wall decode | exits |
|---|---|---|
| raw-pixel control | 77.3% | — |
| **RPDU v1** | **84.1%** | **53** |
| RPDU v2 | 73.6% | 3 |
| DCN v3 | 61.9% | 0 |

Only v1 is above the pixel control, and it is the only one that finds the exit with any
regularity. The DCN, given a map worse than the pixels it was built from, walks into walls.

---

## Verdict

| gate | result |
|---|---|
| L2-1 concept formation | **FAIL** — 14x worse than k-means on the raw patch |
| L2-2 relational aggregation | **FAIL** — mean pooling wins, on both targets |
| L2-3 horizon | **FAIL** — worse than pixels at every tau, worst at the declared one |
| L2-4 five-scalar bottleneck | **PASS** — 1.50x, the only gate this level passes |
| L2-5 phase coordination | **FAIL** — +0.9%, sign inconsistent |
| node is the slow layer | fixed after failing; the fix changed no conclusion |

Level 2 does not pass, so level 3 is not built. That is the methodology working rather than
the methodology failing: the cost of finding this out was one battery, and nothing has been
built on top of it.

**What is actually established.** Two things, both negative and both reusable. Phase is a
clock that has so far bought nothing at any level. And the relational-aggregation constraint
does not generalise beyond the setting it was measured in — which retires the strongest
result the legacy line produced, at least as a design input.

**What to try before declaring the level dead.** In order of what the evidence points at:

1. **The knowledge model consolidates the wrong thing.** It clusters a slow reservoir state,
   in which a moving object is a small perturbation, and loses to k-means on the raw patch.
   Consolidating the *prediction error* instead — what surprised this node — is a different
   mechanism for the same axiom and is testable against the same gate.
2. **Take v2's coalitions seriously.** Synchrony grouping outperforms everything this
   architecture has, and it was discarded on evidence about a different question. The
   honest move is to transplant it in isolation and score it here, which is exactly what
   the shared benchmark was built to allow.
3. **Nothing beats raw pixels.** Until some representation does, on some world, every
   comparison between architectures is a comparison between things that are all losing to
   the input. That is the finding this project keeps arriving at, and it may be the one
   worth attacking directly rather than one level at a time.
