# Aggregate-then-vote — result

*Frozen `corvus-v4.4-alpha` towers, read-only. Three seeds, ~44,700 visible frames each,
capacity swept over {8, 16, 32, 64, 128}. Target: object kind.*

**The hypothesis is refuted, and the refutation locates the problem one level lower than
anyone had it.**

## The numbers, visible condition

| condition | seed 5 | seed 6 | seed 7 | mean |
|---|---|---|---|---|
| best single tower | 0.625 | 0.600 | 0.599 | **0.608** |
| best single assembly (5 towers) | 0.599 | 0.605 | 0.591 | **0.598** |
| all 9 towers concatenated | 0.568 | 0.581 | 0.605 | **0.585** |
| vote by selection | 0.601 | 0.596 | 0.603 | **0.600** |
| mean of hypotheses | 0.596 | 0.595 | 0.598 | **0.596** |
| **raw pixels** *(control)* | 0.760 | 0.811 | 0.774 | **0.782** |
| **k-means on raw patch** *(control)* | 0.441 | 0.460 | 0.515 | **0.472** |

| declared floor | measured | verdict |
|---|---|---|
| assembly beats best single tower | **−0.009** | **FAIL** |
| vote beats best single assembly | **+0.002** | **FAIL** |
| vote beats mean of hypotheses | +0.004 | **FAIL** |
| assembly beats k-means on the raw patch | **+0.126** | **PASS** (3/3 seeds) |
| anything beats raw pixels | **−0.178** | **FAIL** |

## What it means

**Every Corvus-derived condition lands between 0.585 and 0.608.** One tower, five towers, nine
towers, voted, averaged — a spread of 0.023 across conditions that differ by a factor of nine in
how much of the retina they cover.

That is a **ceiling, and one tower already reaches it.**

So the hypothesis fails, but not the way it was expected to. It was not that we aggregated at
the wrong level, or voted at the wrong level, or grouped by the wrong rule. **There is nothing
left to aggregate.** The object-kind signal is worth 0.782 in the pixels and 0.608 after one
tower has processed it; adding eight more towers recovers none of the missing 0.174, because
every tower destroyed the same thing on the way through.

The clearest single statement of it: **an assembly needs 128 components to reach 0.598, and one
tower reaches 0.608 at 32.** Four times the capacity, slightly less accuracy. The extra towers
add dimensions and no information.

## What this settles

**Aggregation is not the missing level.** Six architectures' worth of pooling has now been
measured, and this is the first one aimed at a *hypothesis* rather than at predicting members.
It behaves the same.

**Voting is not the missing level either.** Consensus among nine overlapping assemblies beats
the best single assembly by +0.002. The Thousand Brains premise — that reconciling independent
hypotheses buys something — gets no support on this substrate. Which is consistent with the
reason it should not have: assemblies drawn from towers that all lost the same information hold
nine copies of the same impoverished opinion, and voting between them cannot manufacture what
none of them has.

**Selection beats averaging by +0.004**, same sign on all three visible seeds. Real,
reproducible, and thirteen times below the margin — the same shape as connectivity-over-proximity
last week. Two ideas now, both directionally correct and both negligible.

**The loss happens at Layer 1.** That is the finding, and it is measured rather than argued. Any
v5 that changes L2 or L3 while keeping this L1 will reproduce this table.

## The one thing that passed, and it matters

**Assembly beats k-means on the raw patch by +0.126, on all three seeds.**

Heron's concept formation *lost* to this control by 14×. Corvus's towers carry substantially
more about object kind than unsupervised clustering of the pixels they came from. That is a real
difference between the two architectures, on a control that has killed a layer before, and it is
the first evidence that Corvus's L1 is doing something rather than merely losing less.

It is not enough. 0.598 against 0.782 is still two thirds of the signal thrown away. But "worse
than the input and better than clustering it" is a different position from Heron's, and it says
the tower is not simply a lossy filter.

## Caveats, stated plainly

**The probes are supervised.** Each assembly's hypothesis came from a ridge classifier fitted to
labels. A real assembly would have to learn unsupervised, which is strictly harder. So this
measures *"is the information there to be assembled"*, not *"could an assembly learn to extract
it"*. Since the answer to the first is no, the second does not arise — but if the first had been
yes, this experiment would not have been sufficient.

**Occluded frames are at chance for everything**, raw pixels included (0.503, 0.534, 0.501),
which is the control behaving correctly: pixels cannot see a hidden object. Nothing in the
occluded condition supports or refutes anything.

**Capacity was swept and each condition reported at its own best.** At a single budget of 32 a
lone tower passes through untouched while a five-tower assembly is compressed 4.5:1, so a fixed
budget would have handicapped the condition under test. The sweep is why `best_single_assembly`
peaks at 128 and `best_single_tower` at 32 — and the conclusion holds at every budget, not only
at the chosen one.

## What it costs, and what it saved

One day. The alternative was building a pose world and a new architecture on the assumption that
aggregation-then-voting was the missing piece, and discovering this at the end of it.

Corvus was never unfrozen. `corvus-v4.4-alpha` stands.

## Where this points

To [`docs/SPEC_POSE_WORLD.md`](SPEC_POSE_WORLD.md) and to **Layer 1**.

The information is destroyed before any higher level sees it, so the change must be at the level
that destroys it. And the reason to expect a different L1 to do better is the one thing this
world cannot test: **pose.** Every architecture here has been a compression of what the pixels
already showed, and compression of available information cannot beat the information — which is
`CGE-A-00`, unpassed by five entrants for exactly that structural reason.

Pose-invariant identity is *not* in the pixels. That is the only proposal on the table where the
raw control is at chance by construction, and it is why the world comes before the architecture.
