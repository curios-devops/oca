# Q7 option B — the coordination floor, and what it caught

*`CGE-B-10`, build `corvus-v4.4`, 2026-07-30. 14,000 ticks, three seeds, matched capacity.*

Layer 2 had declared a floor over a job that is off by default and none at all over the job that
runs on every tick. Option B gave coordination a floor with two controls. **It failed the layer
on its first run, which is exactly what a floor is for.**

| condition | seed 0 | seed 1 | seed 2 | role |
|---|---|---|---|---|
| `independent_towers` | 0.6136 | 0.6289 | **0.5991** | control 1 — no grouping at all |
| `random` | 0.6201 | 0.6294 | 0.6048 | sanity |
| `fixed_proximity_membership` | 0.6200 | 0.6355 | 0.6047 | control 2 — what we hard-coded |
| **`connectivity`** | **0.6132** | 0.6300 | 0.5986 | the mechanism |

NRMSE of predicting a tower's own published state 64 ticks ahead — the cluster's declared
horizon — lower is better.

| | margin | measured | |
|---|---|---|---|
| connectivity vs `independent_towers` | +0.05 | **−0.000 ± 0.001** | **FAIL** |
| connectivity vs `fixed_proximity_membership` | +0.05 | **+0.010 ± 0.001** | **FAIL** |

---

## The headline is the first row, not the second

**Coordination contributes nothing.** A tower's cluster-mates tell you nothing about that
tower's future that the tower does not already tell you about itself: −0.000 ± 0.001, three
seeds, at matched capacity and at the layer's own horizon.

That is not a weak result or a tuning problem. It is a null with a standard deviation of 0.001.
Grouping towers is worth zero **however you group them** — the spread across all four membership
rules is 0.007 NRMSE, and the best rule on two of three seeds is *no grouping*.

Layer 2 has now been measured on both of its jobs:

| job | floor | result |
|---|---|---|
| compression | beat `pass_through` | −0.004 ± 0.006 (`CGE-B-03`) |
| coordination | beat `independent_towers` | **−0.000 ± 0.001** (`CGE-B-10`) |

**Both jobs, both nulls.** The layer is not underperforming; it is doing nothing measurable.

## The critique was right, and it did not matter

The second control came from a review arguing that in cortex, connectivity matters more than
proximity — distant columns can belong to the same functional module while neighbours do
unrelated work. Our membership rule was `(c * m + j) % n_towers`: contiguous, positional, never
measured, and paired with retinotopic tower routing, so "proximity is grouping" was assumed in
two places at once.

Made falsifiable, the claim holds and is negligible:

- **Connectivity beats proximity on all three seeds**, +0.010 ± 0.001. Consistent, reproducible,
  correctly signed.
- **It is five times below the margin.**
- Membership overlap between the two rules is **0.53** — they pick genuinely different towers, so
  this is not a case of co-variation quietly reproducing adjacency.
- The hard-coded positional rule is the **worst** of the three grouping rules on all three seeds,
  slightly worse than random.

So the honest reading: *the critique correctly identified an unjustified assumption in our code,
that assumption was indeed slightly harmful, and correcting it changes nothing that matters.*
Cost of finding out: one gate. Cost of the alternative — rebuilding Layer 2 as a synchrony-formed
Dynamic Assembly — would have been an architecture, and the evidence for it would still have been
+0.010.

## What this does to the freeze

R1 is now decidable rather than blocked. The previous obstacle was that L2's floor was
*unreachable*; it is now *reached and failed*, which is a different and much better situation.

| layer | floor | result |
|---|---|---|
| L0 neuron | beat periodic and random sampling | **PASS** +0.917 ± 0.006 |
| L1 tower | beat trivial memory | **PASS** +0.572 ± 0.059 |
| L2 cluster | beat independent towers **and** fixed proximity | **FAIL** −0.000 / +0.010 |

Q7's option **C** — remove Layer 2 from the stack — was rejected earlier on the grounds that it
would throw away a structure *whose measured contribution is unknown*. It is no longer unknown.
It is zero, on both of the layer's jobs, with tight error bars.

**Corvus is a two-layer architecture that has been carrying a third layer.** The decision is now
the user's, and it is a real one because deleting L2 also deletes the only place cross-tower
entity relations currently live — a structure that has never been measured either, because no
gate targets it.

## Contract change this produced

`Floor` now names its `job` and whether that job is `always_on`, and `Layer` rejects any
declaration whose floors all govern optional jobs:

> *"layer 'x' declares floors only over optional jobs. Whatever this layer does on every tick is
> then unmeasured, which is exactly how DCN v3's node layer stayed compliant while being worse
> than the neurons feeding it."*

A `Floor` that names its own job as its baseline is also rejected outright — that is the tie that
blocked the freeze. Both are construction-time errors now, not documentation.

This is a defect fix under R2 by the operational test: *would you require a floor for an
always-on component if the gate did not exist?* Yes. It is the contract's entire premise, and
the contract failed to enforce its own premise.
