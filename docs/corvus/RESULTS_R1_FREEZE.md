# Corvus — the R1 freeze attempt

*Build `corvus-v4.3`, 2026-07-30. Run to decide whether OCA v4 can be frozen and tagged.*

**Verdict: it cannot.** Layer 0 passes cleanly, Layer 1 passes one gate, Layer 2 fails its own
declared floor — and the reason it fails turns out to be structural rather than numerical, which
is the more useful half of this result.

| layer | gate | result | verdict |
|---|---|---|---|
| **L0** neuron | `CGE-A-02` sparse events | **+0.917 ± 0.006**, worst seed +0.911, margin 0.05 | **PASS** |
| **L1** tower | `CGE-A-09` path integration | +0.572 ± 0.059 | **PASS** |
| **L2** cluster | `CGE-B-03` beat pass-through | **−0.004 ± 0.006**, worst seed −0.009 | **FAIL** |

---

## L0 — the inherited number was not inherited

Rule R1 says a result measured on a predecessor does not transfer. Corvus's Layer 0 is Heron's
Layer 0 **minus the phase rotor**, and Heron's +92.9% was measured with the rotor present.
Keeping that number would have been the same substitution this project has already made three
times.

Re-run on Corvus's own activation traces, three seeds, 8,000 ticks, same physics-world stream:

| | vs the best matched-rate control |
|---|---|
| **Corvus L0** (no rotor) | **+0.917 ± 0.006** [+0.911 +0.918 +0.922] |
| Heron L0 (with rotor) | +0.923 ± 0.005 [+0.918 +0.925 +0.927] |

**Removing the rotor cost −0.006.** Nothing. The Q1 decision — no oscillator at any level until
one beats its own ablation — is now supported by a measurement instead of an argument, and the
+92.9% survives being separated from the mechanism it was measured alongside.

1,152 parameters, as-run event rate 0.134. `experiments/corvus_l0.py`.

## L2 — the fifth aggregation failure, and it is a clean null

The floor was declared in `architectures/corvus/cluster.py` before this gate existed:

> `Floor(beats="pass_through", margin=0.05)` — *a cluster must beat the concatenation of its own
> members before it is permitted to summarise them.*

Out-of-view wall decode in the maze, both readouts from the same run (the cluster does not feed
back into the towers, so the summary is a deterministic linear map of the same tick's
pass-through), everything projected to the summary's own 64 components so the comparison is
about compression rather than width:

| seed | summary | pass-through | towers | retina | Δ |
|---|---|---|---|---|---|
| 0 | 0.684 | 0.693 | 0.694 | **0.707** | −0.009 |
| 1 | 0.676 | 0.673 | 0.672 | **0.684** | +0.003 |
| 2 | 0.662 | 0.668 | 0.669 | **0.671** | −0.006 |

**−0.004 ± 0.006.** Not a weak positive; a null. Compression buys nothing over the members it
replaces. That is now **five** aggregation failures with five operators — the legacy assembly's
workspace, Heron's sketched and exact relational operators against a mean-pooling control, and
now a random linear summary. The pattern has stopped being a surprise.

**And the retina wins every seed.** At matched capacity Mirror decodes out-of-view walls better
than the towers *and* the cluster on all three. This is the first time `CGE-A-00` has been
measured on Corvus, and Corvus fails it like everything before it.

## 🚩 The structural defect this exposed

Layer 2's floor is `beats="pass_through"` with a margin of +0.05. Its default mode **is**
pass-through. A tie is not a win.

> **In its compliance mode, Layer 2 cannot clear its own floor by construction — so rule R1
> ("freeze when every layer clears its own floor") can never be satisfied by this stack,
> however good Layers 0 and 1 become.**

Switching to `summarise` does not rescue it either: that is the −0.004 above. So the layer has
two states, and neither can pass.

This is not a tuning problem and it is not fixed by running longer. It is a defect in how the
layer declared its obligations, and it is pinned by
`tests/test_corvus_layers.py::test_layer_2_cannot_clear_its_own_floor_in_its_default_mode` so
that it cannot be quietly forgotten.

**What the layer actually does, and what nobody declared a floor for.** Layer 2 performs two
jobs. *Compression* — optional, off by default, gated by `CGE-B-03`, and now measured at zero.
*Coordination* — member agreement plus cross-tower entity relations, which runs unconditionally
on every tick in both modes, and **has no floor at all.** That is exactly the failure the Corvus
contract was written to prevent: Heron's Layer 2 satisfied its contract completely while being
worse than the layer below it, because no contract ever asked.

Resolving this is a design decision, not an implementation one, and it is
[open](OPEN_QUESTIONS.md).

## Freeze status

`corvus-v4.3` is **not tagged.** R1 is unsatisfied and the blocking item is L2.

| rule | state |
|---|---|
| L0 clears its floor | ✅ +0.917, three seeds |
| L1 clears its floor | ✅ +0.572, three seeds |
| L2 clears its floor | ❌ unreachable in both of its modes |
| **R1 freeze** | **blocked** |

Two gates passed out of thirteen that count. `CGE-A-00` now measured on Corvus and failed. The
honest reading is that Corvus has one working layer, one layer that does one thing well, and one
layer whose obligations were declared in a way that cannot be met.
