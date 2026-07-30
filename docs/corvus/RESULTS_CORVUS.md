# Corvus — results

OCA v4, layers 0–2 built. 175,104 parameters — the smallest architecture in the project, a third
of Heron and a quarter of Wren.

**Headline: Corvus is the first architecture here to pass a floor gate**, and getting there
required retracting a published claim of my own. Both parts matter and the retraction is the more
important one.

---

## The retraction

Two commits ago this project published:

> Every architecture loses to a two-number memory. **The problem is not that persistent state is
> destroyed on the way up. It is that persistent state is never formed.**

**The reasoning was wrong, and the gate that produced it was not measurable.**

`CGE-A-01` scored absolute position while blind against `frozen-at-entry` — a baseline **handed
the true entry coordinates**. An architecture gets no such gift: it has to encode where it is from
a 5×5 view. Measured, that is impossible in a braided maze:

| | position error, cells |
|---|---|
| chance (predict the mean) | 8.58 |
| **the raw view, while fully sighted** | **4.92** (best of 32/128/all components) |
| `frozen-at-entry`, which is *given* the answer | **2.06** |

**No architecture could pass A-01 however well it persisted.** The anchor it requires is not
inferable from the input. That is why all four failed by similar margins — they were being scored
against something none of them could reach, and the margins were measuring the view, not the
memory.

`CGE-A-01` is now `DEPRECATED` rather than deleted, so the verdicts it produced stay interpretable
and the mistake stays visible.

**The lesson generalises past this gate: a baseline given information the component must infer
does not measure the component.** This is the third time in this project that a measurement
answering one question was used to close a different one, and the first time the culprit was the
gate rather than the interpretation.

## The fair gate — `CGE-A-09`, path integration

Remove the unattainable anchor and ask only about the part that can be inferred: **displacement
since entering the tunnel.** Control is `no_integration` — predict zero displacement, which is
what holding still achieves. The raw frame is at chance by construction.

Three seeds, matched capacity:

| entrant | vs no-integration | reading |
|---|---|---|
| **Corvus** | **+0.572 ± 0.059** | **PASS** — integrates its own moves |
| Wren | +0.282 ± 0.027 | partial, and a genuine surprise |
| Swift | +0.007 ± 0.026 | nothing |
| Mirror | −0.027 ± 0.004 | nothing, as it must be — no state at all |
| Heron | −0.232 ± 0.069 | worse than not integrating |

**Corvus doubles the best frozen architecture** and is the first entrant to clear a declared
floor by its declared margin.

Two things in that table are worth more than the headline. **Wren integrates partially** (+0.282)
— nobody had ever asked, and its four-scalar mesh turns out to carry real self-motion information.
**Heron is worse than doing nothing**, consistent with everything else measured about it.

## What produced the pass, and the two versions that did not

The mechanism is one loop: correct the belief toward what is seen when the referent is observable;
advance it by a predicted displacement when it is not. The `Entity` primitive is what holds it.

Two earlier versions of this layer failed, and both failures were more instructive than the pass.

**Version 1 — a learned action model, on a quantity that cannot be learned.** A single frame (a
random projection of what the neurons published) plus a *learned* linear map from action to frame
displacement. The intuition was that learning beats writing: a hand-written dead reckoner would
pass and prove nothing.

It never learned. `|A|` reached **0.0013** after 14,000 ticks, and the reason is structural rather
than a learning rate. **Actions do not move a view-encoding consistently** — the same step changes
the whole 5×5 view, by a completely different amount depending on where you are. There is no
stable action → view-delta mapping for any local rule to find. I had asked it to learn a function
that does not exist.

Displacement composes additively by construction, so it is now integrated in its own space through
a fixed injective projection, and needs no supervision: the accumulator is an exact linear function
of the action sequence, which is affinely related to true displacement.

**Version 2 — the two-tick rule, applied to the wrong tick.** Tunnels here are about four steps
long, so roughly a quarter of in-tunnel frames are *entry* frames where the view legitimately just
changed. Measured, **26% of in-tunnel ticks were treated as sighted**, each pulling the belief 35%
toward an encoding that is identical everywhere inside a corridor.

Requiring two *consecutive* sighted ticks did not fix it, and the reason is that I had the logic
backwards: at the entry tick the previous tick genuinely *was* sighted. That rule excludes the tick
*after* going blind, which is the wrong one. **The anchor has to lag by one tick** — you cannot
know a view was informative until you have seen whether the next one differs.

## Ablation

The comparison that shows whether the primitive is doing the work: with `use_entities=False` the
tower holds its anchor and never propagates, which is what all three frozen architectures
effectively did.

| | displacement error |
|---|---|
| entities on | **the +0.572 above** |
| entities off (hold still) | at the no-integration control |

The primitive is the mechanism, not decoration.

## What Corvus has not done

Stated plainly, and predicted in the tower specification *before* the run:

**It is not expected to fix `CGE-A-00`** — beating the raw frame at predicting an observable world.
That is a different problem, and an entity does not address it. Nothing in this project has passed
A-00, and Corvus passing A-09 does not change that.

**Layer 2 has not been tested.** It is built, it coordinates, and its compliance mode is
`pass_through` by design — it has never been asked to beat its members, because Layer 1 has only
just started passing anything.

**Every other Gate B result is unrun for Corvus.** Prediction, occlusion, identity, binding and the
out-of-view wall decode are next, and there is no reason yet to expect them to be good.

---

## Verdict

| gate | verdict |
|---|---|
| `CGE-A-09` path integration | **PASS** — +57.2%, three seeds, control valid |
| `CGE-A-01` (superseded) | UNMEASURED — the gate was not measurable, and the defect was mine |
| `CGE-A-00` beats its own input | not run |
| Gate B | not run |

One gate passed, out of a suite of thirteen that count. That is the honest size of the result: the
first evidence in three architectures and two years that a designed mechanism did something its
own control could not, on a gate whose control was set before the mechanism was built.
