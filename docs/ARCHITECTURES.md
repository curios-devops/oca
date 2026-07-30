# The architectures

Five entrants share one benchmark. **Four are frozen** and one is a zero-parameter control.

Registry keys (`v1`, `v2`, `dcn`, `raw`) never change — they are written into every scorecard
and log this project has produced, and renaming them would orphan the record. **Codenames are
what appear in tables, figures and conversation**, because "v1, v2 and the other one" stopped
being usable at three.

Corvids, throughout. The newest architecture is named for the birds that pass
object-permanence tests, which is the problem all three frozen ones failed.

| designation | key | status | mechanism | verdict |
|---|---|---|---|---|
| **Mirror** — *no version* | `raw` | control | no state at all — the current frame, and nothing else | the floor, and it beats three of the four |
| **Wren** — OCA v1 | `v1` | frozen | gradient flow on a learned energy landscape | best of the three; still loses to a two-number memory |
| **Swift** — OCA v2 | `v2` | frozen | Stuart–Landau limit-cycle oscillators, phase-gated coupling | best object binding of any version; worse prediction |
| **Heron** — OCA v3 | `dcn` | frozen at `v3.2` | event-driven neurons into a reservoir with a resonance spectrum | fails 4 of 5 gates; loses to Mirror |
| **Corvus** — OCA v4 | `corvus` | **frozen** `v4.4-alpha` | entity beliefs corrected when observable, propagated when not | first architecture here to clear every floor it declared — and last of five on prediction |

Generations, builds and freeze tags are defined in
[EVOLUTION_RULES.md](EVOLUTION_RULES.md#r6). A generation number is permanent: Wren is OCA v1
in the graveyard exactly as it was when live.

Retirement notes for the three frozen versions — goals, gates passed and failed, lessons, and
which principle survives them — are in [`architecture-history/`](../architecture-history/).
The rules governing when a version freezes and what may be changed are in
[EVOLUTION_RULES.md](EVOLUTION_RULES.md).

---

## Mirror — the control, promoted to an entrant

**Where it came from.** Nobody designed Mirror. It is the *raw-frame control* that every
experiment in this project has printed next to its own result since the first one — the "retina
control", key `raw`. Twenty lines in [`cge/registry.py`](../cge/registry.py): reassemble the
frame from the sensory patches, forecast the current frame at every horizon, keep nothing.
Zero parameters, no memory, no learning.

**Why it is in the table.** It was promoted from a caption to a registered entrant for one
reason: **as a control it is a number in a legend that readers skip, and as an entrant it is in
the same table, in the same units, and cannot be skipped.** The promotion earned itself
immediately — Mirror comes second in the maze race with 30 exits, ahead of Swift's 3 and
Heron's 0, and decodes out-of-view walls at 0.773 against Swift's 0.736, Heron's 0.619 and
Corvus's 0.548. It beats three of the four designed architectures at their own task.

**It has no generation number and never will.** It is not `OCA v0`; it is not in the lineage.
It is the floor the lineage has to clear, and `CGE-A-00` is exactly that gate — *beat Mirror on
a world you can see.* Four generations, none has.

Anything that does not beat Mirror is not adding a representation. It is adding latency.

## Wren — `architectures/wren/`

Every unit holds a 16-dimensional state descending its own energy
`E(h) = -½h'UU'h - b·h + ¼‖h‖⁴`. Inputs *tilt* the landscape rather than overwriting the
state, so a strong input annihilates a valley and flips the unit's interpretation while a
weak one leaves it hysteretically in place. Five predictions at three horizons; four-scalar
messages; local learning only, with no error signal crossing a unit boundary.

**What it established.** Local backprop-free learning is genuinely competitive with a
capacity-matched BPTT GRU. Best out-of-view wall decode of any version (84.1% against
Mirror's 77.3%) and the only entrant that finds the maze exit with any regularity.

**What refuted it.** A gradient flow has `dE/dt = -‖∇E‖² ≤ 0` and therefore no periodic
orbits — it provably cannot oscillate, synchronise, or carry a moving quantity. Object
permanence never emerged. Its coalitions carry essentially no object identity (+0.001).

## Swift — `architectures/swift/`

Wren's dynamics replaced by Stuart–Landau limit cycles, because the impossibility above was a
theorem rather than a tuning problem. Amplitude settles while phase keeps advancing, so
synchronisation becomes expressible.

**What it established.** The best object binding any version here has produced: its synchrony
coalitions carry **+0.124** object MI above a shuffled null, fourteen times Heron's +0.009.
That result reverses a decision made against it — see
[FIRST_PRINCIPLES_DCN.md](heron/FIRST_PRINCIPLES_DCN.md).

**What refuted it.** Synchrony as a *representation to predict from* scored exactly 1.00×
persistence, carrying no content. Worse prediction than Wren, worse maze decode than Mirror,
and 3 exits against Wren's 53.

## Heron — `architectures/heron/`

Frozen as of this commit. A blank-page architecture: event-driven neurons that emit only on
significant change, feeding a reservoir with a four-band resonance spectrum, publishing five
scalars and a phase spectrum and nothing else. Levels 1 and 2 built; level 3 specified and
never started.

**What it established.** Level 1 passes cleanly and produced the most portable result in the
project: send-on-delta emission beats the best matched-rate control by **92.9%**, and the
policy transfers to a frozen Wren unit's own trace (NRMSE 0.123 at 15% of its rate). The
five-scalar publication bottleneck costs only 1.50×.

**What refuted it.** Level 2 fails four of five gates. Concept formation loses to k-means on
the raw input patch by 14×. Mean pooling beats relational aggregation at matched width — which
retired the strongest result the whole project had produced. Nothing over Mirror at any
horizon. And in the tunnel maze it scores *worse than chance*.

Full detail: [SPEC_L2_NODE.md](heron/SPEC_L2_NODE.md),
[RESULTS_L2_NODE.md](heron/RESULTS_L2_NODE.md).

## Corvus — `architectures/corvus/`

**Frozen as `corvus-v4.4-alpha`, 2026-07-30 — the first freeze in this project.** Two layers,
**158,208 parameters**, a quarter of Wren's. Specifications in [docs/corvus/](corvus/); results in
[RESULTS_CORVUS.md](corvus/RESULTS_CORVUS.md), [RESULTS_R1_FREEZE.md](corvus/RESULTS_R1_FREEZE.md)
and [RESULTS_COORDINATION.md](corvus/RESULTS_COORDINATION.md).

It inherits exactly one thing in code: the layer contract, which is the third version of that
file and the first written after knowing what the previous two failed to require. Wren, Swift
and Heron all satisfied their contracts completely. Corvus's contract additionally demands
that every layer **name what it must beat** — because the reason three architectures failed
without any gate objecting is that no contract ever asked.

**What it established.** The first floor gate this project has passed: **+0.572 ± 0.059** on
path integration across three seeds, double the best frozen architecture, with the mechanism's
own ablation at the control. One mechanism does it — correct the belief toward what is seen
when the referent is observable, advance it by a predicted displacement when it is not — and
observability is *inferred*, never told. Getting there required retracting a published claim of
this project's own; that retraction is the more important half of the result.

**Both layers clear the floors they declared**, which is why it froze: L0 at **+0.917 ± 0.006**
on sparse event coding — re-measured on its own traces, not inherited, and removing Heron's rotor
cost −0.006 — and L1 at **+0.572 ± 0.059**.

**It had a Layer 2 and no longer does.** Both of the cluster's jobs measured as nulls: compression
−0.004 ± 0.006 against passing its members through, coordination −0.000 ± 0.001 against not
grouping them at all. Removing it changed neither headline to four decimal places, which is the
cleanest possible confirmation that it was inert. Retired to
[CORVUS_L2_CLUSTER.md](../architecture-history/CORVUS_L2_CLUSTER.md).

**And it is the worst entrant on Gate B.** Last of five on prediction (21.2× copy-last), last on
spatial memory (−0.235 against raw pixels), at chance on identity where Mirror scores highest.
Corvus is specialised for knowing where it is while blind and pays for it at forecasting what it
can see. Frozen means *cleared every floor it declared*, not *good*.

---

## What all three have in common

Different primitives, different dynamics, different aggregation, different learning rules, and
one shared result:

> No learned representation beat its own input at predicting an observable world, and none
> formed persistent state — a two-number memory beat all three at knowing where it was while
> blind.

That is the foundation Corvus starts from. [The full ledger.](WHAT_WE_HAVE_LEARNED.md)
