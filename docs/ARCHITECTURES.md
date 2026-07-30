# The architectures

Five entrants share one benchmark. Three are frozen, one is a zero-parameter control, and one
is live.

Registry keys (`v1`, `v2`, `dcn`, `raw`) never change — they are written into every scorecard
and log this project has produced, and renaming them would orphan the record. **Codenames are
what appear in tables, figures and conversation**, because "v1, v2 and the other one" stopped
being usable at three.

Corvids, throughout. The newest architecture is named for the birds that pass
object-permanence tests, which is the problem all three frozen ones failed.

| codename | key | status | mechanism | verdict |
|---|---|---|---|---|
| **Mirror** | `raw` | control | no state at all — the current frame, and nothing else | the floor, and it beats two of the three |
| **Wren** | `v1` | frozen | gradient flow on a learned energy landscape | best of the three; still loses to a two-number memory |
| **Swift** | `v2` | frozen | Stuart–Landau limit-cycle oscillators, phase-gated coupling | best object binding of any version; worse prediction |
| **Heron** | `dcn` | frozen | event-driven neurons into a reservoir with a resonance spectrum | fails 4 of 5 gates; loses to Mirror |
| **Corvus** | `corvus` | **live** | entity beliefs corrected when observable, propagated when not | first floor gate ever passed here (+0.572); 1 of 13 |

Retirement notes for the three frozen versions — goals, gates passed and failed, lessons, and
which principle survives them — are in [`architecture-history/`](../architecture-history/).
The rules governing when a version freezes and what may be changed are in
[EVOLUTION_RULES.md](EVOLUTION_RULES.md).

---

## Mirror — the control, promoted to an entrant

Zero parameters. No memory. At every step it reads what it needs out of the frame in front of
it and forgets. It is registered as an architecture rather than kept as a caption for one
reason: **as a control it is a number in a legend, and as an entrant it is in the same table,
in the same units, and cannot be skipped over.**

It comes second in the maze race with 30 exits, ahead of Swift's 3 and Heron's 0.

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

OCA v4, layers 0–2 built. The smallest architecture in the project: **166,912 parameters**, a
quarter of Wren's. Specifications in [docs/corvus/](corvus/), results in
[RESULTS_CORVUS.md](corvus/RESULTS_CORVUS.md).

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

**What is not yet known.** It has been run on **1 of 13** counting gates. Layer 0's sparse-event
result is inherited from Heron and has never been re-measured without the rotor; Layer 2 is in
`pass_through` and has never been asked to beat its members; every Gate B result is unrun. It is
last in the maze at 0.548 against Mirror's 0.773, and zero maze exits against Wren's 53.

---

## What all three have in common

Different primitives, different dynamics, different aggregation, different learning rules, and
one shared result:

> No learned representation beat its own input at predicting an observable world, and none
> formed persistent state — a two-number memory beat all three at knowing where it was while
> blind.

That is the foundation Corvus starts from. [The full ledger.](WHAT_WE_HAVE_LEARNED.md)
