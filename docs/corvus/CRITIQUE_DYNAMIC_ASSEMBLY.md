# The "Dynamic Cortical Assembly" critique, assessed

*2026-07-30. A review proposing that Layer 2 be redefined as a transient, synchrony-formed
assembly of towers rather than a fixed anatomical cluster, and that Layer 1 be split into
minicolumn / column / hypercolumn.*

**Summary of the assessment: one idea in it is excellent and currently untested in our code.
The rest is a proposal to rebuild two architectures we have already built and measured.**

---

## What the critique gets right, and it matters

**There is no consensus on the cortical column.** This is accurate and well documented — Horton
& Adams (2005), *"The cortical column: a structure without a function"*, is the standard
citation, and the minicolumn / column / hypercolumn / module / patch / domain vocabulary
genuinely does mean different things to different authors. The hypercolumn in particular is a V1
construct (ocular dominance × orientation) and does not generalise across areas.

**The engineering conclusion drawn from it is also right:** do not copy the anatomy, copy the
principle.

But notice where that lands. **If there is no agreed anatomical object, then "copy the anatomy"
was never an option, and the only available discipline is to make each layer declare what it
must beat.** That is what the Corvus contract already does. The absence of consensus is an
argument *for* the method we are using, not for adding levels the neuroscience cannot
adjudicate.

**"Repeat a generalist module massively"** is likewise already our design. Every tower is
identical, generalist, and instantiated seventeen times. No change follows.

---

## The load-bearing objection: the proposed Layer 2 is Swift, and we measured it

The critique proposes:

> *Dynamic Cortical Assembly. Not a fixed object. A set of synchronised columns. It forms,
> processes, dissolves.*

**That is Swift.** Its synchrony coalitions are exactly transient assemblies formed by phase
alignment and dissolved when alignment is lost. It is also the legacy Predictive Assembly
(`docs/wren-swift/SPEC_PA.md`), which specified a shared dynamic workspace and a coalition
manager that holds members together while prediction improves and dissolves them otherwise.

Both were built. Both were measured.

| claim in the critique | what we measured | where |
|---|---|---|
| assemblies form by temporal synchronisation | **confirmed** — Swift's coalitions carry **+0.124** object MI above a shuffled null, 14× Heron's +0.009, 100× Wren's +0.001 | best binding result in the project |
| the assembly is where representation lives | **refuted** — synchrony as a representation to predict from scored **exactly 1.00× persistence**: no content whatsoever | `architecture-history/SWIFT.md` |
| the assembly aggregates its members into something better | **refuted five times** — the Predictive Assembly turned a 0.69× member state into a **6.56×** workspace; Heron's mean-pooling *control* beat both its relational operators; Corvus's cluster summary scores **−0.004 ± 0.006** against its own members | `RESULTS_R1_FREEZE.md` |

So the critique is not proposing something untried. It is proposing the two things we have the
most evidence about, and the evidence splits cleanly: **synchrony groups well and represents
nothing.** Rebuilding it as specified would reproduce both halves.

The interesting question the critique implies but does not ask is the right one: *why did
assemblies bind without representing?* That is answerable with code that already exists, and it
does not require a new level.

## Three L1 levels: no, and R5 says why

Splitting Layer 1 into minicolumn / column / hypercolumn adds two levels of structure to a stack
whose **Layer 2 cannot pass its own floor and whose Layer 1 has passed exactly one gate of
thirteen.** Rule R5 exists for this: no level N+1 until level N clears its floor. Adding levels
*beside* an unpassing one is the same error with a worse cost.

There is a further problem specific to this proposal. The hypercolumn is defined by a property
our world does not have — it organises ocular dominance and orientation columns over a
retinotopic map with two eyes and oriented edge detectors. We have one eye, no orientation
tuning, and a 5×5 view. A hypercolumn here would be a name, not a mechanism.

Our tower already spans a range of integration windows (`tau = 3…40`), which is the functional
content of a minicolumn/column granularity distinction. If that spread turns out to be
insufficient, the fix is a wider spread, not two new layer contracts.

## The critique never addresses `CGE-A-00`

This is the most important omission and it applies to the whole proposal.

**Four architectures have failed to beat their own raw input at predicting an observable
world.** Measured again this week on Corvus: at matched capacity the retina decodes out-of-view
walls better than the towers *and* the cluster on all three seeds.

Every restructuring in the critique happens **above** Layer 1. Reorganising the top of a stack
whose bottom does not beat its own input is precisely the failure mode this project has already
committed three times. If the tower's representation is worth less than the pixels it was
computed from, no assembly of towers can fix that — it inherits the deficit.

## One point actively contradicts our only passing result

> *"Ninguna columna contiene un concepto. El concepto aparece distribuido."*

Wren, Swift and Heron all had fully distributed representations. All three failed persistence,
and all three lost to a two-number memory at knowing where they were while blind.

Corvus passes `CGE-A-09` at **+0.572** precisely because it does the opposite: a **localised,
addressable, persistent** entity belief that is corrected when its referent is observable and
propagated when it is not. Adopting "no unit contains a referent" would undo the single
mechanism in this project that has ever beaten its own control.

Distributed representation is not wrong in general. It is wrong as a *replacement* for
addressable persistence, and that is what the critique proposes.

---

## The one idea worth taking, and it is genuinely good

> *Connectivity matters more than proximity. Distant columns can belong to the same module while
> neighbouring columns do entirely different work.*

**This is a real claim about our code, it is currently assumed rather than tested, and it is
cheap to falsify.**

Two places where we hard-coded proximity:

- `architectures/corvus/cortex.py` routes vision retinotopically: each tower takes a contiguous
  2×2 block of retinal patches, "so that grid adjacency is a spatial neighbourhood rather than an
  arbitrary one". Stated as a design intent, never measured.
- `architectures/corvus/cluster.py` builds membership as `(c * m + j) % n_towers` — **fixed,
  contiguous, proximity-based, and chosen without evidence.**

So the critique has correctly identified an untested assumption sitting exactly where Layer 2's
unmeasured job lives. And it lands on the open question by coincidence:

**Q7 option B asks for a floor on coordination — the job Layer 2 always does and has never been
asked to earn. The critique supplies the second control that floor needs.**

| control | what it tests |
|---|---|
| `independent_towers` | the null: does coordinating buy anything over towers that never relate? |
| **`fixed_proximity_membership`** | **the critique's claim: does connectivity-derived grouping beat the grouping we hard-coded?** |

Coordination must beat both. That makes the strongest claim in the critique falsifiable for the
cost of one gate — no new levels, no architecture change, no rebuild of Swift.

**And it is the correct order.** If connectivity-derived membership beats proximity, *that* is
the evidence that would justify a Dynamic Assembly layer — earned rather than assumed. If it
does not, we have saved ourselves from rebuilding an architecture whose failure mode is already
in the graveyard.

---

## Recommendation

| proposal | verdict |
|---|---|
| Send Corvus to the graveyard | **No.** It holds the project's only two passing gates. Retirement under R2 requires a demonstrated design fault, not a more attractive alternative. |
| Split L1 into minicolumn / column / hypercolumn | **No.** Violates R5; the hypercolumn has no referent in our world; the tower already spans the relevant timescales. |
| Redefine L2 as a synchrony-formed Dynamic Assembly | **Not yet.** This is Swift plus the Predictive Assembly. Both measured: synchrony binds (+0.124) and represents nothing (1.00× persistence). Rebuilding it requires new evidence first. |
| Change L0 | **No.** It passes at +0.917 and nothing in the critique concerns it. |
| **Take "connectivity over proximity" as the second control on the Q7-B coordination floor** | **Yes.** Real, untested, cheap, falsifiable, and it lands exactly on the layer that has never earned its place. |

The critique's own thesis — *copy the principle, not the anatomy* — is the argument against most
of what it proposes. The principle in "dynamic assemblies" is that **grouping should be earned by
function rather than fixed by position.** That principle can be tested in one gate against the
membership rule we already have. Building the anatomy it is usually packaged with is the thing
we should not do, and is what we did in v2.
