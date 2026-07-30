# Open questions — blocking decisions for OCA v4

> **RESOLVED 2026-07-30.** All five are decided and Q6 has been added and decided. The
> decisions are recorded in [DECISIONS.md](DECISIONS.md); the analysis below is kept unchanged
> because the reasoning is the part worth preserving, and because a decision whose evidence has
> been edited out cannot be revisited honestly.
>
> **Q7 opened 2026-07-30, and it blocks the freeze.** See [Q7](#q7) at the foot of this file:
> Layer 2's declared floor cannot be reached in either of the layer's two modes, so rule R1 can
> never be satisfied. Q3's decision produced it, which is why it belongs here rather than in a
> bug tracker.

Five places where the v4 architecture, as specified, is in tension with what Wren, Swift and
Heron actually measured. Each names the decision, the evidence on both sides, and the options.
**None should be resolved by whoever is writing the code at the time**, which is exactly how
the two corrections in [FIRST_PRINCIPLES_DCN.md](../heron/FIRST_PRINCIPLES_DCN.md) happened.

Q1 and Q2 block implementation. Q3–Q5 block the layers they concern.

---

## Q1 — Does the Oscillation Layer stay a required architectural service? **BLOCKING**

**The spec says:** oscillatory coordination is an architecture-level service, coordinating
timing, attention, synchronisation, communication windows, learning windows and offline
consolidation, across multiple temporal scales.

**Half of that is confirmed.** "Oscillations are not memory containers" is well supported: a
synchrony graph read from Swift's real rotor phases scored *exactly* 1.00× persistence in
every multi-object world. Phase carries no content. Good — the spec is right to say so.

**The other half has three measurements against it and none for it:**

| measurement | setting | result |
|---|---|---|
| phase-gated emission | one unit, Heron L0 | ≈ −10% precision |
| phase staggering under channel contention | 128 units sharing one channel, Heron L1 | **+0.9%**, sign flips across 4 budgets |
| removing the oscillation engine entirely | Heron L1 headline | changed by **< 0.5%** |

The second is the important one. Channel contention is the setting where scheduling *should*
pay off — many senders, one wire. It did not, and it was measured across a swept budget
precisely so the operating point could not be chosen afterwards.

**The tension.** Elevating this to a named service, alongside memory and the sensorimotor loop,
spends the specification's complexity budget on the mechanism with the most consistent record
of no measurable benefit. Every implementation would then be obliged to provide it.

**Options:**

- **(a)** Keep it as a required service. Justification would have to be that the payoff appears
  only at Layer 3+, where cross-region timing matters and nothing has been built. This is a
  bet, and should be labelled as one.
- **(b) [my recommendation]** Demote to an **optional service with a mandatory floor**: any
  implementation that provides oscillation must show it beats the same system with oscillation
  ablated, on some gate, by its declared margin. Keep the "not memory containers" clause, which
  is the part that is evidenced. This preserves the idea, costs nothing, and makes it
  falsifiable rather than assumed.
- **(c)** Remove it from the architecture and move it to the Oscillation Specification as an
  experimental extension.

---

## Q2 — Where does entity persistence live? **BLOCKING**

**The spec does not have this concept at all**, and it is the one thing the evidence says is
missing.

Every layer in v4 is about computation (neuron, tower), coordination (cluster, region, cortex)
or service provision (cognitive systems). None of them is about *a thing that continues to
exist*. The nearest is Layer 1's "reference frame", which is about where things are relative to
the observer — not about *this being the same thing as before*.

**Why this blocks.** In the tunnel maze the raw-input control is at chance by construction
(7.02 cells against a chance of 7.02), so the question is properly posed for the only time in
the whole battery. Result:

| entrant | error while blind |
|---|---|
| **store two numbers, once** | **2.26** |
| Wren (2304-dim state, uncompressed) | 4.44 |
| Swift | 5.10 |
| Mirror (no state) | 7.02 |
| Heron | 8.56 |

Every architecture loses to `remember(r, c)`. And Wren's readout is its *entire* mesh state
with nothing pooled, which rules out compression as the cause. **The problem is not that
persistent state is destroyed on the way up the stack. It is that it is never formed.**

A v4 stack could implement all seven layers perfectly, satisfy every contract in the document,
and still lose to two numbers.

**Options:**

- **(a)** A first-class **Entity** abstraction at Layer 1, alongside the reference frame: an
  addressable record with identity, state, uncertainty and lifetime, whose defining property is
  that it survives its referent becoming unobservable. Towers instantiate, maintain and retire
  entities.
- **(b) [my recommendation]** As (a), but at **Layer 0.5 / as a cross-cutting architectural
  concept** rather than owned by one layer — because if entities live only inside a tower, the
  cluster layer has to re-solve identity when an entity crosses a tower boundary, which is the
  same aggregation problem that has failed three times.
- **(c)** Treat persistence as an implementation concern of `local memory` and add no new
  concept. This is what all three frozen architectures effectively did.

Whichever is chosen, the derived requirement is the same and should be stated in the spec: **a
component's state must be identifiable as being about the same referent after the referent has
been unobservable.** That is testable and no previous contract required it.

---

## Q3 — May a Tower Cluster be a pass-through?

**The spec says:** the cluster "provides local consensus before higher-level integration".

**Consensus is compression, and compression is the most-failed operation here:**

- The legacy Predictive Assembly turned a 0.69× mesh state into a 6.56× workspace. The
  aggregation destroyed the signal — a clean, isolated failure.
- Heron's node layer: **mean pooling beat relational aggregation** at matched width on both
  targets (0.598 vs 0.612 sketched, 0.629 exact). This retired the strongest result the
  project had produced.
- Crossing aggregation against workspace dynamics put all six cells within 5%. Nothing about
  the operator mattered, and nothing beat the raw frame.

**The question:** if consensus is a mandatory responsibility, every implementation must
compress, whether or not compression helps. Should the architecture permit a cluster that
provably adds nothing to pass its towers' outputs through unchanged, and remain compliant?

**Recommendation:** yes, and make it the default. Specify the cluster's floor as
`pass_through` — the cluster must beat the concatenation of its own members before it is
allowed to summarise them. That inverts the burden of proof onto the compression step, which is
where three failures say it belongs.

---

## Q4 — Which memory service is required at maturity level 1?

**The spec names five:** short-term, working, episodic, semantic, procedural.

**We have none.** A two-number memory beats every architecture built here. Specifying five
varieties before one exists is precisely the mistake the build order is designed to prevent —
the same mistake as writing a detailed Layer 6 specification while Layer 1 fails.

**Recommendation:** require exactly one for the first maturity level — the one gate `P0`
actually tests, which is persistent state about a referent that is currently unobservable.
Call it what it is rather than borrowing a psychological label prematurely. Defer the other
four to the Memory Specification, marked as unevidenced.

---

## Q5 — Are the eight global cognitive modes evidenced?

**The spec names eight:** focused, exploratory, learning, planning, dreaming, consolidating,
recovery, idle.

Heron's State Adapter is the minimal version of exactly this interface — it received a global
state vector and changed *how* each tower learned without changing *what* it knew. Removing it
altogether moved the headline by 0.4%.

That is not a refutation. A global mode plausibly only matters once there is consolidation and
dreaming to switch between, and neither has been built. But the eight modes are currently
unevidenced, and the specification presents them as settled.

**Recommendation:** keep the abstraction, mark the mode list as illustrative rather than
normative, and require that any implementation providing modes show that at least one mode
changes at least one measurable behaviour.

---

## The pattern behind all five

Both corrections this project has had to publish have the same shape: **a measurement that
answered one question was used to close a different one.** A synchrony graph's failure as a
*representation* was used to discard synchrony as a *grouping*, where it later proved fourteen
times better than what replaced it. Mean pooling's failure in *one mesh predicting its own
members* became a constraint on a whole architecture, where it is false.

Q1 and Q5 are the same risk running the other way — a mechanism kept because it is attractive,
in the absence of any measurement supporting it. The guard is the floor invariant: state what
each thing has to beat, before building it.

---

<a id="q7"></a>

## Q7 — Layer 2 declared a floor it cannot reach. **BLOCKING THE FREEZE**

*Opened 2026-07-30, by the run in [RESULTS_R1_FREEZE.md](RESULTS_R1_FREEZE.md). Q3's decision
produced it, which is why it belongs here and not in a bug tracker.*

**The situation.** Layer 2 declares `Floor(beats="pass_through", margin=0.05)`. Its default mode
**is** pass-through. A tie is not a win, so in its compliance mode the floor is unreachable by
construction. Switching to `summarise` does not rescue it: measured at **−0.004 ± 0.006** over
three seeds, a clean null and the project's fifth aggregation failure with a fifth operator.

Both of the layer's states fail. Under R1 — *freeze when every layer clears its own floor* —
**OCA v4 can never be frozen**, however good Layers 0 and 1 become. L0 is at +0.917 and L1 at
+0.572, and it makes no difference.

**Why this is not simply a bug.** The layer does two jobs, and only one of them was ever given a
floor:

| job | when it runs | floor | measured |
|---|---|---|---|
| compression (summarise) | optional, **off** by default | `CGE-B-03`, beat pass-through | −0.004 ± 0.006 |
| **coordination** — member agreement, cross-tower entity relations | **every tick, both modes** | **none declared** | never measured |

The always-on job has no floor. That is precisely the failure the Corvus contract exists to
prevent — Heron's Layer 2 satisfied its contract completely while being worse than the layer
below it, because no contract ever asked. We have reproduced the failure at the level of the
*contract* rather than the implementation.

### The options

**A — a layer in its compliance mode is exempt from its floor.** Freeze is permitted; Layer 2 is
tagged "no abstraction claimed". *Cost:* a layer can coast indefinitely, which is the Heron
failure with a new name. It also makes R1 weaker than it reads.

**B — declare a floor for coordination, because it is the part that always runs.** The natural
control is `independent_towers`: the same entities, with each tower's entities treated as
unrelated. If relating entities across tower boundaries buys nothing over towers that never
relate, coordination is decoration. Layer 2 then earns its place through the job it actually
does, and can clear R1 in pass-through mode. *Cost:* the contract must carry more than one floor
per layer, and a new gate must be written and, under R3, declared before the mechanism it judges
— which is satisfiable here only because the mechanism already exists and the control does not
yet. That ordering needs stating explicitly if B is chosen.

**C — delete Layer 2 for now.** Two layers, both passing, frozen honestly. Re-introduce a cluster
when there is a measured job for it. *Cost:* the stack loses the only place cross-tower identity
can live, and Q2's decision put `Entity` at cross-cutting scope partly to make that possible.

**Recommendation: B.** It is the option that keeps the rule that has already caught two
architectures — every component names what it must beat — and it is a defect fix under R2 by the
operational test: *would you declare a floor for an always-on component if the gate did not
exist?* Yes. That is the contract's whole premise.

A is a rule change to accommodate a component. C throws away a structure whose measured
contribution is unknown, which is the one thing this project should never do on a guess.
