# Evolution rules

How an architecture in this project is allowed to change, when it is frozen, and what may be
done to the benchmark that judges it.

The model is aeronautical or silicon engineering rather than software iteration: an airframe is
certified, frozen, and superseded — not patched until it passes. The purpose of these rules is
to keep the benchmark independent of the thing it measures.

---

## R1 — Freeze on floor, not on intention

An architecture is frozen and tagged when **every layer it declares has passed its own
`Floor`** — the target that layer named, in its own source, *before it was built* — with a
valid control and at least three seeds.

Two clarifications, both learned the hard way:

**Compliance is not a floor.** A layer in `pass_through` mode satisfies the contract by doing
no harm. "Does no harm" is not "works". A pass-through layer does not count toward the freeze
condition.

**A result measured on a predecessor does not transfer.** If v4's Layer 0 is v3's Layer 0 minus
a component, v3's number is not v4's number. Re-run it.

Once frozen: tagged in Git (`<codename>-v<n>-alpha`), never modified, still executable, still a
registered benchmark entrant. Freezing means *no longer changed*, not *no longer run* —
enforced by `tests/test_frozen_architectures.py`.

## R2 — Never tune to pass a gate. Always fix a defect.

The rule as first stated was *"never modify an architecture to pass an already-defined gate; if
it fails, design another one."* The motive is right and absolute: **a model continuously
adjusted to pass the exam turns the benchmark into a mirror.**

Taken literally it also forbids fixing bugs, and applying it literally would have destroyed
this project's only passing result. Corvus's Layer 1 went through three versions, all *after*
`CGE-A-09` existed: v1 tried to learn an action→view map that does not exist; v2 applied its
observability rule to the wrong tick; v3 passes. Under the literal rule, Corvus is discarded at
v1 and the project has zero passes.

So the distinction is **defect vs. tuning**, and it has an operational test:

> **Would you have made this change if the gate did not exist?**
> **Yes** → it is a defect. Fix it, and say so in the commit.
> **No** → you are studying for the exam. Design another architecture.

A learned map for a function that provably does not exist is a defect. A rule applied to the
wrong tick is a defect. A learning rate raised until the number clears the margin is not.

**Any change invalidates every result the changed component produced.** Re-run the suite; never
patch one number. If the change is a defect fix and the conclusions do not move, say that too —
Heron's leak fix repaired a real defect and **changed no conclusion**, and that is what settled
its retirement.

## R3 — The benchmark is frozen too, and it has been wrong

R2 is only safe if the benchmark is correct. Ours has not always been:

- **`CGE-A-01` was not measurable.** It scored absolute position while blind against a baseline
  **handed the true entry coordinates**. Position from the raw view while fully sighted is 4.92
  cells against that baseline's 2.06 — no architecture could ever have passed it.
- **The horizon gate was confounded.** Persistence degrades fastest, so *every* representation,
  raw pixels included, looks better at longer tau.
- **The contention gate starved high-index neurons** through a first-come drop rule.

So the mirror rule:

> A gate may be changed **only** by publishing a retraction, and the change **invalidates every
> verdict that gate produced.** A retired gate is marked `DEPRECATED`, never deleted, so the
> verdicts stay interpretable and the mistake stays visible.

And the constraint that prevents the same failure prospectively:

> **A gate's control must be declared before the mechanism it will judge is built**, and a gate
> may not be added after inspecting an architecture's state. Otherwise the gates take the shape
> of whatever happens to exist.

Two supporting requirements already enforced in `cge/`: every gate carries a mandatory
`control`, and a gate marked `STABLE` with an empty `has_ever_failed` raises — **a gate that has
never failed anything has not been shown to measure anything.**

## R4 — Audit the graveyard before designing the successor

Before a new version is specified, every retired version is asked one question:
**which properties appeared here that do not exist today, and what principle produced them?**
Never *"should we copy it?"*.

The answer goes in [`architecture-history/`](../architecture-history/), split explicitly into
*principle worth keeping* and *implementation to leave buried*. Swift lost nearly every gate it
entered and still holds the project's best object-binding result by 14×. That principle
survives; its 918,832-parameter substrate does not.

Corollary, learned from finding Wren's +0.282 path integration two versions late: **every new
gate is run against every frozen version, not only the live one.** Retired architectures keep
answering new questions.

## R5 — No level N+1 until level N has passed its floor

Level 3 (functional region) is not specified until Layers 0, 1 and 2 have each cleared R1.

The question *"what emerges when several clusters cooperate?"* is not answerable while the
cluster has never been asked to beat its members. Asking it early does not produce an early
answer; it produces an unfalsifiable one.

---

## Status against these rules, 2026-07-30

| rule | state |
|---|---|
| R1 freeze condition | **not met.** Corvus L0 unmeasured (A-02 never re-run without the rotor); L1 passed 1 of 13; L2 is `pass_through` — compliance, not floor |
| R2 defect vs tuning | in force; Corvus L1 v1→v3 recorded as defect fixes |
| R3 benchmark integrity | in force; one retraction published (`CGE-A-01` → `DEPRECATED`, superseded by `CGE-A-09`) |
| R4 graveyard audit | done — [`architecture-history/`](../architecture-history/) |
| R5 no L3 yet | **blocked by R1**, correctly |

**Two runs stand between the current state and a legitimate freeze:** `CGE-A-02` against
Corvus's own Layer 0, and a declared floor for Layer 2 that is not `pass_through`.
