# Evolution rules

How an architecture in this project is allowed to change, when it is frozen, and what may be
done to the benchmark that judges it.

The model is aeronautical or silicon engineering rather than software iteration: an airframe is
certified, frozen, and superseded — not patched until it passes. The purpose of these rules is
to keep the benchmark independent of the thing it measures.

---

## R0 — Flag before implementing

**A requirement that omits something load-bearing, or that contradicts a measurement already
in the record, is not implemented and then discussed. It is flagged first and resolved
together.** This binds whoever wrote the requirement — architect, agent, or implementer — and
it binds equally when the requirement is well-argued and obviously reasonable.

The two flags that produced this rule are the demonstration:

- A freeze condition that would have tagged an architecture measured on **1 of 13** gates as a
  reference point.
- A rule against tuning-to-pass that, applied literally, would have discarded the project's
  **only** passing result at its first defect.

Both requirements were correct in motive and wrong in a detail that only shows up against the
existing record. Implementing them and discussing afterwards would have put a false reference
in Git and thrown away the one mechanism here that beat its own control.

The flag must name what it costs: which measurement contradicts the requirement, or what
becomes unmeasurable if it ships. *"I disagree"* is not a flag.

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

## R6 — Versioning: generations and builds

The model is a launch vehicle programme. **Each architecture is a ship with a mission — clear
its gates.** It flies, it is repaired between flights, and when the problem turns out to be the
airframe rather than a part, that ship does not fly again and its lessons go into the next one.

The discipline that matters in that analogy is the one R2 already states: **defects are fixed
on the current ship; design changes go into the next ship.** A vehicle is not redesigned to
pass its own static fire.

Three levels of identity, and they are not interchangeable:

| level | form | changes when | example |
|---|---|---|---|
| **generation** | `OCA v<n>` + codename | the *design* changes — a new architecture | `OCA v4 "Corvus"` |
| **build** | `<codename>-v<n>.<b>` | a **defect** is fixed, or a mechanism is replaced before the architecture has ever frozen | `corvus-v4.3`, `jay-v5.2` |
| **freeze tag** | `<codename>-v<n>.<b>-alpha` | every declared floor passes (R1) | `corvus-v4.3-alpha` |

A generation number is **never reused and never retired to make room**. Wren is `OCA v1`
permanently, including in the graveyard — which is why every history file is titled
`# Wren — OCA v1 (RPDU)` and not just `# Wren`.

### A generation is spent at freeze, not at rollout

**A ship that has never flown is not retired — it is redesigned on the pad, and reflown.** Only
a vehicle that has cleared R1 and carries a tag has spent its number.

So replacing a *mechanism* inside an architecture that has never been frozen is a **build**, not
a generation, even when the mechanism is replaced wholesale. Jay's first Layer 1 — feature-at-place
binding — was built, gated and refuted without the architecture ever being tagged; the next
attempt is `jay-v5.2`, not OCA v6.

The alternative burns a generation number on every failed prototype and fills the graveyard with
ships that never left the pad, which destroys the one thing the numbering is for: a permanent,
ordered record of designs that actually flew.

**The distinction is not a loophole, because R2 still binds inside it.** A build may replace a
mechanism, but it may not replace one *in order to pass a gate the previous one failed on its own
terms*. Jay L1 was retired because the property it claimed — pose invariance — was measured
directly and found absent, not because a threshold was missed by a little. What replaces it has
to declare its own floor before it is written, exactly as the first one did.

Retired mechanisms go to `architecture-history/` and stay executable, whether they belonged to a
frozen architecture or to one still on the pad. `CORVUS_L2_CLUSTER.md` and `JAY_L1_BINDING.md`
are both there for the same reason: a refutation that cannot be re-run is a claim.

A build increment requires the R2 sentence in its commit: *would you have made this change if
the gate did not exist?* If the answer is no, it is not a build. It is a new generation.

### Builds so far

| build | what changed | R2 verdict |
|---|---|---|
| `corvus-v4.1` | Layer 1 learned an action→view map | — |
| `corvus-v4.2` | replaced it with additive displacement through a fixed injective projection | **defect** — the learned map was a function that provably does not exist (`\|A\|` = 0.0013 after 14,000 ticks) |
| `corvus-v4.3` | anchor lagged by one tick | **defect** — the observability rule was applied to the wrong tick; 26% of in-tunnel frames were treated as sighted |

`corvus-v4.3` is the build that passes `CGE-A-09`. It is **not tagged `-alpha`**, because R1 is
not satisfied: Layer 0 is unmeasured and Layer 2 is `pass_through`.

Heron carries one retroactive build for the same reason: `heron-v3.2` fixed the node leak
(0.25 → 0.03) after the node was measured decorrelating faster than its own neurons. A genuine
defect — and it **changed no conclusion**, which is what made the retirement a generation
change rather than another build.

### Mirror has no generation number, on purpose

**Mirror is not `OCA v0`.** It is not an architecture in this lineage and was never designed.
It is the zero-parameter control: reassemble the frame from the sensory patches, forecast the
current frame at every horizon, keep nothing. Registered in
[`cge/registry.py`](../cge/registry.py) under the key `raw`.

It was promoted from a caption to a registered entrant for one reason: **as a control it is a
number in a legend that readers skip, and as an entrant it is in the same table, in the same
units, and cannot be skipped.** The promotion immediately earned itself — Mirror comes second
in the maze race with 30 exits, ahead of Swift's 3 and Heron's 0, and decodes out-of-view walls
better than Swift, Heron and Corvus.

Giving it a version number would place it in the lineage, and it is not in the lineage. It is
the floor the lineage has to clear. `CGE-A-00` is exactly that: *beat Mirror on a world you can
see.* Four generations, none has.

---

## Status against these rules, 2026-07-30

| rule | state |
|---|---|
| R0 flag before implementing | in force; three flags raised, all three upheld |
| R1 freeze condition | **met.** L0 +0.917 ± 0.006, L1 +0.572 ± 0.059. L2 failed both its floors and was retired rather than carried |
| R2 defect vs tuning | in force; `corvus-v4.1→v4.4` are all recorded defect fixes, none of which moved a conclusion |
| R3 benchmark integrity | in force; one retraction published (`CGE-A-01` → `DEPRECATED`, superseded by `CGE-A-09`); every gate now declares its OOD condition |
| R4 graveyard audit | done — [`architecture-history/`](../architecture-history/), now including a retired *layer* |
| R5 no L3 yet | in force. Corvus is two layers and L3 is not specified |
| R6 versioning | `corvus-v4.4-alpha` tagged 2026-07-30, the first freeze here. `jay-v5.1` refuted and replaced by `jay-v5.2` on the pad — a build, because Jay has never flown |

### What the tag does and does not mean

`corvus-v4.4-alpha` means **every layer it declares has cleared the floor it declared, before it
was built, with a valid control and three seeds.** It is a reference point.

It does not mean the architecture is good. The same build is **last of five on prediction**
(21.2 × copy-last), **last on spatial memory** (−0.235 against raw pixels), and **at chance on
identity**, where Mirror — which has no state at all — scores highest. `CGE-A-00` is unpassed by
everything including this.

Both statements are true and the tag records only the first. That is the point of freezing on
declared floors rather than on results: a floor is a promise a layer made about itself, and this
is the first architecture here to keep all of them.
