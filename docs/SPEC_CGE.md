# OCA Cognitive Gates Specification (CGE)

**Minimum viable cognitive tests for the Open Cognitive Architecture**
Version 1 · independently versioned from the architecture · implementation: [`cge/`](../../cge/)

---

## Status of this document

This is **not** part of the [architecture specification](SPEC_OCA_ARCHITECTURE.md). It defines
the testing philosophy of OCA and evolves separately.

Its purpose is to answer one engineering question about a component:

> **Does this behave as a valid cognitive abstraction?**

Not "what score did it get". A gate produces a decision about whether a component may advance
to the next development stage.

This version is written after three complete architectures — Wren, Swift and Heron — passed
every gate they were ever set while all three were being beaten by their own sensory input. The
suite that let that happen is the reason several of the rules below exist, and each one names
the failure it descends from.

---

## 1 · Gates, not benchmarks

A benchmark produces a number and invites comparison. A gate produces an engineering decision.

| verdict | meaning |
|---|---|
| **PASS** | the component demonstrates the property; it may advance |
| **CONDITIONAL PASS** | it demonstrates the property with documented limitations |
| **FAIL** | it does not; redesign before continuing |
| **UNMEASURED** | the gate did not run validly; it says nothing either way |

Continuous scoring is avoided wherever a behavioural criterion is available.

### 1.1 · Why there are four verdicts and not three

**UNMEASURED is the addition, and it is not a soft FAIL.** A gate whose own control failed has
produced no information about the component. Recording that as FAIL manufactures evidence;
recording it as PASS hides a broken harness. Both have happened here:

- An unstandardised ridge probe returned decode errors *above chance* — a fit worse than
  predicting the mean. Read as FAIL, it would have condemned a representation for a bug in the
  probe.
- An apparent mutual information of 0.449 was **entirely** label-frequency bias, and survived
  until a paired shuffled null was added.
- The binding gate cannot be computed for an architecture with fewer units than the world has
  objects. Reported as 0.000 it reads as "binds nothing", which the measurement does not
  support.

`UNMEASURED` never enters a denominator and never averages into a summary. A stage that is 4/5
PASS with one UNMEASURED has been evaluated on four gates, not five. Enforced in
[`cge/outcomes.py`](../../cge/outcomes.py).

### 1.2 · Verdicts must be arguable

A verdict other than a clean PASS **must** carry a reason, and a CONDITIONAL PASS **must** list
the limitations it advances with — otherwise it is a PASS pretending to be careful. Both are
enforced at construction, not by review.

---

## 2 · Philosophy

Gates test **functional behaviour**. Not implementation, not algorithms, not neural networks,
not transformers, not symbolic systems. Any implementation may pass if it satisfies the required
cognitive properties.

### 2.1 · Every gate declares its control

The prompt this specification grew from listed twelve required fields. **"Control" was not one
of them, and it is the most important.** Every retracted claim in this project's history traces
to a missing or mismatched control, so it is mandatory and checked when a gate is registered.

Two corollaries, both learned the hard way:

**Matched capacity and matched budget.** Comparing a 3456-dimensional state with a
96-dimensional pooled vector on the same training frames is not a test of which holds more
information — the wider one is underdetermined and overfits. Measured without matching: a
decode error of **993,925** against a chance of 7.8.

**The control gets the anti-test too.** Corrupt the control the same way you corrupt the
component, or graceful degradation cannot be distinguished from the task simply becoming harder.

### 2.2 · Pass criteria are behavioural, and name a baseline

"Maintains stable identity" is unfalsifiable. "Identity survives occlusion better than
frozen-at-entry" is a gate. **Every criterion must be expressed as beating something named**,
which is what turns a behavioural description into a decision.

### 2.3 · A gate that nothing has ever failed is not a gate

The rule this suite most needed. Every gate in the catalogue records **what has actually failed
it**; one that has never discriminated is marked `EXPERIMENTAL` and does **not** count toward
compliance. Two of the thirteen implemented gates are currently in that state
(`CGE-A-06`, `CGE-B-04`), and the catalogue says so when audited.

### 2.4 · The gate must be posed where it can discriminate

The subtlest rule, and the one that cost the most. Most of this project's worlds are observable
enough that the answer is already in the current frame — so a representation can at best
re-encode its input, "beats raw input" is close to unwinnable, and failing it means very little.

The tunnel world is the exception: inside a covered corridor every frame is byte-identical, so
the raw-input control is at chance **by construction**, measured at 7.02 against a chance of
7.02 over three seeds. That is the only place in the whole battery where the central question is
properly posed.

**A gate specification must state the observability condition under which it discriminates**,
and a gate run outside that condition returns UNMEASURED.

---

## 3 · Gate classes

| class | question | precondition |
|---|---|---|
| **A** | Minimum viability. Does this behave as a cognitive abstraction at all? | none |
| **B** | Architectural validation. How does it compare with Wren, Swift, Heron and its peers? | all Gate A passed |
| **C** | External comparison. How does it compare with LLMs, transformers, HTM, Thousand-Brains implementations? | all Gate B passed |

The ordering is a dependency, not a preference. Comparing two components that both lose to
their own input tells you only which is further behind.

### 3.1 · Gate A contains the floor gates

`CGE-A-00` (beats its own input) and `CGE-A-01` (beats a trivial memory while blind) are the
gates whose absence let three architectures fail unnoticed. They are Gate A because nothing
should be compared to anything else before they pass.

Current status: **`CGE-A-01` is unpassed by every architecture ever built here.** It is the
project's standing open challenge, and `CGE-A-00` is unpassed on every world where it
discriminates.

### 3.2 · Baselines are rebuilt from a seed, never loaded

Gate B compares against frozen architectures. Those are rebuilt from a seed on every run rather
than restored from a checkpoint: it costs seconds, and it means a scorecard can never silently
describe a checkpoint that no longer matches the code.

---

## 4 · Anti-tests

Every gate should carry deliberate attempts to break the abstraction: **noise · delayed
feedback · missing inputs · conflicting information · contradictory memories · synchronisation
failure · timing shifts · partial communication loss · false sensory evidence · unexpected
environmental change.**

The pass is **graceful degradation rather than catastrophic failure.**

Anti-tests have already changed conclusions here, which is the argument for them:

| anti-test | what it caught |
|---|---|
| maze braiding | a DFS-carved maze put walls on even coordinates, so a raw-pixel probe scored **94%** on cells it could not see. The gate was measuring parity, not memory. |
| paired shuffled null | an apparent MI of 0.449 was entirely bias |
| the exact operator as well as the sketch | a negative result could otherwise have been blamed on the approximation |
| normalising by the raw frame at the same horizon | the first version of the horizon gate concluded everything on the bench was a 64-tick model |
| swept channel budget | prevented an operating point from being chosen after seeing the numbers |
| keeping the rejected wiring runnable | the failure it documents stays reproducible instead of surviving only as a paragraph |

**Honest limitation:** of the ten anti-test categories above, only noise, held-out time splits
and swept budgets are implemented. Delay, missing input, conflicting evidence and communication
loss are specified and absent.

---

## 5 · Scalable templates, not per-level families

The same gate is asked of a neuron, a tower and a cluster; only the complexity changes.
Inventing a separate gate family per level produces a suite that cannot compare across levels,
which is most of what a suite is for.

Each gate declares `levels` — every layer the template applies to — and four complexity tiers:

| tier | meaning |
|---|---|
| minimum viable | the smallest configuration where the question is well posed |
| intermediate | the default reported configuration |
| advanced | stress: more objects, longer horizons, harder worlds |
| large-scale | the configuration that tests whether the mechanism survives scale |

Scale has already reversed a result here: a +6.8-point maze gain at one size did not survive
larger N. Tiering is not bureaucracy.

---

## 6 · Multi-modal philosophy

The suite should never depend on vision alone. Target modalities: vision, audio, touch, motor
action, spatial reasoning, temporal reasoning, cross-modal reasoning, sensor fusion.

> **Where this suite actually is.** Vision, a coarse 4×4 touch map, and an efference copy of the
> agent's own action. **There is no audio gate and no implemented cross-modal gate.** "Cognition
> is multimodal" is a stated intention here, not a property of the current battery, and
> `cge.catalogue.audit()` reports it as a limitation rather than leaving it to be discovered.

---

## 7 · Gate registration

Every gate receives a permanent identity. Identities are never reused and never renumbered;
semantics change by incrementing `version`, and a superseded gate is marked `DEPRECATED` rather
than deleted so historical verdicts stay interpretable.

Required fields ([`cge/catalogue.py`](../../cge/catalogue.py)):

| field | |
|---|---|
| `id` · `version` · `title` | permanent identity |
| `gate_class` | A, B or C |
| `status` | PROPOSED · EXPERIMENTAL · STABLE · DEPRECATED |
| `objective` | the engineering question, one sentence |
| `required_property` | what must be demonstrated |
| `anti_property` | what must not be true — the failure it exists to catch |
| **`control`** | **mandatory**; registration fails without it |
| `pass_criterion` | behavioural, naming a baseline |
| `levels` · `modalities` · `cost` | scope |
| **`has_ever_failed`** | what has actually failed it; empty ⇒ cannot be STABLE |
| `anti_tests` · `depends_on` · `implemented_by` · `notes` | |

Registration raises rather than warns: a gate marked STABLE with nothing recorded as having
failed it is rejected at import.

---

## 8 · The current suite

24 gates registered; **13 count toward compliance**, 11 are PROPOSED placeholders.
Run `python -m cge.catalogue` for the live audit.

### Gate A — minimum viability

| id | title | status | what has failed it |
|---|---|---|---|
| `CGE-A-00` | Beats its own input | STABLE | **all three architectures**, every horizon |
| `CGE-A-01` | Beats a trivial memory while blind | STABLE | **all three**; frozen-at-entry 2.26 vs Wren 4.44, Swift 5.10, Heron 8.56 |
| `CGE-A-02` | Communication efficiency at matched rate | STABLE | Heron's first oscillation wiring, 19× worse while emitting more |
| `CGE-A-03` | Timescale monotonicity | STABLE | Heron's tower layer, 0.882 vs its members' 0.907 |
| `CGE-A-04` | Self-sustaining dynamics | STABLE | Wren — a gradient flow provably cannot oscillate |
| `CGE-A-05` | Clock is not content | STABLE | Heron's first wiring |
| `CGE-A-06` | Graceful degradation under noise | EXPERIMENTAL | *nothing yet — does not count* |

### Gate B — architectural validation

| id | title | status | what has failed it |
|---|---|---|---|
| `CGE-B-00` | Spatial memory beyond the current view | STABLE | Swift (73.6%) and Heron (61.9%), both below the 77.3% pixel control |
| `CGE-B-01` | Object identity through occlusion | STABLE | Wren, at chance on an object in **plain view** |
| `CGE-B-02` | Binding: does the grouping carry identity? | STABLE | Heron (+0.009, 14× worse than k-means on pixels); Wren (+0.001) |
| `CGE-B-03` | Aggregation earns its place | STABLE | four operators, four times |
| `CGE-B-04` | Narrow interface sufficiency | EXPERIMENTAL | *nothing yet — does not count* |
| `CGE-B-05` | Prediction against persistence | STABLE | all three at τ=1, and all three against the raw frame |

### Gate C — external comparison

Five gates, all PROPOSED, specified in [SBB.md](../SBB.md) — deliberately written before anyone
knows who would win, which is the only moment that is a fair test. **Entry condition: some
architecture must first pass `CGE-A-00` and `CGE-A-01`.** Until then the comparison has no
interpretation.

---

## 9 · The maze, and what qualifies

The maze worlds were built partly because a benchmark that nobody looks at twice does not get
looked at twice. Two of the three things built on them qualify as gates; one does not, and it is
kept deliberately separate.

**Qualifies — `CGE-B-00`, out-of-view wall decode.** Proper control (the raw retina, which must
decode on-screen cells and is the bar off-screen), a behavioural criterion, and a real anti-test
that changed the result (braiding, which exposed a 94% pixel score on unseen cells).

**Qualifies — `CGE-A-01`, the tunnel.** The best gate in the suite. Its control is at chance by
construction, its bar is a two-number memory, and it is the standing open challenge.

**Does not qualify — the four-way race** (`make race`, [`demo/`](../../demo/)). It is a
**demonstration**, and it is labelled as one:

- Exits reached is a continuous score with no behavioural pass criterion.
- It is a downstream task through a shared planner, so it conflates the planner with the
  representation.
- It is single-seed and visibly noisy — Swift decodes 73.6% of out-of-view walls and reaches the
  exit 3 times, while Mirror decodes 77.3% and reaches it 30 times. The ordering is right, the
  magnitudes are not interpretable.

It stays, because it is the clearest thing in the repository to look at and because Mirror
finishing second with zero parameters is the project's central finding made watchable. It
contributes **nothing** to a compliance decision, and its own page says so.

---

## 10 · Future expansion

New gates may be added without invalidating previous verdicts: identities are permanent,
semantics are versioned, and superseded gates are deprecated rather than removed.

Placeholders registered and not implemented: cross-modal integration · novelty detection ·
consensus formation under disagreement · sensorimotor adaptation · recovery after interrupted
communication · knowledge survives frozen learning. Levels 4–6 have no gates yet.

---

## 11 · What this specification is for

A benchmark suite becomes a leaderboard the moment its authors want to win. The defence is
structural, and it is the whole content of this document:

- every gate names its control, and one without a control cannot be registered;
- every pass criterion names what it beats;
- every gate records what has failed it, and one that has never discriminated does not count;
- a gate that cannot run validly returns UNMEASURED, and UNMEASURED is never a score;
- the floor gates come first, so nothing is compared before it beats something cheap.

Three architectures in this repository satisfied every gate they were set and lost to their own
input. This specification exists so the fourth cannot.
