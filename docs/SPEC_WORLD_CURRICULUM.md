# The world curriculum

*How worlds grow alongside the organism, what each level must contain, and — the part that
decides whether any of it means anything — **what the raw-input control scores in it**.*

---

## The organising idea, and its correction

The proposal that produced this document arrived in two versions, and **the second one corrects
the first**. That correction is the foundation here.

The first version gave each architectural component its own world: a neuron lives in a
bacterium's world, a tower in a worm's. The second rejected it:

> *Una neurona nunca existió sola. Lo que evolucionó fue la capacidad del organismo para modelar
> un mundo cada vez más complejo. El protagonista debe ser el organismo, no el componente.*

**That is right and it is adopted without reservation.** We never say *"now we test a neuron."*
We say *"there is now an organism whose maximum capacity is one neuron's worth."* The neuron,
the tower and the cortex are capacities an organism has, not creatures with their own habitats.

The paired principle is adopted too, and it is the sharpest thing in the proposal:

> **Cognitive complexity must never grow faster than world complexity — and no world may require
> a capacity the organism cannot yet develop.**

That closes a loop with a rule this project already has. **R5** says: no level N+1 until level N
clears its floor. **This says**: no world N+1 until the organism has exhausted world N. Together
they are a ratchet, and neither side can run ahead.

---

## 🚩 The hard critique: the most attractive part is the part to skip

The curriculum's charm is that it starts from the bacterium — light and dark, then gradients,
then objects. It is a beautiful story and **it is not where we are stuck.**

Our blocker is `CGE-A-00`: **no representation this project has built beats its own sensory input
at predicting an observable world.** Five entrants, every horizon. And that failure does not
appear at the photic level or the gradient level. **It first appears at the object level**, which
is where our existing worlds already are — and where we already fail.

Building levels 0 and 1 would be going backwards into worlds our components already handle. L0
passes its floor at +0.917 without them. Detection is not the problem.

**So: adopt the curriculum as the frame, and enter it where we are actually blocked.** Levels 0
and 1 are specified below because a future component may need them, and are not built now.

---

## 🚩 The second hard critique: survival is a pressure, never a measurement

The proposal's engine is *"what new evolutionary pressure forces a capacity the organism did not
need?"* — which is an excellent design question and a catastrophic metric.

Survival is a scalar. This project's entire method is: **probe the representation, against a
control that could kill the claim.** Swap that for "did it survive" and every control we have
disappears — and a reflex agent survives most of these worlds. We would be measuring the policy
and calling it cognition, which is the mistake we have already published a retraction for.

> **Survival shapes what a world demands. A probe against a declared control is what we score.**

Concretely: a world may *require* memory to survive in. What we report is a decode of the
remembered quantity against `frozen-at-entry` — not the survival rate.

## 🚩 The third: this is a curriculum, not evolution

There is no population, no mutation, no selection. We hand-design each level. Calling that
"evolutionary" is the same category error as calling gates "intelligence" — and this project has
a document about that ([EIS.md](EIS.md)).

Darwin is the **inspiration for the ordering**, which is genuinely valuable: each level adds one
pressure, and levels accumulate rather than replace. It is not a mechanism we are running. The
honest name is what the project already used: a **developmental curriculum**.

---

## The rule that makes a level count

Every property in the proposal is a good property. None of them, by itself, makes a world
*measure* anything. One rule does:

> **A world level must declare what the raw sensory input scores in it, and that number must
> leave room for a representation to be worth building.**

If raw pixels solve a level, an architecture that solves it has demonstrated nothing. This is not
theoretical: it is why `CGE-A-00` is unpassed and why `CGE-A-01` had to be retracted. Our worlds
have a strong raw control everywhere, so **every architecture has been competing against its own
input and losing.**

**Level 5 — Pose is the first level in this curriculum where the raw control is at chance by
construction.** That is not an incidental property. It is the reason to build it.

---

## The levels

Each level **contains all previous levels**; they accumulate, never replace. Properties are
toggles, so a level is defined by which ones it enables — that part of the proposal is adopted
verbatim and is good engineering.

| # | level | new pressure | raw-input control | status |
|---|---|---|---|---|
| 0 | **Photic** | a signal exists in time; detect and adapt | trivially solves it | **specified, not built** — not where we are stuck |
| 1 | **Gradient** | space and self-motion; go toward | greedy gradient-following solves it perfectly *unless* the field is noisy enough to require integration over time | **specified, not built** |
| 2 | **Objects** | discrete entities with stable identity | **0.782** on object kind — very strong | **built** (`core/world/identity.py`) |
| 3 | **Persistent** | occlusion; the hidden thing still exists | **at chance while occluded** (0.50) — this control is already correct | **built** (`identity.py`, `maze.py` tunnels) |
| 4 | **Predictive** | things move; anticipation beats reaction | copy-last is very strong; only Wren beats it (0.952) | **built** (`core/world/physics.py`) |
| 5 | **Pose** | the same object from a viewpoint never seen | **at chance on held-out poses, by construction** | **next to build** — [SPEC_POSE_WORLD.md](SPEC_POSE_WORLD.md) |
| 6 | **Relational** | *A on B*, *A inside C* — arrangement outranks features | bag-of-features must be at chance | specified, blocked on 5 |
| 7 | **Multisensory** | modalities that can disagree | — | **BLOCKED**: no audio, no cross-modal gate. A real project, not a level |
| 8 | **Agents** | some objects decide | — | blocked; needs an action space with consequences |
| 9 | **Social** | communication, imitation, coalitions | — | blocked |
| 10 | **Symbolic / open** | — | — | **cut.** *"A human baby enters here"* is an aspiration, not a specification |

### What was cut and why

**Levels 9–10 as scheduled work.** Ten levels of which we cannot pass the second is the R5
violation moved up a layer. They stay as direction, not as plan.

**Multisensory as a level.** `cge.catalogue.audit()` already records that this suite is vision
plus a coarse touch map plus an efference copy, and that *"cognition is multimodal"* is a stated
intention here rather than a property of the battery. Scheduling it as level 7 would hide a
project inside a curriculum row.

**Survival as a score.** See above. It stays as the reason a level is hard.

**Component-to-animal mapping** (*neuron = bacterium, tower = worm*). Charming, and it contradicts
the organism principle the same message established. Kept only as **visual** metaphor, where it
is genuinely good — see below.

---

## The reference world: ground truth vs observed

Adopted, with one note: **we already do this**, and it is worth writing down as a rule because it
is the thing that makes every probe here possible.

| | |
|---|---|
| **Ground truth** | the simulator's full state — position, orientation, kind, relations. `world.surrounding_walls()`, `world.kind[i]`, `world.is_fully_occluded(i)`. **The organism never receives it.** |
| **Observed** | what the sensors deliver — a 5×5 view, a patch grid, an efference copy |

The gap between them **is** the cognitive problem, and every gate in `cge/` is a measurement of
how much of the first is recoverable from a representation built only from the second. The
proposal is right that this is missing from most benchmarks. It is not missing from this one, and
stating it explicitly is how it stays that way.

---

## Visualisation — three.js

Worth building, and worth keeping small. The maze demo already proved which rendering carries the
argument: **the truth and the belief, side by side, in the same units.** Everything else is
decoration.

**The design that follows from the organism principle:**

- **The world** is rendered from ground truth. One scene, real geometry.
- **The organism's belief** is rendered as a second, ghosted overlay in the same space — where it
  thinks things are, with uncertainty as transparency. Divergence between the two is the whole
  show, exactly as the maze demo's belief-vs-truth grid is.
- **The body reflects the capacity, and this is where the animal metaphors belong.** A single
  point of light at level 0; something worm-like at level 1, orienting along a gradient; a
  jointed body with a movable fovea from level 5. The organism visibly grows as the curriculum
  advances, which is the evolutionary story rendered without being claimed as a mechanism.
- **Internal state on demand**: neurons as points that flash when they *emit* — not when they
  activate. Send-on-delta is the one mechanism here that passes cleanly and it is invisible in
  every other rendering.

**Scope discipline.** Self-contained page, no external assets, generated from a recorded trace
exactly as `make_maze_demo.py` does — so the visualisation can never diverge from a run, and
never becomes a second implementation of the architecture.

---

## The open question this curriculum does not answer

If the curriculum's own rule holds — *no world may require a capacity the organism cannot yet
develop* — then we are currently in violation, and honestly so:

**Levels 2, 3 and 4 are built, and no organism we have built has passed them.** Five entrants,
all beaten by their own input. The curriculum says do not advance. `CGE-A-00` says the same.

Level 5 is the exception that justifies itself: it is not a harder world, it is **the first
world where the question is asked fairly**, because the raw control is at chance rather than
strong. Building it is not advancing the curriculum. It is going back to fix the measurement.
