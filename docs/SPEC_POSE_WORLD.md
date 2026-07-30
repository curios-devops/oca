# The pose world — specification

***Level 5 of [the world curriculum](SPEC_WORLD_CURRICULUM.md).*** Written before the
architecture that needs it, because if the world cannot separate "recognises the object from a
new viewpoint" from "matches pixels", nothing built on top of it is measurable.

**It is not the next rung of a ladder.** Levels 2–4 are built and no organism here has passed
them, so the curriculum's own rule — *never advance while the current world is unbeaten* — says
stop. This level is the exception that justifies itself: it is not a harder world, it is **the
first world in which the question is asked fairly.** Everywhere else the raw-input control is
strong, so every architecture has been competing against its own input and losing. Here the raw
control is at chance by construction. Building it is going back to fix the measurement, not
forward to a new challenge.

## Why it is needed

Every world in this project so far is **pose-free**. Objects are blobs that translate; they do
not rotate, they have no sides, and they look the same from everywhere. That is why four
architectures could be compared at all — and it is also the reason two of the twelve EIS
capacities cannot be attempted here, and why the Thousand Brains proposal has nothing to grip.

A theory in which each column learns *"the object at pose X relative to me"* is untestable in a
world with no pose. Worse: in a pose-free world, **matching pixels is a complete solution**, so
any architecture that appears to recognise objects has only demonstrated template matching.

The measurement this world exists to make possible:

> **Does the representation identify the object from a viewpoint it was never trained on?**
> Raw pixels cannot, by construction. That is the first time in this project the raw-input
> control will be at chance on a question we care about — which is what `CGE-A-00` has been
> missing.

## The design, deliberately minimal

The temptation is 3D meshes and a rendering pipeline. Resist it: the smallest world that makes
the distinction is enough, and every extra degree of freedom is a confound to control.

**Objects.** A small set of *rigid 2D shapes*, each defined as a set of features at fixed
positions in its own **object-centred frame**. Four to six kinds. Two of them deliberately
**share every local feature and differ only in arrangement** — that pair is what forbids a
bag-of-features solution, and without it the world is not doing its job.

**Pose.** Each object has an orientation, drawn from `N` discrete rotations (start with `N = 8`,
45° apart). The rendered appearance is the shape's features rotated into the sensor frame.

**Movement is sensor-relative, not object-relative.** The agent's action moves *its sensor over
the object*, changing which part it sees and by how much. This is the whole point: a column
learns the object by moving over it, and the efference copy is what tells it how far it moved.
It is the same efference channel Corvus already broadcasts, on a new referent.

**The held-out condition, which is the world's entire reason for existing.** For every object
kind, a subset of poses is **never shown during training**. Scoring happens only on those. An
architecture that memorised appearances scores at chance; one that holds a pose-indexed model
does not.

## Controls, declared now

| control | why |
|---|---|
| **raw pixels** | must be **at chance on held-out poses** and **well above chance on trained poses**. If it is above chance on held-out poses, the poses are not far enough apart and the world is void — the same validity check that `exp08` applies to occlusion |
| **k-means on the raw patch** | the unsupervised control that beat Heron's concept formation by 14× |
| **bag of features** | the same features, positions discarded. If this passes, the shapes are not distinguishable by arrangement and the two confusable kinds were not built correctly |
| **nearest-template** | explicit memorisation of every trained appearance. The thing an architecture must beat to have generalised rather than stored |

## Gates it unlocks

| gate | currently | with this world |
|---|---|---|
| `CGE-A-00` beats its own input | unpassed, and the raw control is *strong* everywhere | the raw control is **at chance** on held-out poses — the first fair setting |
| identity / invariance (EIS 3) | measured only as "same object, same view" | rotated, occluded, novel instance |
| categorization (EIS 4) | `BLOCKED` — no held-out instances | unblocked: new pose of a known kind |
| voting between assemblies | nothing to vote about | each assembly holds a pose hypothesis; consensus is meaningful |

## What it deliberately does not add

No 3D, no lighting, no textures, no new modality, no second agent. Those unlock EIS 9–11 and
each is a separate project. This world does **one** thing: make pose exist.

## Validity checks, before any architecture is run against it

Written as the world's own tests, in the pattern that has already caught three false positives:

1. Raw pixels **above 0.85** on trained poses — the cue is there and the probe works.
2. Raw pixels **at chance** on held-out poses — the task genuinely requires invariance.
3. Bag-of-features **at chance** on the confusable pair — arrangement is load-bearing.
4. Nearest-template **above chance on trained, at chance on held-out** — memorisation is
   possible and insufficient, which is what makes the held-out condition a test.
5. The efference copy is delivered every tick, and is the **only** signal that says how far the
   sensor moved.

**If check 2 fails, the world is void and no result from it counts.** That check is the whole
specification.
