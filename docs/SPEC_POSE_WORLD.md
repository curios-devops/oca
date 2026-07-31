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

**Objects.** Four *rigid 2D shapes*, each three identical blobs at fixed (angle, radius) in its
own **object-centred frame**. Blob size and brightness are the same for every kind, so
**arrangement is the only signal**. Kinds 0 and 1 share their angle multiset *and* their radius
multiset, paired differently — a feature bag and a radial histogram are both at chance between
them, and that pair is the world's hardest control.

**Pose.** Six orientations, 60° apart. **This number was chosen by the checks below, not by
taste** — see "what the first configuration got wrong".

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

## Validity — measured, with a real fovea

`python experiments/validate_pose_world.py --ticks 60000 --seeds 0 1 2`

**The world is VALID.** Four kinds, chance 0.250, 620 trained and 130 held-out *presentations*
per seed. The fovea is 32px on a 120px canvas and the object spans ~106px, so **a single glance
takes in under a third of the object's width.**

| control | trained poses | held-out poses |
|---|---|---|
| **perfect integrator** — every frame stitched back at the fovea position it came from | **0.887** | **0.257** |
| episode's frames pooled, no place information | 0.264 | 0.239 |
| linear probe on the pooled frames | 0.311 | 0.278 |
| feature bag | 0.290 | — |
| **chance** | **0.250** | **0.250** |

### The result that matters, and it is about the problem rather than about us

> **Perfect integration is 0.887 on poses it has seen and 0.257 — chance — on poses it has not.**

The integrator is handed the easy half of the problem: it knows exactly where every fragment came
from, so it reconstructs the object completely. It still cannot recognise that object at an
orientation it was never shown.

**Invariance is not a consequence of integration.** Stitching fragments correctly is necessary
and demonstrably not sufficient, and that reframes what a Layer 1 has to supply: not a better way
to accumulate, but whatever it is that survives a rotation once the accumulation is already
perfect.

It also gives the sharpest floor this project has had. **Beat 0.257 on held-out poses and you
have invariance**, because the thing that beats it cannot be integration — integration is already
at the ceiling and the ceiling is chance.

### What the first two configurations got wrong

| fovea | canvas | object | glance covers | integrator, held-out | |
|---|---|---|---|---|---|
| 64 | 96 | 66px | **97%** | — | not a fovea: one glance took in the whole object |
| 32 | 96 | 66px | 48% | 0.195 | better, but the object still nearly fits |
| **32** | **120** | **106px** | **30%** | **0.257** | ✅ |

The 64px sensor is why two Layer 1 mechanisms could not have worked whatever they did: its
per-tick feature was a *global* descriptor that already encoded the arrangement and therefore the
pose. Binding a pose-dependent feature to a canonical place cannot produce pose invariance.

Both times, the world was tuned against **its own declared checks with no architecture in the
loop**, which is what the checks are for. This is the second such adjustment; a third would mean
the design is wrong rather than the parameters.

### The checks

1. **The cue survives a fovea** — a perfect integrator reaches 0.887 on trained poses, so
   fragments do contain the object even though no single one does.
2. **The raw control is at chance on held-out poses.** ✅ 0.257 vs 0.250.
3. **Feature bag at chance across kinds.** ✅ 0.280.
4. Feature bag at chance on the confusable pair — 0.546 mean against 0.500, one of three seeds
   marginal at 0.602. Reported, and **not** part of the validity criterion: the histogram is
   centred on the *fovea* rather than on the object, so it is slightly more than a pure feature
   bag. Essentially at chance, honestly stated.
5. **Efference copy delivered every tick**, and the only signal saying how far the sensor moved.
6. **Memorisation does not transfer** — the gap, 0.829.

**Validity requires 1, 2 and 6 together.** Not 2 alone: a world where *nothing* works also has
its control at chance, and would pass check 2 vacuously. That is the shape of two of this
project's three retracted claims, and the criterion was corrected before the world was accepted.

### The raw control is the best cheap use of the input, not a linear probe

The first pass scored only a linear probe: 0.337 on trained poses. On that number the world
would have been declared unmeasurable. Nearest-template, on the same frames, scores **0.987**.
Choosing the weaker control and calling the world void is the same error as choosing it and
calling an architecture good — so the control is picked on trained poses, and *that* estimator
is then asked to generalise.

### What the first configuration got wrong

Eight poses at 45°, two held out. **VOID:** nearest-template scored 0.366 on held-out poses
against a chance of 0.250 — a foveal fragment at 45° still matches a memorised fragment at the
pose next door. Six at 60° drops it to 0.158. Four at 90° is *worse* again (0.656), so this is
not monotone in coarseness and could only be settled by measuring:

| poses | held out | trained | held-out |
|---|---|---|---|
| 8 @ 45° | 2 | 0.868 | 0.366 ❌ |
| **6 @ 60°** | **1** | **0.916** | **0.199** ✅ |
| 4 @ 90° | 1 | 1.000 | 0.656 ❌ |
| 4 @ 90° | 2 | 0.999 | 0.320 |

The world was tuned against its **own declared checks, with no architecture in the loop.** That
is what the checks are for, and it is the opposite of shaping a gate around a result — R3
forbids the second and requires the first.
