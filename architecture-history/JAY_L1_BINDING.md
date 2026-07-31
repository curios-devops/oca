# Jay Layer 1 — feature-at-place binding

*Built, gated and refuted on 2026-07-31, in one session. Code kept executable at
[`architectures/jay/retired/tower_binding.py`](../architectures/jay/retired/tower_binding.py).*

**The fastest retirement in this project, and that is the point.** The world existed to test it,
the floors were declared before the code, and the diagnostic that killed it needs no probe at
all. Corvus took three architectures and two years to reach a comparable verdict.

## Goal

Corvus's tower held one thing: *where am I*. It passes `CGE-A-09` at +0.572 and the
aggregate-then-vote experiment showed what that cost — one tower and nine towers both scored
0.60 on object kind against 0.782 in the pixels, so the information was gone at Layer 1 and
nothing above could recover it.

This tower was to hold *what is where*: **bind a feature to a place in an object-centred frame**,
accumulate those pairs as the fovea moves, and read out by the distance between occupied places
— which is unchanged when the object turns.

## Mechanism

1. **Displacement** integrated from the efference copy through the body's own action→step map.
2. **An object-centred frame**: subtract the running centroid of observed feature mass, making
   the frame invariant to where the sensor started.
3. **A feature-at-place map** `M[f, p]`, read out as `R[f, g, b]` — how much features `f` and `g`
   co-occurred at places separated by distance bin `b`.

21,040 parameters in the tower. Nothing learned above Layer 0.

## Gates

**Failed all of them, and the diagnostic is more damning than the accuracy.**

| | measured | chance |
|---|---|---|
| **same object across poses** | **0.541** | — |
| **different objects, same pose** | **0.587** | — |
| **invariance margin** | **−0.047** | must be > 0 |

**Two different objects look more alike than one object at two orientations.** The readout is
more sensitive to pose than to identity, which is exactly backwards, and no probe fitted on top
of it can repair that.

| condition | trained poses | held-out poses |
|---|---|---|
| binding | 0.255 | **0.209** |
| `use_binding=False` | 0.245 | 0.212 |
| `shuffle_locations` | 0.254 | 0.205 |
| `features_only` | 0.241 | 0.259 |
| raw frame *(control)* | 0.442 | 0.113 |
| nearest-template *(control)* | 0.925 | 0.043 |
| **chance** | | **0.250** |

**At chance on orientations it was shown**, not merely on ones it was not. All three declared
controls tie with the mechanism — −0.004, +0.004, −0.050 — so the kill criterion in
[SPEC_L1_TOWER.md](../docs/jay/SPEC_L1_TOWER.md) applies exactly as written.

The floors also read `+0.153` and `+0.083` against their controls and are **void**, because both
controls score *below chance* on held-out poses and a model at chance beats them arithmetically.
That clause was added to the gate after its first run reported PASS for exactly that reason.

## Why it failed

**Structural, not a bug.** The feature code is a random projection of raw patch activations and
is **not rotation-covariant**. Binning away the rotation of *places* leaves the rotation of
*features* untouched, so the (feature, place) pairs a rotated object produces are not the rotated
pairs of the original — they are different pairs entirely. Distance-binning removes a global
rotation of the place field; it cannot remove a rotation that has already changed what each
feature *is*.

> **A reference frame needs an orientation as well as a position. This tower had only a
> position.**

That sentence is the whole finding, and it is the thing the Thousand Brains literature means by
"reference frame" that we had read as "location".

## Four defects found on the way, and three were the measurement's

Recorded because the pattern matters more than this layer does.

1. **The gate declared PASS with the model at chance.** Model 0.254, chance 0.250, PASS on both
   floors — because the controls score *below* chance and a representation that knows nothing
   beats them arithmetically. The `CGE-A-01` shape, again. Floors now require above-chance first.
2. **The "position" was not a position.** A random projection of a one-hot efference does not
   make up and down cancel. Fine in Corvus, where `CGE-A-09` *decodes* displacement with a probe;
   fatal when displacement is an *input* to an object model.
3. **Horizontal movement was never registered at all**, through an index error in reading the
   somatic channel. Every variant sat at chance on trained poses too, and that run would have
   been published as a refutation of binding.
4. **The readout was documented as distances and implemented as a Gram matrix over place-cell
   identities** — which is not pose-invariant at all.

Only after all four were fixed did the mechanism get a fair hearing, and it still failed. **Three
of the four defects would each have produced the same headline for the wrong reason.**

## Principle worth keeping vs implementation to leave buried

**Keep:** the invariance diagnostic. Comparing readouts directly — *does the same object at two
poses look more alike than two objects at one pose?* — answers in seconds what an accuracy number
takes an hour to confound, and it is a property of the representation rather than of whatever
probe was fitted to it. It belongs in every future gate about invariance.

**Keep:** binding a feature to a place, as an idea. It has still never had a fair test, because
this implementation never produced a stable feature to bind.

**Bury:** a fixed random feature encoder in any architecture that must be invariant to a
transformation of its input. Whatever the next Layer 1 is, its features must be covariant with
the transformation the task requires invariance to — or the invariance must be learned, which the
pose world newly makes possible: the fovea moves over a *static* object, so view change is a
consistent function of action here in a way it never was in the maze, where Corvus's v4.1 reached
|A| = 0.0013 trying to learn exactly that.
