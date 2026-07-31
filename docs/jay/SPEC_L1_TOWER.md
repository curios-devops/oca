# Jay — Layer 1, the Tower

*OCA v5. Written before the code, because the floor has to be declared before the mechanism it
judges exists (R3).*

## Why this layer and no other

Corvus is frozen at `v4.4-alpha`. It cleared every floor it declared and lost Gate B to its own
input. The aggregate-then-vote experiment then located the loss exactly:

| condition | object kind |
|---|---|
| raw pixels | **0.782** |
| one Corvus tower | 0.608 |
| five-tower assembly | 0.598 |
| all nine towers | 0.585 |
| voting between assemblies | 0.600 |

**A spread of 0.023 across conditions differing ninefold in how much retina they cover.** One
tower already reached the ceiling. Nothing above Layer 1 can recover what Layer 1 threw away, so
v5 changes Layer 1 and nothing else. Layer 0 carries across unchanged — it passes `CGE-A-02` at
+0.917, re-measured on its own traces.

## What the tower must become

Corvus's tower held **one anchor**: a projection of what its neurons published, corrected when
the referent was observable and advanced by integrated displacement when it was not. That is a
representation of *where I am*. It passes `CGE-A-09` at +0.572 and it says nothing about *what
is there*.

The Thousand Brains claim is that a column learns **a feature at a location in the object's own
frame**, and accumulates those pairs as it moves. Two things follow that Corvus does not have:

**Binding, not pooling.** The tower stores *which* feature was seen *where*, not an average of
features. Pooling has failed six times in this project; binding a feature to a place has never
been tried here.

**An object-centred frame.** Displacement from the efference copy gives a *self*-centred
position — the mechanism Corvus already passes A-09 with. Subtracting the running centroid of
observed feature mass makes it **object-centred**, and therefore invariant to where the sensor
happened to start.

Rotation invariance does not follow from either, and it is what the pose world requires. It is
supplied at readout: **the pairwise distances between the locations at which features were
seen** are unchanged by rotating the object. The pose world was built so that this is exactly
the quantity that separates its confusable pair — kinds 0 and 1 share their angle multiset and
their radius multiset and differ only in the pairing, so their pairwise distances differ and
nothing weaker separates them.

## The floors, declared now

| job | always on | must beat | margin | why |
|---|---|---|---|---|
| **object identity from movement** | yes | `nearest_template_on_held_out_poses` | 0.05 | The raw control, measured at **0.158** on held-out poses against a chance of 0.250. This is `CGE-A-00` in the first setting where the input cannot simply answer. |
| | | `raw_frame` | 0.05 | The standing floor of the whole project. Five entrants, none has cleared it. |
| **self-localisation** | yes | `no_integration` | 0.05 | Carried from `CGE-A-09`, which Corvus passes at +0.572. A v5 tower that loses this has traded a working mechanism for a hypothesis. |

**The ablation that decides whether the mechanism is the mechanism.** `use_binding=False` keeps
everything — the same features, the same accumulator, the same capacity — and stores the feature
sum instead of the feature-at-location map. If that scores the same, binding is decoration and
this layer is Corvus with extra steps.

Two further controls, because a designed readout needs them:

- `location_shuffled` — the same features bound to *permuted* locations. Destroys arrangement,
  keeps everything else. If this passes, the arrangement is not what is being read.
- `features_only` — the pairwise-distance readout computed over *all* observed locations
  regardless of feature. Tests whether the binding matters or only the geometry of where the
  sensor went.

## What is deliberately not built

**No Layer 2.** R5: not until this one clears its floor. The voting experiment already showed
what a Layer 2 over a lossy Layer 1 is worth.

**No oscillator.** Carried from the Q1 decision: three measurements, no payoff, and Corvus
proved removing it costs −0.006.

**No learned action model.** Corvus's tower v4.1 tried to learn action → view displacement and
reached |A| = 0.0013 after 14,000 ticks, because actions do not move a view-encoding
consistently. Displacement composes additively and is integrated through a fixed injective
projection, needing no supervision. That decision stands.

## Result — REFUTED, and the prior was scored

`architecture-history/JAY_L1_BINDING.md` has the full entry. Headline:

    same object across poses        0.541
    different objects, same pose    0.587
    invariance margin              -0.047     NOT POSE-INVARIANT

Two different objects look more alike than one object at two orientations, and the tower sits at
chance (0.255) even on orientations it was shown. All three declared controls tie with the
mechanism, so the kill criterion below applies as written.

**Why: a reference frame needs an orientation as well as a position, and this tower had only a
position.** The feature code is a random projection of raw patch activations and is not
rotation-covariant, so binning away the rotation of *places* leaves the rotation of *features*
untouched.

| prediction | confidence | outcome |
|---|---|---|
| binding beats its own ablation | 65% | **wrong** — −0.004, a tie |
| beats nearest-template on held-out poses | 40% | **void** — the model is at chance |
| clears `CGE-A-00` | 35% | **wrong** |
| still passes `CGE-A-09` | 80% | not run; the layer was retired first |
| `location_shuffled` at chance | 75% | **right**, and so was everything else |

The 65% on the ablation was the most confident prediction here and the most wrong. Worth
recording: the mechanism I was surest about is the one that tied with doing nothing.

## Honest prior

Recorded before running, and scored above.

| prediction | confidence |
|---|---|
| binding beats its own `use_binding=False` ablation | 65% |
| the tower beats `nearest_template` on held-out poses | **40%** |
| the tower beats the raw frame on held-out poses — i.e. clears `CGE-A-00` | **35%** |
| it still passes `CGE-A-09` self-localisation | 80% |
| `location_shuffled` is at chance, confirming arrangement is what is read | 75% |

**Kill criterion.** If `use_binding=False` scores within the margin of the full tower, the
feature-at-location map is not doing the work, and this layer is retired the way Corvus's
Layer 2 was — immediately, with its numbers, into `architecture-history/`.
