# The pose world's ceiling — how much invariance is available, and to what

*2026-07-31. Run before a third Layer 1 mechanism, to establish whether one would be worth
building. It would not be, and the reason is more useful than the mechanism would have been.*

Each row is handed something no component in this project gets: **the perfectly reconstructed
canvas**, every fovea fragment stitched back at the position it came from. None of them is an
architecture. They exist to separate two very different reasons for a null — *the architecture
is failing at a solvable problem* versus *the problem is not solvable here*.

## 1. The invariant exists

Sorted pairwise distances between the three blobs, from the simulator's own shape definitions:

| kind | pairwise distances (px) |
|---|---|
| 0 | 57.3, 58.2, 81.2 |
| 1 | 52.2, 71.3, 71.9 |
| 2 | 39.8, 50.4, 72.1 |
| 3 | 40.9, 64.3, 69.7 |

**The closest pair of kinds differ by 11.4 px.** Well above pixel noise, rotation-invariant,
translation-invariant. The world is not broken and the design claim it was built on holds: the
kinds are separable by the geometry of their parts and by nothing weaker.

## 2–4. And it is not recoverable from the reconstruction

620 trained and 130 held-out presentations, chance 0.250.

| oracle | trained | **held-out** |
|---|---|---|
| integration only — nearest-template on the stitched canvas | 0.859 | **0.284** |
| + rotation-invariant spectrum — radial average of the power spectrum | 0.359 | **0.250** |
| + part detection — peaks located, sorted pairwise distances | 0.425 | **0.320** |

**The best any oracle achieves on an unseen orientation is 0.320, against a chance of 0.250.**

And the middle row is the sharpest: a descriptor that is rotation-invariant *by construction*,
applied to a perfect reconstruction, scores **exactly chance**. Being invariant is not the same
as being informative — the spectrum throws away the arrangement along with the orientation.

## Why the part oracle is weak, stated rather than hidden

| kind | recovered | true |
|---|---|---|
| 0 | 24.1, 44.8, 56.3 | 57.3, 58.2, 81.2 |
| 1 | 25.9, 45.5, 52.8 | 52.2, 71.3, 71.9 |
| 2 | 23.3, 35.7, 49.6 | 39.8, 50.4, 72.1 |
| 3 | 22.7, 43.4, 50.5 | 40.9, 64.3, 69.7 |

Every recovered distance is roughly **half** the true one, and the four kinds come out far more
alike than they are. **The peak detector is failing, not the task.** Non-max suppression is
landing multiple "parts" on one blob, so the distances it measures are within-blob rather than
between-blob.

That is a fixable defect, and fixing it would be **feature engineering, not architecture**. A
sharper detector would raise 0.320 and would tell us nothing about whether a cortical column can
do this — which is the only question the project is asking.

## What this settles

**A third Layer 1 mechanism is not the missing piece.** Two have been built and refuted, and now
three oracles — each strictly more generous than anything an architecture gets — top out at
0.320. The distance between where we are and a solved problem is not one mechanism; it is
**recovering discrete parts from a partial, noisy reconstruction**, which is a perception problem
we never built and never intended to.

Stated as the sequence:

> The invariant exists. Integration recovers the picture (0.859 on trained poses). Neither
> integration alone, nor a principled invariant descriptor, nor crude part detection recovers the
> invariant from that picture at an unseen orientation.

**Verdict: this architecture line does not produce pose-invariant object understanding**, and
the evidence says the next move is not another Layer 1.

## What is explicitly refused

**Tuning the world until the number rises.** The pose world has been adjusted twice, both times
against its own declared validity checks with no architecture in the loop, and it is now frozen
in that respect. Raising 0.257 by making the task easier would produce a number and destroy the
benchmark, which is R3 and is the whole reason the gates are worth anything. The number stands
as measured.

## The question that is now well posed

Not *"what mechanism should Layer 1 use?"* but:

> **What has to be true of a representation for a rotation-invariant quantity to be recoverable
> from it — given that the picture is already complete and the quantity is already there?**

Thousand Brains answers "reference frames". We have now built two things that call themselves
that — a position, and a position plus an estimated orientation — and neither was one. What a
reference frame has to be, operationally, is the open question, and it is a sharper one than the
project had this morning.
