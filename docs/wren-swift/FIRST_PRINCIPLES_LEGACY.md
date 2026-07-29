# Synthetic Brain v1 — First Principles (Legacy, frozen)

This is the RPDU line: `architectures/wren` (gradient-flow mesh), `architectures/swift` (oscillator mesh),
`architectures/swift/pa` (Predictive Assembly). It is **frozen**. Nothing in `legacy/` changes again.

It stays for two reasons. It is the historical record behind every number in
[RESULTS.md](RESULTS.md), and it is a benchmark opponent — a new architecture that cannot
beat it has not earned its complexity. [tests/test_frozen_architectures.py](../../tests/test_frozen_architectures.py)
enforces both: bit-reproducibility, the published parameter counts, and the fact that v1
and v2 remain structurally distinguishable.

The point of writing this down is that the next architecture inherits **nothing by
default**. Every idea below has to re-earn its place on the evidence, and the evidence is
in three piles: what held, what was refuted, and what is still open.

---

## The axioms v1 was built on

Stated as they were actually assumed, so they can be judged individually.

1. Prediction is the primitive. A unit maintains a belief about what happens next, not an answer to a question.
2. Learning is local. A unit sees only its own prediction error; no signal crosses a unit boundary.
3. Knowledge lives in the connections — their topology and gains — not only in weights.
4. Communication is explicit message passing between units.
5. Coalitions form by synchronisation, and represent objects.
6. Structure emerges from prediction alone. Objects, permanence and binding need no separate pressure.
7. One primitive, repeated. Every level is the same computation at a different scale.

---

## What held

**Local learning is competitive with backpropagation through time.** The central claim,
and the one that survived everything. A mesh of locally-learning units beat copy-last at
every horizon, and beat a capacity-matched GRU trained with BPTT (16-step MSE 0.0337
against 0.0413) with no global gradients, no replay, no teacher forcing. The advantage
*widened* with horizon — 0.99x at one step, 0.73x at sixteen — which is the signature of a
dynamical model rather than memorisation.

**The locality guarantee is real and enforced.** No error signal crosses a unit boundary,
under test, in both architectures.

**A dynamics must be checked against the function it has to express.** Learned three
times, expensively. A gradient flow has `dE/dt = -‖∇E‖² ≤ 0` and therefore no periodic
orbits, so v1's units could not oscillate, could not synchronise, and could not carry a
moving quantity. A first-order low-pass `w ← (1−α)w + αc` has the *mean* of its input as
its fixed point and cannot represent a running sum. An unbounded accumulator driven by a
non-zero-mean input ramps forever. Each was provable in advance and none was proved in
advance.

**A layer must be scored at the timescale it integrates over.** No function of the mesh
state beats "assume no change" one tick ahead (1.32x persistence). At 64 ticks a 44% gain
is available. The assembly was failing a test that could not be passed.

---

## What was refuted

**Axiom 6 — that structure emerges from prediction alone.** This is the big one, and it
was refuted repeatedly and from every angle. The mesh built no object representation: a
probe decoded object position *worse* from the mesh state than from the raw retina it had
just been fed (16.3 against 7.5 world units). In a world where object identity was made
**computationally necessary** — passers cross an occluder, bouncers reverse inside it, so
the exit side is determined only by which object entered — the mesh sat at chance on the
identity of an object **in plain view** (51.5% against the retina's 75.9%). The
architecture is not failing to invent objects; it is correctly declining to pay for a
latent variable the objective never rewards. A local motion field is cheaper.

**Axiom 5 — that synchrony is the representation.** v2 made coalitions mechanically
possible and they duly formed (55.6% of units against v1's 0.1%). They carried no object
information: mutual information excess +0.004 ± 0.004 over a paired shuffled null. Later,
measured directly as a representation, a synchrony graph scored *exactly* 1.00x
persistence in every multi-object world. Phase behaves like a clock here, not like
content.

**Averaging as an aggregation operator.** A mean never predicts — 0.99–1.00x persistence
wherever it is numerically stable — and confidence weighting does not rescue it. Pairwise
co-activation over the same members reaches 0.79x, and is the only operator that improves
as the world gets harder. **The predictive code is quadratic in member activity, and every
averaging operator is linear.**

**Voting as a route to binding.** Well-differentiated coalitions formed and remained
uncorrelated with objects (+0.004 ± 0.004). Agreement pressure alone has exactly one
attractor: without task pressure the entire population converged on a single vote
(similarity 0.999).

---

## What is still open

**Axiom 4 — whether explicit message passing is the right abstraction.** Never properly
tested. Every v1 ablation landed within 3%, but on a task where a 99-coefficient linear
filter was near-optimal, so the experiment had no power. The design document's own
alternative — that regions *nudge each other into compatible states* rather than
transmitting representations — remains untried.

**The online/offline gap.** Precisely bounded and unexplained. The same local delta rule
on the same pairwise features reaches 0.82x persistence offline and 2.3x online. It is not
the function class (it nearly matches closed-form ridge), not slow convergence (forty
passes change nothing), and not feature scaling (tested; it made things worse). The
remaining suspects are online decorrelation and learning on top of a mesh that is itself
still learning.

**Whether an active loop changes what gets represented.** The most promising direction and
the least resolved. In a maze with an acting agent the mesh appeared to hold out-of-view
structure (+6.8 points over pixels) — and it did not survive the stress battery, reversing
at a larger lattice (−0.130 ± 0.014). In the tunnel maze, where the pixel control is at
chance *by construction*, no mesh does path integration.

**Sleep and offline consolidation.** Specified in the design, never built.

**Higher-order interactions.** No evidence found above second order — a two-layer network
over quadratic features is worse than ridge — but that is one architecture on one world,
not a general result.

---

## The one sentence worth carrying forward

Local prediction produces the **cheapest** latent representation that satisfies the
objective, and objects are never the cheapest. If the next architecture wants entities,
permanence or binding, something other than "predict your own input" has to make them
worth paying for.
