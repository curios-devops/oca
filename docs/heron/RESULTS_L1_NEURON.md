# Level 1 — Dynamic Neuron: results

Reported per level, as the methodology asks. Levels 2 (DCN) and 3 (DRC) are specified in
[Design.md](Design.md) and not yet built — level 1 has to pass first.

Run with `make dcn-l1`. 32 neurons, 608 parameters, driven by a real sensory stream from
the physics world rather than synthetic noise.

---

## Phase 1 — precision and efficiency

The design's distinctive claim at this level is that a neuron should **emit only when its
state changes significantly**. That is only meaningful against a control: any subsampling
of a smooth signal reconstructs reasonably, so the question is not whether send-on-delta
works but whether it beats the *same number of emissions* spent another way. Thresholds
are solved numerically so every policy is compared at an identical event rate.

Reconstruction NRMSE, where 1.000 is what you get by transmitting nothing at all:

| policy | 0.05 | 0.1 | 0.2 | 0.4 | 0.8 |
|---|---|---|---|---|---|
| **send-on-delta** | **0.762** | **0.555** | **0.312** | **0.111** | 0.013 |
| periodic | 1.116 | 1.076 | 0.845 | 0.282 | 0.000 |
| random | 1.147 | 1.060 | 0.919 | 0.659 | 0.232 |

Area under the curve: send-on-delta **0.143**, periodic 0.320, random 0.490.

**PASS — send-on-delta is 55.2% better than the best control at matched rate.** Emitting
on significant change is doing real work, not merely emitting less. The margin is largest
in the sparse regime (0.05–0.2), which is the regime the design cares about.

## Oscillation

With **no input at all**: phase advance 0.285 rad/tick, amplitude 0.853, amplitude drift
0.0000, still moving.

**PASS — the neuron sustains its own rhythm.** Checked directly rather than assumed,
because the legacy line made the opposite mistake three times: a gradient flow provably
cannot oscillate, and that was discovered empirically each time after a full experimental
cycle. The Stuart-Landau form makes it a property of the mathematics.

## Noise robustness

| input noise σ | NRMSE | event rate |
|---|---|---|
| 0.0 | 0.236 | 0.251 |
| 0.1 | 0.231 | 0.254 |
| 0.3 | 0.220 | 0.274 |

Graceful: the adaptive threshold absorbs noise by spending slightly more events rather
than by degrading.

## Ablations — and one mechanism that does not earn its place

| variant | event rate | NRMSE | energy |
|---|---|---|---|
| full | 0.153 | 0.672 | 0.206 |
| **no oscillation** | 0.131 | **0.039** | 0.183 |
| no plasticity | 0.153 | 0.657 | 0.205 |
| fixed threshold | 0.306 | 0.187 | 0.359 |

**Oscillation, as currently wired, actively hurts.** Turning it off improves
reconstruction seventeen-fold (0.039 against 0.672) while *also* emitting less. The reason
is mechanical: the local rhythm is added into the activation, so the neuron spends its
event budget transmitting its own oscillation instead of the signal. The rhythm is real —
the oscillation gate passes — but coupling it additively into the transmitted value makes
the neuron talk about itself.

This does not refute axiom 3; it supports it. The axiom says **waves coordinate
synchronisation, they do not store information**. Adding the wave into the activation is
exactly treating it as content, and the measurement says that is the wrong wiring. Phase
should gate *when* a neuron speaks, not modulate *what* it says.

**Plasticity currently earns nothing** (0.657 against 0.672), which is unsurprising given
axiom 2 — neurons hold no knowledge worth naming — and worth revisiting only if level 2
gives the weights something to do.

**The adaptive threshold trades precision for rate**, holding 0.153 where a fixed
threshold drifts to 0.306. It is doing what it was built for; whether that trade is
favourable is the rate-distortion curve above, and there it is.

---

## Phase 2 — versus the frozen legacy line

The only axis on which these are honestly comparable. A legacy unit publishes its state
every tick, so its event rate is 1.0 and its reconstruction is exact by construction. The
question is not accuracy — that is settled in advance — but what silence costs.

| unit | event rate | NRMSE |
|---|---|---|
| legacy v1 unit | 1.000 | 0.000 (publishes every tick) |
| legacy v2 unit | 1.000 | 0.000 (publishes every tick) |
| dynamic neuron | **0.153** | 0.672 |

**A legacy unit publishes 6.5x more often.** And applying the DN's emission policy to a
legacy unit's own trace shows the exchange rate is favourable on that signal too:

| trace | at rate 0.15 | at rate 0.30 |
|---|---|---|
| legacy v1 unit | NRMSE 0.123 | 0.072 |
| legacy v2 unit | NRMSE 0.277 | 0.122 |

So the emission policy transfers: a legacy v1 unit could have published at 15% of its rate
for a reconstruction error of 0.12. That is a property of the *policy*, independent of the
architecture that produced the signal, and it is the most transferable thing level 1 has
produced.

---

## Verdict

| gate | result |
|---|---|
| precision vs efficiency | **PASS** (+55.2% over the best matched-rate control) |
| sustains its own oscillation | **PASS** |
| noise robustness | PASS (degrades gracefully) |
| oscillation earns its place | **FAIL** — hurts precision 17x as wired |
| plasticity earns its place | inconclusive — no effect at this level |

Level 1 passes the gates it was set. The open item before level 2 is the oscillation
coupling: the rhythm exists and is stable, but feeding it into the transmitted value
contradicts axiom 3 and the measurement agrees. The next change is to make phase gate
*when* a neuron emits rather than modulate *what* it emits — and to re-run this same
battery, which is the point of having it.
