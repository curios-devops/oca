# Dynamic Cortical Network — Architecture v1

## What is established, and what we are proposing

Kept separate on purpose, because the value of this work depends on not confusing the two.

**Reasonably well supported by neuroscience**

- Cortical columns exist as an anatomical organisation, though their exact computational role remains debated.
- The brain oscillates in multiple bands (delta, theta, alpha, beta, gamma).
- Phase synchronisation influences communication between regions.
- Global neuromodulatory systems (dopamine, acetylcholine, noradrenaline, serotonin) shift the functional state of wide networks.

**Our hypothesis** is how to translate those principles into a computational system.
Everything below this line is proposal, not established fact.

---

## Level 1 — Dynamic Neuron (DN)

A neuron stops being a summer. Its responsibilities:

- integrate local signals
- maintain a small internal state
- learn weights
- **oscillate** — it participates in a local rhythm rather than waiting for a tower to synchronise it
- **emit events only when its state changes significantly**

Components: weights, internal potential, activation, local phase, plasticity, energy.

## Level 2 — Dynamic Cortical Node (DCN)

The first cognitive unit. It does not represent a concept; it represents a **dynamic
hypothesis**. Hundreds to thousands of neurons.

- **Knowledge Model** — consolidated knowledge.
- **Working State** — what is happening now.
- **Prediction Engine** — predicts sensory input, its own future state, and its neighbours.
- **Neighbourhood Model** — a probabilistic model of nearby DCNs. It does not know their weights, only their expected behaviour.
- **Oscillation Engine** — not one frequency but a **resonance spectrum** (e.g. gamma 0.91, beta 0.55, alpha 0.18, theta 0.05), so one node participates simultaneously in several conversations at different timescales.
- **State Adapter** — receives the Global State Vector. It does not change knowledge; it changes *how the node thinks*.

**A DCN does not send information.** It publishes only its state: prediction error,
confidence, phase spectrum, activation, novelty, energy. Everything else emerges.

## Level 3 — Dynamic Resonance Cluster (DRC)

Not an anatomical region — a temporary functional structure, defined by **coherence**
rather than position. It appears when many DCNs enter resonance and may last 30 ms, 500 ms
or several seconds, then dissolves.

A DRC does not compute. It coordinates: resolving conflicts between hypotheses,
synchronising phases, amplifying consistent patterns, and creating context — not by
sending messages but by generating a **field** that modifies the DCNs inside it.

## Level 4 — Regional Resonance Field

Coordinates a whole cortex rather than one functional group: all of visual, all of motor. A
region is not designed, it is **observed** — specialisation should emerge because the
clusters inside it repeatedly reduce each other's prediction error, and the measurement is
whether it appears without being asked for.

## Level 5 — Global Dynamic Field

The slowest wave, and the closest thing to an operating system: it sets the mode of the
whole system — waking, consolidating, dreaming, exploring, fatigued. It changes *how* every
level below thinks and never *what* any of them knows. The level-2 State Adapter is the
minimal version of this interface, already built, and currently measuring no effect.

## Level 6 — Global World Model

**There is no world-model database.** The world model is the current synchronised state of
every field — alive, not stored. This is where "what the system believes about objects,
agents, physics and itself" becomes a readable quantity, and the gate is whether it can be
read at all: decoding identity, position and persistence from field state, against the raw
input control that has so far beaten every representation in this project.

## Level 7 — Executive System

Small, deliberately: attention, goal selection, curiosity, planning, task switching, energy
allocation. It never understands the world directly — it allocates. The five scalars a node
publishes were chosen to be exactly what an executive needs to allocate on.

## Outside the system — the Developmental Curriculum Engine

Not part of the brain; part of its childhood. It generates progressively harder worlds,
measures developmental milestones, chooses what to experience next, controls how long the
system sleeps, adjusts exploration against exploitation, and decides when the cortex is
ready for more. A newborn is not shown calculus on day one.

Likely to be built early and out of order, because it is the only upper level that can be
tested against the levels that already exist: the worlds in `core/world/` already form a
difficulty ladder, and choosing among them by measured milestone rather than by hand is
self-contained and falsifiable.

**Levels 4 to 7 and the curriculum engine are one-paragraph definitions with entry
conditions, not specifications.** Writing a detailed spec for level 6 while level 2 fails is
the exact mistake the build order exists to prevent. Full stack map:
[SPEC_DCN_STACK.md](SPEC_DCN_STACK.md).

## Waves

Not one wave but a hierarchy, as slow and fast components coexist in an ECG.

| scale | approximate role |
|---|---|
| **Global** (~0.1–4 Hz) | whole-system mode: waking, sleep, focus, fatigue, exploration — the operating system |
| **Regional** | coordinates an entire cortex, e.g. all of visual cortex |
| **Cluster** | synchronises one functional group; appears and disappears quickly |
| **Local** | inside a DCN, coordinating its neurons |

## The fractal property

Every level obeys the same rules — state → prediction → error → resonance →
synchronisation → learning. Only the spatial and temporal scale changes.

## The most important change from the previous architecture

The physical hierarchy exists; the **computational** hierarchy does not. Processing does
not "climb" level by level as in a deep network. At every instant, multiple resonance
fields interact in parallel and constrain the dynamics of the levels below.

```
Global Dynamic Field
        │
Regional Resonance Fields
        │
Dynamic Resonance Clusters
        │
Dynamic Cortical Nodes
        │
Dynamic Neurons
```

DCNs do not build cognition by themselves; cognition emerges when thousands of them are
temporarily coupled by these fields.

**The candidate central claim: the fundamental unit of computation is neither the neuron
nor the DCN, but multiscale resonance.** Knowledge lives in the DCNs; intelligence emerges
from dynamic synchronisation between them, modulated simultaneously by local, cluster,
regional and global waves.

---

## Methodology — build it like an aircraft

You do not build a whole aeroplane and then see whether it flies. You test each system
separately.

### Phase 1 — component benchmarks

Each module gets its own battery, and results are reported **per level of abstraction**
rather than in one place, so a change that helps one layer and hurts another is visible
immediately.

| component | what is measured |
|---|---|
| Neuron | precision and efficiency |
| DCN | concept formation |
| Resonance | memory association |
| Waves | coordination between modules |
| Sleep | consolidation and noise removal |
| Cortex | complex problem solving |

### Phase 2 — architecture A versus B

Not a single score: a capability radar — continual learning, generalisation, learning
speed, long-term memory, noise robustness, explainability, energy, inference time,
creativity, cross-domain transfer.

If A wins on memory and B on reasoning, A is not discarded. What makes it better is
documented, and we ask whether that mechanism can be transplanted in isolation.

### Phase 3 — comparison against an LLM

Deliberately *not* MMLU-style knowledge tests, where LLMs are already strong. The tests
target where they are weak: continual learning, consolidation, episodic memory, transfer,
discovery, adaptation. Specified in [SBB.md](../SBB.md); not part of this stage.

---

## Build order

**Each level is implemented and gated before the next is started**, and results are reported
per level rather than in one place.

| level | state | spec | results |
|---|---|---|---|
| 1 — Dynamic Neuron | built, passes its gates | [SPEC_L1_NEURON.md](SPEC_L1_NEURON.md) | [RESULTS_L1_NEURON.md](RESULTS_L1_NEURON.md) |
| 2 — Dynamic Cortical Node | built, **fails 4 of 5 gates** | [SPEC_L2_NODE.md](SPEC_L2_NODE.md) | [RESULTS_L2_NODE.md](RESULTS_L2_NODE.md) |
| 3 — Dynamic Resonance Cluster | specified, not started | [SPEC_L3_CLUSTER.md](SPEC_L3_CLUSTER.md) | — |
| 4–7 + curriculum engine | proposal | [SPEC_DCN_STACK.md](SPEC_DCN_STACK.md) | — |

Level 3 is deliberately not started. Level 2 loses to raw pixels at every horizon, loses to
both frozen legacy versions on the shared benchmark, and four of its six mechanisms change
its headline by less than 0.5%. Building a coordination layer over nodes that do not yet
hold anything worth coordinating would be building the aeroplane to find out whether the
wing works.

Two things measured at level 2 changed decisions made on the blank page, and both are
written up in [FIRST_PRINCIPLES_DCN.md](FIRST_PRINCIPLES_DCN.md): the relational-aggregation
constraint is retired, and coalitions-by-synchrony — discarded from the new architecture —
turns out to bind objects fourteen times better than what replaced it.
