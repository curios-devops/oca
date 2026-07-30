# Heron — OCA v3 (Dynamic Cortical Node, DCN)

*Retired 2026-07. Code frozen at [`architectures/heron/`](../architectures/heron/), still
executable, still a registered benchmark entrant.*

Named for a bird that stands still for a very long time and then moves once — the architecture
built around emitting only on significant change.

## Goals

A blank page rather than a patch. Abandon the mesh-of-identical-units premise entirely and
build a layered stack from a different starting point: event-driven neurons that stay silent
unless something changes, feeding a reservoir with a resonance spectrum, publishing five
scalars and nothing else.

## Mechanism

**Layer 1 (neuron).** Leaky integration, **send-on-delta** emission — a unit transmits only
when its activation has drifted past a threshold from the last value it sent — plus a
per-neuron adaptive threshold, local plasticity, and a phase rotor.

**Layer 2 (node).** An echo-state reservoir (spectral radius 0.95) with a four-band resonance
spectrum, aggregating its neurons relationally rather than by mean, publishing five scalars and
a phase spectrum.

**Layer 3 (cluster).** Specified, never built.

## Strengths

- **The most portable result in the project.** Send-on-delta beat the best matched-rate control
  by **+92.9%** — and the policy *transferred*: applied to a frozen Wren unit's own trace it
  reconstructed at NRMSE 0.123 while emitting at 15% of the rate. A result that survives being
  moved to a different architecture is a different class of evidence from one that does not.
- **The five-scalar publication bottleneck costs only 1.50×.** A hard interface limit is
  affordable.
- **Smallest of the three legacy versions** at 293,485 parameters.
- Level 1 passed its gates cleanly — the only layer in the legacy line that did.

## Weaknesses

- **Level 2 fails four of five gates.**
- **Concept formation loses to k-means on the raw input patch by 14×.**
- **Mean pooling beats relational aggregation at matched width** — which retired the strongest
  result the project had produced at the time, on both targets.
- **Nothing over Mirror at any horizon.**
- **Worse than chance in the tunnel maze**, and **−0.232** on path integration: actively worse
  than not integrating at all. The only entrant with a negative that large.
- The phase rotor never paid: −10% for a lone unit, +0.9% under channel contention with the
  sign flipping, <0.5% on ablation.

## Gates

**Passed** — `CGE-A-02` sparse event coding (+92.9%, three seeds, transferred); publication
bottleneck (1.50×).

**Failed** — `CGE-A-00`; concept formation (14× worse than k-means on pixels); relational
aggregation vs mean pooling; horizon advantage over Mirror; tunnel navigation; path integration
(−0.232).

## Why it was replaced

Layer 1 was correct and Layer 2 inverted every claim built on top of it. The stack's whole
argument was that a node abstracts over its neurons; measured, the node **decorrelated faster
than its own neurons** at its declared horizon (0.882 vs 0.907) — the opposite of abstraction,
and **no gate in the battery caught it.** Fixing the leak (0.25 → 0.03) repaired the defect and
**changed no conclusion**, which settled it: the failure was the design, not the tuning.

## Lessons learned

1. **Sparse, change-triggered communication is a genuine architectural constraint**, not an
   efficiency preference. It is the only legacy mechanism carried forward into Corvus by
   requirement.
2. **A layer must be required to beat its own members.** Heron's Layer 2 satisfied its contract
   completely while being worse than the layer below it, because no contract ever asked. This
   is the direct origin of the `Floor` field in the v4 layer contract, where every layer must
   **name what it must beat** before it is built.
3. **Relational aggregation does not generalise.** It won once, spectacularly, and lost to mean
   pooling at matched width on both targets. Matched capacity is not optional.
4. **Broadcast proprioception; do not route it.** Heron delivered the efference copy to exactly
   one node of seventeen, leaving sixteen unable to distinguish *"I moved"* from *"the world
   moved"* — plausibly part of why it scored worse than chance at knowing where it was while
   blind. Proprioception is not localised in the visual field, so it must not be routed like
   something that is. Corvus broadcasts it to every tower.
5. **A gate that cannot discriminate must say so.** `gate_coalitions` reported 0.000 for a
   condition it could not measure — three objects cannot label seventeen nodes. It now returns
   `UNMEASURED` with a reason. Silence and failure must not print the same number.

## Principle worth keeping vs implementation to leave buried

**Keep:** send-on-delta emission (already carried into Corvus L0 as a requirement); the
five-scalar publication bottleneck; a population spanning a **range** of time constants rather
than sharing one, since a single integration window cannot support any hierarchy above it, and
it costs nothing to avoid.

**Bury:** the phase rotor — three measurements, no payoff, one sign flip, and the version that
wired it into the activation path cost 92% of the layer's own result. Corvus has **no
oscillator at any level**; temporal coordination is an optional declared service whose
mechanism is unspecified, and anything claiming it must beat its own ablation first. Also bury
relational aggregation as a default, and the echo-state reservoir as a memory: a low-pass
filter cannot integrate.
