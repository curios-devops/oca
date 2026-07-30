# Wren — OCA v1 (RPDU)

*Retired 2026-07. Code frozen at [`architectures/wren/`](../architectures/wren/), still
executable, still a registered benchmark entrant.*

Named for a small bird that builds many nests and uses one.

## Goals

Show that a mesh of small, autonomous predictive units, learning **only from local signals**,
could match a backpropagation-trained recurrent network — and that intelligence-like behaviour
would follow from the interaction of the units rather than from any one of them.

## Mechanism

Each unit holds a 16-dimensional state descending its own energy landscape

```
E(h) = -½ h'UU'h - b·h + ¼‖h‖⁴
```

Inputs **tilt the landscape** instead of overwriting the state, so a strong input can
annihilate a valley and flip the unit's interpretation, while a weak one leaves it
hysteretically where it was. Five predictions at three horizons; four-scalar messages between
units; no error signal ever crosses a unit boundary.

## Strengths

- **Local learning is competitive.** Beat or matched a capacity-matched BPTT-trained GRU. This
  is the project's most robust positive result and it has never been contradicted.
- **Best out-of-view wall decode of any version: 0.841**, against the retina's 0.773. The mesh
  state genuinely contains structure the current frame does not.
- **The only entrant that navigates.** 53 maze exits in 900 steps, against Mirror's 30,
  Swift's 3, and zero for both Heron and Corvus.
- **Partial path integration: +0.282** — discovered a year late, because nobody thought to ask
  a gradient-flow architecture whether it knew where it was.

## Weaknesses

- **No object permanence, ever.** Occluded objects were not tracked.
- **Coalitions carry no identity: +0.001** object MI above a shuffled null — indistinguishable
  from noise.
- **Loses to Mirror's own input** at predicting an observable world.
- Largest-but-one parameter count in the project (614,960) for the second-best gate record.

## Gates

**Passed** — local-vs-BPTT parity; held-out prediction; out-of-view wall decode above retina;
partial path integration (+0.282, measured after freezing).

**Failed** — `CGE-A-00` beats-its-own-input; occlusion tracking; object binding; identity
decode (0.510, chance is 0.500).

## Why it was replaced

Not by a measurement — by a **theorem**. A gradient flow satisfies

```
dE/dt = -‖∇E‖² ≤ 0
```

and therefore has no periodic orbits. It cannot oscillate, cannot synchronise, and cannot
carry a quantity around a cycle. Whatever the missing ingredient for binding was, this dynamics
provably could not contain it. That is a strictly stronger reason to move on than "it scored
badly", and it is the cleanest retirement in this history.

## Lessons learned

1. **Local learning is not the bottleneck.** Two more architectures assumed it might be. It
   never was. Stop re-testing it.
2. **A closed-form impossibility beats a benchmark.** One line of calculus retired an
   architecture that months of gates had not.
3. **Measure the old versions against new gates.** Wren's +0.282 on path integration sat
   undiscovered through two successor architectures. The frozen versions must be re-run on
   every new gate, not just the live one — which is now enforced.

## Principle worth keeping vs implementation to leave buried

**Keep:** inputs that *bias* a dynamical state rather than overwrite it. Hysteresis — the
property that a weak input leaves the interpretation alone — is real, cheap, and nothing since
has reproduced it. It is a strong candidate for why Wren's mesh holds out-of-view structure
better than anything built after it.

**Bury:** the energy-descent formulation itself. It buys the hysteresis at the cost of a
proof that the system can never do half of what the architecture was for.
