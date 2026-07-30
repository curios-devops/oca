# Swift — OCA v2 (RPDU mesh v2)

*Retired 2026-07. Code frozen at [`architectures/swift/`](../architectures/swift/), still
executable, still a registered benchmark entrant.*

Named for a bird that stays airborne for months without landing — the architecture whose units
never settle.

## Goals

Answer Wren's impossibility directly. If a gradient flow provably cannot oscillate, replace the
dynamics with one that must, and see whether synchronisation gives the mesh the binding
mechanism it lacked.

## Mechanism

Every unit becomes a **Stuart–Landau oscillator** in normal form: the amplitude settles to a
limit cycle while the phase keeps advancing. Coupling between units is **phase-gated** — units
influence each other in proportion to how aligned their phases are. Units that fall into
synchrony form a **coalition**, and the coalition, not the unit, was to be the carrier of an
object.

## Strengths

- **The best object binding this project has produced, by a wide margin.** Synchrony coalitions
  carry **+0.124** object mutual information above a shuffled-label null — **fourteen times**
  Heron's +0.009 and a hundred times Wren's +0.001.
- The mechanism does what the theory says: amplitude and phase decouple cleanly, synchrony is
  expressible, coalitions form and dissolve.
- Provided the coalition machinery (`phase_coherence`, `vote_coalitions`) that every later
  measurement of binding has been scored against.

## Weaknesses

- **Synchrony carries no content.** As a representation to predict *from*, it scored exactly
  **1.00× persistence** — i.e. nothing above simply repeating the last value.
- **Worse prediction than Wren**, and worse out-of-view wall decode than Mirror (0.736 vs
  0.773) — it is beaten by having no memory at all.
- **3 maze exits** against Wren's 53.
- **No path integration: +0.007.** Indistinguishable from zero.
- The most expensive architecture in the project: **918,832 parameters** for the worst
  navigation record of the three legacy versions.

## Gates

**Passed** — object binding / coalition MI (+0.124, the only version to pass it).

**Failed** — `CGE-A-00` beats-its-own-input; held-out prediction (worse than Wren); out-of-view
wall decode (below Mirror); identity decode (0.528); path integration (+0.007); maze
navigation.

## Why it was replaced

Because it answered its own question and the answer was *"yes, and it isn't enough."*
Synchrony **did** bind — that result stands and has never been beaten. But binding a set of
units together turned out to be orthogonal to representing what they were bound *about*. The
coalition knew which units belonged to the same object and could not say anything about the
object.

## Lessons learned

1. **Phase as a clock, never as content.** Swift's decisive negative — synchrony at exactly
   1.00× persistence — is the origin of the axiom now written into the v4 specification: phase
   may gate *when* a unit emits, never *what* it emits. Heron later violated the spirit of this
   by putting a rotor into the activation path, and lost 92% of its Layer-0 result until the
   rotor was moved to gate the emission threshold instead.
2. **A structural mechanism can succeed while the architecture around it fails.** Judging the
   idea by the version's overall score would have thrown away the best binding result in the
   project. This is the single strongest argument for keeping this directory.
3. **The reason this version was overridden was itself later reversed.** The decision that set
   Swift aside is contradicted by its own binding number; see
   [FIRST_PRINCIPLES_DCN.md](../docs/heron/FIRST_PRINCIPLES_DCN.md).

## Principle worth keeping vs implementation to leave buried

**Keep:** *synchrony as a grouping signal.* +0.124 is the highest binding evidence in the
project and it came from units agreeing about timing, not about content. Nothing in Corvus
currently reproduces it — Corvus's cluster layer coordinates through member agreement and
entity relations, and has never been measured against Swift's number. **That comparison is
outstanding and Corvus is currently expected to lose it.**

**Bury:** Stuart–Landau limit cycles as the *substrate for state*. Making every unit an
oscillator to obtain a grouping signal costs 918,832 parameters and destroys the unit's ability
to hold a value. The grouping signal does not require the substrate.
