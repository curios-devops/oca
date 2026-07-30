# Architecture history

Every architecture this project has retired, and what survived it.

The purpose is not nostalgia and it is not a leaderboard. It is one question, asked of each
retired version:

> **Which properties appeared here that do not exist today — and what principle produced
> them?**

Never *"should we copy it?"*. An architecture can be poor overall and still contain one
correct idea. Swift lost almost every gate it entered and still holds the best object-binding
result in the project by a factor of fourteen. That principle deserves to survive. Its
implementation almost certainly does not.

Each file records: goals, strengths, weaknesses, gates passed, gates failed, lessons, and the
reason for replacement — plus the section that matters most, **the principle worth keeping vs
the implementation to leave buried.**

| version | codename | retired | file |
|---|---|---|---|
| v1 | **Wren** | 2026-07 | [WREN.md](WREN.md) |
| v2 | **Swift** | 2026-07 | [SWIFT.md](SWIFT.md) |
| v3 | **Heron** (DCN) | 2026-07 | [HERON.md](HERON.md) |
| — | **Mirror** | *never retired* | the zero-parameter control, still a live entrant |
| v4 | **Corvus** | live | [docs/corvus/](../docs/corvus/) |

Live code lives in [`architectures/`](../architectures/). The frozen versions are still
executable and still registered as benchmark entrants — freezing means *"no longer modified"*,
not *"no longer run"*. `tests/test_frozen_architectures.py` enforces that.

---

## The property audit

Two tables, deliberately, because merging them would be dishonest.

### A. Measured head-to-head, same harness, same seeds

This is the only table where a comparison between columns means anything. Today it has
**one row**, because `path_integration` is the sole gate every entrant has been run through.

| property | Mirror | Wren | Swift | Heron | Corvus |
|---|---|---|---|---|---|
| **path integration** (vs `no_integration`, 3 seeds) | −0.027 | **+0.282** | +0.007 | **−0.232** | **+0.572** ✅ |

Three facts in one row, and two of them are surprises:

- **Corvus is the only entrant to clear a declared floor by its declared margin.** First in the
  project.
- **Wren integrates partially, and nobody had ever asked.** Its four-scalar mesh carries real
  self-motion information that went unmeasured for two versions.
- **Heron is worse than doing nothing.** Not "weak" — actively harmful, consistent with every
  other measurement of it.

### B. Measured per version, under different harnesses — **not comparable across columns**

Everything below was measured when that version was live, against the controls that existed
then. Reading down a column is valid. Reading across a row is not, and the ✅/❌ table shape
invites exactly that error, so the cells carry numbers instead of ticks.

| property | Wren | Swift | Heron | Corvus |
|---|---|---|---|---|
| out-of-view wall decode (retina = 0.773) | **0.841** | 0.736 | 0.619 | 0.548 |
| maze exits, 900 steps (Mirror = 30) | **53** | 3 | 0 | 0 |
| object binding, coalition MI vs null | +0.001 | **+0.124** | +0.009 | not run |
| identity decode (chance = 0.500) | 0.510 | 0.528 | 0.523 | not run |
| sparse event coding vs matched-rate control | — | — | **+92.9%** | not re-run |
| publication bottleneck cost | — | — | **1.50×** | not run |
| parameters (maze-race config) | 614,960 | 918,832 | 293,485 | **166,912** |
| beats its own input on an observable world | ❌ | ❌ | ❌ | ❌ |

**The V4 column is not all-green, and that is the point.** Wren decodes hidden walls better
than Corvus by 29 points and finds the exit 53 times to Corvus's zero. Swift binds objects
fourteen times better than anything that replaced it. A property-audit table written by the
author of the newest version tends to come out all-✅ down the last column; this one does not,
and the two red cells are the most useful entries in it.

**The last row is the standing open problem.** Four architectures, four different primitives,
four different learning rules, one shared result: nothing here has beaten the raw frame at
predicting a world it can see. That is `CGE-A-00`, and it is unpassed.

---

## Why this directory exists

Two reasons, and the second is the real one.

**It turns each failure into accumulated knowledge** rather than a deleted branch. Every entry
below names a mechanism that sounded right, and says what killed it.

**It prevents re-deriving a dead idea in five years.** The specific hazard here is not
forgetting that Swift failed — it is forgetting *why*, and rebuilding phase-gated coupling
because the argument for it is genuinely persuasive. It was persuasive the first time too. The
counter-evidence is in [SWIFT.md](SWIFT.md), with the numbers.
