# corvus/

OCA v4. Nearly empty by design — see [../../docs/corvus/](../../docs/corvus/).

Named for the corvids, the canonical animals that pass object-permanence tests. That is the
problem all three frozen architectures failed: in the tunnel maze, storing the two coordinates
you walked in at beats every one of them.

Rules for anything added here:

- Import from `core/` only. Never from `legacy/`, which now holds three frozen architectures.
- Every layer declares its prediction horizon, its measured integration window, and **what it
  must beat**. The third one is new in v4, and it is the requirement whose absence let three
  architectures fail without any gate objecting.
- Every layer talks only to the layer directly below it.
- New gates go in `bench/`, never here, so every architecture is scored on the same code path.

Two blocking questions must be answered before Layer 0 or Layer 1 is implemented:
[Q1 (oscillation) and Q2 (entity persistence)](../../docs/corvus/OPEN_QUESTIONS.md).
