# dcn/

Architecture v2. Empty by design — see [../../docs/heron/FIRST_PRINCIPLES_DCN.md](../../docs/heron/FIRST_PRINCIPLES_DCN.md).

Rules for anything added here:

- Import from `core/` only. Never from `legacy/`.
- Every level declares its prediction horizon explicitly.
- Every level talks only to the level directly below it.
- New gates go in `bench/gates.py`, never here, so every architecture is scored on the
  same code path.
