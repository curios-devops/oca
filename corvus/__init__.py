"""Corvus — OCA v4. A blank page, for the third time.

Named for the corvids, which are the canonical animals that pass object-permanence and
tool-use tests. That is not decoration: **object permanence is this project's open problem.**
In the tunnel maze — the one world where the image provably cannot contain the answer —
storing the two coordinates you walked in at beats every architecture built here, and the
most recent one scores worse than storing nothing. A crow would not have that difficulty.

Three architectures are frozen in `legacy/`: Wren (gradient flow), Swift (limit-cycle
oscillators) and Heron (reservoir and resonance). None of them beat their own input. They are
kept because they are the opponents, and because the reason each failed is now written down.

The rule that governs this package, unchanged and enforced by
`tests/test_architecture_contract.py`:

    Nothing enters by compatibility. Everything enters because it has a stated function,
    and because it names what it would have to beat.

`corvus/` imports from `core/` and from nothing in `legacy/`. What it does inherit is the
contract in `contract.py`, which is the third version of that file and the first one written
after knowing what the previous two failed to require.

See `docs/CORVUS/` for the specifications, and `docs/WHAT_WE_HAVE_LEARNED.md` for the
evidence every one of them is checked against.
"""

from .contract import LAYERS, Floor, Layer, check_stack, register

__all__ = ["Floor", "Layer", "LAYERS", "check_stack", "register"]
