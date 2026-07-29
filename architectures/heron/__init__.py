"""Dynamic Cortical Network — architecture v2, built on a blank page.

Deliberately empty of architecture. What lives here is the contract a DCN has to satisfy
to be scored, so that the design can be filled in without ever needing to touch the
benchmark, and so that the first line of real code is written against a test rather than
against the old model.

See docs/DCN/FIRST_PRINCIPLES_DCN.md for the axioms. The rule that governs this package:
nothing enters by compatibility with `legacy/`, only by stated function. It imports from
`core/` (worlds, sensors, probes, metrics, baselines) and from nothing in `legacy/` --
enforced by tests/test_dcn_contract.py.
"""

from .contract import LEVELS, DCNLevel, register_dcn

__all__ = ["DCNLevel", "LEVELS", "register_dcn"]
