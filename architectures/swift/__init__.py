"""RPDU architecture v2.

Built against the four diagnostics in RESULTS.md's appendix. v1's units were gradient
flows, which provably cannot oscillate (so coalitions were impossible) and cannot carry a
moving quantity (so nothing survived an occlusion); and its objective was locally
satisfiable, so nothing ever forced an object-level representation. v2 replaces the
dynamics with limit-cycle oscillators, gates communication on phase agreement, spreads
integration timescales across units, and masks sensory input so a unit must reconstruct
its own patch from its neighbours.

v1 is untouched and still reproduces its published numbers.
"""

from .dynamics import phase, phase_alignment, radius, rollout, step
from .mesh import build_mesh2, predicted_retina2, tick2
from .state import Config2, MeshState2

__all__ = [
    "Config2", "MeshState2",
    "build_mesh2", "tick2", "predicted_retina2",
    "step", "rollout", "phase", "radius", "phase_alignment",
]
