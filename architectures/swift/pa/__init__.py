"""Predictive Assembly — the level above the RPDU (docs/RPDU/SPEC_PA.md).

A read-only layer over an unmodified mesh: it observes unit state, maintains one slow
shared field per assembly, predicts, and exports five scalars. It never writes back into
the mesh, so a failure here is evidence about aggregation rather than about feedback.
"""

from .assembly import (Assembly, AssemblyConfig, build_assembly, export,
                       pooled_members, step_assembly, workspace)

__all__ = ["Assembly", "AssemblyConfig", "build_assembly", "step_assembly",
           "export", "pooled_members", "workspace"]
