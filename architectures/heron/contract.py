"""What a DCN level must provide, and nothing about how.

Axiom 5 says higher structures never address individual neurons: each level speaks only to
the level below through a stated interface. That is the axiom most easily lost to
convenience -- reaching down a level for a quantity that happens to be in scope -- so it is
written as an explicit contract here rather than left as an intention.

A level declares:

* `horizon` -- how far ahead it predicts, in ticks. Not optional and not a default. The
  legacy line spent an entire experimental cycle discovering that a slow layer scored on a
  one-tick prediction is competing with persistence and cannot win: no function of the mesh
  state beat "assume no change" at tau=1, while at tau=64 a 44% gain was available. A level
  that will not state its horizon cannot be scored honestly.
* `inputs_from` -- the name of the level directly below, or None for the sensory level.
  Anything else is a violation of axiom 5.
* `state` and `readout` -- what it holds, and what the level above is allowed to see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

LEVELS: dict[str, "DCNLevel"] = {}


@dataclass
class DCNLevel:
    """One level of the hierarchy: neurons, DCN, column, region, cortex."""

    name: str
    horizon: int
    inputs_from: str | None
    build: Callable[..., object]
    step: Callable[[object, object], dict]
    readout: Callable[[object], object]
    describe: Callable[[object], dict] = field(default=lambda s: {})

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(
                f"level {self.name!r} must declare a prediction horizon of at least 1 tick")


def register_dcn(level: DCNLevel) -> DCNLevel:
    LEVELS[level.name] = level
    return level
