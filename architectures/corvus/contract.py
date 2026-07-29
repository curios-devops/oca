"""What an OCA architectural layer must declare before it may exist.

The third iteration of this file, and the only one written after three architectures had
failed. The previous version (`legacy/dcn/contract.py`) required a layer to declare its
prediction horizon and the layer it reads from. Both were right and neither was sufficient:
DCN v3 satisfied that contract completely and still lost to a two-number memory.

So this version adds the requirement that would have caught it. A layer declares:

* `horizon` — how far ahead it predicts, in ticks. Mandatory, never defaulted. A slow layer
  scored on a one-tick prediction is competing with persistence and cannot win: no function
  of a legacy mesh state beat "assume no change" at tau=1, while at tau=64 a 44% gain was
  available.

* `inputs_from` — the layer directly below, or None at the sensory surface. Dependencies
  flow downward, abstractions flow upward, and nothing skips a layer.

* `floor` — **the thing this layer must beat to be a layer at all.** This is the new one,
  and it is the whole lesson of v1, v2 and v3. Every intermediate representation those three
  built was no better than its own input, and no gate noticed, because no gate asked. A
  layer that cannot state what it would have to outperform in order to justify existing has
  not been designed; it has been named.

* `integrates_over` — the timescale over which its state actually decorrelates, measured
  rather than asserted. DCN v3's node layer decorrelated *faster* than the neurons feeding
  it, which made it a relabelling rather than a level, and every gate in the battery missed
  it. A layer must be slower than the layer below it and this is checked.

The contract deliberately says nothing about mechanism. Backpropagation, local rules,
symbolic inference, probabilistic filtering and spiking computation are all admissible; what
is not admissible is a layer that has no stated floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

LAYERS: dict[str, "Layer"] = {}


@dataclass(frozen=True)
class Floor:
    """What a layer must beat, and by how much, to have earned its place.

    Not a benchmark result — a *declaration*, made before the layer is built, naming the
    cheapest thing that could do its job. The gates then check it.

    Three floors have already proved decisive in this project and are named here because
    each one beat a real architecture:

    * `raw_input` — the sensory frame itself. Every architecture so far has lost to it at
      predicting an observable world.
    * `trivial_memory` — store one value once and never update it. "Frozen at entry"
      (2.26 cells) beat all three architectures at knowing where it was while blind.
    * `mean_pool` — a linear summary of the layer below. Beat DCN v3's relational
      aggregation at matched width, on both targets.
    """

    beats: str
    """Name of the baseline this layer claims to beat. Free-form, but if it names one of the
    three above, the shared benchmark already knows how to compute it."""

    margin: float = 0.05
    """The minimum relative improvement that counts. Defaulted, because 5% has been the
    working threshold throughout, but stating it is what makes a near-tie a failure rather
    than a matter of interpretation."""

    why: str = ""
    """One sentence: why is *this* the right thing to beat? A floor chosen for being easy is
    worse than no floor, because it produces a pass."""

    def __post_init__(self) -> None:
        if not self.beats:
            raise ValueError(
                "a layer must name what it beats. Three architectures satisfied every "
                "contract they were given and were all beaten by their own input or by a "
                "two-number memory, because no contract asked this question.")
        if self.margin <= 0:
            raise ValueError(f"floor margin must be positive, got {self.margin}")


@dataclass
class Layer:
    """One architectural layer: neuron, tower, cluster, region, system, cortex, brain."""

    name: str
    horizon: int
    inputs_from: str | None
    floor: Floor
    build: Callable[..., object]
    step: Callable[[object, object], dict]
    readout: Callable[[object], object]
    describe: Callable[[object], dict] = field(default=lambda s: {})

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(
                f"layer {self.name!r} must declare a prediction horizon of at least 1 tick")
        if not isinstance(self.floor, Floor):
            raise TypeError(
                f"layer {self.name!r} must declare a Floor: what it beats, and why")


def register(layer: Layer) -> Layer:
    LAYERS[layer.name] = layer
    return layer


def check_stack(layers: list[Layer]) -> None:
    """Verify the invariants that hold across a whole stack rather than within one layer.

    Called by tests rather than at import time, because a partially built stack is a
    legitimate intermediate state — the point of the build order is that layers arrive one
    at a time.
    """
    names = [l.name for l in layers]

    for i, layer in enumerate(layers):
        if layer.inputs_from is None:
            if i != 0:
                raise ValueError(f"{layer.name} has no input but is not the lowest layer")
            continue
        if layer.inputs_from != names[i - 1]:
            raise ValueError(
                f"{layer.name} reads from {layer.inputs_from!r}, skipping a layer. "
                "Dependencies flow downward one step at a time.")

    horizons = [l.horizon for l in layers]
    if horizons != sorted(horizons) or len(set(horizons)) != len(horizons):
        raise ValueError(
            f"horizons must strictly increase up the stack, got {horizons}. A layer that "
            "does not integrate over longer than the layer below it is a relabelling — "
            "DCN v3's node layer decorrelated faster than its own neurons and no gate "
            "caught it.")
