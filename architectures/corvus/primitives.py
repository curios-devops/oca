"""The two architectural primitives: `Entity` and `Event`.

Neither belongs to a layer. They are to OCA what `process`, `address` and `message` are to an
operating system — concepts every layer manipulates, specified before any layer so that the
interface contracts have something to be written in terms of.

**Entity is the only concept v4 adds, and the only one with a measurement behind it.** In the
tunnel world, where the raw-input control is at chance by construction, storing two numbers beat
every architecture ever built here. Wren's readout is its entire 2304-dimensional mesh state with
nothing pooled and it lost too, so persistence is not destroyed on the way up the stack — it is
never formed. Memory answers *what happened*; identity answers *which thing happened*.

**Event is the carrier.** Sparse, timestamped, and referencing an entity. Not a dense state
vector: the one architecture that made a shared state vector its carrier destroyed the signal
(0.69x -> 6.56x). Not a bare scalar either, because a scalar cannot say what it is about, which
is exactly the persistence failure above. An event referencing an entity is sparse *and*
attributable.

Ownership is layered even though the type is not. That distinction is the whole reason this file
enforces anything: a cross-cutting primitive any layer may create is the architectural
equivalent of a global variable, and it would quietly dissolve the layering invariant.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

import numpy as np

_ids = itertools.count(1)


class OwnershipError(RuntimeError):
    """Raised when a layer tries to do something to an entity it does not own.

    Deliberately an error and not a warning. Layer 5 minting something that addresses what
    Layer 1 owns is exactly what the layering invariant forbids, and the only reliable way to
    keep that true is to make it impossible rather than discouraged.
    """


@dataclass
class Entity:
    """A thing that continues to exist.

    The defining property is not that it has state -- all three frozen architectures had plenty
    of state. It is that the state **survives its referent becoming unobservable** and can
    afterwards be recognised as being about the same referent.
    """

    owner: str
    """The layer that created it, and the only layer permitted to retire it."""

    attributes: np.ndarray
    """What is currently believed about it. Shape is fixed at creation."""

    uncertainty: float = 1.0
    """How much that belief is trusted. Grows while unobservable, shrinks on correction. This is
    what makes a stale entity distinguishable from a fresh one, rather than both being numbers."""

    id: int = field(default_factory=lambda: next(_ids))
    created_at: int = 0
    last_seen: int = 0
    observable: bool = True
    relations: tuple[int, ...] = ()

    _retired: bool = False

    # -- immutability of identity ------------------------------------------
    #
    # `id` and `created_at` are writable in the dataclass sense; nothing in Python makes them
    # genuinely const. They are documented as immutable and the guard test checks that no code
    # assigns to them. That is the honest position: enforced by test, not by the type system.

    @property
    def age(self) -> int:
        return self.created_at

    def ticks_unobserved(self, now: int) -> int:
        return max(now - self.last_seen, 0)

    @property
    def retired(self) -> bool:
        return self._retired

    # -- operations any layer may perform ----------------------------------

    def annotate(self, values: np.ndarray) -> None:
        """Attach information without claiming to have observed the referent.

        Permitted to any layer. This is how a higher layer contributes what it knows -- context,
        expectation, relation -- without pretending to have seen anything, which would corrupt
        `last_seen` and with it the entity's own account of how stale it is.
        """
        self._check_live()
        self.attributes = self.attributes + values

    def relate(self, other: "Entity") -> None:
        self._check_live()
        if other.id not in self.relations:
            self.relations = self.relations + (other.id,)

    # -- operations only the owner may perform ------------------------------

    def observe(self, owner: str, attributes: np.ndarray, now: int,
                trust: float = 0.5) -> None:
        """The referent is visible: correct the belief toward it and reduce uncertainty."""
        self._require_owner(owner, "observe")
        self.attributes = (1.0 - trust) * self.attributes + trust * attributes
        self.uncertainty *= (1.0 - trust)
        self.last_seen = now
        self.observable = True

    def propagate(self, owner: str, delta: np.ndarray, growth: float = 0.05) -> None:
        """The referent is *not* visible: advance the belief by a prediction, and grow doubt.

        This is the operation the whole primitive exists for, and the one no frozen architecture
        had. Nothing here observes anything -- the belief moves because the entity is expected to
        have moved, and the uncertainty records that the movement was inferred.
        """
        self._require_owner(owner, "propagate")
        self.attributes = self.attributes + delta
        self.uncertainty = min(self.uncertainty + growth, 1.0)
        self.observable = False

    def retire(self, owner: str) -> None:
        self._require_owner(owner, "retire")
        self._retired = True

    # -- guards ------------------------------------------------------------

    def _require_owner(self, who: str, what: str) -> None:
        self._check_live()
        if who != self.owner:
            raise OwnershipError(
                f"{who!r} tried to {what} entity {self.id}, owned by {self.owner!r}. Only the "
                "layer that can observe a referent may create, correct, propagate or retire it; "
                "any layer may read, annotate and relate. Cross-cutting as a type, layered as "
                "an authority.")

    def _check_live(self) -> None:
        if self._retired:
            raise OwnershipError(f"entity {self.id} is retired and must not be used again")

    def snapshot(self) -> dict:
        return {"id": self.id, "owner": self.owner, "uncertainty": self.uncertainty,
                "observable": self.observable, "last_seen": self.last_seen,
                "n_relations": len(self.relations)}


@dataclass(frozen=True)
class Event:
    """The unit of communication. Sparse, timestamped, attributable.

    Frozen: an event is a record of something that happened, and a mutable one is a state
    variable pretending to be a message.
    """

    source: str
    kind: str
    value: float
    at: int
    entity: int | None = None
    """Which entity this is about. `None` is legal -- a neuron reporting its own activation is
    not talking about an entity -- but an event that *could* name one and does not is how
    identity gets lost at a layer boundary."""

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("an event must say what kind of claim it makes")

    def about(self, entity: Entity) -> "Event":
        return replace(self, entity=entity.id)


class Registry:
    """The entities one layer owns, and the only place they are created.

    Deliberately not global. A registry belongs to a layer instance, so "who owns this" is a
    structural fact rather than a convention -- and a layer cannot accidentally create an entity
    in someone else's registry because it has no reference to one.
    """

    def __init__(self, owner: str):
        self.owner = owner
        self._live: dict[int, Entity] = {}

    def create(self, attributes: np.ndarray, now: int, uncertainty: float = 1.0) -> Entity:
        e = Entity(owner=self.owner, attributes=np.asarray(attributes, dtype=np.float64).copy(),
                   uncertainty=uncertainty, created_at=now, last_seen=now)
        self._live[e.id] = e
        return e

    def retire(self, entity: Entity) -> None:
        entity.retire(self.owner)
        self._live.pop(entity.id, None)

    def __iter__(self):
        return iter(list(self._live.values()))

    def __len__(self) -> int:
        return len(self._live)

    def get(self, entity_id: int) -> Entity | None:
        return self._live.get(entity_id)

    def stalest(self, now: int) -> Entity | None:
        """The entity whose referent has been unobservable longest -- the retirement candidate."""
        return max(self._live.values(), key=lambda e: e.ticks_unobserved(now), default=None)
