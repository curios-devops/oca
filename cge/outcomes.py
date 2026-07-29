"""Gate verdicts. A Cognitive Gate returns an engineering decision, not a score.

A benchmark produces a number and invites you to compare it with someone else's number. A gate
produces a decision about whether a component may advance. That difference is the point of the
CGE, and it is enforced here rather than left to convention.

There are **four** verdicts, not three. The fourth is the one this project had to learn.

    PASS              the component demonstrates the property; it may advance
    CONDITIONAL_PASS  it demonstrates the property with a documented limitation
    FAIL              it does not; redesign before continuing
    UNMEASURED        the gate did not run validly, so it says nothing either way

**UNMEASURED is not a soft FAIL and must never be recorded as one.** A gate whose own control
failed has produced no information about the component. Collapsing that into FAIL manufactures
evidence, and collapsing it into PASS hides a broken harness. Both have happened here:

* An unstandardised ridge probe returned decode errors *above chance* -- a fit so wild it was
  worse than predicting the mean. Read as FAIL, it would have condemned a representation on the
  strength of a bug in the probe.
* An apparent mutual information of 0.449 turned out to be entirely label-frequency bias, and
  survived until a paired shuffled null was added.
* The binding gate cannot be computed at all for an architecture with fewer units than the
  world has objects. Reported as 0.000 it would have read as "this architecture binds nothing",
  which is a claim the measurement does not support.

So every gate carries its control, and a gate whose control has failed returns UNMEASURED with
a reason. `Verdict.reason` is mandatory for everything except a clean PASS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Outcome(str, Enum):
    PASS = "PASS"
    CONDITIONAL_PASS = "CONDITIONAL_PASS"
    FAIL = "FAIL"
    UNMEASURED = "UNMEASURED"

    @property
    def may_advance(self) -> bool:
        """Only PASS and CONDITIONAL_PASS permit a component to progress a stage."""
        return self in (Outcome.PASS, Outcome.CONDITIONAL_PASS)

    @property
    def is_evidence(self) -> bool:
        """UNMEASURED is not evidence, and must not be averaged into a summary."""
        return self is not Outcome.UNMEASURED


@dataclass
class Verdict:
    """One gate's decision about one component, with everything needed to argue with it."""

    gate: str
    outcome: Outcome
    reason: str = ""
    """Why. Mandatory for anything other than a clean PASS -- a FAIL without a reason is not
    actionable, and an UNMEASURED without one is indistinguishable from a crash."""

    observed: dict = field(default_factory=dict)
    """The measurements behind the decision. Reported so the verdict can be re-derived."""

    control: dict = field(default_factory=dict)
    """What the component was measured *against*. A verdict with no control is not a verdict --
    this is the field whose absence explains most of this project's retracted claims."""

    limitations: list[str] = field(default_factory=list)
    """For CONDITIONAL_PASS: the documented limitations the component advances *with*. A
    conditional pass with an empty list is a PASS pretending to be careful."""

    def __post_init__(self) -> None:
        if self.outcome is not Outcome.PASS and not self.reason:
            raise ValueError(
                f"gate {self.gate!r} returned {self.outcome.value} with no reason. A verdict "
                "that cannot be argued with is not an engineering decision.")
        if self.outcome is Outcome.CONDITIONAL_PASS and not self.limitations:
            raise ValueError(
                f"gate {self.gate!r} returned CONDITIONAL_PASS with no limitations listed. "
                "The limitations are the whole difference from PASS.")

    def __str__(self) -> str:
        head = f"{self.gate}: {self.outcome.value}"
        return head if not self.reason else f"{head} -- {self.reason}"


def summarise(verdicts: list[Verdict]) -> dict:
    """Roll up a set of verdicts without letting UNMEASURED masquerade as a result.

    Deliberately reports `unmeasured` as its own count rather than folding it into a
    denominator. A stage that is 4/5 PASS with one UNMEASURED has not been evaluated on five
    gates; it has been evaluated on four.
    """
    counts = {o: 0 for o in Outcome}
    for v in verdicts:
        counts[v.outcome] += 1
    measured = [v for v in verdicts if v.outcome.is_evidence]
    return {
        "n_gates": len(verdicts),
        "n_measured": len(measured),
        "passed": counts[Outcome.PASS],
        "conditional": counts[Outcome.CONDITIONAL_PASS],
        "failed": counts[Outcome.FAIL],
        "unmeasured": counts[Outcome.UNMEASURED],
        # a stage advances only if every *measured* gate advances, and nothing is unmeasured
        "may_advance": bool(measured
                            and counts[Outcome.UNMEASURED] == 0
                            and all(v.outcome.may_advance for v in measured)),
        "blocking": [v.gate for v in verdicts if not v.outcome.may_advance],
    }
