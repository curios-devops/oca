"""The gate catalogue: every Cognitive Gate's identity, class, scope and status.

Separated from the implementations on purpose. `gates.py`, `components.py` and `nodes.py` say
*how* a gate is computed; this says what it is, which levels it applies to, what it costs, what
it must beat, and — the field that matters most — **what has already failed it**.

Three rules govern this file.

**A gate that nothing has ever failed is not a gate.** Every entry records at least one
architecture or configuration that FAILED it, or is marked `PROPOSED` and does not count toward
compliance. This is the rule that would have caught the most expensive mistake in this project:
a battery of gates that three architectures passed while all three were being beaten by their
own sensory input.

**A gate is a template instantiated per level, not a family per level.** `rate_distortion` is
the same question of a neuron, a tower and a cluster; only the scale changes. Inventing a new
gate family for each level produces a suite that cannot compare across levels, which is most of
what the suite is for.

**Identity is permanent.** A gate's `id` never changes and never gets reused. Semantics change
by incrementing `version`, and a gate that is superseded is marked `DEPRECATED` rather than
deleted, so a historical verdict stays interpretable.

See `docs/SPEC_CGE.md` for the specification this implements. It sits at the top of
`docs/` rather than under any architecture's directory, deliberately: a benchmark filed
inside the thing it judges is a benchmark that will drift toward it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GateClass(str, Enum):
    """Which question a gate answers, and therefore when it may be run.

    Ordering is a dependency, not a preference: B is meaningless before A passes, because
    comparing two components that both lose to their own input tells you only which is further
    behind. C is meaningless before B.
    """

    A = "A"   # minimum viability, including the floor gates. Required before any comparison.
    B = "B"   # architectural validation: against Wren, Swift, Heron and each other.
    C = "C"   # external comparison: LLMs, transformers, HTM, Thousand-Brains implementations.


class Status(str, Enum):
    PROPOSED = "PROPOSED"        # specified, not implemented; does not count toward compliance
    EXPERIMENTAL = "EXPERIMENTAL"  # implemented, but nothing has failed it yet
    STABLE = "STABLE"            # implemented, and has discriminated at least once
    DEPRECATED = "DEPRECATED"    # superseded; kept so old verdicts stay readable


class Modality(str, Enum):
    VISION = "vision"
    TOUCH = "touch"
    PROPRIOCEPTION = "proprioception"   # efference copy: the agent's own action
    AUDIO = "audio"
    CROSS_MODAL = "cross_modal"


@dataclass(frozen=True)
class Gate:
    """One Cognitive Gate's contract. Fields follow the CGE specification."""

    id: str
    """Permanent. `CGE-A-03`. Never reused, never renumbered."""

    version: int
    title: str
    gate_class: GateClass
    status: Status

    objective: str
    """The engineering question, in one sentence."""

    required_property: str
    """What the component must demonstrate. Behavioural, not a percentage."""

    anti_property: str
    """What must *not* be true. The failure this gate exists to catch."""

    control: str
    """What the component is measured against. **Mandatory.** A gate with no control cannot
    produce a verdict, only a number, and this project has three retracted claims to show what
    happens then."""

    pass_criterion: str
    """Behavioural, and expressed as beating a named baseline. "Maintains stable identity" is
    unfalsifiable; "identity survives occlusion better than frozen-at-entry" is a gate."""

    levels: tuple[int, ...]
    """Which architectural layers this template applies to. Most apply to several."""

    modalities: tuple[Modality, ...]
    cost: str
    """Order of magnitude of wall-clock time: "seconds", "minutes", "hours"."""

    has_ever_failed: str = ""
    """**What has actually failed this gate.** Empty means the gate has never discriminated,
    and a gate that has never discriminated does not count toward compliance."""

    ood: str = ""
    """**Which distribution shift this gate applies between fitting and scoring.**

    Mandatory in the same sense as `control`: it may say "none", but it may not be silent.

    Every gate in this project fits and scores on the *same* world with a *time* split. Nothing
    is ever asked to transfer. That is precisely the hole through which a benchmark rewards
    task-fitting rather than capability, and it applies to our own headline results -- nobody
    knows whether Corvus's +0.572 on path integration survives a maze it was not trained in,
    because no gate asked.

    Declaring it per gate rather than adding a thirteenth "generalisation gate" is deliberate.
    A gate runs once and is forgotten; a mandatory field puts the caveat on **every result the
    gate has ever produced**, which is how `control` earned its place.
    """

    anti_tests: tuple[str, ...] = ()
    """Deliberate attempts to break the abstraction: noise, delay, missing input, conflicting
    evidence, timing shifts, partial communication loss. Graceful degradation is the pass."""

    depends_on: tuple[str, ...] = ()
    notes: str = ""
    implemented_by: str = ""
    """Dotted path to the callable, or empty for PROPOSED gates."""

    def __post_init__(self) -> None:
        if not self.control:
            raise ValueError(f"{self.id} declares no control")
        if self.status is not Status.PROPOSED and not self.ood:
            raise ValueError(
                f"{self.id} declares no OOD condition. Say which distribution shift it applies "
                "between fitting and scoring, or say 'none' -- but a gate that fits and scores "
                "on one distribution and does not admit it is measuring the task.")
        if not self.levels:
            raise ValueError(f"{self.id} declares no applicable levels")
        if self.status is Status.STABLE and not self.has_ever_failed:
            raise ValueError(
                f"{self.id} is marked STABLE but nothing has ever failed it. A gate that "
                "nothing fails is decoration; mark it EXPERIMENTAL until it discriminates.")
        if self.status is not Status.PROPOSED and not self.implemented_by:
            raise ValueError(f"{self.id} is not PROPOSED but names no implementation")

    @property
    def counts_toward_compliance(self) -> bool:
        return self.status in (Status.EXPERIMENTAL, Status.STABLE)


CATALOGUE: dict[str, Gate] = {}


def register(gate: Gate) -> Gate:
    if gate.id in CATALOGUE:
        raise ValueError(f"gate id {gate.id} is already taken; ids are never reused")
    CATALOGUE[gate.id] = gate
    return gate


def by_class(cls: GateClass) -> list[Gate]:
    return [g for g in CATALOGUE.values() if g.gate_class is cls]


def by_level(level: int) -> list[Gate]:
    return [g for g in CATALOGUE.values() if level in g.levels]


# ===========================================================================================
# Gate A — minimum viability. Required before any architecture comparison.
# ===========================================================================================
#
# A0 and A1 are the floor gates, and they are the reason this catalogue exists. Wren, Swift and
# Heron passed every gate they were ever set while all three were beaten by their own sensory
# input, because nothing in the old battery asked them to beat anything cheap.

register(Gate(
    id="CGE-A-00", version=1,
    title="Beats its own input",
    gate_class=GateClass.A, status=Status.STABLE,
    objective="Does this component make more of the world readable than its own input already "
              "does?",
    required_property="A representation that carries something the input does not.",
    anti_property="A re-encoding: the component's state is a lossy copy of what it received.",
    control="The raw sensory frame, at matched capacity and through the identical probe.",
    pass_criterion="Beats the raw input at the component's own declared horizon by its "
                   "declared margin. Must be run on a world where the answer is NOT fully "
                   "present in the current observation, or the gate cannot discriminate.",
    levels=(0, 1, 2, 3), modalities=(Modality.VISION, Modality.TOUCH), cost="minutes",
    has_ever_failed="Wren, Swift and Heron. All three, at every horizon (1, 4, 16, 64 ticks). "
                    "Mirror -- literally the raw frame -- outscores all three.",
    anti_tests=("input noise sigma 0.0/0.1/0.3", "held-out time split, never random"),
    ood="none. Fitted and scored on one physics world with a time split. The gate has never been asked whether an architecture that beats its input on world A still does on world B, and since nothing has passed it, that omission has cost nothing yet. It would the moment something passes.",
    implemented_by="cge.nodes.predictive_gain",
    notes="Most of this project's worlds are observable enough that the answer is already in "
          "the frame, which makes this gate close to unwinnable on them and is most of why "
          "'nothing beats raw pixels' went unexplained for so long. Run it on CGE-A-01's world.",
))

register(Gate(
    id="CGE-A-01", version=1,
    title="Beats a trivial memory while blind (SUPERSEDED -- not measurable)",
    gate_class=GateClass.A, status=Status.DEPRECATED,
    objective="Does the component hold anything about a referent it currently cannot observe?",
    required_property="State that survives its referent becoming unobservable and can be read "
                      "back as being about the same referent.",
    anti_property="State that is a fading function of the recent past and nothing more.",
    control="`frozen-at-entry`: store one value once, never update it. Plus the raw frame, "
            "which is at chance BY CONSTRUCTION in this world -- measured 7.02 against a "
            "chance of 7.02, to two decimals, over three seeds.",
    pass_criterion="Lower position error while blind than frozen-at-entry, by the declared "
                   "margin. This is the project's standing open challenge (P0).",
    levels=(1, 2, 3), modalities=(Modality.VISION, Modality.PROPRIOCEPTION), cost="minutes",
    has_ever_failed="Every architecture built here. frozen-at-entry 2.26 cells; Wren 4.44, "
                    "Swift 5.10, Mirror 7.02 (= chance), Heron 8.56 (worse than chance).",
    anti_tests=("length of blindness swept 1-2, 3-4, 5-8, 9+ steps",
                "the pixel control must land at chance or the corridors are leaking"),
    ood="none, and irrelevant -- the gate was not measurable in-distribution either.",
    implemented_by="cge.gates.gate_tunnel",
    notes="DEPRECATED, superseded by CGE-A-09. This gate was not measurable and the defect was "
          "mine. It scored absolute position against `frozen-at-entry`, a baseline HANDED the "
          "true entry coordinates -- while an architecture must encode where it is from a 5x5 "
          "view. Measured, that is impossible here: position decodes from the raw view at 4.92 "
          "cells WHILE FULLY SIGHTED, against a chance of 8.58 and the baseline's 2.06. So no "
          "architecture could pass it however well it persisted, and all four failed by similar "
          "margins for a reason unrelated to persistence. Kept, not deleted, so the verdicts it "
          "produced stay interpretable and the mistake stays visible.",
))

register(Gate(
    id="CGE-A-09", version=1,
    title="Path integration while blind",
    gate_class=GateClass.A, status=Status.STABLE,
    objective="Does the component integrate its own moves while it cannot see?",
    required_property="State that accumulates self-motion and can be read back as displacement.",
    anti_property="State that is a fading function of the recent past; or a gate that rewards "
                  "information the component was given rather than inferred.",
    control="`no_integration` -- predict zero displacement, which is what holding still achieves "
            "-- plus the raw frame, at chance by construction inside a corridor.",
    pass_criterion="Lower displacement error than no-integration, by the declared margin. The "
                   "target is displacement SINCE ENTERING, not absolute position, so both sides "
                   "are asked only about the part that can be inferred.",
    levels=(1, 2, 3), modalities=(Modality.VISION, Modality.PROPRIOCEPTION), cost="minutes",
    has_ever_failed="Heron (-0.232, worse than not integrating), Swift (+0.007, nothing) and "
                    "Mirror (-0.027, as it must be). Wren integrates partially (+0.282). Corvus "
                    "is the first architecture in this project to clear a floor gate: +0.572.",
    anti_tests=("the raw frame must carry no displacement information, or the corridors leak",
                "displacement rather than position, which removes the unattainable anchor"),
    depends_on=("CGE-A-01",),
    ood="none, and this is the most expensive gap in the suite. Corvus's +0.572 was fitted and scored in the same braided maze. Whether path integration transfers to a maze it never saw is the difference between a capability and a fitted trajectory, and it is unmeasured. **First OOD variant to build.**",
    implemented_by="cge.gates.gate_path_integration",
    notes="Replaces CGE-A-01. The lesson generalises past this gate: a baseline given information "
          "the component must infer does not measure the component.",
))

register(Gate(
    id="CGE-A-02", version=1,
    title="Communication efficiency at matched rate",
    gate_class=GateClass.A, status=Status.STABLE,
    objective="Does the component's emission policy beat the same budget spent another way?",
    required_property="Precision per event better than a schedule or a coin flip at the "
                      "identical event rate.",
    anti_property="A policy whose apparent benefit is explained by emitting less.",
    control="Periodic and random sampling, thresholds solved numerically so the event rate "
            "matches to within 5%.",
    pass_criterion="Lower reconstruction error than the best matched-rate control, by the "
                   "declared margin, across the swept rate range.",
    levels=(0, 1, 2), modalities=(Modality.VISION, Modality.TOUCH), cost="seconds",
    has_ever_failed="Heron's first oscillation wiring: 19x worse reconstruction while emitting "
                    "MORE, because the rhythm was added into the transmitted value.",
    anti_tests=("noise sigma sweep", "a hard per-tick channel budget with dropped events"),
    ood="partial, and the strongest transfer evidence in the project: the send-on-delta policy was applied to a FROZEN Wren unit's own trace -- a different architecture, different dynamics -- and reconstructed at NRMSE 0.123 at 15% of its rate. That is cross-architecture rather than cross-world, but it is a real shift.",
    implemented_by="cge.components.gate_rate_distortion",
    notes="Heron L0 passes at +92.9%, and the policy transferred to a frozen Wren unit's own "
          "trace. The strongest positive result in the project.",
))

register(Gate(
    id="CGE-A-03", version=1,
    title="Timescale monotonicity",
    gate_class=GateClass.A, status=Status.STABLE,
    objective="Does this layer integrate over a longer window than the layer below it?",
    required_property="State autocorrelation exceeding its own inputs' at its declared horizon.",
    anti_property="A relabelling: a layer that decorrelates as fast as, or faster than, its "
                  "own members.",
    control="The layer's own members, measured at the identical lags.",
    pass_criterion="Higher autocorrelation than its members at every lag up to its declared "
                   "horizon. Driven by a real sensory stream, never white noise -- noise "
                   "decorrelates in two ticks and every layer then looks slow.",
    levels=(1, 2, 3), modalities=(Modality.VISION,), cost="seconds",
    has_ever_failed="Heron's tower layer, first build: 0.882 against its own members' 0.907 at "
                    "its declared horizon. Not one gate in the old battery detected it.",
    anti_tests=("real stream vs white noise, to show the comparison is not an artefact",),
    ood="none. Timescales are measured on the same stream the component was fitted to.",
    implemented_by="cge.nodes.gate_horizon",
    notes="Cheap, and it caught a defect that would otherwise have been attributed to "
          "mechanisms. Fixing it changed no other conclusion, which made the other failures "
          "stronger rather than weaker.",
))

register(Gate(
    id="CGE-A-04", version=1,
    title="Self-sustaining dynamics",
    gate_class=GateClass.A, status=Status.STABLE,
    objective="Can the component's dynamics express what the architecture asks of it?",
    required_property="If the design requires a rhythm, the component sustains one with no "
                      "input, at stable amplitude.",
    anti_property="A dynamics that provably cannot do its job -- a gradient flow asked to "
                  "oscillate, a low-pass filter asked to integrate, an accumulator asked not "
                  "to drift.",
    control="Zero input, plus an ablation with the mechanism removed.",
    pass_criterion="Phase advances, amplitude does not drift, and the rhythm is measured on "
                   "the internal oscillator rather than on the transmitted value.",
    levels=(0, 1, 2, 3), modalities=(), cost="seconds",
    has_ever_failed="Wren. A gradient flow has dE/dt <= 0 and therefore no periodic orbits: it "
                    "could not oscillate, synchronise, or carry a moving quantity. Discovered "
                    "empirically three times before it was proved on paper in minutes.",
    anti_tests=("ablate the mechanism and confirm the rhythm stops",),
    ood="not applicable -- the component is driven with no input at all, so there is no fitting distribution to shift away from.",
    implemented_by="cge.components.gate_oscillation",
    notes="Belongs in Gate A because it is checkable before anything is trained, and because "
          "each of the three impossibilities above cost a full experimental cycle.",
))

register(Gate(
    id="CGE-A-05", version=1,
    title="Clock is not content",
    gate_class=GateClass.A, status=Status.STABLE,
    objective="Does the component's timing mechanism stay out of the values it transmits?",
    required_property="With no input, the transmitted value is exactly still however fast the "
                      "internal oscillator is turning.",
    anti_property="Phase used as content -- the component spending its channel describing its "
                  "own rhythm.",
    control="The same component with the mixing restored, kept runnable as a named ablation.",
    pass_criterion="Transmitted-value activity below 1e-6 with zero input, while the "
                   "oscillator is demonstrably moving.",
    levels=(0, 1, 2), modalities=(), cost="seconds",
    has_ever_failed="Heron L0's first wiring: 19x worse reconstruction while emitting more.",
    anti_tests=("the rejected wiring must still reproduce the failure it documents",),
    ood="none.",
    implemented_by="cge.components.gate_oscillation",
    notes="Kept separate from A-04 because they pull in opposite directions: A-04 wants the "
          "rhythm to exist, A-05 wants it out of the payload. A single combined gate passed "
          "the wrong wiring.",
))

register(Gate(
    id="CGE-A-06", version=1,
    title="Graceful degradation under noise",
    gate_class=GateClass.A, status=Status.EXPERIMENTAL,
    objective="Does precision degrade smoothly as the input is corrupted?",
    required_property="No cliff. Degradation proportional to corruption.",
    anti_property="Catastrophic failure, or a silent switch to a degenerate constant output.",
    control="The same corruption applied to the control, so degradation is distinguishable "
            "from the task simply becoming harder.",
    pass_criterion="Monotone, bounded degradation across the sweep, with the component still "
                   "beating its floor at the highest noise level it claims to tolerate.",
    levels=(0, 1, 2, 3), modalities=(Modality.VISION, Modality.TOUCH), cost="seconds",
    has_ever_failed="",
    anti_tests=("input noise sweep", "missing input", "delayed feedback", "timing shift"),
    ood="input noise, swept 0.0 / 0.1 / 0.3, applied at scoring only. A corruption shift rather than a world shift, and the only shift any gate here currently applies.",
    implemented_by="cge.components.gate_noise_robustness",
    notes="EXPERIMENTAL: nothing has failed it yet, so it does not count toward compliance. "
          "The likely reason is that the sweep is too gentle -- sigma 0.3 on a normalised "
          "signal is mild, and the anti-tests beyond noise are specified but not implemented.",
))

# ===========================================================================================
# Gate B — architectural validation. Only meaningful once Gate A passes.
# ===========================================================================================

register(Gate(
    id="CGE-B-00", version=1,
    title="Spatial memory beyond the current view",
    gate_class=GateClass.B, status=Status.STABLE,
    objective="Does the component represent structure that is currently out of sight?",
    required_property="Decoding of out-of-view structure above what the visible frame supports.",
    anti_property="Apparent memory that is deducible from the visible part -- a maze whose "
                  "regularity lets pixels predict cells they cannot see.",
    control="The raw retina on the same cells. Its accuracy ON-screen must be high (the probe "
            "works) and its accuracy off-screen is the bar.",
    pass_criterion="Out-of-view decode above the raw-frame control by the declared margin.",
    levels=(1, 2, 3), modalities=(Modality.VISION, Modality.PROPRIOCEPTION), cost="minutes",
    has_ever_failed="Swift (73.6%) and Heron (61.9%) both fall below the 77.3% pixel control. "
                    "Only Wren clears it, at 84.1%.",
    anti_tests=("maze braiding, because a DFS-carved maze let pixels score 94% on cells they "
                "could not see -- the gate was measuring parity, not memory",),
    ood="none. One maze topology, one seed per run, time split.",
    implemented_by="cge.gates.gate_maze",
    notes="The braiding anti-test is the clearest example in the suite of a gate that passed "
          "for the wrong reason until its control was taken seriously.",
))

register(Gate(
    id="CGE-B-01", version=1,
    title="Object identity through occlusion",
    gate_class=GateClass.B, status=Status.STABLE,
    objective="Can the component say WHICH object is hidden, not merely that something is?",
    required_property="Identity that survives the object becoming invisible.",
    anti_property="Identification by elimination from the objects still visible, which looks "
                  "exactly like memory and is not.",
    control="Pixels local to the object: must decode identity when visible (the probe works) "
            "and must be at chance when hidden (nothing is leaking).",
    pass_criterion="Balanced accuracy above chance on the hidden object, with both control "
                   "conditions satisfied.",
    levels=(1, 2, 3), modalities=(Modality.VISION,), cost="minutes",
    has_ever_failed="Wren, at chance on the identity of an object in PLAIN VIEW. This gate "
                    "refuted the project's original headline claim of emergent object "
                    "permanence.",
    anti_tests=("probe local to the object, never whole-frame, to block elimination",
                "balanced classes so a constant guess scores 0.5"),
    ood="none. Occlusion is present during fitting as well as scoring.",
    implemented_by="cge.gates.gate_identity",
    notes="The local-probe requirement is not fussiness: a whole-frame probe can name the "
          "hidden object from the three still visible.",
))

register(Gate(
    id="CGE-B-02", version=1,
    title="Binding: does the grouping carry object identity?",
    gate_class=GateClass.B, status=Status.STABLE,
    objective="Whatever the architecture calls a group -- coalition, concept, cluster -- does "
              "membership track objects in the world?",
    required_property="Mutual information between grouping and object identity, above a null.",
    anti_property="Mutual information that is an artefact of label frequency.",
    control="A paired shuffled-label null, plus clustering the raw input at matched k. Each "
            "architecture labels its OWN units at its own granularity, or the finer one wins "
            "for being finer.",
    pass_criterion="Excess over the shuffled null exceeding three standard errors, AND above "
                   "clustering the raw input at the same number of groups.",
    levels=(2, 3), modalities=(Modality.VISION,), cost="minutes",
    has_ever_failed="Heron's knowledge concepts: +0.009 excess, and 14x WORSE than k-means on "
                    "the raw input patch. Wren's coalitions: +0.001, indistinguishable from "
                    "the null. Swift is the only pass, at +0.124.",
    anti_tests=("paired shuffled null, which exposed an apparent MI of 0.449 as entirely bias",
                "matched granularity across architectures"),
    ood="none.",
    implemented_by="cge.gates.gate_coalitions",
    notes="The one gate where the OLDEST architecture wins. Swift's synchrony grouping binds "
          "14x better than the mechanism that replaced it -- and it was discarded on evidence "
          "about synchrony as a representation, which is a different question.",
))

register(Gate(
    id="CGE-B-03", version=1,
    title="Aggregation earns its place",
    gate_class=GateClass.B, status=Status.STABLE,
    objective="Does combining a population into a shared state beat leaving the population "
              "alone?",
    required_property="A summary that preserves what its members carried.",
    anti_property="Compression that destroys the signal, at any level of sophistication.",
    control="Pass-through (the members concatenated) and mean pooling, both at matched width "
            "and matched capacity.",
    pass_criterion="Beats BOTH controls at the declared horizon by the declared margin.",
    levels=(2, 3), modalities=(Modality.VISION,), cost="minutes",
    has_ever_failed="Four times, with four operators. The legacy assembly turned a 0.69x mesh "
                    "state into a 6.56x workspace. Heron: mean pooling 0.598 BEAT the "
                    "bilinear sketch 0.612 and exact pairwise 0.629, on both targets.",
    anti_tests=("cross the operator against the workspace dynamics -- all six cells landed "
                "within 5%, so neither factor was responsible",
                "the exact operator as well as the sketch, so a negative cannot be blamed on "
                "the approximation"),
    ood="none. Compression is fitted and scored on the same trajectory.",
    implemented_by="cge.nodes.gate_relational",
    notes="This gate retired the strongest result the project ever produced. Its floor should "
          "be pass-through, which inverts the burden of proof onto the compression step.",
))

register(Gate(
    id="CGE-B-04", version=1,
    title="Narrow interface sufficiency",
    gate_class=GateClass.B, status=Status.EXPERIMENTAL,
    objective="Is a component's published state enough for the layer above, or must it "
              "transmit its contents?",
    required_property="A reader given only the published scalars performs comparably to a "
                      "reader given the full internal state.",
    anti_property="A publication so narrow that the layer above cannot function, or so wide "
                  "that it is the internal state under another name.",
    control="The same reader given the full internal state, at matched capacity.",
    pass_criterion="Published-only within a declared factor of full-state, AND the published "
                   "form still beats the component's own floor.",
    levels=(1, 2, 3), modalities=(), cost="minutes",
    has_ever_failed="",
    anti_tests=("verify the publication contains no state vector, or the gate compares the "
                "full state against a copy of itself",),
    ood="none.",
    implemented_by="cge.nodes.gate_publication",
    notes="EXPERIMENTAL. Heron passes at 1.50x -- its only pass -- but the result needs its "
          "caveat: the bottleneck was cheap relative to a full state that was itself worse "
          "than raw input. 'Cheap to compress' is not 'worth transmitting', and the second "
          "clause of the pass criterion exists to close that hole.",
))

register(Gate(
    id="CGE-B-05", version=1,
    title="Prediction against persistence",
    gate_class=GateClass.B, status=Status.STABLE,
    objective="Does the component predict the world better than assuming nothing changes?",
    required_property="Error below persistence at the component's declared horizon.",
    anti_property="A win that is really the baseline decaying -- persistence degrades faster "
                  "than anything else as the horizon grows, so every representation looks "
                  "better the further out it is asked to predict.",
    control="Copy-last, AND the raw frame at the identical horizon, which removes the trend "
            "above.",
    pass_criterion="Beats persistence at the declared horizon, and beats the raw frame there "
                   "too.",
    levels=(0, 1, 2, 3), modalities=(Modality.VISION,), cost="minutes",
    has_ever_failed="All three architectures at tau=1 (Wren 3.41x, Swift 4.06x, Heron 4.85x -- "
                    "all far worse than copy-last), and all three against the raw frame at "
                    "every horizon.",
    anti_tests=("normalise by the raw frame at the same tau, which changed the reading of "
                "every horizon curve in the project",),
    ood="a held-out test stream from the same world -- a time shift, not a distribution shift. The world, the objects and the dynamics are identical.",
    implemented_by="cge.gates.gate_prediction",
    notes="The anti-test is the important part. Scored against persistence alone, the first "
          "version of this gate concluded that everything on the bench was a 64-tick model.",
))

register(Gate(
    id="CGE-B-10", version=1,
    title="Coordination earns its place",
    gate_class=GateClass.B, status=Status.STABLE,
    objective="Does knowing which components belong together tell you anything that the "
              "components do not already tell you about themselves?",
    required_property="A component's group-mates improve prediction of that component's own "
                      "future beyond what the component alone supports -- and the grouping "
                      "rule beats whatever grouping was hard-coded.",
    anti_property="A layer that groups on every tick with nothing to beat. Compliance without "
                  "contribution: it satisfies its contract while doing no measurable work.",
    control="Two, and both are required. `independent_towers` -- the component alone, no "
            "grouping at all. `fixed_proximity_membership` -- the positional rule already in "
            "the code, so a win cannot be 'more inputs help'.",
    pass_criterion="The grouping rule beats BOTH controls at the layer's declared horizon by "
                   "the declared margin, at matched capacity.",
    levels=(2, 3), modalities=(Modality.VISION,), cost="minutes",
    has_ever_failed="Corvus L2, first run. Connectivity-derived membership scored -0.000 "
                    "+/- 0.001 against no grouping at all: coordination contributes nothing. "
                    "It did beat the hard-coded positional rule, consistently across three "
                    "seeds, by +0.010 -- a real effect five times below the margin.",
    anti_tests=("match capacity across conditions, or the gate rewards width: a group offers "
                "m*width numbers where a component alone offers width",
                "report membership overlap between the rules, which separates 'grouping does "
                "not matter' from 'the two rules picked the same components'",
                "derive every grouping from ONE trajectory, since the layer does not feed back "
                "into the components it groups",
                "grow every rule from identical seeds, so the rules differ only in which "
                "components they pull in"),
    ood="none. Membership is derived and scored on one trajectory, deliberately, because the layer does not feed back into the towers it groups. That makes the comparison clean and says nothing about transfer.",
    implemented_by="experiments.corvus_l2_coordination",
    notes="Written for Q7 option B after Layer 2 was found to have declared a floor only over "
          "an optional job while its unconditional job -- coordination -- had none. The gate "
          "failed the layer on its first run, which is what a floor is for. The second control "
          "came from a cortical-column critique proposing connectivity over proximity; making "
          "that claim falsifiable cost one gate instead of an architecture.",
))

# ===========================================================================================
# Gate C — external comparison. Only after Gate B. Nothing here is implemented.
# ===========================================================================================
#
# Deliberately empty of implementations. Running a Gate C comparison today would measure which
# of two systems is further behind its own input. The tasks are specified in docs/SBB.md,
# written before anyone knows who would win, which is the only moment that is a fair test.

for _id, _title, _obj in (
    ("CGE-C-00", "Incremental learning without forgetting",
     "Learn A, then B, then C; measure retention on A against a freshly-trained control."),
    ("CGE-C-01", "Consolidation improves performance without new data",
     "Does an offline phase with no input make the model better AND simpler?"),
    ("CGE-C-02", "Episodic recall after long intervening experience",
     "Retrieve a specific past episode, not a summary, with distractors present."),
    ("CGE-C-03", "Cross-domain transfer of dynamics",
     "Learn structure in one world; measure learning speed in a second sharing it."),
    ("CGE-C-04", "Novel regularity discovery",
     "Find a regularity never demonstrated and not lookup-able, verified on held-out cases."),
):
    register(Gate(
        id=_id, version=0, title=_title,
        gate_class=GateClass.C, status=Status.PROPOSED,
        objective=_obj,
        required_property="Specified in docs/SBB.md.",
        anti_property="Specified in docs/SBB.md, including how each task could be unfair.",
        control="A fine-tuned baseline tested for catastrophic forgetting; retrieval "
                "augmentation where memory is the subject; the shared planner where action is.",
        pass_criterion="To be fixed before the first run, never after.",
        levels=(3, 4, 5, 6), modalities=(Modality.VISION, Modality.CROSS_MODAL), cost="hours",
        notes="Entry condition: some architecture must first pass CGE-A-00 and CGE-A-01. "
              "Until then this comparison has no interpretation.",
    ))


# ===========================================================================================
# Placeholders — gates the specification names and the suite does not yet have.
# ===========================================================================================

for _id, _title, _obj, _lvl in (
    ("CGE-A-07", "Cross-modal integration",
     "When two modalities disagree, which evidence dominates, and is the arbitration sensible?",
     (1, 2, 3)),
    ("CGE-A-08", "Novelty detection",
     "Is an anomaly in a familiar sequence flagged, above a false-positive-matched control?",
     (0, 1, 2)),
    ("CGE-B-06", "Consensus formation under disagreement",
     "With members holding incompatible hypotheses, is the wrong one suppressed more often "
     "than chance and more often than with the coordinating layer removed?", (2, 3)),
    ("CGE-B-07", "Sensorimotor adaptation",
     "Does acting improve prediction, measured against the same agent acting at random?",
     (1, 2, 3)),
    ("CGE-B-08", "Recovery after interrupted communication",
     "After partial message loss, does coordination return to its prior level, and how fast?",
     (2, 3)),
    ("CGE-B-09", "Knowledge survives frozen learning",
     "With plasticity switched off mid-run, does performance hold -- i.e. is it memory rather "
     "than ongoing fitting?", (1, 2, 3)),
):
    register(Gate(
        id=_id, version=0, title=_title,
        gate_class=GateClass.A if "-A-" in _id else GateClass.B,
        status=Status.PROPOSED,
        objective=_obj,
        required_property="To be specified with the implementation.",
        anti_property="To be specified with the implementation.",
        control="Must be named before implementation begins. A gate whose control is decided "
                "afterwards is a number.",
        pass_criterion="Behavioural, and expressed as beating a named baseline.",
        levels=_lvl, modalities=(Modality.VISION,), cost="unknown",
        notes="PROPOSED: does not count toward compliance.",
    ))


def audit() -> dict:
    """What the suite can and cannot currently decide. Printed by `python -m cge.catalogue`."""
    counting = [g for g in CATALOGUE.values() if g.counts_toward_compliance]
    never_failed = [g.id for g in counting if not g.has_ever_failed]
    no_modality = [g.id for g in counting if not g.modalities]
    audio = [g.id for g in CATALOGUE.values() if Modality.AUDIO in g.modalities]
    return {
        "total": len(CATALOGUE),
        "counting_toward_compliance": len(counting),
        "proposed_only": len(CATALOGUE) - len(counting),
        "by_class": {c.value: len(by_class(c)) for c in GateClass},
        "implemented_but_never_discriminated": never_failed,
        "modality_independent": no_modality,
        "audio_gates": audio,
        "honest_limitation": (
            "No audio gate exists, and no cross-modal gate is implemented. The suite is "
            "vision plus a coarse touch map plus an efference copy. 'Cognition is multimodal' "
            "is a stated intention here, not a property of the current battery."),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(audit(), indent=2))
