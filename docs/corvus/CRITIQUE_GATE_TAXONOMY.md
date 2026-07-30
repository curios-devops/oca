# The capability-taxonomy critique, assessed

*2026-07-30. A review proposing that gates be organised around **cognitive capacities** rather
than tasks, with twelve capacities drawn from comparative psychology, each measurable by several
challenges.*

**Verdict: adopt the principle, adopt one property outright, and put most of the list in the EIS
rather than the CGE.** One idea in the critique is the most valuable contribution anyone has made
to this benchmark. One is a naming inconsistency in our own code that the critique spotted before
we did. The twelve-capacity list is excellent and is a better draft of the *Emergent Intelligence
Specification* than the one already in `docs/EIS.md`.

---

## 1. The naming critique is correct, and it is our bug

The critique is right that a benchmark named after tasks overfits to tasks. It is right *about
our code specifically*:

```python
GATES = {
    "prediction",        # a capacity
    "path_integration",  # a capacity
    "identity",          # a capacity
    "maze",              # a WORLD
    "tunnel",            # a WORLD
    "coalitions",        # a MECHANISM
}
```

Three of six are named after the thing they run in rather than the thing they measure.

But note where the inconsistency is. **The `cge/catalogue.py` already does this correctly** —
every gate there carries a `title`, an `objective`, a `required_property` and an `anti_property`,
and none of them names a world. `CGE-B-03` is *"Aggregation earns its place"*, not *"maze"*. The
task names live only in `cge/gates.py`, the runner.

So this is a real fix and a cheap one: align the runner with the catalogue that is already right.

| runner key | measures | should be |
|---|---|---|
| `maze` | out-of-view wall decode | `spatial_memory` |
| `tunnel` | *(deprecated — `CGE-A-01` was not measurable)* | — |
| `coalitions` | whether synchrony groups carry object identity | `object_binding` |

**One correction to the critique.** It maps `coalitions` onto `social_reasoning`. That is wrong:
the gate measures whether groups of units carry information about *objects* — perceptual binding,
not social cognition. Renaming it that way would make the suite less accurate, not more.

## 2. OOD is the best idea in the message, and it is not a gate

> *"El agente entrena en un conjunto de condiciones y debe superar variantes nuevas. Ahí es donde
> realmente se mide si aprendió una capacidad cognitiva y no simplemente una tarea específica."*

**We have no out-of-distribution testing anywhere.** Every gate in this project trains and tests
on the *same world* with a *time split*. The `--sides 12 24` stress rebuilds the architecture at a
new size; it does not ask a trained one to transfer.

That is a structural hole, and it is precisely the hole that lets a benchmark reward task-fitting.
It also has teeth against our own results: **we do not currently know whether Corvus's +0.572 on
path integration transfers to a maze it was not trained in.** Nobody asked. It is the same class
of omission as Wren's +0.282 sitting undiscovered for two architectures.

**But OOD is not a thirteenth gate. It is a property every gate must declare**, at the same level
as `control` and `has_ever_failed`:

> **`ood`** — which distribution shift this gate applies between fitting and scoring, or an
> explicit admission that it applies none. A gate with no declared shift is measuring performance
> on the distribution it was fitted to, and must say so.

That placement matters. As a gate, OOD gets run once and forgotten. As a mandatory field, **every
existing result acquires a caveat it currently hides**, which is the more useful outcome and is
exactly how `control` earned its place.

## 3. The other two properties: one already satisfied, one already our rule

**"Continuous metrics, not pass/fail."** Already true. Every gate returns a continuous `headline`
plus its control; `PASS / CONDITIONAL_PASS / FAIL / UNMEASURED` is a *decision layer* over the
number, not a replacement for it. `cge/outcomes.py` requires a reason for every non-pass. No
change.

**"Validated in humans and animals."** Agreed and worth writing down, with a caveat the critique
does not mention: an animal protocol validated on animals is not automatically valid on an
architecture with a 5×5 view and four actions. Delayed match-to-sample is meaningful for a crow
because a crow *could* solve it many ways; here it can be passed by a filter with a long enough
time constant. **Every imported protocol still needs its anti-test** — the cheapest system that
passes it without the capability. That requirement already exists in `docs/EIS.md` and it does not
relax because a paradigm is famous.

## 4. The answer to the direct question: two batteries, and this list is the EIS

> *¿Existen dos baterías (EIS y CGE) o concentramos todo en gates CGE?*

**Two.** And the boundary is already written; the proposed list mostly falls on the far side of it.

The operational test is one question: **can you ask a single layer this?**

| capacity | askable of one layer? | battery |
|---|---|---|
| prediction | yes — probe its readout | **CGE** |
| path integration | yes | **CGE** |
| identity / invariance | yes | **CGE** |
| object binding | yes | **CGE** |
| working memory | no — a behaviour of the whole agent | **EIS** |
| object permanence *(Piaget)* | no | **EIS** |
| categorization | no | **EIS** |
| relational reasoning | no | **EIS** |
| cognitive flexibility | no | **EIS** |
| spatial navigation | no | **EIS** |
| planning | no | **EIS** |
| causal reasoning | no | **EIS** |
| social reasoning | no | **EIS** |

You cannot ask a neuron whether it does reversal learning. That is the whole distinction.

**And the same capability legitimately appears in both, at different levels.** Object permanence
is the clearest case: `CGE-A-09` asks whether a *layer's state* contains displacement while its
referent is unobservable; a Piaget hiding task asks whether the *system behaves* as though the
object still exists. Both are worth having and neither substitutes for the other — which is the
aircraft analogy exactly. CGE: can it take off, turn, hold stability, land. EIS: can it cross the
Atlantic.

So the recommendation is to **replace the seven levels currently in `docs/EIS.md` with these
twelve.** The existing list was written by me against nothing; this one carries actual
experimental protocols from comparative psychology, which is strictly better material.

## 5. What we cannot build, stated plainly

Four of the twelve are not gate designs. They are world-building projects, and calling them gates
would put unbuildable items in a catalogue that is already 12/26 unimplemented:

| capacity | what is missing |
|---|---|
| **planning** | a task with a horizon where greedy fails. Tower of Hanoi and Sokoban need an agent that manipulates objects; ours moves through a maze |
| **causal reasoning** | interventions. A lever that dispenses food and then breaks requires an action space with consequences beyond locomotion |
| **social reasoning** | a second agent. There is no multi-agent world here at all |
| **categorization** | labelled object classes with **held-out instances**. Our worlds have three object kinds and no notion of a novel member of a known class |

**Working memory (DMTS), object permanence, cognitive flexibility (reversal learning) and
relational reasoning are all buildable now**, in worlds close to what exists. Those four are the
right next batch, and DMTS is the cheapest — it is close to the delayed cause-and-effect world
already specified in the archived plan.

## 6. On the consciousness remark

> *"predecir la otra mente... creo que ahí puede emerger por primera vez la conciencia sintética"*

Modelling another agent to predict it is a real, implementable and genuinely interesting research
direction, and it is the natural top of this ladder. It is also the point where the EIS's own
discipline has to hold hardest: **there is no measurement that distinguishes "models another
agent" from "is conscious"**, and a system that predicts a second agent well would be evidence of
the first and none of the second.

That is not a reason to avoid the gate. It is the reason the gate needs its anti-test written
first — the cheapest thing that predicts another agent without modelling it is a model of the
*environment* the other agent is reacting to, which is exactly what would be mistaken for theory
of mind.

---

## Recommendation

| proposal | verdict |
|---|---|
| Organise gates around capacities, not tasks | **Yes.** Cheap, and the catalogue already does it — the runner is the inconsistency |
| Rename `maze` → `spatial_memory`, `coalitions` → `object_binding`, keep old keys as aliases | **Yes**, same pattern as the registry's codename aliases |
| Map `coalitions` → `social_reasoning` | **No.** It measures perceptual binding |
| Add OOD | **Yes, and as a mandatory `Gate` field**, not a gate. The single most valuable idea here |
| Continuous metrics | already satisfied |
| Twelve capacities as CGE gates | **No** — as the **EIS**, replacing the seven currently drafted |
| Build planning / causal / social / categorization now | **No.** They need worlds and an action space we do not have; say so rather than list them as gates |
| Build DMTS, object permanence, reversal learning, relational reasoning | **Yes**, in that order. All four are reachable from the worlds we have |

The critique's own principle — *a capability must be measurable by more than one challenge, or you
have measured the challenge* — is the argument for putting it in the EIS. A component contract is
measured one way by definition. A capability is not.
