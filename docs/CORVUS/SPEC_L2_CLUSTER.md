# SPEC L2 — Tower Cluster

**OCA v4 · Layer 2 · status: specified, blocked on [Q3](OPEN_QUESTIONS.md) and on Layer 1**

A collection of cooperative towers. Responsible for solving coherent local cognitive problems,
providing local consensus before higher-level integration, and coordinating specialised tower
populations.

**This layer's core responsibility is the single most-failed operation in this project's
history.** Aggregating a population into a shared state has now been tried four times, with
four different operators, and has never once beaten leaving the population alone.

---

## Component contract

| field | specification |
|---|---|
| **Purpose** | Make a population of towers into something that solves a problem no single tower can. |
| **Responsibilities** | Coordinate specialised towers; resolve conflicting local hypotheses; maintain cross-tower referent identity; expose one coherent local result. |
| **Internal State** | Membership · conflict/agreement structure · whatever survives [Q3](OPEN_QUESTIONS.md). |
| **Inputs** | The published states of its member towers. Never their internals. |
| **Outputs** | A cluster-level published state, or — see Q3 — its members' states unchanged. |
| **Operations** | `recruit` · `coordinate` · `resolve` · `dissolve` |
| **Communication Model** | Sparse and event-driven. Membership is itself a communicated fact. |
| **Lifecycle** | **Transient by design.** A cluster forms while it is useful and dissolves when it is not, and "useful" must be a measured quantity, not a configured one. |
| **Dependencies** | Layer 1 only. |
| **Temporal Horizon** | Declared, **> its towers'**. |
| **Integration Window** | Measured, **> its towers'**. Checked by `corvus.contract.check_stack`. |
| **Floor** | Beats **`pass_through`** — the concatenated published states of its own members, at matched capacity — by ≥ 5%. |
| **Failure Modes** | Aggregation that destroys what its members carried (measured four times); consensus that hides a real conflict; membership that never changes, making it a fixed region. |
| **Scalability** | Relations between members grow as N². Any implementation must state how it stays sub-quadratic, or cap N. |
| **Known Limitations** | Cannot create information its towers do not have. Its ceiling is `pass_through`, and the whole question is how much of that ceiling survives. |

## Why the floor is `pass_through`, and why that inverts the burden of proof

This is the substantive change v4 makes to this layer.

Four attempts, all measured with the members, target and protocol held fixed:

| attempt | operator | result |
|---|---|---|
| legacy Predictive Assembly | mean workspace | mesh state 0.69× → workspace **6.56×**. The aggregation destroyed the signal. |
| Heron L1 | random bilinear sketch of co-activation | 0.612 |
| Heron L1 | **exact** pairwise co-activation | 0.629 |
| Heron L1 | **mean pooling — the control** | **0.598, best of the three** |

The last row retired the strongest result this project ever produced ("the predictive code is
relational, not additive"). Both escape routes were closed: the exact operator loses too, so it
is not the sketch; and crossing the operator against the workspace dynamics put all six cells
within 5%, so it is not the filter either. Nothing about the aggregation mattered, and none of
it beat the raw frame.

So: **a cluster must prove it beats leaving its towers alone before it is permitted to summarise
them.** If it cannot, the compliant behaviour is to pass through. That is
[Q3](OPEN_QUESTIONS.md), and it is the recommendation.

This is not a claim that consensus is impossible. It is a claim that consensus is the step that
has failed every time it has been measured, and should therefore carry the burden of proof
rather than be granted it.

## ⚑ Challenged against the record

**"Provides local consensus before higher-level integration" should not be an unconditional
responsibility.** As written, every implementation must compress whether or not compression
helps. See above and Q3.

**Transient membership is supported, and by the mechanism v4 discarded.** The one thing that
worked at this scale was **Swift's synchrony coalitions**: they carry **+0.124** object MI above
a shuffled null, against Heron's knowledge-based concepts at **+0.009**. Fourteen times better
binding, from the mechanism the newer architecture threw out.

It was thrown out on a measurement about synchrony as a *representation to predict from* (exactly
1.00× persistence), which said nothing about synchrony as a *grouping*. That is one of the two
documented cases of using a measurement to close a question it did not answer.

**Concretely: dynamic membership by coherence is the best-evidenced mechanism available for this
layer**, and it should be transplanted in isolation and scored here rather than reinvented.
Note precisely what it is evidence for — grouping, not representation. Membership may be decided
by coherence; the cluster's *published state* may not be a synchrony graph.

**Cross-tower identity is unowned, and that is a real gap.** If entities live inside towers
(Q2 option a), then when a referent crosses a tower boundary this layer has to re-establish that
it is the same referent — which is the aggregation problem again, in its hardest form. If
entities are cross-cutting (Q2 option b), this layer inherits them and the problem does not
arise. **Q3 cannot be settled independently of Q2.**

**Conflict resolution has never been tested.** It is a plausible and completely unmeasured
responsibility. Its gate should be written before it is built: with two towers holding
incompatible hypotheses, does the cluster suppress the wrong one more often than chance, and
more often than the same towers with the cluster removed?

**Blocked** on Layer 1 passing its floor. A coordination layer over towers that hold no
persistent state cannot produce persistent state, and building it first is how three
architectures got as far as they did before anyone noticed.
