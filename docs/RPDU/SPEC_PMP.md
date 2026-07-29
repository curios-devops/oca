# SPEC_PMP — the Predictive Mesh Protocol

Specification of what passes between units. Companion to [SPEC_RPDU.md](SPEC_RPDU.md).

The architecture document's closing argument is that this file may matter more than that
one — that intelligence lives in the protocol by which units negotiate a shared
interpretation, not in the units themselves. This spec is written so that claim is
testable: the protocol has its own ablations, and E3 measures how much of the effect it
carries.

## 1. Topology

Units sit on a 2-D lattice (`√N × √N`, wrapping). Each unit has a fixed in-degree budget:

| class | count | initialised | plastic? |
|---|---|---|---|
| local | 32 | 8-neighbourhood out to radius 2 | gain only |
| long-range | 8 | uniformly random non-local | **yes — rewired** |
| executive | 1 | to one of the 20 executive units | no |

Stored as an `(N, 41)` int array of source indices plus parallel `(N, 41)` arrays for
gains and per-link state. The out-link view is derived once per rewiring, not per tick.

The budget is deliberately hard. Attention lets every token reach every other token, and
the architecture document's bet is that the cortex does not and does not need to. A fixed
41-link budget makes the mesh `O(N)` per tick instead of `O(N²)`, so it is also the reason
this runs on a laptop.

## 2. The message

Each directed link carries exactly four scalars per tick:

```
msg[i→j] = ( confidence c, expectation e, novelty n, importance ι )
```

Four numbers is almost nothing, and that is the point of the design — but four scalars
alone cannot carry a belief. **The resolution: the basis lives in the link, the
coefficient lives in the message.**

- The **sender** owns a projector `u_ij ∈ R^d` per out-link and sends
  `e = u_ij · ĥ_i^{τ=1}` — the coefficient of its predicted next state along that
  link's private channel.
- The **receiver** owns a read vector `a_ji ∈ R^d` and injects
  `w_ji · γ · e · a_ji` into its drive (SPEC_RPDU §2).

So a link is a learned 1-D communication channel between two 16-dimensional dynamical
systems, and the wire format stays four floats. Bandwidth per tick is `4 · 41 · N` floats,
which at `N = 512` is 84k numbers — against roughly 34M for dense 16-d all-to-all
attention over the same population.

The remaining three fields:

```
c = 1/(1 + σ̂_i)              sender's precision — how much it trusts this expectation
n = S_i / (μ_i + ϵ)           novelty — normalised surprise, "something changed here"
ι = tanh(κ_i · n · c)         importance — the sender's bid for the receiver's bandwidth
```

`κ_i` is a per-unit learned scalar, adapted so that a unit whose messages consistently
help its receivers bids higher (§4). The receiver combines them into one gate:

```
γ_ji = c · (1 + ι)            confidence-weighted, importance-boosted
```

Precision weighting is the mechanism the design document is reaching for when it says the
brain behaves differently when uncertain: an uncertain sender's expectation is
automatically discounted by every receiver, with no controller deciding that.

## 3. Timing

The mesh is **synchronous with a one-tick delay**. All units read the buffer written at
`t-1` and write the buffer for `t+1`. There is no in-tick propagation and no sweep order.

Consequences, all wanted: the update is embarrassingly parallel; the result does not
depend on unit indexing; runs are bit-reproducible; and information physically takes one
tick per hop, so the mesh has a real signal-propagation speed and distant units genuinely
cannot coordinate instantaneously. Coalition formation therefore has to happen through
dynamics rather than through a global read.

## 4. Structural plasticity — rewiring

Every `T_rewire = 200` ticks, each unit re-evaluates its 8 long-range in-links. This is
the architecture document's "who consistently helps me reduce prediction error", made
operational.

Credit per in-link, updated every tick as an EMA (`λ_cr = 0.005`):

```
cr_ik ← (1-λ_cr) · cr_ik + λ_cr · ( ê_ik(t) - e_ik(t) )² ⁻ ¹ · ΔS_i(t)
                                    ^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^
                                    how predictable this      how much this
                                    link was                  unit's surprise fell
```

Read plainly: a link earns credit when its traffic is *both* predictable by the receiver
*and* coincident with the receiver getting less surprised. A link that is pure noise is
unpredictable and earns nothing; a link that is perfectly predictable but tells the
receiver nothing new produces no `ΔS` and also earns nothing. Both failure modes are
pruned by the same rule.

At each rewiring event, per unit:

1. Prune the 2 long-range links with the lowest credit.
2. Grow 2 replacements, sampled with probability `∝ exp(corr(n_i, n_j) / T_samp)` over
   candidate sources `j`, using a novelty correlation the executive maintains over a
   random projection of the population (`T_samp = 0.3`). Units that are surprised together
   are likely to be looking at the same thing.
3. Initialise new links with `w = 0.1`, random `u`, `a`, and credit at the population
   median (so a new link gets a grace period rather than being pruned on its first
   evaluation).
4. Update `κ_i` by the mean credit the unit's *out*-links earned, so bidding is calibrated
   by usefulness rather than by self-assessed importance.

Step 4 is the only place a scalar crosses a unit boundary in the direction of learning,
and it is deliberately impoverished — one number, every 200 ticks, about a unit's own
broadcasting, never a gradient. If this turns out to be doing the heavy lifting, that is a
finding and it must be reported as one, not hidden. The E3 no-rewiring ablation isolates
exactly this.

## 5. The executive

20 units with two jobs and no intelligence of their own. They are the last 20 lattice
indices rather than a separate population: ordinary RPDUs whose only distinction is that
every unit holds one in-link to one of them, making them the mesh's shared bus. Keeping
them inside the lattice avoids a second code path for a population of 20.

**Global context broadcast.** Each tick the executive maintains

```
g(t) = tanh( R · h̄(t) ) ∈ R^q ,    h̄ = population mean state, R a fixed random (q × d) projection
```

This is the target for every unit's global-context head (SPEC_RPDU §3.4) — the answer to
"what brain state am I participating in?". `R` is fixed, not learned, so `g` is a cheap
summary rather than a second model.

**Novelty map.** A running `(N,)` novelty EMA and its random-projection covariance, used
only by the rewiring sampler in §4.

The executive does not select actions, gate attention, or route information in this POC.
The architecture document assigns it goals, curiosity and planning; those need a task with
a reward, and this POC has none by design. Left as a stub with the interface in place.

## 6. Coalitions — measured, not programmed

Nothing in this spec creates coalitions. They are detected, and if the hypothesis is wrong
the detector will simply find nothing stable.

Every 10 ticks, build a coherence graph over existing links:

```
coh_ij = corr_window( h_i, h_j )·  γ_ij  , window = 32 ticks
edge kept if coh_ij > θ_coh = 0.6
```

Run 5 rounds of asynchronous label propagation over the kept edges. Log per snapshot:
count, size histogram, membership churn against the previous snapshot, and lifetime
distribution of labels tracked by Jaccard overlap ≥ 0.5.

One and only one feedback path back into the dynamics: units sharing a label multiply
their mutual `γ` by `1 + 0.2`. This is the transient synchronisation of SPEC_RPDU, and it
is the mechanism by which a settled coalition becomes briefly self-reinforcing — the
"settling of the network" the design document calls thought. It is also an ablation.

## 7. Protocol ablations (feed E3)

| ablation | what it kills | prediction if the hypothesis holds | measured |
|---|---|---|---|
| message width 4 → 1 (expectation only) | precision weighting, bidding | error rises; uncertainty handling degrades most | **1.00x** |
| no rewiring | structural plasticity | slower learning, worse long-range structure | 1.03x |
| no long-range links | small-world topology | coalitions stay spatially local; global context head fails | 1.02x |
| no coalition feedback | transient synchronisation | coalitions still detected but shorter-lived | 0.99x |
| no oscillator | shared clock | phase alignment between distant units disappears | 0.98x |
| one-tick delay → instant | dynamics-based settling | error may *improve*, which would be evidence the timing story is unnecessary | 1.03x |

The last row was included on purpose, as the ablation most likely to embarrass the
hypothesis. In the event it barely mattered — but neither did any of the others.

**These predictions were wrong.** Every row lands within 3% of the full model, and this
protocol is, on the world tested, essentially inert. See [RESULTS.md](RESULTS.md) for the
measurements and for what would have to change to give the protocol something to do.
