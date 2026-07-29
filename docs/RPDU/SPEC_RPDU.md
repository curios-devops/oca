# SPEC_RPDU — the local machine

Formal specification of one Recursive Predictive Dynamical Unit. Companion to
[SPEC_PMP.md](SPEC_PMP.md), which specifies what happens *between* units.

Everything here is local: a unit reads its own state, its own parameters, and the
messages that arrived on its in-links. It never reads another unit's state, another
unit's error, or any global gradient. That restriction is the whole point — it is the
claim the experiments are built to falsify.

## 0. Notation and sizes

| symbol | meaning | default |
|---|---|---|
| `N` | units in the mesh | 512 |
| `d` | latent width per unit | 16 |
| `r` | rank of the landscape | 3 |
| `p` | sensory patch width (sensory units only) | 16 |
| `q` | global context summary width | 8 |
| `T` | prediction horizons | {1, 4, 16} |
| `D` | in-links per unit (32 local + 8 long + 1 exec) | 41 |
| `L` | relaxation sub-steps per tick | 2 |

The mesh is stored as arrays, never as `N` Python objects. Every quantity below with a
unit subscript `i` is one row of an `(N, …)` array, and every rule is applied to all
units at once with vectorized numpy.

## 1. State

Each unit owns:

| tensor | shape | role | changes on |
|---|---|---|---|
| `h_i` | `(d,)` | latent state — the unit's current belief | every tick |
| `U_i` | `(d, r)` | landscape basis | slow (plasticity) |
| `b_i` | `(d,)` | landscape resting tilt | slow |
| `P_i^τ` | `(p, d)` | sensory head, per horizon | slow |
| `M_i^τ` | `(D, d)` | neighbour-message head | slow |
| `C_i^τ` | `(q, d)` | global-context head | slow |
| `v_i, v0_i` | `(d,)`, scalar | surprise head | slow |
| `a_ik` | `(d,)` per in-link | receiver-owned read vector | slow |
| `u_ik` | `(d,)` per out-link | sender-owned channel projector | structural only |
| `w_ik` | scalar per in-link | link gain | slow |

Parameter count per unit is `d·r + d + |T|·(p+D+q)·d + d + D·2d ≈ 2.9k` at defaults, so
512 units is ≈1.5M parameters — comparable to the GRU baseline it must beat.

## 2. Internal geometry: the landscape

The design document's "tiny physical landscape" is a per-unit energy

```
E_i(h) = -½ hᵀ W_i h  -  b_i^eff · h  +  (α/4)‖h‖⁴ ,        W_i = U_i U_iᵀ  (PSD, rank r)
```

with gradient flow

```
ḣ = -∇E_i(h) = W_i h + b_i^eff - α‖h‖² h
```

The quadratic term amplifies whatever lies in `span(U_i)`; the quartic term saturates it.
With `b^eff = 0` the fixed points sit at `±√(λ_k/α)·e_k` for each eigenvalue `λ_k > 0` of
`W_i` — so a rank-`r` landscape gives up to `2r` valleys, and those valleys are the unit's
stable hypotheses. `α = 1` by default.

**Input reshapes the landscape, it does not overwrite the state.** All incoming
influence enters through the tilt:

```
b_i^eff(t) = b_i + drive_i(t)
```

Tilting moves the valleys and can annihilate one entirely (a saddle-node), which is what
makes "incoming information reshapes the landscape" a different claim from an RNN adding
its input to a hidden vector. The state itself only ever moves by descending the current
energy.

Drive has three sources:

```
drive_i(t) = Σ_{k ∈ in(i)} w_ik · γ_ik · e_ik · a_ik        (messages, see SPEC_PMP §2)
           + g_s · S_i · s_i(t)                              (retina, sensory units only)
           + g_o · o(t) · ω_i                                 (global oscillator)
```

`ω_i` is a fixed random unit vector per unit and `o(t) = sin(2πt/T_osc)` with `T_osc = 40`
ticks. The oscillator is the shared clock that lets distant units phase-align, and it is
one of the four ingredients the design document says knowledge lives in. It is ablatable.

The two gains are not free parameters — they were forced by measurement. With
`g_s = g_o = 1` the drive at a sensory unit decomposes as 0.45 from messages, 0.31 from
the retina and 0.62 from the oscillator, so the mesh's own chatter and its clock together
drown out the world at the very units whose job is to see, and no head can then predict
better than the state permits. `g_o = 0.15` makes the oscillator modulatory, and
`g_s = 3.0` makes sensory units afferent-dominated the way cortical layer 4 is; that
combination roughly halves the best achievable sensory prediction error.

**State update** — `L` Euler steps per tick, step size `η_h = 0.15`:

```
for _ in range(L):
    h_i ← h_i + η_h · (W_i h_i + b_i^eff - α‖h_i‖² h_i) + σ_n · ξ,   ξ ~ N(0, I)
h_i ← h_i · min(1, h_max / ‖h_i‖)        # hard safety clip, h_max = 10
```

`σ_n = 0.01`. The clip should never bind in a healthy run; if it binds, that is a bug and
the metrics log counts it.

## 3. The five predictions

All heads are linear in `h`, which is what keeps the learning rule closed-form and local.
Each is emitted at every horizon `τ ∈ T`, except surprise which is one-step only.

**(1) Sensory future** — sensory units only, others emit nothing:
```
ŝ_i^τ = s_i(t) + P_i^τ h_i              target: s_i(t+τ)
```
The head predicts the *change* from the patch the unit is currently receiving. A sensory
unit physically has its own afferent input, so forcing it to re-derive that input from a
lossy 16-d state before it can say anything about the future is a plumbing handicap
rather than a scientific one — and it is a handicap the baselines do not share, so the
GRU is given the same residual output for the comparison to be about learning rules. With
this in place, copy-last is the mesh's zero-initialisation.

**(2) Neighbour future** — the coefficient expected to arrive on each in-link:
```
m̂_i^τ = M_i^τ h_i  ∈ R^D                target: e_ik(t+τ) for each in-link k
```

**(3) Self future** — *not* a separate head. The unit predicts itself by rolling its own
landscape forward with the drive frozen:
```
ĥ_i^τ = relax^τ(h_i ; b_i^eff frozen)   target: h_i(t+τ)
```
This is deliberate. If self-prediction were its own weight matrix, the landscape would be
ornamental — a nice picture bolted onto a standard recurrent net. Making the landscape
*be* the forward model means the energy geometry is load-bearing, and the E3 ablation
that replaces it with a linear map is then a real test rather than a formality.

**(4) Global context**:
```
ĝ_i^τ = C_i^τ h_i  ∈ R^q                target: g(t+τ), the executive broadcast (SPEC_PMP §5)
```

**(5) Surprise** — the unit predicts the size of its own error:
```
σ̂_i = softplus(v_i · h_i + v0_i)        target: S_i(t+1), the scalar total error below
```

## 4. Surprise and precision

At each tick a unit compares the predictions it made `τ` ticks ago against what actually
happened. This needs a ring buffer of depth `max(T) = 16` holding past predictions and
past states.

```
ε^sens = s_i(t) - ŝ_i^τ(t-τ)
ε^msg  = e_i(t) - m̂_i^τ(t-τ)
ε^self = h_i(t) - ĥ_i^τ(t-τ)
ε^glob = g(t)   - ĝ_i^τ(t-τ)

S_i(t) = Σ_τ  ρ_τ · ( β_s‖ε^sens‖² + β_m‖ε^msg‖² + β_h‖ε^self‖² + β_g‖ε^glob‖² )
```

with horizon discount `ρ_τ = 1/τ` and channel weights `β = (1, 1, 1, 0.5)`.

Precision — how much the unit trusts itself right now — is the inverse of its *predicted*
surprise, not its actual surprise:

```
c_i = 1 / (1 + σ̂_i)
```

Using the prediction rather than the measurement is what makes the uncertainty head do
work: a unit that has learned it is reliably wrong in some regime down-weights its own
learning and its own outgoing confidence there, before the error arrives.

Novelty is the surprise normalised against the unit's own recent history, so units on
different scales stay comparable on the wire:

```
n_i(t) = S_i(t) / (μ_i(t) + ϵ),    μ_i ← (1-λ)μ_i + λ S_i,   λ = 0.01
```

## 5. Local learning — fast loop, every tick

Every head is linear in `h`, so the delta rule is exact and needs nothing but quantities
the unit already holds:

```
ΔΘ_i^τ = η · c_i · ε^τ ⊗ h_i(t-τ) / (‖h_i(t-τ)‖² + ϵ)
```

`η = 0.08` for `P`, `M`, `C`; `η_σ = 1e-3` for the surprise head
(`Δv_i = η_σ (S_i - σ̂_i) h_i`, plus the same on `v0`).

The `1/‖h‖²` factor is normalised LMS, and it is load-bearing rather than cosmetic: a
unit's state is strongly autocorrelated and its scale drifts as the landscape learns, so
a single fixed step size cannot serve both regimes. With the plain delta rule the mesh
converged so slowly it sat near the predict-the-mean solution after thousands of ticks.
The normaliser uses nothing but `h_i`, so locality is preserved.

The landscape learns from one-step self-error only. Because `ĥ^1 = h + η_h(U Uᵀ h + b^eff
- α‖h‖²h)`, the derivative of `½‖ε^self‖²` with respect to `U` is analytic:

```
ΔU_i = η_U · c_i · η_h · ( ε^self hᵀ + h ε^selfᵀ ) U_i
Δb_i = η_U · c_i · η_h · ε^self
```

`η_U = 1e-3`. This is one step of gradient descent through **one** tick of **one** unit's
own dynamics. It is not backpropagation through time and not through the mesh; no error
signal ever crosses a unit boundary. Receiver-owned read vectors `a_ik` and link gains
`w_ik` learn by the same one-step rule, since drive enters `ĥ^1` linearly through them.

Weight decay `1e-5` on everything, and `U` is renormalised to `‖U‖_F ≤ 3` after each
update so the landscape cannot run away.

## 6. The tick, in order

```
1. receive        read the previous tick's message buffer for all in-links
2. update state   assemble drive → b^eff → L relaxation steps
3. predict        all heads, all horizons; push into the ring buffer
4. measure        compare buffered predictions against reality → ε, S_i, n_i, c_i
5. adjust         fast delta rules (§5)
6. choose         every T_rewire=200 ticks, update link credit and rewire (SPEC_PMP §4)
7. broadcast      emit (confidence, expectation, novelty, importance) per out-link
```

Steps 1 and 7 are the protocol boundary. Because every unit reads the *previous* tick's
buffer, the whole mesh update is order-independent and bit-reproducible under a fixed
seed — there is no sequential sweep and no hidden ordering bias.

## 7. Invariants the tests must hold

- No NaN or Inf in any state or parameter after 10⁵ ticks.
- The safety clip in §2 binds on fewer than 0.1% of unit-ticks.
- Two runs with the same seed produce bit-identical logs.
- With plasticity disabled and input frozen, `h` converges to a fixed point (the landscape
  is a genuine gradient flow, so this must hold; if it does not, `∇E` is wrong).
- With `r = 1` and a bistable input, a lone unit's state settles into one of exactly two
  valleys and hysteretically stays there — the minimum demonstration that the geometry is
  real. This is the gate for build step 3.
