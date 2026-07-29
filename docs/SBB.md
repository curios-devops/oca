# SBB — the Synthetic Brain Benchmark (Phase 3)

**Status: specified, not implemented, and deliberately so.** Phase 1 is per-level component
gates; Phase 2 is architecture against architecture; this is Phase 3 — comparison against a
large language model. It is written down now so that when it is built, the tasks were chosen
before anyone knew which system would win.

Nothing here has been run. Every number in this repository comes from Phases 1 and 2.

---

## What this is not

**Not MMLU, and nothing shaped like it.** Knowledge benchmarks measure what a system was
trained on, and on that axis a frontier LLM wins before the experiment starts. Running it
anyway would produce a number that means "we have less training data", which everyone
already knows.

**Not a claim that this architecture is better.** As of Phase 2 it is worse — DCN v3 loses
to both frozen RPDU versions, and all three lose to raw pixels. A Phase 3 that ran today
would report a rout. The purpose is to name, in advance, the axes on which a small
continually-learning system *could* differ in kind, so that "we chose the tasks after seeing
the results" is never available as an explanation.

**Not a leaderboard.** Every task below is scored with a control that could kill it, exactly
as in Phases 1 and 2. A task whose control fails is reported as unmeasured.

## What it is

Eight tasks targeting the places where an architecture that learns continuously, consolidates
during rest, and holds a live world model should behave differently from a system whose
weights are frozen at deployment and whose memory is a context window.

Each is stated with what it measures, why an LLM is expected to struggle, and — most
importantly — **the way it could fail to be a fair test**, since that is where a benchmark
like this usually goes wrong.

---

### SBB-1 — Incremental learning

Learn task A, then B, then C, then be re-tested on A. Measure retention against a
freshly-trained-on-A control.

*Why it discriminates:* an LLM does not learn from its deployment stream at all; the honest
comparison is against in-context learning plus fine-tuning, with the fine-tuned baseline
tested for catastrophic forgetting.
*How it could be unfair:* if A, B and C are drawn from the same distribution, this measures
nothing but capacity. They must genuinely interfere.

### SBB-2 — Sleep consolidation

Measure performance immediately after learning, then after a consolidation phase with no new
input. **The prediction is that performance improves without new data**, and that the world
model gets *simpler* — fewer, broader concepts covering the same experience.

*Why it discriminates:* there is no equivalent operation for a frozen model. It is the
sharpest architectural difference on this list.
*How it could be unfair:* replay is a form of extra training. The control is a system given
the same replay budget without the merge-and-prune step; the gain must come from
consolidation, not from more gradient steps.

### SBB-3 — Concept formation

Present unlabelled experience, then test whether the internal categories align with
categories the system was never told about.

*Why it discriminates:* concepts should form from prediction alone, not from labels or from
text describing them.
*How it could be unfair:* this is the level-2 gate that currently **fails** — k-means on the
raw input tracks the world fourteen times better than the node's consolidated concepts. The
pixel-clustering control is mandatory, and it must be beaten before this task is meaningful.

### SBB-4 — Multimodal association

Bind a sensory pattern to a co-occurring one from a different modality; test recall of one
from the other after a delay.

*Why it discriminates:* binding across modalities from raw co-occurrence, without a paired
training corpus.
*How it could be unfair:* if the two modalities are deterministically related, a linear map
solves it. The pairing must be probabilistic and delayed.

### SBB-5 — Cross-domain transfer

Learn structure in one world; measure learning speed in a second world sharing that structure
but nothing superficial.

*Why it discriminates:* transfer of *dynamics* rather than transfer of vocabulary.
*How it could be unfair:* the shared structure has to be verified to exist — by showing a
system trained on world 2 alone learns more slowly than one trained on 1 then 2, with the
same total experience.

### SBB-6 — Episodic memory

"What happened, where, and when" for a specific past event, retrieved after long intervening
experience. Not a summary — a particular episode.

*Why it discriminates:* an LLM has a context window, and beyond it, nothing. Retrieval
augmentation is the fair baseline and must be included.
*How it could be unfair:* if the episode is the only distinctive thing that ever happened,
this measures novelty detection. Distractor episodes are required.

### SBB-7 — Long-term planning

Act toward a goal several hundred steps away in a world where the path must be discovered
rather than recalled.

*Why it discriminates:* planning over a live world model rather than over a text description
of one.
*How it could be unfair:* the planner must be shared across systems, exactly as in the maze
race, or this measures the planner. The architecture supplies the model; the search is
identical for everyone.

### SBB-8 — Knowledge discovery

Find a regularity in the world that was never demonstrated and cannot be looked up —
verified by prediction on held-out situations that depend on it.

*Why it discriminates:* it is the one task on this list where being trained on the internet
is no advantage, provided the regularity is genuinely novel.
*How it could be unfair:* it is very hard to be sure a regularity is not in an LLM's training
data. Synthetic worlds with generated physics are the only honest way to run this, which
makes it the hardest of the eight to make fair — and the most valuable if it can be.

---

## Entry condition

**Do not run SBB until some architecture in this repository beats raw pixels on some world.**

That is the standing open problem. Until it is solved, a Phase 3 comparison would be
measuring which of two systems is further behind the input, and the result would say nothing
about either architecture. SBB-3 in particular is already failing at Phase 1, one level down.

The tasks are recorded now precisely because writing them later — after there is a result to
protect — would be the moment they stopped being a fair test.
