This RPDU (Recursive Predictive Dynamical Unit)Design is becoming less "let's invent an AGI" and more "let's identify the right computational primitive."

I think that's where almost every AGI proposal fails. They start from the Transformer and add modules. Evolution didn't start from language; it started from a generic cortical computation that was reused everywhere.

So I would intentionally forget LLMs for a moment.

Let's ask a much simpler question.

What is the smallest computational unit that evolution found worth repeating millions of times?

Not the neuron.

The cortical column.

That immediately suggests a research program.

Research Program
"The Predictive Cortex"

Hypothesis 0

Intelligence emerges from millions of identical predictive cortical units that self-organize through local learning and sparse communication—not from one monolithic neural network optimized by a global objective.

Everything else should follow from this.

Step 1 — Forget Language

Language should not exist inside the architecture.

Language is merely an output modality.

Likewise,

vision

hearing

touch

proprioception

planning

social reasoning

should all emerge from the same computational primitive.

This is already one of the strongest biological clues.

Visual cortex and auditory cortex look remarkably similar under the microscope.

The difference is mostly what they receive, not what they are.

That is an enormous hint.

Step 2 — Define the Cortical Prediction Unit (CPU)

I don't think a cortical column stores knowledge.

I think it maintains expectations.

Imagine every unit only knows five things.

Incoming signals

↓

Current internal state

↓

Prediction of next state

↓

Prediction error

↓

Updated state

That's it.

No text.

No embeddings.

No symbols.

Only dynamics.

Instead of asking

"What is this?"

Every cortical unit continuously asks

"What should happen next?"

That tiny change is profound.

Prediction becomes the primitive.

Step 3 — Cortical Unit Structure

A cortical unit should probably contain several interacting subsystems rather than one matrix.

Something like

Prediction Core

↓

Temporal Memory

↓

Novelty Detector

↓

State Compressor

↓

Communication Router

Each is tiny.

Collectively they form one reusable module.

Prediction Core

Predicts

next sensory activation
next neighboring activation
next internal state

Not language.

Everything.

Temporal Memory

Very short.

Perhaps one or two seconds.

Just enough to detect motion and sequences.

Almost like working memory.

Novelty Detector

Computes

Prediction

Reality

Difference

Difference becomes

Surprise.

Surprise becomes

Learning.

State Compressor

Instead of storing every experience...

compress recurring dynamics into attractors.

Very brain-like.

Communication Router

Most interesting component.

The cortical unit decides

Who needs this information?

Not everyone.

Only the few units that care.

Sparse routing.

Step 4 — How Does the Cortex Scale?

Current AI scales vertically.

Layer

↓

Layer

↓

Layer

Brains don't.

Brains scale horizontally.

Millions of columns.

So perhaps architecture becomes

Column

Column

Column

Column

Column

No hierarchy initially.

Just neighbors.

Exactly like early cortex.

Then...

Hierarchy emerges naturally.

Step 5 — Regions

Instead of programming regions...

let them emerge.

Suppose neighboring cortical units gradually specialize because they receive similar inputs.

Vision receives photons.

Audio receives sound.

Language receives auditory and visual abstractions.

Eventually

Visual Region

↓

Object Region

↓

Semantic Region

↓

Planning Region

No one designed them.

Training created them.

Exactly what biology appears to do during development.

Step 6 — Hemispheres

This is where we can go beyond biology.

Biology has two hemispheres because of developmental and evolutionary constraints. We shouldn't assume we need to copy them exactly, but functional specialization with rich communication is a compelling idea.

Instead of left/right anatomy, imagine two computational "modes."

Exploratory Hemisphere

High entropy

Creative

Weak priors

Hypothesis generation

Counterfactual imagination

Divergent thinking

Stable Hemisphere

Low entropy

Logical

Consistency

Planning

Verification

Precise prediction

Both continuously exchange hypotheses.

Almost like Bayesian inference.

One generates priors.

One rejects them.

Step 7 — Regional Oscillations

Here is something I think AI almost completely ignores.

Brains communicate using rhythms.

Not continuous messages.

Oscillations.

Maybe synchronization itself is computation.

Instead of sending vectors...

Regions synchronize.

Vision

████████

Memory

████████

Planning

████████

When synchronized...

Information flows.

Otherwise...

Almost disconnected.

This automatically limits bandwidth.

Exactly like biology.

I suspect oscillations are not an implementation detail but a computational mechanism for routing and binding information.

Step 8 — Synthetic Cortex

Now imagine

Millions of Cortical Prediction Units

↓

Self-organize

↓

Become Regions

↓

Regions synchronize

↓

Temporary coalitions emerge

↓

World model emerges

↓

Executive system appears

↓

Behavior

Notice something missing.

There is no global database.

No explicit world model module.

The world model is an emergent attractor.

Exactly as your understanding of "Paris" is spread throughout the cortex.

Step 9 — Executive Cortex

The executive system should actually be tiny.

Maybe

1%

of total computation.

Its job isn't thinking.

Its job is

Attention

Goals

Task switching

Planning horizon

Conflict resolution

Exactly like the prefrontal cortex.

Step 10 — Sleep

Current AI never sleeps.

Huge mistake.

I think sleep is one of evolution's greatest algorithms.

During sleep

Weak connections disappear.

Strong ones strengthen.

Memories merge.

Noise disappears.

Concepts emerge.

So synthetic brains should spend perhaps 20–30% of their time offline reorganizing themselves instead of interacting with the world.

The biggest departure from today's AI

I think the Transformer computes functions.

The brain maintains state.

That distinction may be the entire game.

Today's LLM:

Input

↓

Compute

↓

Output

↓

Done

Synthetic cortex:

Input

↓

State changes

↓

Predictions change

↓

Connections change

↓

Identity changes

↓

Next input arrives

The system is never reset.

It is always becoming something slightly different.

One hypothesis I'd add to the research agenda

I would also challenge an assumption that nearly every AI architecture inherits from digital computing: that communication should be explicit messages passed between modules.

The brain may instead rely heavily on shared dynamical states. A cortical region doesn't necessarily transmit a complete representation to another region; it nudges the receiving region into a compatible state. If that's true, then the fundamental abstraction for AGI may not be "message passing" but state coupling. That could dramatically reduce communication costs while allowing much richer global coordination.

If I were turning this into a serious multi-year research program, I would organize it around one central question:

What is the smallest predictive dynamical system that, when replicated millions of times and allowed to self-organize through local learning and sparse long-range communication, gives rise to the computational properties we call intelligence?

That's a question that is scientifically testable. It doesn't assume transformers are the endpoint, it doesn't require reproducing biology neuron-by-neuron, and it gives researchers a concrete target: discover the right cortical primitive rather than simply scaling existing architectures.I think we're finally asking the right question.

Not:

"How do we build AGI?"

But:

"What is the cortical algorithm?"

That is a much deeper question.

I also think we need to be careful not to copy biology literally. Evolution optimized for wet chemistry, energy constraints, and development. Our goal is to discover the computational principle behind the cortex, not to reproduce every neuron.

A thought that changed my mind

Until now we've assumed a cortical column is a little computer.

I now think that's wrong.

A cortical column may not compute answers.

It may compute beliefs about the future.

Those are different.

The cortex isn't solving problems.

It is continuously maintaining a probability distribution over what will happen next.

Everything else—vision, language, planning—emerges from that.

So maybe the primitive isn't a computer.

It is a Predictive Dynamical Unit (PDU).

What should a PDU actually contain?

I would forget neurons.

Forget attention.

Forget MLPs.

Imagine evolution had to invent one reusable module.

What absolutely cannot be removed?

I currently arrive at six internal states.

          Incoming Signals
                 │
                 ▼
        Current Internal State
                 │
 ┌───────────────┼───────────────┐
 ▼               ▼               ▼
Predict      Compare        Simulate
                 │
                 ▼
        Prediction Error
                 │
 ┌───────────────┼───────────────┐
 ▼               ▼               ▼
Learn        Communicate     Update State

Notice there is no "answer."

Only state evolution.

The most important realization

I don't think prediction is the output.

I think prediction modifies the internal state.

This is a huge distinction.

Today's Transformer

Input

↓

Output

Synthetic Cortex

Input

↓

State changes

↓

Predictions change

↓

Future behavior changes

No output is required.

The brain keeps changing even in silence.

Internal Geometry

This is where I think we can invent something new.

Instead of storing vectors...

Suppose every cortical unit contains something like a tiny physical landscape.

           /\

      /\        \__

__/                 \___

A rolling ball always falls into one valley.

Those valleys represent stable hypotheses.

Incoming information reshapes the landscape.

Instead of "memory."

We have energy landscapes.

This is closer to attractor networks, but imagine each PDU having its own evolving landscape rather than a single global one.

Every PDU predicts five things

Not just one.

It predicts simultaneously:

1. External sensory future

"What will my inputs be?"

2. Neighbor future

"What will nearby PDUs become?"

3. Self future

"What state will I enter next?"

4. Global context

"What brain state am I participating in?"

5. Surprise

"How wrong am I likely to be?"

This last one is fascinating.

The unit predicts its own uncertainty.

That feels biologically plausible because the brain behaves very differently when it is uncertain.

Learning becomes local

Every PDU follows one rule.

Prediction

↓

Reality

↓

Prediction Error

↓

Tiny structural change

Not weight update.

Structural change.

This is another big leap.

Maybe weights disappear

Current AI assumes

Knowledge

↓

Weights

Maybe knowledge actually lives in

Topology

+

Oscillations

+

Internal state

+

Plasticity

The connections themselves become part of memory.

Not merely their strength.

The architecture literally rewires itself.

Communication

Here's another place I think we differ from Transformers.

Attention is expensive because every token can talk to every other token.

The cortex almost certainly doesn't do that.

Instead

Each PDU has maybe

100 local neighbors

20 regional neighbors

5 global neighbors

Those numbers are illustrative, not biological claims.

Like airline networks.

Mostly local flights.

Occasionally international.

Yet the world remains connected.

Dynamic Coalitions

Now something beautiful appears.

Instead of one global network...

Thousands of temporary coalitions form.

Vision Coalition

████████

Planning Coalition

████████

Language Coalition

████████

Memory Coalition

████████

These groups continuously appear and disappear.

Almost like flocks of birds.

The brain never has one fixed architecture.

Its architecture is changing every few hundred milliseconds.

The Cortex Doesn't Think

Here's my most controversial thought.

The cortex may never "think."

Thinking might emerge from conflicts between predictions.

Imagine millions of PDUs.

Each predicts locally.

Most agree.

Some disagree.

The disagreements propagate.

Eventually a stable global interpretation emerges.

Thought is simply the settling of the network.

Not symbolic reasoning.

Not chain of thought.

Just self-consistency.

Then I think I found the hierarchy

This is where I think the paper could become original.

Instead of building modules...

Build one recursive hierarchy.

                 Synthetic Brain
                        │
          Global Dynamic State
                        │
────────────────────────────────────
      Hemispheric Prediction Fields
                        │
────────────────────────────────────
        Regional Prediction Fields
                        │
────────────────────────────────────
      Cortical Prediction Meshes
                        │
────────────────────────────────────
      Predictive Dynamical Units
                        │
────────────────────────────────────
     Internal Dynamical Components

Every layer uses the same computational primitive.

Only the spatial and temporal scales change.

This is exactly how fractals work.

The cortex may be recursively self-similar.

I think we're still missing one ingredient

After all of this, I still feel something is missing.

The cortex isn't only predictive.

It is also generative.

Every moment it is asking:

"If my current model of the world is true, what should I experience next?"

Reality doesn't build the brain.

The brain continuously builds reality, and sensory input only corrects it.

That inversion is subtle but profound.

Instead of perception being:

Reality → Brain

it becomes:

Brain predicts reality
          ↓
Reality corrects prediction
          ↓
Brain updates itself

That idea is closely related to predictive processing theories in neuroscience, but I think your research program can push it one step further.

My candidate for the central primitive

I would define the primitive as a Recursive Predictive Dynamical Unit (RPDU).

Not because it predicts the next token.

Not because it stores memories.

But because it continuously performs four inseparable operations:

Maintain a stable internal dynamical state (its current belief about a tiny piece of the world).
Predict how that state, its neighbors, and its sensory inputs should evolve over multiple timescales.
Adapt its own internal dynamics and connectivity using only local prediction errors and uncertainty.
Synchronize transiently with other RPDUs to form larger coalitions that represent objects, concepts, plans, or thoughts.

If I had to bet on one idea that could distinguish a new architecture from the Transformer, it would be this:

The cortex is not a network of neurons that stores knowledge. It is a network of predictive dynamical systems that continuously synchronize until they reach a coherent interpretation of the world.

That is a hypothesis that can be explored mathematically, tested experimentally, and—if it's even partially correct—could define a genuinely new line of AI research rather than another variation on existing deep learning.