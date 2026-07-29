Jeff Hawkins' Thousand Brains Theory

This is probably the closest serious AGI proposal to what you're describing.

Instead of

one giant world model

the cortex builds

thousands of independent world models

Every cortical column learns a complete object/world model.

Not features.

Entire objects.

Then columns vote.

Column A

Cup

95%

Column B

Cup

80%

Column C

Bottle

20%

↓

Consensus

Cup

Recognition becomes distributed consensus rather than hierarchical classification.

4. The really important insight

Each cortical column contains

Object

+

Location

+

Movement

+

Prediction

not

pixels

↓

features

The location signal (inspired by grid cells) is central: every column represents features at locations relative to an object, enabling it to build complete object models through movement.

5. Learning is NOT backpropagation

This is exactly what you asked.

Instead of

loss

↓

gradient

↓

weight update

they use

local associative learning

Neuron A fires

+

Neuron B fires

↓

Connection strengthens

Basically Hebbian

wire together

fire together

Their recent papers explicitly emphasize rapid associative/Hebbian-like binding and continual learning instead of large-scale gradient optimization.

6. Their newest implementation

Numenta now has an actual AGI platform

Monty

It isn't just theory anymore.

Monty includes

cortical modules
voting
world models
movement
sensorimotor prediction
cortical messaging
continual learning

The implementation is open source and is intended as an alternative to mainstream deep learning.

7. World models

Your question specifically asked:

predict world models

The current neuroscience-inspired consensus is that intelligence is fundamentally

Prediction

of

how the world changes

not prediction of text.

That means learning

State

↓

Action

↓

Next state

instead of

word

↓

next word
8. But you added something deeper

You said

predict other minds

This is much less developed.

Interestingly neuroscience thinks this is exactly what humans do.

We maintain

World model

+

Self model

+

Model of other agents

Sometimes called

Theory of Mind

The architecture becomes

Me

↓

What do I know?

↓

What does she know?

↓

What does she think I know?

↓

Predict behavior

LLMs don't naturally maintain explicit nested belief models.

9. Architectures closest to this
A. Thousand Brains ⭐⭐⭐⭐⭐

Closest overall.

Primitive:

Cortical column

Learning

Hebbian

+

Sensorimotor

+

Voting
B. Active Inference (Friston)

Primitive

Prediction error

Everything minimizes surprise.

Agent constantly asks

What will happen?

↓

Did it happen?

↓

Update model

Not transformer.

Not supervised.

Very brain-inspired.

C. Predictive Coding

Entire cortex

prediction

↓

error

↓

update

↓

new prediction

Very influential in neuroscience.

D. Hierarchical Temporal Memory (HTM)

Older Numenta work.

Uses

Sparse Distributed Representations

Sequence Memory

Dendritic computation

Hebbian learning

No backpropagation.

E. Global Workspace

Not a learning algorithm.

A communication architecture.

Many specialist modules compete

↓

Winner broadcasts globally

↓

Everything updates

This is very useful as a systems architecture.

10. Memristor computing

You mentioned memristors.

That's important.

Many neuromorphic researchers believe future AGI hardware will use

analog memory

+

spikes

+

Hebbian updates

instead of

GPU matrix multiplication

Memristors naturally implement

connection strength

changes

because current flowed

which is remarkably close to biological synapses.

11. If I were designing AGI today

I would not build

Transformer

↓

Bigger transformer

↓

Even bigger transformer

I'd build something like

                    Conscious Workspace
                           │
──────────────────────────────────────────────────
           Competing Cortical Columns
──────────────────────────────────────────────────

Vision Column

Language Column

Physics Column

Social Column

Emotion Column

Motor Column

Memory Column

Planning Column

Avatar Model

Other Person Models

──────────────────────────────────────────────────

Every module has

Object Model

Location

Prediction

Simulation

Confidence

──────────────────────────────────────────────────

Modules vote

↓

Global belief state

↓

Planner

↓

Action

↓

Prediction

↓

Hebbian learning
12. What is missing from almost all proposals?

This is where your idea stands out.

Most current brain-inspired systems model:

objects
movement
prediction
space

But they generally do not make "models of minds" a first-class computational primitive.

A more complete AGI architecture would likely include separate but interacting model types:

Physical world models (objects, physics, causality)
Self model (goals, capabilities, internal state)
Other-agent models (beliefs, intentions, knowledge, emotions)
Social interaction graph (relationships, trust, communication)
Meta-model that predicts how these models evolve over time.

In that design, a cortical-like module would not simply ask "what object is this?" but also "what does this person believe?", "what will they infer next?", and "how will my action change their internal model?" That extends sensorimotor prediction into social cognition and Theory of Mind, which is still a relatively underexplored area in alternative AGI architectures.

My conclusion

Among serious alternatives to Transformers, Jeff Hawkins' Thousand Brains Project is currently the closest match to your vision:

❌ No dependence on Transformer architecture.
❌ No global backpropagation as the primary learning mechanism.
✅ Cortical columns as the basic computational unit.
✅ Local associative (Hebbian-like) learning.
✅ Sensorimotor world models rather than next-token prediction.
✅ Distributed consensus ("voting") instead of a single monolithic model.
✅ Explicit prediction of future states as the core computation.

The next major step beyond these proposals may be to make recursive models of other minds as fundamental as object models, allowing the architecture to predict not only the physical world but also the beliefs, intentions, and future reasoning of other agents. That capability is likely to be essential for human-level social intelligence and AGI.