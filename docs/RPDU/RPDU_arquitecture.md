A research paper asks:

"What might the cortex do?"

An engineering program asks:

"What is the smallest architecture that could demonstrate this hypothesis?"

Those are very different.

I think we should aim for something that can run on a single GPU with maybe 100–1000 RPDUs, not millions. If the primitive is right, scaling should mostly mean adding more units, not redesigning the architecture.

Phase 1: Forget the Brain

This sounds strange, but I think the first POC should not try to be an entire brain.

Instead it should answer only one question:

Can a network of RPDUs spontaneously build a world model?

Everything else is secondary.

What is the minimum RPDU?

I think the RPDU should not be a neural network.

This is my first disagreement with almost every AGI project.

A cortical column is probably a dynamical system, not a feed-forward network.

So imagine each RPDU has an internal state.

RPDU

Internal State
Prediction State
Connection State
Plasticity State

Notice something.

No "weights."

Instead it has an evolving internal state.

Weights still exist inside the implementation, but conceptually they become secondary. The identity of an RPDU is its state, not a frozen parameter vector.

Every timestep

Every RPDU executes exactly the same algorithm.

Receive Signals

↓

Update Internal State

↓

Predict Future

↓

Measure Surprise

↓

Adjust Internal Dynamics

↓

Choose Neighbors

↓

Broadcast Tiny Summary

That is it.

No attention.

No decoder.

No token generation.

The biggest change

Current transformers move information.

I think RPDUs move beliefs.

Instead of sending vectors.

Each RPDU sends something like

Confidence

Expectation

Novelty

Importance

Four numbers.

Not 4096-dimensional embeddings.

This dramatically reduces communication.

Locality

Every RPDU should only know maybe

32 local neighbors

+

8 long-range neighbors

+

1 executive neighbor

Think of a social network.

You don't know humanity.

You know your friends.

Yet information propagates.

Dynamic Wiring

This might be the biggest experiment.

Connections are not fixed.

Every few hundred steps.

The RPDU asks

Who consistently helps me reduce prediction error?

Those links strengthen.

Others disappear.

The network literally rewires itself.

Almost like growing axons.

Memory

I no longer think memory should exist as a separate module.

Instead

Memory = Stable Network State

Example

Yesterday's experience slightly changes

topology
dynamics
attractors

The brain itself becomes the memory.

Exactly like muscles don't contain a "memory file."

The muscle changes.

The World Model

Here's where I think the architecture becomes beautiful.

Don't build a world model.

Let it emerge.

Imagine watching

red ball rolls behind blue box

Initially

Prediction error explodes.

Eventually

Many RPDUs synchronize.

Now the network expects

ball will emerge

Nobody programmed object permanence.

The network invented it because it minimizes prediction error.

That's the key experiment.

Hierarchy

Instead of programming regions.

Use recursive clustering.

1000 RPDUs

↓

communities appear

↓

those become regions

↓

regions become systems

↓

systems become hemispheres

Exactly like cities emerge from houses.

Executive

Very small.

Maybe 20 RPDUs.

Responsibilities

goals
attention
planning horizon
curiosity
action selection

Not intelligence.

Management.

Language

This is another radical choice.

Language lives entirely outside the cortex.

Brain

↓

Language Adapter

↓

LLM Decoder

The cortex thinks.

The decoder speaks.

That separation gives you the freedom to replace the language system without changing the cognitive architecture.

The first experiment

Forget internet.

Forget Wikipedia.

Forget text.

Create a tiny world.

Like

□ □ ○ △

objects

movement

physics

collisions

hidden objects

light

sound

The synthetic cortex receives

vision

touch

time

Then ask

Can it predict tomorrow better than today?

If yes...

The architecture is learning.

Then I had one idea that I think could be the real breakthrough

I think we've been assuming that the RPDU is the primitive.

What if it isn't?

What if the primitive is the conversation between RPDUs?

Think about biology. A cortical column in complete isolation is almost useless. Its computational power comes from the continual exchange of predictions with neighboring columns. The intelligence may therefore reside less in the unit itself than in the protocol by which units negotiate a shared interpretation of reality.

That suggests designing the POC around two independent specifications rather than one:

1. RPDU Specification

A well-defined local machine:

Internal latent state
Local predictor
Surprise estimator
Plasticity rule
Routing policy
2. Predictive Mesh Protocol (PMP)

A communication protocol that every RPDU obeys:

What information is exchanged (predictions, uncertainty, novelty, confidence)
When messages are sent
How synchronization emerges
How temporary coalitions form and dissolve
How new long-range links are created and old ones are pruned

I suspect this protocol could end up being more important than the RPDU itself, just as TCP/IP mattered as much as the computers it connected.

If I were leading this project, I'd tell the engineering team: don't start by implementing a synthetic brain. Start by implementing one excellent RPDU and one excellent Predictive Mesh Protocol. If those two pieces are correct, scaling from 100 to 100,000 units should largely be an engineering problem rather than a conceptual one.

That, to me, feels like the first milestone that could realistically be built and tested over a few months, rather than a decades-long neuroscience project.