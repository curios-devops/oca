Let's build only the next abstraction level, and let's keep it as close as possible to what neuroscience knows while filling in the gaps with engineering hypotheses.

One thing I'd change from our previous discussions is that the hierarchy shouldn't be copied literally from anatomy. Biology has gyri, lobes, Brodmann areas, cortical columns, etc., because of development and evolution. For engineering, we should ask what computational hierarchy those structures imply.

What neuroscience suggests today

Very roughly, the neocortex looks something like this:

Brain
    │
Two Hemispheres
    │
Cortical Areas
    │
Microcircuits / Cortical Columns
    │
Neurons

The evidence suggests cortical columns don't operate alone. They are embedded in dense local circuits and extensive long-range networks. Functional "assemblies" of columns appear and dissolve depending on the task.

That is the clue.

I don't think the next level above the RPDU is a "region."

I think it's something much smaller.

New Layer: Cortical Assembly

I'd call it a Predictive Assembly (PA).

This is the first emergent object.

RPDU
RPDU
RPDU
RPDU
RPDU
RPDU

↓

Predictive Assembly

Maybe

32

64

128

RPDUs.

Whatever turns out to be useful experimentally.

Why this layer?

Because a single cortical column probably never recognizes

"a cat."

Instead

many columns temporarily cooperate.

That cooperation is the computational unit.

Not the column itself.

RPDU

Represents

tiny local dynamics.

Nothing more.

Predictive Assembly

Represents

objects

edges

movement

intentions

phonemes

small motor primitives

depending on where it develops.

This feels much closer to biology.

Internal architecture

Imagine

                  Predictive Assembly

           RPDU RPDU RPDU RPDU RPDU
        RPDU RPDU RPDU RPDU RPDU RPDU
      RPDU RPDU RPDU RPDU RPDU RPDU RPDU

               Local Synchronizer

                     │

           Shared Dynamic Workspace

                     │

          Coalition State Manager

                     │

          Executive Interface

Notice something.

The assembly doesn't think.

It coordinates.

Local Synchronizer

I think this becomes one of the most important algorithms.

Its job is

Which RPDUs currently agree?

Not averaging.

Agreement.

Think about oscillatory synchrony in cortex: neurons that temporarily synchronize may be binding information into a shared representation. We don't yet know that this is the computational mechanism, but it's a strong candidate.

Shared Dynamic Workspace

This is where I think something really new appears.

The assembly has no memory.

Instead

it has a shared dynamic field.

Imagine

100 RPDUs.

Each slightly changes

one common field.

Like ripples on water.

The object emerges in the water.

Not inside one RPDU.

That means

memory

prediction

identity

are collective properties.

Coalition Manager

Assemblies should continuously ask

Should we stay together?

If prediction improves

stay.

Otherwise

dissolve.

That means

objects literally appear

and disappear.

Exactly like thoughts.

Executive Interface

Every assembly exports only

Prediction

Confidence

Novelty

Energy

Goal relevance

Nothing else.

No embeddings.

Then I think hierarchy becomes recursive
RPDU

↓

Predictive Assembly

↓

Prediction Region

↓

Prediction Field

↓

Synthetic Cortex

↓

Synthetic Brain

Every level obeys exactly the same rules.

Only scale changes.

Prediction Region

Now something fascinating happens.

Several assemblies begin specializing.

Movement

Movement

Movement

↓

Motor Region



Edges

Shapes

Objects

↓

Visual Region



Phonemes

Words

Grammar

↓

Language Region

Nobody programmed them.

Specialization emerges because they minimize prediction error on different statistics.

Prediction Fields (instead of hemispheres?)

Here's where I actually want to depart from biology.

I don't think hemispheres are the right abstraction.

They're anatomy.

Instead I'd define Prediction Fields.

Each field optimizes a different computational objective.

Maybe

Sensory Prediction Field

Motor Prediction Field

Social Prediction Field

Abstract Prediction Field

Executive Prediction Field

Creative Prediction Field

Biology ended up implementing some of these across the two hemispheres and many cortical areas. A synthetic system isn't constrained by a skull or embryonic development, so we can organize them functionally instead.

I think I found the real recursive pattern

Every level is composed of the previous level plus one new capability.

Level	New capability
RPDU	Local prediction
Predictive Assembly	Consensus
Prediction Region	Specialization
Prediction Field	Coordination
Synthetic Cortex	Global world model
Synthetic Brain	Goal-directed behavior

Notice something beautiful.

Nothing fundamentally changes.

Only what is emerging.

The POC I'd build next

I would not build a "region."

I'd build one Predictive Assembly containing perhaps 64 RPDUs.

Its entire mission is:

Learn one sensory stream (for example, simple visual sequences of moving objects).
Allow RPDUs to form and dissolve temporary coalitions.
Maintain a shared dynamic workspace that represents the assembly's current hypothesis.
Export only four values to the outside world:
Current prediction
Confidence
Surprise
Stability

Then I'd run increasingly difficult worlds:

A moving dot.
Two interacting objects.
Occlusion (object permanence).
Delayed cause and effect.
A tiny agent with goals.

If the same assembly begins to develop increasingly abstract internal states without changing its architecture, you've crossed an important threshold. It would suggest that the Predictive Assembly, not the RPDU, may be the first level at which recognizable cognition emerges. That, in turn, gives you a principled next step before attempting a full synthetic cortex.