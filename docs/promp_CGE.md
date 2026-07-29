

Create a new document for the OCA repository.

This document is NOT part of the architecture specification.

It defines the official testing philosophy of OCA.

The document should become one of the most important specifications in the repository.

Its purpose is to determine whether a cognitive component is mature enough to advance to the next development stage.

Write everything in excellent technical English.

The tone should resemble engineering standards rather than academic papers.

Document Title
OCA Cognitive Benchmark Specification (CBS)

Subtitle:

Minimum Viable Cognitive Tests for Open Cognitive Architecture

Purpose

The purpose of this document is not to maximize benchmark scores.

Its purpose is to determine whether a cognitive component demonstrates the minimum functional properties required to become part of OCA.

This benchmark suite should answer a simple engineering question:

Does this component behave as a valid cognitive abstraction?

Every benchmark should produce one of three outcomes:

PASS

CONDITIONAL PASS

FAIL

Avoid continuous scoring whenever possible.

The benchmark is designed as an engineering gate rather than a leaderboard.

Philosophy

The benchmark is designed around functional behavior.

Not implementation.

Not algorithms.

Not neural networks.

Not transformers.

Not symbolic systems.

Any implementation may pass if it satisfies the required cognitive properties.

Every benchmark must define

Objective

Required properties

Anti-properties

Metrics

Pass criteria

Failure modes

Expected scalability

Expected robustness

Applicable cognitive levels

Difficulty scaling

Estimated execution time

Reproducibility requirements

Anti-tests

Introduce a dedicated concept called

Anti-tests

These intentionally attempt to break the abstraction.

Examples

Noise

Delayed feedback

Missing inputs

Conflicting information

Contradictory memories

Synchronization failures

Timing shifts

Partial communication loss

False sensory evidence

Unexpected environmental changes

The benchmark should verify graceful degradation rather than catastrophic failure.

Benchmark Gates

Every benchmark belongs to one of three categories.

Gate A

Minimum viability.

Required before any architecture comparison.

Gate B

Architectural validation.

Compare against previous OCA architectures.

Examples:

DCN v1

DCN v2

DCN v3

Future architectures.

Gate C

External comparison.

Only after passing Gate B.

Possible comparisons include:

LLMs

Transformer-based systems

HTM

Thousand Brains inspired implementations

Other cognitive architectures

The benchmark should encourage understanding rather than competition.

Multi-modal Philosophy

The benchmark suite should never depend on vision alone.

Cognition is multimodal.

Benchmarks should gradually include:

Vision

Audio

Touch

Motor actions

Spatial reasoning

Temporal reasoning

Cross-modal reasoning

Sensor fusion

The same benchmark should ideally support multiple sensory modalities.

Progressive Complexity

Each benchmark should specify complexity levels.

The exact same benchmark should be reusable from low-level neurons to high-level cortical regions.

Only the complexity changes.

Avoid creating unrelated benchmark families for every level.

Instead create scalable benchmark templates.

Initial Scope

This version defines benchmarks only for

Level 1

Synthetic Neuron

Level 2

Cortical Tower

Level 3

Tower Cluster

Future levels will be added later.

Leave placeholders.

Benchmark Categories

Think carefully.

Do not simply reuse existing AI benchmarks.

The benchmark should evaluate cognitive behavior.

Potential categories include:

Perception

Prediction

Temporal continuity

Object permanence

Local memory

Working memory

Sensorimotor adaptation

Pattern abstraction

Novelty detection

Context preservation

Selective attention

Multi-modal integration

Consensus formation

Error recovery

Planning

Causal reasoning

Self-consistency

Robustness

Generalization

Continual learning

Energy efficiency

Communication efficiency

Synchronization

Wave coordination

Global-state adaptation

Graceful degradation

Scalability

Explain why every category exists.

Example Cognitive Tests

Design reusable benchmark templates.

Examples may include:

A moving ball disappears behind a wall.

Can the system predict where and when it will reappear?

A sound continues after the object becomes invisible.

Can auditory evidence update the prediction?

A bouncing ball changes direction after collision.

Can the model infer the hidden collision?

A partially observed object rotates.

Can the identity remain stable?

A motor action changes perception.

Can action improve prediction?

A sequence contains an anomaly.

Can novelty be detected?

Two sensory modalities disagree.

Which evidence dominates?

Multiple towers disagree.

Can consensus emerge?

Communication is delayed.

Does synchronization recover?

Learning pauses.

Does previous knowledge remain usable?

Sleep occurs.

Does performance improve afterwards?

Do not limit yourself to these examples.

Invent better ones.

Benchmark Scaling

Every benchmark should indicate:

Minimum viable version

Intermediate version

Advanced version

Large-scale version

The benchmark should naturally grow with the architecture.

Pass Criteria

Avoid arbitrary percentages.

Prefer behavioral criteria.

Examples:

Maintains stable identity.

Recovers after interruption.

Preserves context.

Forms consensus.

Improves after consolidation.

Gracefully degrades.

Learns incrementally.

Transfers previous knowledge.

Rejects inconsistent hypotheses.

Each benchmark should define objective observations.

Benchmark Independence

The benchmark suite should not assume:

Backpropagation

Transformers

Language

Text

Specific memory implementations

Specific neural representations

Any architecture capable of cognitive behavior should be eligible.

Future Expansion

The benchmark framework must support adding new tests without breaking previous versions.

Each benchmark should receive:

Unique ID

Version

Applicable levels

Dependencies

Expected computational cost

Status

Experimental

Stable

Deprecated

Final Goal

The benchmark suite should eventually become one of the defining contributions of OCA.

It should allow the community to evaluate cognitive architectures using functional behavior instead of implementation details.

Encourage the AI agent to challenge this proposal.

It should compare these ideas with existing cognitive benchmarks, neuroscience-inspired evaluations, robotics benchmarks, developmental psychology experiments, and AI evaluation methodologies.

The AI agent should improve, replace, or reorganize anything if it discovers a more elegant engineering solution.

The final document should be significantly better than this prompt.

The benchmark suite should become a living specification maintained independently from the architecture itself.

Creo que hay una idea que aún no hemos explotado y que podría ser diferencial para OCA.

No llamaría a estos simplemente benchmarks. Los llamaría Cognitive Gates. Asi el documeto a crear seria 
Cognitive Gates Specification SPEC_CGE.md
Un benchmark normalmente produce un número.

Un Gate produce una decisión de ingeniería:

PASS → el componente puede avanzar.
CONDITIONAL PASS → puede avanzar con limitaciones documentadas.
FAIL → debe rediseñarse antes de continuar.

Eso cambia el enfoque de "competir por una puntuación" a "demostrar propiedades cognitivas mínimas". Si OCA mantiene esa filosofía desde el principio, cualquier arquitectura futura —DCN u otra— tendrá que demostrar primero que posee las capacidades funcionales esperadas antes de compararse con modelos existentes. Creo que ese enfoque puede convertirse en uno de los rasgos más distintivos del proyecto