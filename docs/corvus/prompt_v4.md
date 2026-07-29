Prompt for AI Agent

Create the official architecture specification for the Open Cognitive Architecture (OCA).

This document defines the architecture itself.

It must NOT define the benchmark suite.

Benchmarking, validation, Cognitive Gates, curricula, datasets, evaluation protocols, and experimental procedures belong to separate specifications.

The architecture document should only reference them when appropriate.

The final document should read like a senior engineering architecture specification suitable for an open-source research project.

Use excellent technical English.

Avoid marketing language.

Avoid implementation details whenever possible.

Define contracts, responsibilities, interfaces, and functional behavior.

Document Title
OCA Architecture Specification

Subtitle

Version 4

Open Cognitive Architecture

Purpose

OCA is an open cognitive architecture intended to support continual learning, adaptive reasoning, long-term memory, sensorimotor intelligence, and modular cognitive systems.

OCA is not a machine learning model.

It is an architectural framework capable of hosting multiple cognitive implementations.

The architecture intentionally separates:

Architecture
Training
Curriculum
Cognitive Gates
Benchmarks
Runtime implementations

Each evolves independently.

Design Philosophy

The architecture follows several core principles.

Modular
Hierarchical
Stateful
Event-driven
Sensorimotor
Multi-modal
Continual learning
Replaceable components
Biology-inspired but engineering-first
Implementation agnostic

No implementation strategy is assumed.

Backpropagation, local learning, symbolic reasoning, probabilistic inference, spiking computation, hybrid systems, and future approaches should all be compatible if they satisfy the architectural contracts.

Functional Hierarchy

The architecture is organized into seven functional levels.

These levels describe responsibility, not implementation.

Level 0

Synthetic Neuron

Smallest computational unit.

Responsible for local computation, local adaptation, sparse communication, and maintaining minimal internal state.

Level 1

Cortical Tower

Primary cognitive building block.

Responsible for local world modeling through perception and action.

Maintains prediction, local memory, reference frame, confidence, and sensorimotor state.

Learns continuously through interaction.

Level 2

Tower Cluster

Collection of cooperative towers.

Responsible for solving coherent local cognitive problems.

Provides local consensus before higher-level integration.

Coordinates specialized tower populations.

Level 3

Functional Region

Coordinates multiple clusters.

Represents functional capabilities rather than anatomical structures.

Possible examples include:

Vision

Language

Motor Control

Spatial Reasoning

Executive Processing

Planning

Social Cognition

The architecture does not require biological anatomical fidelity.

Level 4

Cognitive Systems

Cross-regional cognitive services.

Examples include:

Working Memory

Semantic Memory

Episodic Memory

Attention

Planning

Motivation

Sleep

Memory Consolidation

These systems coordinate information across multiple regions.

Level 5

Synthetic Cortex

Global cognitive coordinator.

Responsible for:

Global planning

Cross-region coordination

Conflict resolution

Attention routing

Global synchronization

High-level reasoning

The cortex never manages individual neurons or towers directly.

Level 6

Synthetic Brain

Complete cognitive agent.

Includes:

Goals

Identity

Long-term objectives

External memory interfaces

Tools

Robotic interfaces

Environment interaction

Execution policies

The Synthetic Brain represents the deployable cognitive system.

External Systems

The following are intentionally outside the architecture.

Training

Curriculum

Datasets

Simulation

Teacher agents

Human interaction

Robotics environments

Evaluation

Benchmarking

These systems interact with OCA but are not architectural layers.

Component Contract

Every architectural component must define:

Purpose

Responsibilities

Internal State

Inputs

Outputs

Operations

Communication Model

Lifecycle

Dependencies

Failure Modes

Scalability Considerations

Known Limitations

Components should be specified through functional contracts rather than implementation details.

Communication Model

Communication is hierarchical and sparse.

The architecture supports communication at multiple scales.

Neuron

Tower

Cluster

Region

Cognitive System

Global Cortex

Global Brain

Communication should favor structured events over continuous dense state sharing.

Synchronization mechanisms are architecture-level services.

Internal State

Every architectural component maintains internal state.

State is considered a first-class architectural concept.

Stateless cognitive components are discouraged except for explicitly defined utility modules.

Oscillation Layer

OCA defines oscillatory coordination as an architectural service.

Oscillations are synchronization mechanisms.

They are not memory containers.

They coordinate:

Timing

Attention

Synchronization

Communication windows

Learning windows

Offline consolidation

Multiple temporal scales should be supported.

The exact implementation is intentionally unspecified.

Global Cognitive State

The architecture defines a global cognitive state abstraction.

Its purpose is to coordinate overall system behavior.

Possible modes include:

Focused

Exploratory

Learning

Planning

Dreaming

Consolidating

Recovery

Idle

The architecture defines responsibilities rather than implementation.

Future implementations may represent this state differently.

Memory

Memory is distributed.

The architecture distinguishes:

Short-Term Memory

Working Memory

Episodic Memory

Semantic Memory

Procedural Memory

No specific storage mechanism is required.

Sensorimotor Loop

Perception and action are symmetric architectural concepts.

Every cognitive level may receive observations, generate actions, receive feedback, and adapt.

The architecture is intentionally grounded in interaction rather than passive observation.

Extensibility

Every level may have multiple implementations.

For example:

Different neuron models

Different tower models

Different memory systems

Different synchronization systems

Different planning systems

All implementations must satisfy the architectural contract.

Architecture Validation

This specification intentionally does not define evaluation procedures.

Architecture validation is performed through the independent OCA Cognitive Gates Specification (CBS).

CBS defines:

Functional validation

Behavioral validation

Robustness

Scalability

Efficiency

Regression testing

Architectural comparison

No architecture should be considered OCA-compliant until it satisfies the required Cognitive Gates for its maturity level.

Non Goals

OCA does not attempt to:

Replicate biological anatomy.

Model molecular neuroscience.

Optimize for benchmark scores.

Depend on a specific learning algorithm.

Require transformer architectures.

Require language as the primary cognitive modality.

Specify implementation details.

Constrain future cognitive architectures.

Future Specifications

The architecture specification is intentionally complemented by independent documents.

Examples include:

OCA Cognitive Gates Specification

Training Specification

Curriculum Specification

Communication Protocol Specification

Memory Specification

Oscillation Specification

Reference Implementations

Engineering APIs

Each specification evolves independently under semantic versioning.

Closing Statement

OCA defines a cognitive architecture, not a single model.

Its objective is to establish stable architectural contracts that enable multiple cognitive implementations to evolve, interoperate, and be evaluated through a common engineering framework.

I would make one final architectural change for OCA v4 that I think will make it look much more like a mature engineering project.

Instead of calling the seven layers simply "levels", I'd define them as Architectural Layers and explicitly state that dependencies flow downward while abstractions flow upward. That single rule becomes a fundamental invariant of the architecture:

Lower layers never know higher-level semantics.
Higher layers never manipulate lower-layer implementation details directly.
Each layer communicates only through well-defined contracts.

That principle is common in successful software and systems architecture, and it maps naturally onto the cognitive hierarchy you're designing. It also makes it much easier for independent contributors to replace or improve a layer without breaking the rest of OCA.