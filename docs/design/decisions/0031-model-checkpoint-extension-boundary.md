# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0031: Separate [model](../../glossary.md#term-model) [Checkpoint](../../glossary.md#term-checkpoint) from training continuation


## Editorial reading context

Checkpoint code carries durable bytes, NN interprets module state, and Engine materializes computed values on the host. Keeping those roles distinct prevents file handling from becoming model semantics or optimizer logic. The empty-project decision below records a foundation at that time, not the present module location or a ready-to-use persistence API.

Use the [current Checkpoint placement and namespace migration](../../developer-guide/repository-layout.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — 2026-10-13, after independent Class C review and the repository [validation](../../glossary.md#term-validation) gate. The [architecture root](../../../ARCHITECTURE.md#dependency-rules)
and [extensions contract](../../architecture/contracts/extensions-training.md#extensionscheckpoint)
are the authority for this boundary; this record explains the decision.

## Context

[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) owns [module](../../glossary.md#term-module) [parameters](../../glossary.md#term-parameter), persistent [buffers](../../glossary.md#term-buffer), and their state paths, but its state dictionary
contains [Tensor](../../glossary.md#tensor) references rather than durable bytes. [Engine](../../glossary.md#term-engine) owns the public boundary for
[detached](../../glossary.md#detached-model-results) [host](../../glossary.md#term-host) [materialization](../../glossary.md#materialization). Training has a distinct in-memory [optimizer](../../glossary.md#term-optimizer)/session [snapshot](../../glossary.md#term-snapshot);
an exact continuation also needs NN values, effective forward contexts, and caller-owned progress
or random state. None of those [owners](../../glossary.md#term-owner) currently supplies a durable [checkpoint](../../glossary.md#term-checkpoint) [API](../../glossary.md#term-application-programming-interface-api) or joint
restore [transaction](../../glossary.md#term-transaction). A [persistence](../../glossary.md#term-persistence) module placed in NN would pull storage concerns inward; one
module for all recovery would make [model](../../glossary.md#term-model)-only use depend on Training.

## Decision drivers

- Keep the [model](../../glossary.md#term-model)-only [dependency](../../glossary.md#term-dependency) footprint free of Training and Text.
- Preserve [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) state meaning and [Engine](../../glossary.md#term-engine) [materialization](../../glossary.md#materialization) instead of bypassing [backend](../../glossary.md#backend) boundaries.
- Leave [exact training continuation](../../glossary.md#exact-training-continuation) with its state [owners](../../glossary.md#term-owner) and a separately gated coordinator.
- Express the chosen boundary in Gradle and an executable dependency-direction test now, without
  implying an artifact schema or save/load API exists.<span data-glossary-navigation="true"> ([Gradle glossary](../../glossary.md#term-gradle); [artifact glossary](../../glossary.md#term-artifact); [schema glossary](../../glossary.md#term-schema); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>

## Options considered

| Option | Advantage | Cost or rejection reason |
|---|---|---|
| Add [persistence](../../glossary.md#term-persistence) to [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) | Direct access to [module](../../glossary.md#term-module) state paths | Moves file/[codec](../../glossary.md#term-codec) concerns into NN and couples [inference](../../glossary.md#term-inference) composition to persistence. |
| Use one [Checkpoint](../../glossary.md#term-checkpoint) extension for models and training | One [artifact](../../glossary.md#term-artifact) [owner](../../glossary.md#term-owner) | Would make [model](../../glossary.md#term-model)-only use [carry](../../glossary.md#term-carry) Training and obscure owner-specific restore [validation](../../glossary.md#term-validation). |
| Separate model Checkpoint and optional Training Checkpoint adapter | Model-only users retain a small [dependency](../../glossary.md#term-dependency) set; training state stays with its owner | Requires a later adapter and coordinated cross-owner validation. |

The third option is selected. The comparison is about [ownership](../../glossary.md#term-ownership-meanings) and [dependency direction](../../glossary.md#java-platform-and-build-tools), not a
choice of file format or recovery protocol.

## Decision

Add an empty `extensions/checkpoint` [Gradle](../../glossary.md#term-gradle) project downstream of `extensions/nn` and the public
`modules/engine` [facade](../../glossary.md#term-facade). Those are its only current direct project dependencies. It has no Java
types, [persistence](../../glossary.md#term-persistence) behavior, or [artifact](../../glossary.md#term-artifact) format in this task. [Checkpoint](../../glossary.md#term-checkpoint) will own [model](../../glossary.md#term-model)-only
durable recovery while [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) retains [module](../../glossary.md#term-module)-state semantics and installation and [Engine](../../glossary.md#term-engine) retains
public [host](../../glossary.md#term-host) [materialization](../../glossary.md#materialization). Checkpoint must not depend on Training, Text, or lower execution and
[backend](../../glossary.md#backend) internals; inward modules, NN, and Training must not depend on it. A later public
Checkpoint [API](../../glossary.md#term-application-programming-interface-api) that directly names Model types may justify a direct Model edge via coordinated
architecture and test updates. The current empty project's exact two-edge test does not ban
that future decision.

Reserve a separate optional `extensions/training-checkpoint` adapter downstream of Training and
Checkpoint for [exact training continuation](../../glossary.md#exact-training-continuation). It is not included in settings or implemented until
its later gate. It cannot substitute for Training-owned state [validation](../../glossary.md#term-validation) or the caller's
progress/external-state [handoff](../../glossary.md#term-handoff) described by [ADR 0030](0030-exact-training-resume-state-ownership.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>.

## Rationale

A [model](../../glossary.md#term-model)-only application needs [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s state contract and [Engine](../../glossary.md#term-engine)'s public [materialization](../../glossary.md#materialization), not an
[optimizer](../../glossary.md#term-optimizer). Separating the adapter prevents Training state from becoming part of the model [schema](../../glossary.md#term-schema)
and keeps NN free of reverse dependencies. Requiring a later gate avoids claiming exact or [atomic](../../glossary.md#resource-transaction-and-rollback)
[resume](../../glossary.md#schema-serialization-and-checkpoint) before supported [owner](../../glossary.md#term-owner) [validation](../../glossary.md#term-validation) and installation exist.

## Consequences

### Positive

The new project and [architecture test](../../glossary.md#term-architecture-test) make the [model](../../glossary.md#term-model)-only boundary visible and enforceable.
Future model [persistence](../../glossary.md#term-persistence) can be designed without making Training or Text mandatory.

### Negative and risks

The empty project cannot save or load anything. The future adapter adds an [artifact](../../glossary.md#term-artifact) and a
cross-[owner](../../glossary.md#term-owner) coordination problem; it cannot be implemented merely by serializing today's
`TrainingState` or [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) `StateDictionary`.

### Migration, testing, and follow-up

There is no Java [API](../../glossary.md#term-application-programming-interface-api) or persisted data to migrate. The focused [architecture test](../../glossary.md#term-architecture-test) checks the
present exact [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[Engine](../../glossary.md#term-engine) edges, absence of reverse edges, and absence of the optional adapter.
Later [artifact](../../glossary.md#term-artifact), [host](../../glossary.md#term-host)-value [handoff](../../glossary.md#term-handoff), and training-[resume](../../glossary.md#schema-serialization-and-checkpoint) tasks must update tests and contracts if
they add an API or [dependency](../../glossary.md#term-dependency) and must establish their own [validation](../../glossary.md#term-validation) and recovery semantics.

## Related documentation

- [Architecture dependency rules](../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([dependency glossary](../../glossary.md#term-dependency))</span>
- [Owning extensions contract](../../architecture/contracts/extensions-training.md#extensionscheckpoint)
- [Module boundary explanation](../../architecture/module-boundaries.md#extensionscheckpoint)<span data-glossary-navigation="true"> ([Module glossary](../../glossary.md#term-module))</span>
- [Dependency direction explanation](../../architecture/dependency-rules.md#generic-persistence-and-nn-recovery-have-distinct-owners)<span data-glossary-navigation="true"> ([Dependency direction glossary](../../glossary.md#java-platform-and-build-tools))</span>
- [Checkpoint 0001 brief](../../planning/extensions/checkpoint/tasks/0001-checkpoint-module-boundary.md)<span data-glossary-navigation="true"> ([Checkpoint glossary](../../glossary.md#term-checkpoint))</span>
- [Checkpoint dependency test](../../../testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/CheckpointDependencyContractTest.java)
