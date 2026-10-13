# ADR 0031: Separate model Checkpoint from training continuation

## Status

Accepted — 2026-10-13, after independent Class C review and the repository validation gate. The [architecture root](../../../ARCHITECTURE.md#dependency-rules)
and [extensions contract](../../architecture/contracts/extensions-training.md#extensionscheckpoint)
are the authority for this boundary; this record explains the decision.

## Context

NN owns module parameters, persistent buffers, and their state paths, but its state dictionary
contains Tensor references rather than durable bytes. Engine owns the public boundary for
detached host materialization. Training has a distinct in-memory optimizer/session snapshot;
an exact continuation also needs NN values, effective forward contexts, and caller-owned progress
or random state. None of those owners currently supplies a durable checkpoint API or joint
restore transaction. A persistence module placed in NN would pull storage concerns inward; one
module for all recovery would make model-only use depend on Training.

## Decision drivers

- Keep the model-only dependency footprint free of Training and Text.
- Preserve NN state meaning and Engine materialization instead of bypassing backend boundaries.
- Leave exact training continuation with its state owners and a separately gated coordinator.
- Express the chosen boundary in Gradle and an executable dependency-direction test now, without
  implying an artifact schema or save/load API exists.

## Options considered

| Option | Advantage | Cost or rejection reason |
|---|---|---|
| Add persistence to NN | Direct access to module state paths | Moves file/codec concerns into NN and couples inference composition to persistence. |
| Use one Checkpoint extension for models and training | One artifact owner | Would make model-only use carry Training and obscure owner-specific restore validation. |
| Separate model Checkpoint and optional Training Checkpoint adapter | Model-only users retain a small dependency set; training state stays with its owner | Requires a later adapter and coordinated cross-owner validation. |

The third option is selected. The comparison is about ownership and dependency direction, not a
choice of file format or recovery protocol.

## Decision

Add an empty `extensions/checkpoint` Gradle project downstream of `extensions/nn` and the public
`modules/engine` facade. Those are its only current direct project dependencies. It has no Java
types, persistence behavior, or artifact format in this task. Checkpoint will own model-only
durable recovery while NN retains module-state semantics and installation and Engine retains
public host materialization. Checkpoint must not depend on Training, Text, or lower execution and
backend internals; inward modules, NN, and Training must not depend on it. A later public
Checkpoint API that directly names Model types may justify a direct Model edge via coordinated
architecture and test updates. The current empty project's exact two-edge test does not ban
that future decision.

Reserve a separate optional `extensions/training-checkpoint` adapter downstream of Training and
Checkpoint for exact training continuation. It is not included in settings or implemented until
its later gate. It cannot substitute for Training-owned state validation or the caller's
progress/external-state handoff described by [ADR 0030](0030-exact-training-resume-state-ownership.md).

## Rationale

A model-only application needs NN's state contract and Engine's public materialization, not an
optimizer. Separating the adapter prevents Training state from becoming part of the model schema
and keeps NN free of reverse dependencies. Requiring a later gate avoids claiming exact or atomic
resume before supported owner validation and installation exist.

## Consequences

### Positive

The new project and architecture test make the model-only boundary visible and enforceable.
Future model persistence can be designed without making Training or Text mandatory.

### Negative and risks

The empty project cannot save or load anything. The future adapter adds an artifact and a
cross-owner coordination problem; it cannot be implemented merely by serializing today's
`TrainingState` or NN `StateDictionary`.

### Migration, testing, and follow-up

There is no Java API or persisted data to migrate. The focused architecture test checks the
present exact NN/Engine edges, absence of reverse edges, and absence of the optional adapter.
Later artifact, host-value handoff, and training-resume tasks must update tests and contracts if
they add an API or dependency and must establish their own validation and recovery semantics.

## Related documentation

- [Architecture dependency rules](../../../ARCHITECTURE.md#dependency-rules)
- [Owning extensions contract](../../architecture/contracts/extensions-training.md#extensionscheckpoint)
- [Module boundary explanation](../../architecture/module-boundaries.md#extensionscheckpoint)
- [Dependency direction explanation](../../architecture/dependency-rules.md#model-only-recovery-is-downstream-of-nn-and-engine)
- [Checkpoint 0001 brief](../../planning/extensions/checkpoint/tasks/0001-checkpoint-module-boundary.md)
- [Checkpoint dependency test](../../../testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/CheckpointDependencyContractTest.java)
