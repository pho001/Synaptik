# ADR 0040 Fluent Model builder independent of checkpoint admission

## Status

Accepted policy — 2026-10-19; implementation awaits independent Class C review and the main gate.

## Context

Owned Models already cache public Engine preparations and delegate updates to TrainingSession.
Their GraphCheckpoint acquisition admits eight operation kinds. Requiring that acquisition for
every construction path would prevent existing convolutional and static recurrent layers from
using the Model lifecycle merely because their expressions cannot be serialized.

## Decision drivers

Keep one execution lifecycle, private working state and current live guards. Reuse existing layers
and Tensor semantics without widening Compiler reconstruction, checkpoint formats or adjoints.

## Options considered

Keeping mandatory graph acquisition preserves the status quo but excludes executable compositions.
Extending reconstruction/serialization for every layer couples fluent construction to unrelated
format and compiler decisions. Retaining a closed, privately owned layer chain separates those
concerns while preserving ordinary provenance for compile and autograd.

## Decision and rationale

Use one public ModelBuilder returning the existing Model specialization. Copy supplied leaves
once by identity through the existing zero-node raw copier before expressions; automatic recipes
create fresh private layers. Retain a frozen Sequential and exact prototype outputs. Cold forward
composes that chain; compute/training consume fixed outputs and existing cached bindings.
Save alone enforces the unchanged eight-kind format domain before file work. A failed save leaves
the Model usable and Engine selection unfrozen. No callback entry point, registry, second graph,
compiler edge, backend change or execution owner is introduced.

## Consequences

MLPs retain v1/v2 round trips; CNNs and all-output TBF recurrent stacks can execute without a format
extension. Unsupported saves remain explicit failures. Static recurrent unrolling increases cold
graph size and is not scan BPTT; Conv3d backward remains unsupported. Exact signatures, state guards,
native admission, exclusion and closure remain shared with existing owned Models.

Tests must compare compute and SGD against independent Tensor/Engine equations, cover private
acquisition and save failure isolation, and preserve existing lifecycle evidence. Independent review
and the main native-enabled repository gate remain required before acceptance.

## Related documentation

- [Root lifecycle](../../../ARCHITECTURE.md#core-lifecycle)
- [Owning NN contract](../../architecture/contracts/extensions-training.md#owned-tensor-models)
- [Lifecycle explanation](../../architecture/lifecycle.md#compile-lifecycle)
- [Implementation brief](../../planning/extensions/nn/tasks/0029-fluent-model-builder-and-checkpoint-independent-execution.md)
