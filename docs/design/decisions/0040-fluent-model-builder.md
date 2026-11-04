# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0040 [Fluent Model builder](../../glossary.md#fluent-model-builder) independent of [checkpoint](../../glossary.md#term-checkpoint) [admission](../../glossary.md#term-admission)

## Status

Accepted policy — 2026-10-19; implementation awaits independent Class C review and the main gate.

## Context

Owned Models already [cache](../../glossary.md#term-cache) public [Engine](../../glossary.md#term-engine) preparations and delegate updates to [TrainingSession](../../glossary.md#training-session--trainingsession).
Their GraphCheckpoint acquisition admits eight [operation](../../glossary.md#operation) kinds. Requiring that acquisition for
every construction path would prevent existing convolutional and static recurrent layers from
using the [Model](../../glossary.md#term-model) [lifecycle](../../glossary.md#lifecycle) merely because their expressions cannot be serialized.

## Decision drivers

Keep one execution [lifecycle](../../glossary.md#lifecycle), private working state and current live guards. Reuse existing layers
and [Tensor](../../glossary.md#tensor) semantics without [widening](../../glossary.md#term-widening) [Compiler](../../glossary.md#term-compiler) reconstruction, [checkpoint](../../glossary.md#term-checkpoint) formats or adjoints.

## Options considered

Keeping mandatory [graph](../../glossary.md#graph) acquisition preserves the status quo but excludes executable compositions.
Extending reconstruction/[serialization](../../glossary.md#term-serialization) for every layer couples fluent construction to unrelated
format and [compiler](../../glossary.md#term-compiler) decisions. Retaining a closed, privately owned layer chain separates those
concerns while preserving ordinary [provenance](../../glossary.md#provenance) for [compile](../../glossary.md#compile) and [autograd](../../glossary.md#autograd).

## Decision and rationale

Use one public ModelBuilder returning the existing [Model](../../glossary.md#term-model) [specialization](../../glossary.md#term-specialization). Copy supplied leaves
once by [identity](../../glossary.md#term-identity-meanings) through the existing zero-[node](../../glossary.md#node) raw copier before expressions; automatic recipes
create fresh private layers. Retain a frozen Sequential and exact [prototype](../../glossary.md#term-prototype) outputs. Cold forward
composes that chain; compute/training consume fixed outputs and existing cached [bindings](../../glossary.md#term-binding).
Save alone enforces the unchanged eight-kind format domain before file work. A failed save leaves
the Model usable and [Engine](../../glossary.md#term-engine) selection unfrozen. No [callback](../../glossary.md#term-callback) entry point, [registry](../../glossary.md#term-registry), second [graph](../../glossary.md#graph),
[compiler](../../glossary.md#term-compiler) edge, [backend](../../glossary.md#backend) change or execution [owner](../../glossary.md#term-owner) is introduced.

## Consequences

MLPs retain v1/v2 round trips; CNNs and all-output [TBF](../../glossary.md#term-tbf) recurrent stacks can execute without a format
extension. Unsupported saves remain explicit failures. Static recurrent [unrolling](../../glossary.md#term-unrolling) increases cold
[graph](../../glossary.md#graph) size and is not [scan](../../glossary.md#term-cumulative-scan) [BPTT](../../glossary.md#term-backpropagation-through-time-bptt); [Conv3d](../../glossary.md#conv3d--ncdhw) backward remains unsupported. Exact [signatures](../../glossary.md#term-signature), state guards,
native [admission](../../glossary.md#term-admission), exclusion and closure remain shared with existing owned Models.

Tests must compare compute and [SGD](../../glossary.md#term-stochastic-gradient-descent-sgd) against independent [Tensor](../../glossary.md#tensor)/[Engine](../../glossary.md#term-engine) equations, cover private
acquisition and save failure isolation, and preserve existing [lifecycle](../../glossary.md#lifecycle) evidence. Independent review
and the main native-enabled repository gate remain required before acceptance.

## Related documentation

- [Root lifecycle](../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Owning NN contract](../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([NN glossary](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [Lifecycle explanation](../../architecture/lifecycle.md#compile-lifecycle)
- [Implementation brief](../../planning/extensions/nn/tasks/0029-fluent-model-builder-and-checkpoint-independent-execution.md)
