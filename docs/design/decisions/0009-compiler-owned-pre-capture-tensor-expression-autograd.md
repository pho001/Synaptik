# ADR 0009: Compiler-owned pre-capture Tensor-expression autograd

## Status

Accepted — 2026-07-26

## Context

The previous compiler plan captured and optimized an immutable forward graph, reconstructed
placeholder Tensors for `ValueId` values, expressed derivative formulas over those placeholders,
and captured only the backward fragment. That design introduced a conversion boundary between
public Tensor expressions and captured graph values, obscured hidden auxiliary output identity,
and split graph-local identity assignment across forward and backward construction.

The current model already provides a backend-independent Tensor expression vocabulary capable of
expressing adjoints. It also represents one producer with ordered multi-output slots, including
the dropout mask and batch-normalization saved statistics required by backward formulas. The
remaining model gap is that a producer retains descriptors, but not the canonical Tensor wrapper
for every slot.

## Decision drivers

- keep derivative-rule ownership in the compiler;
- use one existing Tensor operation vocabulary rather than a second algebra language;
- assign `NodeId` and `ValueId` once for the combined computation;
- preserve exact same-occurrence hidden outputs without reconstructing wrappers;
- fail before partial backward construction when an operation or policy is unsupported;
- optimize and validate the complete immutable graph;
- keep Tensor free of gradient/backward lifecycle state; and
- preserve an explicit path to higher derivatives.

## Options considered

### Captured-IR autograd with placeholder Tensors

Capture and optimize forward graph state first, create placeholder Tensors for forward
`ValueId` values, stage gradients, and translate the backward fragment back into graph state. This
retains the completed forward pipeline directly, but creates a placeholder/binding subsystem,
requires two capture-like identity domains, and cannot obtain exact hidden output wrappers from
the current producer contract.

### Direct immutable-graph algebra

Build derivative `Operation`, `CompiledNode`, and `GraphValue` objects directly through a new
compiler algebra. This avoids temporary Tensors, but duplicates the public operation construction
vocabulary and its Shape, descriptor, provenance, and validation behavior.

### Compiler-owned pre-capture Tensor-expression autograd

Preflight the original forward Tensor expression, construct gradients with ordinary public Tensor
operations, and capture the forward outputs and gradient roots together once. Retain exact
canonical output wrappers on each producer so hidden saved outputs participate by identity.

## Decision

Synaptik adopts compiler-owned pre-capture Tensor-expression autograd.

For `FORWARD_AND_BACKWARD` and the initial `TRAINING_STEP`, the compiler inventories every
backward-reachable producer occurrence and exact derivative-relevant attributes or policies. It
fails closed before constructing backward expressions if any required rule is absent or
ambiguous. `FORWARD_ONLY` skips this work.

Named compiler components such as `ElementwiseGradientRules` dispatch formulas and call only
ordinary public Tensor methods. One compile request may use identity-based maps to collect
contributions and accumulated gradients; accumulation uses `Tensor.add`. These maps are
short-lived compiler bookkeeping, not Tensor state or a graph representation.

Seeds and derivative constants are storage-free Tensor leaves or expressions registered
explicitly as compile-time constant splats. Storage, factory history, and provenance absence never
imply constant status.

One phase-aware capture receives forward outputs, gradient roots and target roles, the original
forward-producer identity set, and explicit constant facts. It assigns graph-local IDs once and
classifies every captured producer as `FORWARD` or `BACKWARD`. Result roles may map multiple
targets to one gradient `ValueId`; the output boundary retains each distinct gradient value once.

Inference and validation follow capture. Canonicalization, the exact rewrites and folding already
selected by compiler tasks 0003A and 0003B, dead-code elimination, and phase-local
common-subexpression elimination then operate on the immutable combined graph where their
existing guards are proved safe. Changed candidates are revalidated through Compiler 0002. No new
rewrite is implied by this ADR.

`TensorProducer` must retain and return the canonical exact Tensor wrapper for every output slot.
`TensorFactory` atomically constructs the shared producer, wrappers, and indexed provenance.
This introduces a collectable object cycle, not mutable ownership: final fields and unpublished
construction provide safe publication, and ordinary garbage collection reclaims the whole
unreachable occurrence. No wrapper is reconstructed.

## Rationale

This option keeps one semantic construction language and one graph capture. Exact Tensor identity
is available when reverse traversal needs a saved auxiliary, while graph-local identity remains a
compiler concern. Compiler ownership is explicit without adding public registries, model
derivative rules, ThreadLocal scope, or mutable Tensor gradients.

Preflight avoids predictable partial backward construction, while post-capture inference remains
the authoritative semantic check. Consequently, later failures may consume temporary Tensor IDs;
that cost is compatible with their existing opaque, monotonic, non-reusable contract.

## Consequences

### Positive

- forward and backward nodes receive graph-local identities once;
- all optimization and logical planning can see one immutable combined graph;
- hidden same-occurrence outputs are exact Tensor expressions available to compiler rules;
- generated gradients remain ordinary differentiable Tensor expressions; and
- public Tensor methods and ergonomic multi-output result carriers remain unchanged.

### Negative and risks

- the model producer now retains its result wrappers, creating an intentional object cycle;
- compiler task 0004 must adapt completed forward-only orchestration for combined phases;
- fail-closed preflight and later full validation are separate checks; and
- failed compilation can consume temporary Tensor IDs.

### Migration, testing, and follow-up

Model task 0025 is the next implementation frontier and adds canonical producer outputs with
focused identity, construction, publication, ID-side-effect, and hidden-output tests. Compiler
task 0004 remains Draft until that prerequisite is complete. It must remove the obsolete
placeholder/`ValueId` conversion design and implement the one-capture pipeline.

Higher derivatives remain future. A later task must define create-graph or derivative-order
lifecycle, cover every operation used in gradient formulas, and represent derivative order
alongside node phase. It must not add `Tensor.gradient`, `Tensor.backward`, mutable gradient
fields, or hidden thread-local state.

This decision changes no module dependency direction. Existing architecture tests therefore need
no dependency-rule update; implementation tasks add focused model/compiler behavior tests.

## Related documentation

- [Architecture contract](../../../ARCHITECTURE.md)
- [Lifecycle](../../architecture/lifecycle.md)
- [Training graph](../../architecture/training-graph.md)
- [Autograd strategy](../notes/autograd-strategy.md)
- [Model master plan](../../planning/modules/model/master-plan.md)
- [Compiler master plan](../../planning/modules/compiler/master-plan.md)
- [Superseded ADR 0005](0005-training-combined-forward-backward-graph.md)
