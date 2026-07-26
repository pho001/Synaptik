# Autograd strategy

> [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) is authoritative; this note explains its
> compiler-owned pre-capture automatic-differentiation design.

## Purpose and status

Automatic differentiation (autograd) derives gradient computations from a forward expression.
The design is accepted, but implementation remains planned. Model task 0025 is Complete and every
producer can return its canonical exact Tensor wrapper for each output position. Compiler task
0004 remains Draft and awaits its dedicated planning pass.

## Mental model

```text
original forward Tensor expression DAG
  -> fail-closed compiler preflight
  -> reverse traversal using named compiler gradient rules
  -> ordinary Tensor expressions for contributions and accumulation
  -> combined forward + gradient Tensor expression DAG
  -> one phase-aware capture
  -> immutable combined compiler graph
  -> inference, validation, exact combined optimization, final validation
  -> publication and planning
```

Tensor expression objects are the single construction language before capture.
`CompiledGraphModel` is the immutable graph state after capture. The compiler does not convert
captured `ValueId` values back into placeholder Tensors and does not maintain a second algebra.

## Ownership and reverse accumulation

- `modules/model` owns public Tensor operations, exact producer occurrence identity, canonical
  output wrappers, and immutable indexed provenance. It owns no derivative rule.
- `modules/compiler` owns preflight, output seeds, differentiation targets, rule dispatch,
  deterministic reverse traversal, contribution accumulation, phase-aware capture, validation,
  and combined optimization.
- Planning assigns backend ownership after expansion. Concrete backends prepare assigned regions.
  Runtime executes prepared work and never derives gradients.

Named compiler components such as `ElementwiseGradientRules` call only existing public methods
such as `mul`, `add`, `sumToShape`, and `transpose`. During one compile request, identity-based
maps associate exact Tensor objects with ordered contributions and accumulated gradients. The
compiler combines contributions with ordinary `Tensor.add`. These maps are temporary bookkeeping,
not graph IR, public Tensor state, a tape, or a registry.

## Preflight and construction failures

Before constructing a backward expression, the compiler inventories every backward-reachable
producer occurrence, output role, exact attributes, and required derivative policy. Unsupported
or ambiguous work fails closed. This prevents a known incomplete rule matrix from creating a
partial backward expression.

Preflight is not full graph inference. The compiler performs authoritative inference and
validation after the one combined capture. A later Tensor construction, capture, inference,
validation, or optimization failure can therefore consume temporary `TensorId` values. This is
compatible with the existing opaque, monotonic, non-reusable ID contract.

## Constants and hidden outputs

Unit seeds, disconnected zeros, and other derivative constants are storage-free Tensor leaves or
expressions. The compiler registers each leaf explicitly with one exact logical-splat fact for
the combined capture. Host storage, labels, factory history, layout, and provenance absence never
imply constant status.

Some formulas need producer outputs omitted from a public ergonomic result. Dropout, for example,
returns the public result and next RNG state while its same-occurrence keep mask is hidden.
Batch-normalization training similarly hides saved batch statistics. Model task 0025 makes each
producer retain the canonical Tensor wrapper for every slot and exposes the smallest indexed
retrieval contract needed by compiler. It never reconstructs an equal wrapper.

This creates an intentional reference cycle:

```text
Tensor -> TensorProvenance -> TensorProducer -> canonical outputs -> Tensor
```

The cycle is immutable expression metadata. Factory construction finishes all final fields before
publishing any output, and ordinary garbage collection can reclaim the whole unreachable
occurrence. There is no global registry, weak-reference protocol, graph membership, or runtime
resource ownership.

## One phase-aware capture

Capture receives:

- ordered forward outputs;
- ordered gradient roots with target-specific roles;
- the identity set of original forward producers; and
- explicit constant-splat facts.

It traverses the combined expression once, assigns `NodeId` and `ValueId` once, and gives every
producer occurrence a per-node `FORWARD` or `BACKWARD` phase. A single positional
`backwardStartIndex` cannot replace this phase map.

Multiple targets may share the exact same accumulated-gradient Tensor and therefore one captured
gradient `ValueId`. Result roles still map those targets independently. The graph output boundary
lists each distinct gradient value once; no manufactured identity node is needed.

## Combined optimization

The immutable combined graph, not a forward-only prefix, enters optimization. Compiler task 0004
must reassess the completed task-0003, 0003A, and 0003B orchestration:

- canonicalization remains mandatory;
- the already selected exact rewrites and constant folds may apply in either phase only when
  their existing guards remain valid;
- dead-code elimination sees the whole graph;
- common-subexpression elimination remains phase-local initially; and
- every changed candidate returns through Compiler 0002 validation.

This migration authorizes no new rewrite, relaxed arithmetic, or physical constant
materialization.

## Compile modes and future derivatives

`FORWARD_ONLY` skips autograd. `FORWARD_AND_BACKWARD` and the initial `TRAINING_STEP` build the
combined expression before capture. `TRAINING_STEP` does not add optimizer updates yet.

Generated gradients are ordinary differentiable Tensor expressions, preserving a route to higher
derivatives. Higher derivatives are not part of Compiler 0004. A later task must define an
explicit create-graph or derivative-order lifecycle contract, provide rules for every operation
used in gradient formulas, and represent derivative order in addition to graph phase.

The design adds no `Tensor.gradient`, `Tensor.backward`, mutable gradient field, ThreadLocal
compilation scope, model-owned derivative rule, public compiler registry, physical saved buffer,
or backend-owned global autograd.

See [Training graph](../../architecture/training-graph.md),
[ADR 0009](../decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md), the
[model master plan](../../planning/modules/model/master-plan.md), and the
[compiler master plan](../../planning/modules/compiler/master-plan.md).
