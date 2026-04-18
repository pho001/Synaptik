# Operations Package

## Purpose

The `operations` package contains the canonical graph primitive descriptors.

An `Operation` answers the question:

- “what primitive does this tensor node represent?”

It does **not** answer:

- “how do we build this node from public API?”
- “how is the backward graph wired?”
- “which backend loop executes it?”

Those concerns live in other layers:

- public graph construction: [tensor/](../tensor)
- family builders: [tensor/ops/](../tensor/ops)
- graph rewrites and fusion: [graph/](../graph)
- runtime execution: [backend/](../backend)

## Reading Guide

This package has to stay readable for three different audiences:

1. Tensor family implementers
   - add or update descriptors when a tensor surface is truly primitive-backed
2. Graph / optimizer implementers
   - use descriptor shape and immutable parameters for rewrites, fusion, and lowering
3. Backend / runtime implementers
   - dispatch execution from descriptor type and descriptor metadata

That distinction matters because a descriptor is intentionally not:

- a public API builder
- an autograd formula holder
- a kernel implementation
- a runtime cache container

## Where `operations` Sits In The Stack

The package stack is:

1. `Tensor` / `TensorOps`
2. `tensor.ops.*`
3. `operations.*`
4. `graph/*` optimizer and compiler
5. `backend/*` runtime kernels

The important contract is:

- `Tensor` is user-facing
- `tensor.ops.*` decides how to build the graph
- `operations.*` describes the primitive node that ended up in the graph

Another way to read the layering:

- `tensor` asks: "what graph do we want to build?"
- `operations` answers: "what primitive node is this?"
- `graph` asks: "can this node be rewritten, fused, or lowered?"
- `backend` asks: "how do we execute this primitive efficiently?"

## What A Descriptor Should Contain

An operation descriptor should usually contain only:

- `opType()`
- immutable descriptor parameters needed to interpret the primitive
- optional readable expression text for debugging

Typical descriptor state:

- reduction axis and `keepDims`
- scalar exponent or scalar multiplier
- reshape/permute metadata
- attention scale and mask-presence flags
- loss reduction mode or ignore-index metadata

An operation descriptor should usually **not** contain:

- forward graph-building logic
- backward lambdas
- backend dispatch heuristics
- execution caches
- generic fallback loop implementations
- mutable runtime state

That split is deliberate.
If a descriptor starts accumulating family logic, the architecture collapses back into one giant mixed layer.

## Descriptor Taxonomy

The package contains more than one kind of primitive descriptor.
That is expected, but the distinction should stay explicit.

### Canonical forward descriptors

These represent stable first-class graph semantics that the rest of the stack should be able to reason about directly.

Examples:

- `add`, `mul`, `relu`, `sigmoid`
- `sum`, `softmax`, `logSoftmax`
- `matmul`, `linear`
- `scaledDotProductAttention`
- `conv2d`, `maxPool2d`, `avgPool2d`
- `crossEntropyLoss`, `crossEntropyLossIndices`

These are the descriptors most likely to correspond to:

- public tensor surfaces
- rewrite/lowering anchors
- backend dispatch families

### Auxiliary backward / helper descriptors

These are still real primitives, but they usually exist because some backward path or hot internal path needs a stable runtime contract.

Examples:

- `gatherGrad`
- `takeAlongAxisGrad`
- `reduceMinGrad`, `reduceMaxGrad`
- `softmaxGrad`
- `scaledDotProductAttentionWeights`
- `scaledDotProductAttentionBackward`
- `conv2dBackwardInput`, `conv2dBackwardWeight`

These are valid descriptors, but they should not be documented as if they were automatically public ergonomic tensor APIs.

### Optimizer / fused descriptors

These exist because the graph compiler needs to represent lowered or fused structure explicitly.

Examples:

- `FusedOperation`
- descriptors produced only after rewrite or lowering phases

These are still declarative descriptors.
They are not executable kernels and they are not a second public modeling API.

## Main Components

- base primitive interface
  - [operations/Operation.java](../operations/Operation.java)
- fused descriptor support
  - [operations/fused/FusedOperation.java](../operations/fused/FusedOperation.java)
  - [operations/fused/FusedOperationFactory.java](../operations/fused/FusedOperationFactory.java)

## Package Layout

The package is no longer flat.
Descriptors are grouped by the same broad families used in `tensor.ops.*` and CPU backend dispatch.

Current layout:

```text
operations/
  Operation.java
  fused/
  elementwise/
    binary/
    unary/
    compare/
    logical/
    where/
  layout/
  index/
  reduction/
  normalization/
  linalg/
  nn/
    conv/
    pool/
  loss/
```

That split is intentional:

- `tensor.ops.*` builds graphs by semantic families
- `graph` rewrites and lowerings reason about descriptors by family
- `backend/kernels/cpu/*` dispatches kernels by family

Keeping `operations` grouped the same way makes descriptor ownership easier to read and avoids one flat directory of unrelated primitives.

### `elementwise/binary`

File family:

- [operations/elementwise/binary/add.java](../operations/elementwise/binary/add.java)
- [operations/elementwise/binary/sub.java](../operations/elementwise/binary/sub.java)
- [operations/elementwise/binary/mul.java](../operations/elementwise/binary/mul.java)
- [operations/elementwise/binary/div.java](../operations/elementwise/binary/div.java)
- [operations/elementwise/binary/min.java](../operations/elementwise/binary/min.java)
- [operations/elementwise/binary/max.java](../operations/elementwise/binary/max.java)
- [operations/elementwise/binary/minGrad.java](../operations/elementwise/binary/minGrad.java)
- [operations/elementwise/binary/maxGrad.java](../operations/elementwise/binary/maxGrad.java)

These are pure elementwise numeric binary descriptors plus the explicit min/max backward helper descriptors that naturally belong to the same semantic family.

### `elementwise/unary`

Examples:

- [operations/elementwise/unary/neg.java](../operations/elementwise/unary/neg.java)
- [operations/elementwise/unary/abs.java](../operations/elementwise/unary/abs.java)
- [operations/elementwise/unary/inv.java](../operations/elementwise/unary/inv.java)
- [operations/elementwise/unary/log.java](../operations/elementwise/unary/log.java)
- [operations/elementwise/unary/exp.java](../operations/elementwise/unary/exp.java)
- [operations/elementwise/unary/fastExp.java](../operations/elementwise/unary/fastExp.java)
- [operations/elementwise/unary/tanh.java](../operations/elementwise/unary/tanh.java)
- [operations/elementwise/unary/fastTanh.java](../operations/elementwise/unary/fastTanh.java)
- [operations/elementwise/unary/pow.java](../operations/elementwise/unary/pow.java)
- [operations/elementwise/unary/mulScalar.java](../operations/elementwise/unary/mulScalar.java)
- [operations/elementwise/unary/relu.java](../operations/elementwise/unary/relu.java)
- [operations/elementwise/unary/sigmoid.java](../operations/elementwise/unary/sigmoid.java)
- [operations/elementwise/unary/clampMin.java](../operations/elementwise/unary/clampMin.java)
- [operations/elementwise/unary/clampMax.java](../operations/elementwise/unary/clampMax.java)

The rule here is simple:

- if the primitive acts elementwise on one input tensor and is not compare/bool-specific, it belongs here

### `elementwise/compare`

Examples:

- [operations/elementwise/compare/greaterThan.java](../operations/elementwise/compare/greaterThan.java)
- [operations/elementwise/compare/greaterOrEqual.java](../operations/elementwise/compare/greaterOrEqual.java)
- [operations/elementwise/compare/lessThan.java](../operations/elementwise/compare/lessThan.java)
- [operations/elementwise/compare/lessOrEqual.java](../operations/elementwise/compare/lessOrEqual.java)
- [operations/elementwise/compare/equalTo.java](../operations/elementwise/compare/equalTo.java)
- [operations/elementwise/compare/notEqualTo.java](../operations/elementwise/compare/notEqualTo.java)

These map cleanly to the compare kernel family in the CPU backend.

### `elementwise/logical`

Examples:

- [operations/elementwise/logical/logicalAnd.java](../operations/elementwise/logical/logicalAnd.java)
- [operations/elementwise/logical/logicalOr.java](../operations/elementwise/logical/logicalOr.java)
- [operations/elementwise/logical/logicalNot.java](../operations/elementwise/logical/logicalNot.java)

These are separated from numeric compare ops because they operate on boolean semantics, not numeric ordering semantics.

### `elementwise/where`

- [operations/elementwise/where/where.java](../operations/elementwise/where/where.java)

`where` stays in its own tiny family because the runtime has its own broadcast planning and execution path for ternary elementwise selection.

### `layout`

Examples:

- [operations/layout/contiguous.java](../operations/layout/contiguous.java)
- [operations/layout/reshape.java](../operations/layout/reshape.java)
- [operations/layout/expand.java](../operations/layout/expand.java)
- [operations/layout/permute.java](../operations/layout/permute.java)
- [operations/layout/expandDims.java](../operations/layout/expandDims.java)
- [operations/layout/squeeze.java](../operations/layout/squeeze.java)
- [operations/layout/select.java](../operations/layout/select.java)
- [operations/layout/noop.java](../operations/layout/noop.java)

`select` is intentionally grouped with layout descriptors instead of index descriptors:

- its `OpType` is `SELECT`
- the CPU backend treats it as an alias-view style layout remap
- it does not materialize indexed gather output like `gather` or `takeAlongAxis`

### `index`

Examples:

- [operations/index/gather.java](../operations/index/gather.java)
- [operations/index/gatherGrad.java](../operations/index/gatherGrad.java)
- [operations/index/takeAlongAxis.java](../operations/index/takeAlongAxis.java)
- [operations/index/takeAlongAxisGrad.java](../operations/index/takeAlongAxisGrad.java)
- [operations/index/scatterAdd.java](../operations/index/scatterAdd.java)

These are true indexed read/write primitives with dedicated backend contracts.

### `reduction`

Examples:

- [operations/reduction/sum.java](../operations/reduction/sum.java)
- [operations/reduction/mean.java](../operations/reduction/mean.java)
- [operations/reduction/reduceMin.java](../operations/reduction/reduceMin.java)
- [operations/reduction/reduceMax.java](../operations/reduction/reduceMax.java)
- [operations/reduction/reduceAll.java](../operations/reduction/reduceAll.java)
- [operations/reduction/reduceAny.java](../operations/reduction/reduceAny.java)
- [operations/reduction/reduceMinGrad.java](../operations/reduction/reduceMinGrad.java)
- [operations/reduction/reduceMaxGrad.java](../operations/reduction/reduceMaxGrad.java)
- [operations/reduction/softmax.java](../operations/reduction/softmax.java)
- [operations/reduction/softmaxGrad.java](../operations/reduction/softmaxGrad.java)
- [operations/reduction/logSoftmax.java](../operations/reduction/logSoftmax.java)
- [operations/reduction/logSoftmaxGrad.java](../operations/reduction/logSoftmaxGrad.java)

`softmax` and `logSoftmax` stay here because the CPU backend executes them in the reduction family, even though they are semantically normalization-like.

### `normalization`

Examples:

- [operations/normalization/layerNorm.java](../operations/normalization/layerNorm.java)
- [operations/normalization/rmsNorm.java](../operations/normalization/rmsNorm.java)

This matches the public tensor family split better than hiding everything under a generic `nn` root.

### `linalg`

Examples:

- [operations/linalg/matmul.java](../operations/linalg/matmul.java)
- [operations/linalg/linear.java](../operations/linalg/linear.java)
- [operations/linalg/scaledDotProductAttention.java](../operations/linalg/scaledDotProductAttention.java)
- [operations/linalg/scaledDotProductAttentionWeights.java](../operations/linalg/scaledDotProductAttentionWeights.java)
- [operations/linalg/scaledDotProductAttentionBackward.java](../operations/linalg/scaledDotProductAttentionBackward.java)

This is the family for matrix-style primitives and attention primitives that are fundamentally linalg workloads.

### `nn/conv`

Examples:

- [operations/nn/conv/conv2d.java](../operations/nn/conv/conv2d.java)
- [operations/nn/conv/conv2dGemm.java](../operations/nn/conv/conv2dGemm.java)
- [operations/nn/conv/conv2dBackwardInput.java](../operations/nn/conv/conv2dBackwardInput.java)
- [operations/nn/conv/conv2dBackwardInputGemm.java](../operations/nn/conv/conv2dBackwardInputGemm.java)
- [operations/nn/conv/conv2dBackwardWeight.java](../operations/nn/conv/conv2dBackwardWeight.java)
- [operations/nn/conv/conv2dBackwardWeightGemm.java](../operations/nn/conv/conv2dBackwardWeightGemm.java)

The nested `nn/conv` split mirrors the CPU backend's `nn` family while still keeping the concrete convolution descriptors together.

### `nn/pool`

Examples:

- [operations/nn/pool/maxPool2d.java](../operations/nn/pool/maxPool2d.java)
- [operations/nn/pool/maxPool2dBackwardInput.java](../operations/nn/pool/maxPool2dBackwardInput.java)
- [operations/nn/pool/avgPool2d.java](../operations/nn/pool/avgPool2d.java)
- [operations/nn/pool/avgPool2dBackwardInput.java](../operations/nn/pool/avgPool2dBackwardInput.java)

### `loss`

Examples:

- [operations/loss/nllLoss.java](../operations/loss/nllLoss.java)
- [operations/loss/crossEntropyLoss.java](../operations/loss/crossEntropyLoss.java)
- [operations/loss/crossEntropyLossIndices.java](../operations/loss/crossEntropyLossIndices.java)
- [operations/loss/crossEntropyLossIndicesGrad.java](../operations/loss/crossEntropyLossIndicesGrad.java)

### `fused`

Examples:

- [operations/fused/FusedOperation.java](../operations/fused/FusedOperation.java)
- [operations/fused/FusedOperationFactory.java](../operations/fused/FusedOperationFactory.java)

These are optimizer/runtime-owned descriptors produced after rewrite or fusion phases.

Not every descriptor is guaranteed to have a one-to-one public `Tensor` instance method.
Some exist mainly because the optimizer or backend needs a stable primitive contract.

The public API source of truth remains:

- [tensor/API.md](../tensor/API.md)

## Relation To `tensor.ops.*`

The family builders under `tensor.ops.*` own:

- user-facing input validation
- dtype/broadcast/layout checks
- deciding whether an operation should be primitive-backed or composed
- attaching backward lambdas

Examples:

- [tensor/ops/unary/TensorUnaryOps.java](../tensor/ops/unary/TensorUnaryOps.java)
- [tensor/ops/binary/TensorBinaryOps.java](../tensor/ops/binary/TensorBinaryOps.java)
- [tensor/ops/reduction/TensorReduceOps.java](../tensor/ops/reduction/TensorReduceOps.java)
- [tensor/ops/linalg/TensorAttentionOps.java](../tensor/ops/linalg/TensorAttentionOps.java)
- [tensor/ops/loss/TensorLossOps.java](../tensor/ops/loss/TensorLossOps.java)

This means:

- descriptor classes are intentionally small
- family builders are where the semantic assembly happens

One useful rule:

- if you are deciding tensor semantics, you are probably in `tensor.ops.*`
- if you are deciding primitive identity and immutable primitive parameters, you are probably in `operations/**`

## Primitive-Backed vs Composed Surface

This distinction matters when deciding whether a new descriptor belongs here.

### Primitive-backed public surface

These are public tensor operations whose graph node is represented directly by a descriptor in this package.

Examples:

- `add`, `sub`, `mul`, `div`
- `relu`, `sigmoid`, `exp`, `tanh`, `log`
- `softmax`, `logSoftmax`
- `matmul`, `linear`
- `scaledDotProductAttention`
- `conv2d`, `pool2d`
- indexed loss primitives
- `layerNorm`, `rmsNorm`

These deserve descriptors because they have real runtime meaning.

### Composed public surface

These are public tensor helpers that are expressed using other tensor operations and therefore do **not** need their own canonical descriptor.

Current examples:

- `minimum`
- `maximum`
- `clamp`
- the `batchNorm(...)` composition path

If an operation is naturally expressed as graph algebra and the runtime does not need a first-class primitive, keep it composed.

### Internal optimizer/runtime-only primitives

Some descriptors exist mainly for optimizer or runtime reasons rather than because users should build them directly.

Examples:

- fused descriptors
- dedicated grad/output auxiliaries such as `softmaxGrad`
- specialized attention/loss backward helpers

These are still valid primitives, but they are not necessarily part of the ergonomic modeling surface.

## Descriptor Lifecycle

In normal graph construction, a primitive-backed operation flows through these stages:

1. `tensor.ops.*` validates inputs and decides to build a primitive-backed node.
2. The builder instantiates an `Operation` descriptor from this package.
3. The graph compiler inspects descriptor type and immutable parameters.
4. Rewrites or fusion may replace that descriptor with another descriptor.
5. Backend code dispatches execution from the final compiled descriptor shape.

That lifecycle is why descriptors must remain:

- immutable
- declarative
- free of builder logic
- free of runtime-owned mutable state

## How Descriptors Are Consumed

Once a tensor node has a descriptor:

- the graph layer uses `opType()` and descriptor state for optimizer reasoning
- fusion and rewrite logic decide whether the node should be replaced, clustered, or lowered
- the backend uses the descriptor to dispatch the correct kernel family
- debug/reporting layers use descriptor text for traceability

This is why descriptors should stay declarative.
They are shared input for multiple downstream subsystems.

## Fused Operations

`FusedOperation` is the descriptor used when the optimizer groups several primitive ops into one fused node.

Important points:

- fused ops are still descriptors, not executable kernels
- the fused descriptor records the fused primitive structure
- backend code chooses how to execute that fused cluster
- the fused descriptor should expose enough structure for traceability, tuning, and backend selection, but not embed backend implementation logic

That keeps the layering intact:

- fusion belongs to the graph compiler
- execution belongs to the backend

## How To Add A New Operation

Use this checklist.

1. Decide whether the operation is:
   - a composed tensor helper
   - a primitive-backed operation
   - an internal optimizer/runtime primitive
2. If it is primitive-backed, add a small descriptor class in the correct `operations/*` family package.
3. Add the public builder in the correct `tensor.ops.*` family.
4. Keep backward wiring in the family builder unless there is a strong reason for a dedicated backward primitive.
5. Add rewrite/backend support only if the new primitive truly needs it.
6. Update the public docs:
   - [tensor/README.md](../tensor/README.md)
   - [tensor/API.md](../tensor/API.md)
   - this file if the package-level architecture changed

## Practical Decision Rules

Choose a new descriptor when at least one of these is true:

- the operation has a distinct runtime contract
- the optimizer needs to recognize it as a first-class node
- the backend has a dedicated kernel or lowering path for it
- using only composed graph algebra would make the graph artificially noisy or slow
- the descriptor boundary makes the graph semantically clearer for downstream tooling

Do **not** add a descriptor only because:

- the API would read nicer
- the backward formula is long
- there is already a similar primitive nearby
- a benchmark would look cleaner with a synthetic node that has no real graph/runtime meaning

Those are not sufficient reasons on their own.

## Common Mistakes

- putting forward graph-building logic into the descriptor class
- putting backend-specific loop logic into the descriptor class
- adding a primitive for what should stay as simple tensor composition
- documenting a descriptor as if it were automatically a public API method
- mixing semantic configuration types into support/internal helpers instead of using public packages such as:
  - [tensor/options](../tensor/options)
  - [tensor/loss](../tensor/loss)
- letting one public tensor surface accidentally map to multiple semantically different descriptors without documenting that split
- introducing optimizer-only helper descriptors and then treating them as canonical public modeling concepts

The package stays clean only if descriptors remain small, declarative, and boring.
