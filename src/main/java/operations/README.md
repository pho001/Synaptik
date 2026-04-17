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
  - [operations/FusedOperation.java](../operations/FusedOperation.java)
  - [operations/FusedOperationFactory.java](../operations/FusedOperationFactory.java)

The package contains descriptors for several broad groups.

### Elementwise arithmetic and unary primitives

Examples:

- `add`, `sub`, `mul`, `div`
- `neg`, `abs`, `inv`, `sqrt`
- `log`, `exp`, `fastExp`, `tanh`, `fastTanh`, `sigmoid`
- `pow`, `mulScalar`
- `relu`, `clampMin`, `clampMax`
- `min`, `max`

### Compare, bool, and selection primitives

Examples:

- `greaterThan`, `greaterOrEqual`, `lessThan`, `lessOrEqual`
- `equalTo`, `notEqualTo`
- `logicalAnd`, `logicalOr`, `logicalNot`
- `where`

### Layout and indexing primitives

Examples:

- `contiguous`, `reshape`, `expand`, `permute`, `expandDims`, `squeeze`
- `select`, `gather`, `gatherGrad`
- `takeAlongAxis`, `takeAlongAxisGrad`
- `scatterAdd`

### Reduction and normalization-adjacent primitives

Examples:

- `sum`, `mean`
- `reduceMin`, `reduceMax`, `reduceAll`, `reduceAny`
- `reduceMinGrad`, `reduceMaxGrad`
- `softmax`, `softmaxGrad`
- `logSoftmax`, `logSoftmaxGrad`
- `layerNorm`, `rmsNorm`

### Linalg and spatial primitives

Examples:

- `matmul`, `linear`
- `scaledDotProductAttention`
- `scaledDotProductAttentionWeights`
- `scaledDotProductAttentionBackward`
- `conv2d`, `conv2dGemm`
- `conv2dBackwardInput`, `conv2dBackwardWeight`
- `maxPool2d`, `avgPool2d`
- `maxPool2dBackwardInput`, `avgPool2dBackwardInput`

### Loss primitives

Examples:

- `nllLoss`
- `crossEntropyLoss`
- `crossEntropyLossIndices`
- `crossEntropyLossIndicesGrad`

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
- if you are deciding primitive identity and immutable primitive parameters, you are probably in `operations/*`

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
2. If it is primitive-backed, add a small descriptor class in `operations/`.
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
