# Operations Package

The `operations` package contains immutable primitive descriptors.

An `Operation` answers one question:

- what primitive does this tensor node represent?

It does not answer:

- how the public API should build this node
- how backward should be wired
- which CPU loop or kernel should execute it

Those belong to:

- `tensor` / `tensor.ops.*`
- `graph/*`
- `backend/*`

## Core Contract

Base interface:

- [Operation.java](../operations/Operation.java)

The key surface is:

- `opType()`
- `getExpression()`
- optional cheap/non-cheap hint through `isCheap()`

The descriptor should carry only immutable semantic parameters such as:

- exponent for `pow`
- scalar for `mulScalar`
- reduction dimension / `keepDims`
- reshape/permute metadata
- attention options
- loss reduction metadata

It should not carry:

- mutable runtime state
- backward lambdas
- kernel loops
- backend dispatch hints

## `OpType` Taxonomy

`Operation.OpType` currently divides primitives by broad category:

- elementwise
- reduction
- layout
- linear algebra
- special
- fused

Some examples:

- elementwise numeric:
  - `ADD`, `SUB`, `MUL`, `DIV`, `NEG`, `EXP`, `TANH`, `POW`
- elementwise compare/logical:
  - `GT`, `LE`, `EQ`, `LOGICAL_AND`
- reductions:
  - `SUM`, `MEAN`, `REDUCE_MAX`, `SOFTMAX`, `LOG_SOFTMAX`
- layout:
  - `RESHAPE`, `PERMUTE`, `EXPAND`, `SELECT`
- special:
  - `LINEAR`, `CROSS_ENTROPY_LOSS_INDICES`, `SCALED_DOT_PRODUCT_ATTENTION`, `CONV2D_GEMM`
- fused:
  - `FUSED`

The `isFusable()` bit on `OpType` is what the graph `FUSE` stage uses as the primary fusable/non-fusable gate.

## Package Layout

Descriptors are grouped by semantic family:

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

That mirrors the structure used by:

- `tensor.ops.*`
- CPU backend kernel families

## Family Examples

### Elementwise binary

Examples:

- [elementwise/binary/add.java](../operations/elementwise/binary/add.java)
- [elementwise/binary/sub.java](../operations/elementwise/binary/sub.java)
- [elementwise/binary/mul.java](../operations/elementwise/binary/mul.java)
- [elementwise/binary/div.java](../operations/elementwise/binary/div.java)
- [elementwise/binary/min.java](../operations/elementwise/binary/min.java)
- [elementwise/binary/max.java](../operations/elementwise/binary/max.java)

These describe pairwise numeric elementwise semantics.

### Elementwise unary

Examples:

- [elementwise/unary/neg.java](../operations/elementwise/unary/neg.java)
- [elementwise/unary/log.java](../operations/elementwise/unary/log.java)
- [elementwise/unary/exp.java](../operations/elementwise/unary/exp.java)
- [elementwise/unary/fastExp.java](../operations/elementwise/unary/fastExp.java)
- [elementwise/unary/tanh.java](../operations/elementwise/unary/tanh.java)
- [elementwise/unary/fastTanh.java](../operations/elementwise/unary/fastTanh.java)
- [elementwise/unary/relu.java](../operations/elementwise/unary/relu.java)
- [elementwise/unary/sigmoid.java](../operations/elementwise/unary/sigmoid.java)
- [elementwise/unary/pow.java](../operations/elementwise/unary/pow.java)
- [elementwise/unary/mulScalar.java](../operations/elementwise/unary/mulScalar.java)

### Compare and logical

Compare:

- `greaterThan`
- `greaterOrEqual`
- `lessThan`
- `lessOrEqual`
- `equalTo`
- `notEqualTo`

Logical:

- `logicalAnd`
- `logicalOr`
- `logicalNot`

These are split because numeric ordering and boolean logic have different semantics and backend handling.

### `where`

`where` is isolated in its own micro-family because ternary select has:

- its own broadcast contract
- its own backend execution path
- special fusion considerations

### Layout

Examples:

- `contiguous`
- `reshape`
- `expand`
- `permute`
- `expandDims`
- `squeeze`
- `select`
- `noop`

`select` intentionally lives with layout-like descriptors rather than indexed gather descriptors because it behaves as a view-style remap rather than as a materialized indexed read.

### Index

Examples:

- `gather`
- `gatherGrad`
- `takeAlongAxis`
- `takeAlongAxisGrad`
- `scatterAdd`

These are true indexed access/update primitives.

### Reduction

Examples:

- `sum`
- `mean`
- `reduceMin`
- `reduceMax`
- `reduceAll`
- `reduceAny`
- `softmax`
- `softmaxGrad`
- `logSoftmax`
- `logSoftmaxGrad`

### Linalg and structured special primitives

Examples:

- `matmul`
- `linear`
- `scaledDotProductAttention`
- `scaledDotProductAttentionBackward`
- `scaledDotProductAttentionWeights`

These are important because many optimizer passes lower decomposed subgraphs back into these higher-level primitives.

### Loss

Examples:

- `nllLoss`
- `crossEntropyLoss`
- `crossEntropyLossIndices`
- `crossEntropyLossIndicesGrad`

### NN / conv / pool / normalization

Examples:

- `conv2d`
- `conv2dGemm`
- `conv2dBackwardInput`
- `conv2dBackwardWeight`
- `maxPool2d`
- `avgPool2d`
- `layerNorm`
- `rmsNorm`

## Primitive vs Composed Surface

Not every public tensor operation must map 1:1 to a primitive.

Good reasons to introduce or retain a primitive:

- optimizer should recognize it directly
- backend has a specialized kernel family for it
- it appears frequently enough that the decomposed form is noisy or slow

Examples of strong primitive candidates in the current codebase:

- `LINEAR`
- `SOFTMAX_GRAD`
- `LOG_SOFTMAX_GRAD`
- `CROSS_ENTROPY_LOSS_INDICES`
- `CROSS_ENTROPY_LOSS_INDICES_GRAD`
- `SCALED_DOT_PRODUCT_ATTENTION`
- `SCALED_DOT_PRODUCT_ATTENTION_BACKWARD`

## Worked Examples

### Example 1: `mulScalar`

Public tensor call:

```java
Tensor y = x.mul(2.5);
```

Underlying descriptor:

- `Operation.OpType.MUL_SCALAR`
- scalar parameter `2.5`

This lets:

- `AR` reason about scalar identities
- `FUSE` fuse it as a cheap or non-cheap op according to current policy
- backend resolve a direct scalar kernel path

### Example 2: lowered linear

Original public graph:

```text
add(matmul(input, weight), bias)
```

After `AR`:

```text
LINEAR(input, weight, bias)
```

The descriptor is now simpler for:

- CSE
- backend dispatch
- trace reporting

## What Operations Must Not Become

This package should not grow into:

- a second public modeling API
- a runtime cache layer
- a kernel implementation layer
- a place for ad hoc benchmark-only abstractions

Descriptors are semantic vocabulary.
Nothing more.
