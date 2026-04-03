# Tensor (src/main/java/tensor)

## Purpose

The `tensor` package provides the core tensor node abstraction used by:

- graph construction
- runtime data storage
- reverse-mode autodiff
- backend dispatch entry

`Tensor` is both:

- a value container (`shape`, `strides`, dtype-specific storage)
- a graph node (`operation`, `prevTensors`, gradient references, backward hook)

It is not the primary owner of explicit compile/runtime artifacts. Those live in:

- [src/main/java/graph/CompiledGraph.java](../graph/CompiledGraph.java)
- [src/main/java/graph/execution/PreparedExecution.java](../graph/execution/PreparedExecution.java)

## Main Components

- Core tensor type:
  - [src/main/java/tensor/Tensor.java](../tensor/Tensor.java)
- API reference:
  - [src/main/java/tensor/API.md](../tensor/API.md)
- Metadata:
  - [src/main/java/tensor/TensorMetadata.java](../tensor/TensorMetadata.java)
- Graph-building helpers:
  - [src/main/java/tensor/TensorOps.java](../tensor/TensorOps.java)
  - [src/main/java/tensor/TensorBinaryOps.java](../tensor/TensorBinaryOps.java)
  - [src/main/java/tensor/TensorUnaryOps.java](../tensor/TensorUnaryOps.java)
  - [src/main/java/tensor/TensorReduceOps.java](../tensor/TensorReduceOps.java)
  - [src/main/java/tensor/TensorLayoutOps.java](../tensor/TensorLayoutOps.java)
  - [src/main/java/tensor/TensorNaryOps.java](../tensor/TensorNaryOps.java)
- Operation descriptors and addition guide:
  - [src/main/java/operations/README.md](../operations/README.md)
- Layout remap utility:
  - [src/main/java/tensor/TensorRemap.java](../tensor/TensorRemap.java)
- Storage/type abstractions:
  - [src/main/java/tensor/TensorStorage.java](../tensor/TensorStorage.java)
  - [src/main/java/tensor/DataType.java](../tensor/DataType.java)
  - [src/main/java/tensor/Float16Storage.java](../tensor/Float16Storage.java)
  - [src/main/java/tensor/Float32Storage.java](../tensor/Float32Storage.java)
  - [src/main/java/tensor/Float64Storage.java](../tensor/Float64Storage.java)

## Data Model

`Tensor` stores:

- tensor storage (`FLOAT16`, `FLOAT32`, `FLOAT64`, `BOOL`)
- metadata (`shape`, `strides`, `label`, `requiresGrad`)
- producing operation (`Operation`) or `null` for leaves/constants
- graph input links (`prevTensors`)
- gradient tensor reference (`gradient`)
- backward lambda (`backwardFunction`)
- optional forced backend (`forcedBackend`)

`Tensor` itself no longer owns compile/runtime cache artifacts.

The intended model is:

- `CompiledGraph` = explicit compile artifact
- `PreparedExecution` = explicit runtime artifact

## Execution Model

Current execution flow is:

1. Build a tensor expression graph through tensor operations.
2. Compile the graph into [`CompiledGraph`](../graph/CompiledGraph.java).
3. Prepare a runtime-bound [`PreparedExecution`](../graph/execution/PreparedExecution.java).
4. Execute in one of the engine modes:
   - `FORWARD`
   - `FORWARD_BACKWARD`

Preferred high-level entry points on `Tensor` are:

- `prepare(ExecutionProfile profile)`
- `compute(ExecutionProfile profile)`
- `compute(PreparedExecution execution, ExecutionMode mode)`

`Tensor` no longer exposes the older optimizer/runtime convenience layer directly.
Compilation and runtime binding are explicit through:

- [src/main/java/graph/CompiledGraph.java](../graph/CompiledGraph.java)
- [src/main/java/graph/execution/PreparedExecution.java](../graph/execution/PreparedExecution.java)

## Public Tensor Operations

The current public graph-building operation surface on `Tensor` is:

### Layout / Shape

- `contiguous()`
- `reshape(int... newShape)`
- `expand(int... newShape)`
- `permute(int... axes)`
- `transpose()`
- `expandDims(int axis)`
- `squeeze(int axis)`

### Binary Arithmetic / Comparison

- `add(Tensor second)`
- `sub(Tensor second)`
- `mul(Tensor second)`
- `div(Tensor second)`
- `min(Tensor second)`
- `max(Tensor second)`
- `matmul(Tensor second)`
- `greaterThan(Tensor second)`
- `greaterOrEqual(Tensor second)`
- `lessThan(Tensor second)`
- `lessOrEqual(Tensor second)`
- `equalTo(Tensor second)`
- `notEqualTo(Tensor second)`
- `where(Tensor condition, Tensor ifTrue, Tensor ifFalse)` (static helper)
- `minimum(Tensor second)`
- `maximum(Tensor second)`
- `logicalAnd(Tensor second)`
- `logicalOr(Tensor second)`
- `logicalNot()`

### Unary / Scalar

- `relu()`
- `neg()`
- `log()`
- `exp()`
- `fastExp()`
- `tanh()`
- `fastTanh()`
- `pow(double exponent)`
- `mul(double scalar)`
- `inv()`
- `sqrt()`
- `sigmoid()`
- `clamp(double minValue, double maxValue)`
- `clampMin(double minValue)`
- `clampMax(double maxValue)`

### Reduction

- `sum(int dimension)`
- `sum(int dimension, boolean keepDims)`
- `sum()`
- `mean(int dimension)`
- `mean(int dimension, boolean keepDims)`
- `mean()`
- `min(int dimension)`
- `min(int dimension, boolean keepDims)`
- `min()`
- `max(int dimension)`
- `max(int dimension, boolean keepDims)`
- `max()`
- `all(int dimension)`
- `all(int dimension, boolean keepDims)`
- `all()`
- `any(int dimension)`
- `any(int dimension, boolean keepDims)`
- `any()`

### Helper / Internal Execution Anchor

- `forwardOutput()`

Not every operation descriptor in `src/main/java/operations/` is exposed as a direct public `Tensor` method.
For example, some descriptors exist for backend/optimizer reasons even when there is no matching public `Tensor` convenience method yet.

The full public API reference remains in:

- [src/main/java/tensor/API.md](../tensor/API.md)

## Contiguous and Materialization Contract

`Tensor` supports both:

- dense contiguous tensors
- non-contiguous or view-like tensors defined by shape/stride metadata

Important current behavior:

- `permute(...)` creates a view-like tensor with reordered strides
- `expand(...)` creates a zero-stride broadcast alias view
- `expandDims(...)` creates a stride-preserving alias view with one inserted size-`1` axis
- `squeeze(...)` creates a stride-preserving alias view with one removed size-`1` axis
- `reshape(...)` changes shape interpretation while preserving element count
  - for contiguous inputs it aliases the same storage as a reshape view
  - for non-contiguous inputs it may materialize a dense reshaped result at runtime
- `contiguous()` is the canonical explicit materialization path

Use `contiguous()` when:

- a later kernel benefits from dense row-major layout
- you want to materialize an expanded zero-stride broadcast view
- you want a stable dense tensor independent of the current view layout

## Broadcasting Contract

The core broadcasting contract used by tensor operations is:

- ranks align from the right
- missing leading dimensions behave as `1`
- two dimensions are compatible if:
  - they are equal
  - or one side is `1`
- the output dimension size is the maximum of the two compatible dimensions

Examples:

- `[2, 3, 4]` with `[3, 4]` produces `[2, 3, 4]`
- `[3, 4]` with `[2, 1, 4]` produces `[2, 3, 4]`
- `[1, 1, 2, 4]` with `[2, 3, 1, 4]` produces `[2, 3, 2, 4]`

This contract is used by:

- binary arithmetic ops
- comparison ops
- logical binary bool ops
- `where(condition, x, y)` via common broadcast shape resolution across all three inputs

Backward note:

- if an operand was broadcast in forward execution, its gradient is reduced back to the original operand shape

### Broadcast-Aware Operations

Broadcast-aware binary operations in the current tensor surface are:

- `add`
- `sub`
- `mul`
- `div`
- `min`
- `max`

Comparison ops follow the same binary broadcasting contract:

- `greaterThan`
- `greaterOrEqual`
- `lessThan`
- `lessOrEqual`
- `equalTo`
- `notEqualTo`

They produce `BOOL` tensors and are nondifferentiable.

`where(condition, x, y)`:

- requires `condition` to have dtype `BOOL`
- requires `x` and `y` to have numeric dtypes
- broadcasts all three inputs to a common output shape
- returns the promoted numeric dtype of `x` and `y`
- propagates gradient only through the selected data branches
- does not propagate gradient through `condition`

`minimum(second)` and `maximum(second)` are explicit compare/select-based piecewise surfaces:

- `minimum(a, b)` behaves like `where(a < b, a, b)`
- `maximum(a, b)` behaves like `where(a > b, a, b)`
- they intentionally use where-style branch semantics on ties
- that tie behavior is different from the specialized `min/max` ops, which keep their existing tie-gradient contract

### Logical Bool Broadcasting

Logical bool ops:

- `logicalAnd`
- `logicalOr`
- `logicalNot`

require `BOOL` tensors, produce `BOOL` tensors, and are nondifferentiable.
`logicalAnd` and `logicalOr` follow the same binary broadcasting contract as other broadcast-aware binary ops.

Their shape contract is:

- ranks align from the right
- missing leading dimensions behave as `1`
- dimensions are compatible if they are equal or one of them is `1`
- output dimension size is the maximum of the two

Examples:

- `[2, 3, 4]` + `[3, 4]` -> `[2, 3, 4]`
- `[2, 3, 4]` + `[4]` -> `[2, 3, 4]`
- `[2, 1, 4]` + `[3, 4]` -> `[2, 3, 4]`

Backward semantics:

- gradients are reduced back to the original input shape for every operand that was broadcast in the forward pass

Additional rank-mismatch examples:

- `[3, 4]` + `[2, 1, 4]` -> `[2, 3, 4]`
- `[1, 1, 2, 4]` + `[2, 3, 1, 4]` -> `[2, 3, 2, 4]`

`expand(int... newShape)` is the first explicit broadcast-shape operation on `Tensor`.
It is implemented as a true broadcasted alias view with zero strides on expanded axes.
`expandDims(...)` and `squeeze(...)` follow the same view-oriented philosophy: they rewrite shape/stride metadata without forcing dense materialization.
If a dense materialized tensor is required, use `contiguous()` explicitly.

## Reduction Shape Policy

Current reduction surface:

- `sum()`
- `sum(int dimension)`
- `sum(int dimension, boolean keepDims)`
- `mean()`
- `mean(int dimension)`
- `mean(int dimension, boolean keepDims)`
- `min()`
- `min(int dimension)`
- `min(int dimension, boolean keepDims)`
- `max()`
- `max(int dimension)`
- `max(int dimension, boolean keepDims)`

For axis reduction:

- `keepDims=false` removes the reduced axis
- `keepDims=true` preserves the reduced axis with size `1`

`mean(...)` follows the same shape policy as `sum(...)`, but scales the reduced value by the size of the reduced axis (or by the total element count for `mean()`).

`min(...)` and `max(...)` follow the same shape policy as `sum(...)`.
Their backward semantics route gradient only to winning elements; if multiple values tie for the extremum, the gradient is split evenly across the winners.

`all(...)` and `any(...)` are `BOOL`-only reductions.
They follow the same shape policy as other reductions, but they are nondifferentiable.

`clamp(minValue, maxValue)` is a piecewise numeric transform built on top of compare/select semantics.
It keeps values inside the interval and replaces values below/above the interval bounds by the corresponding boundary value.

`clampMin(minValue)` and `clampMax(maxValue)` are the one-sided variants:

- `clampMin` only raises values below the lower bound
- `clampMax` only lowers values above the upper bound

## Gradient and Backward

Autodiff is reverse-mode:

- forward nodes carry a backward lambda
- compilation of a differentiable graph builds explicit backward nodes
- root gradient is seeded with `onesLike(root)`
- `PreparedExecution.execute(FORWARD_BACKWARD)` runs forward and then backward

## Backend Resolution

Tensor-level backend selection is intentionally simple:

1. use `forcedBackend` if explicitly set
2. otherwise default to `CPU`

`Operation` no longer advertises backend preference.
Backend-specific runtime metadata is prepared later during graph preparation, not stored on the operation descriptor.

## Non-Contiguous Layout Handling

`TensorRemap` and backend remap plans support execution across non-contiguous layouts.

Current behavior:

- identical layout copies use fast typed paths
- small non-contiguous tensors can use strided element-wise fallback
- larger non-contiguous inputs can be materialized to temporary contiguous buffers
- reduction and broadcast paths have their own resolved runtime metadata

This is handled in backend planning/execution, not directly in tensor graph construction.

## DType / Storage Notes

Storage is dtype-native:

- `BoolStorage`
- `Float16Storage`
- `Float32Storage`
- `Float64Storage`

Important current behavior:

- `BOOL` is a first-class dtype, not a numeric `0/1` workaround
- implicit `BOOL <-> numeric` dtype conversion is intentionally not supported
- non-`FLOAT64` tensors do not maintain a mirrored `double[]` cache
- `markDataViewStale()` is now effectively a no-op compatibility hook
- `toDoubleArrayCopy()` is the canonical generic readback path
- `toBooleanArrayCopy()` is the canonical logical readback path for `BOOL`
- `copyDataFrom(...)` is the internal typed tensor-to-tensor sync helper used by runtime code

## Related Modules

- Operation descriptors and "how to add a new operation":
  - [src/main/java/operations/README.md](../operations/README.md)
- Graph orchestration:
  - [src/main/java/graph/README.md](../graph/README.md)
- Backend dispatch:
  - [src/main/java/backend/README.md](../backend/README.md)
- Optimizer:
  - [src/main/java/graph/optimizer/README.md](../graph/optimizer/README.md)
- Numerics harness:
  - [src/main/java/numerics/README.md](../numerics/README.md)
