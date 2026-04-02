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

- tensor storage (`FLOAT16`, `FLOAT32`, `FLOAT64`)
- metadata (`shape`, `strides`, `label`, `requiresGrad`)
- producing operation (`Operation`) or `null` for leaves/constants
- graph input links (`prevTensors`)
- gradient tensor reference (`gradient`)
- backward lambda (`backwardFunction`)
- optional forced backend (`forcedBackend`)

There is also a transitional compile/execution cache in `Tensor` today:

- `compiledGraph`
- `lastPreparedExecution`

This still exists in code for compatibility, but the intended architectural direction is to treat:

- `CompiledGraph` as the explicit compile artifact
- `PreparedExecution` as the explicit runtime artifact

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

Convenience overloads using only `RuntimeConfig` still exist and derive optimizer defaults from the graph shape / gradient requirements:

- `prepare(RuntimeConfig runtimeConfig)`
- `compute(RuntimeConfig runtimeConfig, ExecutionMode mode)`

Legacy optimizer-centric methods are still present, but deprecated:

- `compile(...)`
- `prepare(GraphOptimizer, ...)`
- `compute(GraphOptimizer, ...)`
- `compute()`

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

### Unary / Scalar

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

### Helper / Internal Execution Anchor

- `forwardOutput()`

Not every operation descriptor in `src/main/java/operations/` is exposed as a direct public `Tensor` method.
For example, some descriptors exist for backend/optimizer reasons even when there is no matching public `Tensor` convenience method yet.

The full public API reference remains in:

- [src/main/java/tensor/API.md](../tensor/API.md)

## Broadcasting Contract

Broadcast-aware binary operations in the current tensor surface are:

- `add`
- `sub`
- `mul`
- `div`
- `min`
- `max`

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

## Gradient and Backward

Autodiff is reverse-mode:

- forward nodes carry a backward lambda
- compilation of a differentiable graph builds explicit backward nodes
- root gradient is seeded with `onesLike(root)`
- `PreparedExecution.execute(FORWARD_BACKWARD)` runs forward and then backward

The explicit form is:

- compile graph
- prepare execution
- call `PreparedExecution.backward()`

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

- `Float16Storage`
- `Float32Storage`
- `Float64Storage`

Important current behavior:

- non-`FLOAT64` tensors do not maintain a mirrored `double[]` cache
- `markDataViewStale()` is now effectively a no-op compatibility hook
- `toDoubleArrayCopy()` is the canonical generic readback path

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
