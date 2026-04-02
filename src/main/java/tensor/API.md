# Tensor API Reference

This document summarizes the current public surface of `tensor.Tensor`.

Primary class:
- [src/main/java/tensor/Tensor.java](../tensor/Tensor.java)

## Current API Shape

`Tensor` still exposes two layers of execution API:

- preferred profile/prepared-execution entry points
- legacy optimizer-centric overloads kept as transitional compatibility shims

If you are writing new code, prefer:

- `prepare(ExecutionProfile profile)`
- `compute(ExecutionProfile profile)`
- `compute(PreparedExecution execution, ExecutionMode mode)`

Avoid building new code on:

- `compile(...)`
- `prepare(GraphOptimizer, ...)`
- `compute(GraphOptimizer, ...)`
- `compute()`

These remain in the class today, but are explicitly marked deprecated in code and are scheduled for cleanup.

## Constructors

- `Tensor(Object multiDimArray, List<Tensor> previous, String label)`
- `Tensor(Object multiDimArray, List<Tensor> previous, String label, DataType dataType)`
- `Tensor(int[] dimensions, List<Tensor> previous, String label)`
- `Tensor(int[] dimensions, List<Tensor> previous, String label, DataType dataType)`
- `Tensor(int[] shape, List<Tensor> previous, Operation operation, String label)`
- `Tensor(int[] shape, List<Tensor> previous, Operation operation, String label, DataType dataType)`
- `Tensor(double[] data, int[] shape, List<Tensor> previous, String label)`
- `Tensor(double[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)`
- `Tensor(double[] data, int[] shape, int[] strides, List<Tensor> previous, String label)`
- `Tensor(double[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType)`
- `Tensor(float[] data, int[] shape, List<Tensor> previous, String label)`
- `Tensor(float[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)`
- `Tensor(float[] data, int[] shape, int[] strides, List<Tensor> previous, String label)`
- `Tensor(float[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType)`
- `Tensor(short[] data, int[] shape, List<Tensor> previous, String label)`
- `Tensor(short[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)`
- `Tensor(short[] data, int[] shape, int[] strides, List<Tensor> previous, String label)`
- `Tensor(short[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType)`

## Static Factories

- `scalar(double value)`
- `scalar(double value, DataType dataType)`
- `onesLike(Tensor other)`
- `zerosLike(Tensor other)`

## Preferred Execution API

- `prepare(RuntimeConfig runtimeConfig)`
- `prepare(ExecutionProfile profile)`
- `compute(RuntimeConfig runtimeConfig, ExecutionMode mode)`
- `compute(ExecutionProfile profile)`
- `compute(PreparedExecution execution, ExecutionMode mode)`
- `backward()`

Execution-mode values currently used by the engine:

- `ExecutionMode.FORWARD`
- `ExecutionMode.FORWARD_BACKWARD`

## Legacy Execution / Compile API

These methods still exist today, but are transitional and should not be used as the preferred public API:

- `compute()`
- `compute(GraphOptimizer optimizer)`
- `prepareCompiledGraph(GraphOptimizer optimizer)`
- `compile(GraphOptimizer optimizer)`
- `compile(OptimizerConfig optimizerConfig)`
- `prepare(GraphOptimizer optimizer, RuntimeConfig runtimeConfig)`
- `compute(GraphOptimizer optimizer, RuntimeConfig runtimeConfig, ExecutionMode mode)`
- `compute(GraphOptimizer optimizer, RuntimeConfig runtimeConfig)`
- `getCompiledGraph()`
- `resetCompiledGraph()`

## Tensor Operations (Graph Building)

### Layout

- `contiguous()`
- `reshape(int... newShape)`
- `expand(int... newShape)`
- `permute(int... axes)`
- `transpose()`
- `expandDims(int axis)`
- `squeeze(int axis)`

### Binary

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

### Execution Anchor / Autodiff Helpers

- `forwardOutput()`
- `buildBackwardGraph()`
- `setBackwardFunction(Runnable backwardFunction)`

## Complete Current Public Tensor Operation Catalog

These are the instance-level graph-building operations currently exposed directly on `Tensor`.

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

## Descriptor Set vs Public Tensor Surface

The `operations/` package contains more operation descriptors than are currently exposed as direct public `Tensor` instance methods.

Examples:

- `relu`
- `noop`
- `batchNorm` wiring exists in `TensorOps`, but there is no direct public `Tensor.batchNorm(...)` method today

So there are three separate questions:

1. does an operation descriptor exist?
2. does a backend kernel exist?
3. is the operation exposed as a public `Tensor` method?

This document treats the third point as the source of truth for the user-facing `Tensor` operation surface.

## Metadata and Layout Accessors

- `getShape()`
- `getShapeUnsafe()`
- `getStrides()`
- `getStridesUnsafe()`
- `getStride(int index)`
- `getDimensionAt(int index)`
- `getFlatDataSize()`
- `isContiguous()`
- `getFlatIndex(int[] indices)`
- `getSpatialIndex(int index)`
- `computeStrides(int[] shape)`
- `computeStrides()`
- `calculateSize(int[] dimensions)`

## Data Access and Conversion

- `getByFlatIndex(int index)`
- `setDataAt(int flatIndex, double value)`
- `getData()` (`FLOAT64` only)
- `setData(double[] data)`
- `setData(float[] data)`
- `setData(short[] data)`
- `setFloat32Data(float[] data)` (`FLOAT32` only)
- `getFloat64Data()`
- `getFloat32Data()`
- `getFloat16Data()`
- `toDoubleArrayCopy()`
- `scalarAsDouble()`
- `markDataViewStale()`
- `aliasRuntimeFrom(Tensor source)`

## Graph and Runtime Wiring

- `getOperation()`
- `setOperation(Operation operation)`
- `getPrevTensors()`
- `setPrevTensors(List<Tensor> prevTensors)`
- `topologicalSort()`
- `setGradient(Tensor gradient)`
- `getGradient()`
- `setBackward(boolean backward)`
- `isBackward()`
- `setIntermediates(double[] intermediates)`
- `getIntermediates()`

## Backend Selection

- `setBackend(ComputeBackend backend)`
- `resolveBackend()`

Current fallback behavior:

- explicit `forcedBackend` if set
- otherwise `CPU`

`Operation` no longer participates in backend preference resolution.

## Label / Grad / Type / Storage

- `getLabel()`
- `setLabel(String label)`
- `getRequiresGrad()`
- `setRequiresGrad(boolean requiresGrad)`
- `getDataType()`
- `setDataType(DataType dataType)`
- `getStorage()`

## Debug Formatting

- `toStructString()`

## Reduction Semantics

- `sum()`
  - reduces all elements to one scalar-shaped tensor
  - current output shape is `[1]`
- `sum(int dimension)`
  - reduces one axis
  - removes the selected axis from output shape
  - `dimension` is zero-based
- `sum(int dimension, boolean keepDims)`
  - when `keepDims=false`, removes the selected axis from output shape
  - when `keepDims=true`, keeps the selected axis with size `1`
  - `dimension` is zero-based
- `mean()`
  - reduces all elements to shape `[1]`
  - value is `sum / numberOfElements`
- `mean(int dimension)`
  - reduces one axis and removes that axis from output shape
  - value is `sumAlongAxis / axisSize`
- `mean(int dimension, boolean keepDims)`
  - same shape policy as `sum(int dimension, boolean keepDims)`
  - values are scaled by the reduced axis size
- `min()`
  - reduces all elements to shape `[1]`
  - value is the minimum element
- `min(int dimension)`
  - reduces one axis and removes that axis from output shape
  - value is the minimum along the selected axis
- `min(int dimension, boolean keepDims)`
  - same shape policy as `sum(int dimension, boolean keepDims)`
- `max()`
  - reduces all elements to shape `[1]`
  - value is the maximum element
- `max(int dimension)`
  - reduces one axis and removes that axis from output shape
  - value is the maximum along the selected axis
- `max(int dimension, boolean keepDims)`
  - same shape policy as `sum(int dimension, boolean keepDims)`
- reduction gradient semantics for `min/max`
  - gradient flows only to winning elements
  - if multiple elements tie for the extremum, the gradient is split evenly across all winners

## Broadcasting Contract

Binary broadcast-aware tensor operations (`add`, `sub`, `mul`, `div`, `min`, `max`) follow this contract:

- ranks are aligned from the right
- missing leading dimensions are treated as `1`
- two dimensions are compatible if:
  - they are equal
  - or one of them is `1`
- output dimension size is `max(leftDim, rightDim)`

Examples:

- `[2, 3, 4]` + `[3, 4]` -> `[2, 3, 4]`
- `[2, 3, 4]` + `[4]` -> `[2, 3, 4]`
- `[2, 1, 4]` + `[3, 4]` -> `[2, 3, 4]`

Incompatible example:

- `[2, 3, 4]` + `[2, 4]`

Backward contract:

- if an operand was broadcast in forward execution, its gradient is reduced back to the original operand shape
- reduction is applied across all axes that were introduced or expanded by broadcasting

Additional examples:

- `[3, 4]` + `[2, 1, 4]` -> `[2, 3, 4]`
- `[1, 1, 2, 4]` + `[2, 3, 1, 4]` -> `[2, 3, 2, 4]`

## Explicit Expand Operation

`expand(int... newShape)` is a first-class shape/broadcast ergonomics op.

Current contract:

- output rank may be equal to or greater than input rank
- rank alignment follows the same right-aligned broadcasting rules as binary broadcast ops
- a dimension may be expanded only if:
  - source dimension already matches target dimension
  - or source dimension is `1`
- expanding a non-singleton dimension to a different size is illegal

Current implementation note:

- `expand(...)` creates a zero-stride broadcast alias view
- expanded axes are logical read views over shared backing storage
- `contiguous()` is the canonical explicit materialization path
- backward reduces gradients back to the original input shape via broadcast reduction semantics
