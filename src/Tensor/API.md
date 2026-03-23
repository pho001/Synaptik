# Tensor API Reference

This document summarizes public constructors, factory methods, runtime methods, and tensor operations exposed by `Tensor`.

Primary class:
- [src/Tensor/Tensor.java](src/Tensor/Tensor.java)

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

## Static Factories

- `scalar(double value)`
- `scalar(double value, DataType dataType)`
- `onesLike(Tensor other)`
- `zerosLike(Tensor other)`

## Execution and Autodiff

- `compute()`
- `compute(GraphOptimizer optimizer)`
- `backward()`
- `buildBackwardGraph()`
- `setBackwardFunction(Runnable backwardFunction)`

## Tensor Operations (Graph-Building)

- Layout:
  - `contiguous()`
- Binary:
  - `add(Tensor second)`
  - `sub(Tensor second)`
  - `mul(Tensor second)`
  - `div(Tensor second)`
  - `min(Tensor second)`
  - `max(Tensor second)`
- Unary:
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
- Reduction:
  - `sum(int dimension)`
  - `sum()`
- Execution anchor:
  - `forwardOutput()`

## Metadata and Layout Accessors

- `getShape()`
- `getStrides()`
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
- `getFloat64Data()`
- `getFloat32Data()`
- `getFloat16Data()`
- `toDoubleArrayCopy()`
- `scalarAsDouble()`
- `syncDataToStorage()`
- `syncStorageToData()` (`FLOAT64` only)
- `markDataViewStale()`
- `aliasRuntimeFrom(Tensor source)`

## Graph and Runtime Wiring

- `getOperation()`
- `setOperation(Operation operation)`
- `getPrevTensors()`
- `setPrevTensors(List<Tensor> prevTensors)`
- `topologicalSort()`
- `getCompiledGraph()`
- `resetCompiledGraph()`
- `setGradient(Tensor gradient)`
- `getGradient()`
- `setBackward(boolean backward)`
- `isBackward()`
- `setIntermediates(double[] intermediates)`
- `getIntermediates()`

## Backend and Kernel Resolution

- `setBackend(ComputeBackend backend)`
- `resolveBackend()`
- `getResolvedBackend()`
- `setResolvedBackend(ComputeBackend resolvedBackend)`
- `getResolvedCpuKernel()`
- `setResolvedCpuKernel(CpuKernel resolvedCpuKernel)`

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

