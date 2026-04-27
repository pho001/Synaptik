<!-- generated-by: gsd-doc-writer -->
# Tensor API Guide

Navigation: [Index](index.md) | [Public API](public-api.md) | [Examples](examples.md) | [Compute Flow](compute-flow.md) | [Graph Optimizer](graph-optimizer.md) | [Troubleshooting](troubleshooting.md)

This guide describes the public `tensor.Tensor` API as implemented in the Java source and exercised by the runtime tests. It focuses on how tensor objects are built, how graph operations are represented, what each public operation returns, and the gradient behavior that matters when composing models.

## Table Of Contents

- [API Surface And Conventions](#api-surface-and-conventions)
- [Constructors, Storage, And Dtype](#constructors-storage-and-dtype)
- [Metadata, Data Access, And Mutation](#metadata-data-access-and-mutation)
- [Graph Lifecycle And Execution](#graph-lifecycle-and-execution)
- [Operation Catalog](#operation-catalog)
- [Layout And View Operations](#layout-and-view-operations)
- [Binary Broadcasting And Scalar Arithmetic](#binary-broadcasting-and-scalar-arithmetic)
- [Comparisons, Boolean Logic, And Selection](#comparisons-boolean-logic-and-selection)
- [Indexing, Gather, Scatter, And Take Along Axis](#indexing-gather-scatter-and-take-along-axis)
- [Unary Math](#unary-math)
- [Reductions, Softmax, And LogSoftmax](#reductions-softmax-and-logsoftmax)
- [Matrix, Linear, And Attention Operations](#matrix-linear-and-attention-operations)
- [Convolution And Pooling](#convolution-and-pooling)
- [Normalization](#normalization)
- [Loss Functions](#loss-functions)
- [Dtype, Shape, And Edge-Case Rules](#dtype-shape-and-edge-case-rules)
- [Implementation Source Map](#implementation-source-map)

## API Surface And Conventions

Where it lives in the code:

- `src/main/java/tensor/Tensor.java` is the main user-facing API. It owns constructors, metadata/data accessors, graph lifecycle methods, and instance operation methods.
- `src/main/java/tensor/TensorOps.java` exposes static counterparts for operation families and delegates into `src/main/java/tensor/ops/**`.
- `src/main/java/operations/**` contains operation primitives that compiled graphs execute.
- Runtime behavior is covered by tests under `src/test/java/*Tensor*Test.java`, `src/test/java/*ExecutionTest.java`, `src/test/java/*Broadcast*Test.java`, and `src/test/java/*Gradient*Test.java`.

`Tensor` operation methods are graph-building methods unless they are explicit data/metadata accessors or mutators. For example, `x.add(y)` creates a graph tensor with an `ADD` operation; kernels run when the graph is computed or compiled and executed.

Shapes are `int[]` arrays. An empty shape is normalized to `[1]`, so scalar-like tensors in this API are rank-1 tensors of shape `[1]`. Logical data is row-major. `toDoubleArrayCopy()` reads logical row-major values even for non-contiguous views.

Dtypes are defined by `DataType`: `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, and `BOOL`. Numeric math operations are for floating dtypes. `INT32` is primarily for index tensors, and `BOOL` is for masks and boolean reductions.

## Constructors, Storage, And Dtype

Where it lives in the code:

- Constructors and public methods: `src/main/java/tensor/Tensor.java`
- Data factories: `src/main/java/tensor/TensorDataFactory.java`
- Storage conversion: `src/main/java/tensor/TensorStorageSupport.java`
- Metadata and dtype defaults: `src/main/java/tensor/TensorMetadata.java`, `src/main/java/tensor/DataType.java`

### Signatures

```java
// Leaf/data constructors
new Tensor(Object multiDimArray, List<Tensor> previous, String label)
new Tensor(Object multiDimArray, List<Tensor> previous, String label, DataType dataType)

new Tensor(int[] dimensions, List<Tensor> previous, String label)
new Tensor(int[] dimensions, List<Tensor> previous, String label, DataType dataType)

new Tensor(double[] data, int[] shape, List<Tensor> previous, String label)                      // FLOAT32 default
new Tensor(double[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)
new Tensor(double[] data, int[] shape, int[] strides, List<Tensor> previous, String label)       // FLOAT32 default
new Tensor(double[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType)

new Tensor(float[] data, int[] shape, List<Tensor> previous, String label)                       // FLOAT32
new Tensor(float[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)
new Tensor(float[] data, int[] shape, int[] strides, List<Tensor> previous, String label)        // FLOAT32
new Tensor(float[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType)

new Tensor(short[] data, int[] shape, List<Tensor> previous, String label)                       // FLOAT32 default
new Tensor(short[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)
new Tensor(short[] data, int[] shape, int[] strides, List<Tensor> previous, String label)        // FLOAT32 default
new Tensor(short[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType)

new Tensor(byte[] data, int[] shape, List<Tensor> previous, String label)                        // BOOL
new Tensor(byte[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)
new Tensor(byte[] data, int[] shape, int[] strides, List<Tensor> previous, String label)         // BOOL
new Tensor(byte[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType)

new Tensor(int[] data, int[] shape, List<Tensor> previous, String label)                         // INT32
new Tensor(int[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)
new Tensor(int[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType)

// Manual graph-node constructors used mostly by internals and low-level tests
new Tensor(int[] shape, List<Tensor> previous, Operation operation, String label)
new Tensor(int[] shape, List<Tensor> previous, Operation operation, String label, DataType dataType)
new Tensor(int[] shape, int[] strides, List<Tensor> previous, Operation operation, String label, DataType dataType)
new Tensor(int[] shape, int[] strides, int storageOffset, List<Tensor> previous, Operation operation, String label, DataType dataType)

// Factories
Tensor.scalar(double value)
Tensor.scalar(double value, DataType dataType)
Tensor.onesLike(Tensor other)
Tensor.zerosLike(Tensor other)
```

### Behavior

- `data` is flat logical row-major data. Its length must equal the product of `shape`.
- `shape` and `dimensions` are normalized by `TensorMetadata`; an empty shape becomes `[1]`.
- `strides` and storage offsets are used by view operations and by advanced constructors.
- `previous` supplies graph predecessors for manual graph construction. Ordinary leaves pass `null`.
- Constructors that omit `DataType` mostly use `TensorMetadata.DEFAULT_DATA_TYPE`, currently `FLOAT32`: object, dimension, `double[]`, `float[]`, and `short[]` overloads all default this way. The two intentional exceptions are `byte[]`, which defaults to `BOOL`, and `int[]`, which defaults to `INT32`.
- Passing `dataType = null` normalizes metadata to `FLOAT32`. That is valid for floating/numeric source arrays that can be represented as floating storage, but it is not a compatibility conversion for every storage kind: `byte[]` with `null` becomes a failed BOOL-to-numeric conversion, and `int[]` with `null` becomes a failed INT32-to-floating conversion.
- `Tensor.scalar(value, DataType.INT32)` accepts integral values only and returns shape `[1]`.
- `onesLike` and `zerosLike` preserve shape and preserve `INT32` or floating storage. They do not support `BOOL` inputs because numeric factory conversion to `BOOL` is rejected.

### Example

```java
Tensor f64 = new Tensor(
        new double[]{1.0, 2.0, 3.0, 4.0},
        new int[]{2, 2},
        null,
        "f64",
        DataType.FLOAT64
);
// f64 = [
//   [1.0, 2.0],
//   [3.0, 4.0]
// ]
// f64 shape = [2, 2]
// f64 dtype = FLOAT64

Tensor defaultTyped = Tensor.scalar(1.5);
// defaultTyped = [1.5]
// defaultTyped dtype = FLOAT32

Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", DataType.INT32);
// indices = [2, 0]
// indices.toDoubleArrayCopy() = [2.0, 0.0]

Tensor mask = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "mask", DataType.BOOL);
// mask = [
//   [true, false],
//   [true, false]
// ]
```

## Metadata, Data Access, And Mutation

Where it lives in the code:

- Accessors and mutators: `src/main/java/tensor/Tensor.java`
- Logical copy/debug helpers: `src/main/java/tensor/TensorDebugSupport.java`
- Metadata: `src/main/java/tensor/TensorMetadata.java`
- Remapping/copy behavior: `src/main/java/tensor/TensorRemap.java`

### Common Methods

```java
int[] getShape()
int[] getShapeUnsafe()
int[] getStrides()
int[] getStridesUnsafe()
int getStorageOffsetUnsafe()
int getFlatDataSize()
int getFlatIndex(int[] indices)
int[] getSpatialIndex(int index)
boolean isContiguous()
boolean hasStorageOffset()

DataType getDataType()
void setDataType(DataType dataType)
String getLabel()
void setLabel(String label)
List<Tensor> getPrevTensors()
Operation getOperation()

double getByFlatIndex(int index)
void setDataAt(int flatIndex, double value)
void setData(double[] data)
void setData(float[] data)
void setData(short[] data)
void setData(byte[] data)
void setData(int[] data)
void copyDataFrom(Tensor source)
double[] toDoubleArrayCopy()
boolean[] toBooleanArrayCopy()
double scalarAsDouble()

float[] getFloat32Data()
double[] getFloat64Data()
short[] getBFloat16Data()
int[] getInt32Data()
byte[] getBoolData()

boolean getRequiresGrad()
void setRequiresGrad(boolean requiresGrad)
Tensor getGradient()
```

### Behavior And Edge Cases

- `getShape()` and `getStrides()` return copies. `getShapeUnsafe()` and `getStridesUnsafe()` expose internal arrays for runtime code.
- `getByFlatIndex` and `setDataAt` use logical row-major flat indices.
- `setDataAt` rejects writes through broadcast views.
- `copyDataFrom` requires matching shape and dtype and remaps data according to source and target strides.
- `getData()` is only supported for `FLOAT64`; use typed storage getters or `toDoubleArrayCopy()` for other dtypes.
- `setDataType` allows floating-to-floating conversion, but rejects implicit conversion to/from `INT32` and between `BOOL` and numeric dtypes.

### Example

```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "x", DataType.FLOAT64);
// x = [
//   [1, 2],
//   [3, 4]
// ]

x.setDataAt(2, 30.0);
// x = [
//   [1, 2],
//   [30, 4]
// ]

int[] spatial = x.getSpatialIndex(2);
// spatial = [1, 0]

double[] copy = x.toDoubleArrayCopy();
// copy = [1.0, 2.0, 30.0, 4.0]
```

## Graph Lifecycle And Execution

Where it lives in the code:

- Graph lifecycle entrypoints: `src/main/java/tensor/Tensor.java`
- Execution support: `src/main/java/tensor/TensorExecutionSupport.java`
- Compiled graph runtime: `src/main/java/graph/CompiledGraph.java`, `src/main/java/graph/execution/PreparedExecution.java`

### Methods

```java
List<Tensor> topologicalSort()
PreparedExecution prepare(ExecutionProfile profile)
CompiledGraph compile()
CompiledGraph compile(CompileMode compileMode)
Tensor compute()
Tensor compute(CompileMode compileMode)
Tensor compute(ComputeOptions options)
void compute(ExecutionProfile profile)
void compute(PreparedExecution execution, ExecutionMode mode)
Tensor forwardOutput()
```

### Behavior

`topologicalSort()` walks the graph ending at the current tensor. `compute()` and `compile()` route through `TensorExecutionSupport`; training-capable graphs use `CompileMode` or runtime execution modes that include backward support. Gradients are attached to source tensors through backward functions registered by operation builders.

`forwardOutput()` wraps a tensor in a `noop` operation with label `System_Forward_Output`. It is a graph boundary marker, not a numeric transform.

### Example

```java
Tensor x = new Tensor(new double[]{2, 3}, new int[]{2}, null, "x", DataType.FLOAT64);
// x = [2, 3]

Tensor y = x.mul(x).sum();
// formula: y = sum(x * x)
// y before compute = graph node, not executed data

y.compute();
// y = [13]
```

## Operation Catalog

The sections below document each important public operation as its own subsection or as a named overload group. Instance methods and `TensorOps` static methods delegate to the same implementation unless noted.

Gradient notation:

- `g` means upstream gradient `dL/dout`.
- Broadcasted operands reduce their gradients back to their original shapes with `TensorBroadcastOps.sumToShape`.
- Shape references use logical row-major layout.
- "No gradient" means the operation is built with a no-gradient primitive or has no backward function in the source.

## Layout And View Operations

Implementation: `src/main/java/tensor/ops/layout/TensorLayoutOps.java`, `src/main/java/tensor/ops/layout/LayoutSupport.java`, `src/main/java/tensor/TensorLayoutTransform.java`, `src/main/java/tensor/TensorPrimitiveBuilder.java`, `src/main/java/tensor/TensorRemap.java`, `src/main/java/backend/cpu/kernels/layout/LayoutExecutor.java`, `src/main/java/tensor/ops/index/TensorIndexOps.java`

Mental model: a tensor is not just a flat data array. It is `(storage, shape, strides, storageOffset)`.

- `shape` says how many logical coordinates exist.
- `strides` say how a logical coordinate maps to storage.
- `storageOffset` says where the logical tensor starts inside storage.
- A view operation usually creates a new tensor node with different metadata but the same runtime storage.
- A materialization operation creates dense row-major storage where logical flat index `0, 1, 2, ...` maps directly to adjacent storage cells.
- `reshape` is mixed: contiguous inputs are metadata-only views, while non-contiguous inputs become execution-time layout nodes.
- `contiguous` is the canonical materialization request: it preserves logical values, shape, and dtype, but intentionally does not preserve aliasing.

For a rank-2 tensor, the storage offset for logical coordinate `[row, col]` is:

```text
offset = storageOffset + row * strides[0] + col * strides[1]
```

That formula is the key to understanding `permute`, `expand`, `select`, and `contiguous`.

### `contiguous`

```java
Tensor contiguous()
static Tensor TensorOps.contiguous(Tensor input)
```

What problem this solves: view tensors can be logically correct but physically awkward. A transpose has swapped strides, a broadcast expand has zero strides, and a selected slice may have a non-zero storage offset. Many kernels can execute strided inputs, but dense row-major storage is simpler and often faster for downstream kernels. `contiguous()` is the explicit "materialize this logical tensor into dense storage" operation.

Implementation mechanism:

1. `TensorLayoutOps.contiguous(input)` creates a graph node with operation type `CONTIGUOUS`, the same shape, the same dtype, and dense row-major output metadata.
2. The CPU backend dispatches this node through `CpuContiguousKernel`.
3. `CpuContiguousKernel` calls `LayoutExecutor.contiguous(...)`.
4. `LayoutExecutor.contiguous(...)` calls `TensorRemap.apply(src, dst, contiguousMaterializeThreshold)`.
5. `TensorRemap` walks logical coordinates in row-major order, converts each logical coordinate into the source storage offset using source strides and `storageOffset`, and writes to the dense destination offsets.
6. If source and destination are already raw-copy compatible, `TensorRemap` uses `System.arraycopy`; otherwise it uses sequential or chunked parallel remapping depending on `contiguousMaterializeThreshold`.

Forward: `out[i] = input.logicalValueAt(i)` for every logical row-major index `i`, but `out` is stored densely with standard row-major strides. The result does not alias the source runtime storage after materialization.

Autograd note: unlike `reshape`, `expand`, `permute`, `expandDims`, `squeeze`, and `select`, this builder currently does not attach a Java-level backward function. Treat it primarily as an execution/materialization primitive unless a backend-specific path handles it in a larger compiled graph.

Performance note: `contiguous()` is useful when repeated downstream work benefits from dense storage. It is not free: for non-contiguous inputs it copies every logical element once.

```java
Tensor base = new Tensor(
        new double[]{1, 2, 3, 4, 5, 6},
        new int[]{2, 3},
        null,
        "base",
        DataType.FLOAT64
);
// base = [
//   [1, 2, 3],
//   [4, 5, 6]
// ]
// base shape = [2, 3]
// base strides = [3, 1]

Tensor transposed = base.transpose();
// transposed is a view:
// transposed shape = [3, 2]
// transposed strides = [1, 3]
// transposed logical values = [
//   [1, 4],
//   [2, 5],
//   [3, 6]
// ]

Tensor dense = transposed.contiguous();
// formula: dense[row, col] = transposed[row, col]
// dense logical values = [
//   [1, 4],
//   [2, 5],
//   [3, 6]
// ]
// dense shape = [3, 2]
// dense strides = [2, 1]
// dense flat storage order = [1, 4, 2, 5, 3, 6]
```

### `reshape`

```java
Tensor reshape(int... newShape)
static Tensor TensorOps.reshape(Tensor input, int[] newShape)
```

What problem this solves: the same linear sequence of logical values often needs a different rank or dimension grouping. `reshape` changes the interpretation of the element sequence without changing element count.

Implementation mechanism:

1. `TensorLayoutTransform.inferReshape(oldShape, requestedShape)` validates that element count is unchanged.
2. Exactly one `-1` dimension may be inferred from the old size and the known requested dimensions.
3. If the input is contiguous, `TensorLayoutOps.reshape` creates a `unaryView` with dense strides for the new shape and the same storage offset.
4. If the input is not contiguous, it creates a graph operation node.
5. During CPU execution, `LayoutExecutor.reshapeLike(...)` aliases the runtime source only when that runtime source is contiguous; otherwise it copies the logical row-major sequence with `TensorLayoutTransform.copyLinearized(...)`.

Forward: preserve row-major logical order, then reinterpret the sequence with `newShape`.

Gradient: `dL/dinput = reshape(g, input.shape)`.

```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "x", DataType.FLOAT64);
// x = [
//   [1, 2],
//   [3, 4]
// ]

Tensor y = x.reshape(4);
// formula: y = reshape(x, [4])
// row-major sequence before reshape = [1, 2, 3, 4]
// y = [1, 2, 3, 4]

Tensor z = x.reshape(-1, 1);
// formula: infer -1 as 4 because old element count is 4
// z = [
//   [1],
//   [2],
//   [3],
//   [4]
// ]
// if g has shape [4, 1], dL/dx = reshape(g, [2, 2])
```

### `expand`

```java
Tensor expand(int... newShape)
static Tensor TensorOps.expand(Tensor input, int[] newShape)
```

What problem this solves: many operations need a small tensor, such as a bias vector, to act as if it had a larger shape. `expand` does this without copying by using zero strides on broadcasted axes.

Implementation mechanism:

1. `TensorLayoutTransform.inferExpandShape` validates that every requested dimension is positive and that target rank is not smaller than source rank.
2. Source dimensions align to the right side of target dimensions, matching NumPy-style broadcasting.
3. `LayoutSupport.buildExpandedStrides` keeps the original stride when the dimension already matches.
4. When a source dimension of `1` expands to a larger target dimension, the output stride for that axis becomes `0`.
5. A zero stride means every logical index along that axis reads the same storage cell.

Edge case: writing through a broadcast view is rejected by the mutation API because one logical write could correspond to many aliases of the same physical storage cell.

Forward: `out[expandedCoord] = input[sourceCoord]`, where broadcasted coordinates map back to index `0` in the source dimension.

Gradient: sum the upstream gradient back to the original shape.

```java
Tensor bias = new Tensor(new double[]{10, 20}, new int[]{1, 2}, null, "bias", DataType.FLOAT64);
// bias = [[10, 20]]

Tensor y = bias.expand(3, 2);
// formula: y = broadcast(bias)
// y strides include a zero stride on the expanded row axis
// y = [
//   [10, 20],
//   [10, 20],
//   [10, 20]
// ]
// if g is all ones with shape [3, 2], dL/dbias = [[3, 3]]
```

### `permute`

```java
Tensor permute(int... axes)
static Tensor TensorOps.permute(Tensor input, int[] axes)
```

What problem this solves: many algorithms need to move an axis without copying data, for example turning `[batch, time, channels]` into `[batch, channels, time]`. `permute` changes axis order by reordering shape and stride metadata.

Implementation mechanism:

1. `TensorLayoutTransform.normalizeAxes(rank, axes)` validates that every axis appears exactly once and normalizes negative axes.
2. For each output axis `i`, `outShape[i] = inputShape[axes[i]]`.
3. For each output axis `i`, `outStrides[i] = inputStrides[axes[i]]`.
4. The output aliases the same storage with the same storage offset.

Forward: `out[i0, i1, ...] = input[coordAtAxisPermutation]`.

Gradient: permute `g` by the inverse axis order.

```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
// x = [
//   [1, 2, 3],
//   [4, 5, 6]
// ]
// x shape = [2, 3]
// x strides = [3, 1]

Tensor y = x.permute(1, 0);
// formula: y[col, row] = x[row, col]
// y shape = [3, 2]
// y strides = [1, 3]
// y = [
//   [1, 4],
//   [2, 5],
//   [3, 6]
// ]
// inverse axes = [1, 0], so dL/dx = g.permute(1, 0)
```

### `transpose`

```java
Tensor transpose()
```

Convenience method for rank-2 tensors. Equivalent to `permute(1, 0)` and throws for non-rank-2 tensors.

Gradient: transpose the upstream gradient.

```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
// x = [
//   [1, 2, 3],
//   [4, 5, 6]
// ]

Tensor y = x.transpose();
// formula: y = transpose(x)
// y = [
//   [1, 4],
//   [2, 5],
//   [3, 6]
// ]
```

### `expandDims`

```java
Tensor expandDims(int axis)
static Tensor TensorOps.expandDims(Tensor input, int axis)
```

What problem this solves: reductions often remove dimensions, but later broadcast-aware operations may need that axis back. `expandDims` inserts a size-1 axis without copying.

Implementation mechanism: `TensorLayoutTransform.normalizeInsertAxis` accepts insertion positions from `0` through `rank` and normalizes negative axes against `rank + 1`. Existing shape and stride entries are copied around the inserted axis. `LayoutSupport.insertedAxisStride` chooses a stride that preserves dense-layout reasoning: if inserting before an existing axis, it uses `strides[axis] * shape[axis]`; appending at the end uses stride `1`. The storage offset is preserved.

Forward: each old coordinate maps to the same storage cell with one extra coordinate fixed at `0`.

Gradient: `squeeze(g, axis)`.

```java
Tensor x = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "x", DataType.FLOAT64);
// x = [1, 2, 3]
// x shape = [3]

Tensor y = x.expandDims(0);
// formula: y[0, col] = x[col]
// y shape = [1, 3]
// y = [[1, 2, 3]]
// if g shape is [1, 3], dL/dx = g.squeeze(0)
```

### `squeeze`

```java
Tensor squeeze(int axis)
static Tensor TensorOps.squeeze(Tensor input, int axis)
```

What problem this solves: `squeeze` removes a dimension that carries no choice because its size is `1`. This is commonly used after `gather`, loss reductions, or `expandDims`.

Implementation mechanism: the selected axis is normalized, validated to have size `1`, then removed from both shape and stride metadata. The output aliases the same storage and offset. If the result would otherwise have no dimensions, tensor metadata normalizes the public shape back to `[1]`, so squeezing a single scalar-like `[1]` tensor does not expose a rank-0 public tensor.

Gradient: `expandDims(g, axis)`.

```java
Tensor x = new Tensor(new double[]{1, 2, 3}, new int[]{1, 3}, null, "x", DataType.FLOAT64);
// x = [[1, 2, 3]]
// x shape = [1, 3]

Tensor y = x.squeeze(0);
// formula: y[col] = x[0, col]
// y shape = [3]
// y = [1, 2, 3]
// if g shape is [3], dL/dx = g.expandDims(0)
```

### `select`

```java
Tensor select(int dimension, int index)
static Tensor TensorOps.select(Tensor input, int dimension, int index)
```

What problem this solves: `select` takes one slice out of a tensor without copying. It is a layout operation, not a gather kernel: the selected slice is represented by removing one dimension and shifting the storage offset.

Implementation mechanism:

1. Normalize `dimension`.
2. Normalize `index`, including negative indices.
3. Remove `dimension` from the output shape and strides.
4. Compute `outStorageOffset = input.storageOffset + index * input.strides[dimension]`.
5. Build a view node with the reduced shape, reduced strides, and shifted offset.

Edge cases: negative indices normalize against the selected axis size. Out-of-range indices throw `IndexOutOfBoundsException`. Selecting from a rank-1 tensor returns public shape `[1]`, not rank `0`.

Gradient: create a zero tensor shaped like the input and `scatterAdd` the upstream gradient into the selected slice.

```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
// x = [
//   [1, 2, 3],
//   [4, 5, 6]
// ]

Tensor row = x.select(0, 1);
// formula: row = x[1, :]
// row = [4, 5, 6]
```

## Binary Broadcasting And Scalar Arithmetic

Implementation: `src/main/java/tensor/ops/binary/TensorBinaryOps.java`, `src/main/java/tensor/TensorBroadcastOps.java`, `src/main/java/tensor/TensorPiecewiseOps.java`

All binary floating operations use NumPy-style broadcast planning. Result dtype is promoted by `TensorDataTypeUtil.promote`: `FLOAT64` wins over `FLOAT32`, `FLOAT32` wins over `BFLOAT16`, and `BOOL`/`INT32` are rejected for floating math.

Mechanism: binary operations first compute a broadcast plan, then create a graph node whose output shape is the broadcasted shape. During backward, gradients are first computed in the broadcasted output shape and then reduced with `TensorBroadcastOps.sumToShape(...)` so each input receives a gradient with its original shape.

### `add`

```java
Tensor add(Tensor second)
static Tensor TensorOps.add(Tensor first, Tensor second)
```

Forward: `out = first + second`.

Gradient: `dL/dfirst = sumToShape(g, first.shape)`, `dL/dsecond = sumToShape(g, second.shape)`.

```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "x", DataType.FLOAT64);
// x = [
//   [1, 2],
//   [3, 4]
// ]

Tensor bias = Tensor.scalar(10.0, DataType.FLOAT64);
// bias = 10

Tensor y = x.add(bias);
// formula: y = x + broadcast(bias)
// broadcast(bias) = [
//   [10, 10],
//   [10, 10]
// ]
// y = [
//   [11, 12],
//   [13, 14]
// ]
// if g is all ones, dL/dbias = [4]
```

### `sub`

```java
Tensor sub(Tensor second)
static Tensor TensorOps.sub(Tensor first, Tensor second)
```

Forward: `out = first - second`.

Gradient: `dL/dfirst = sumToShape(g, first.shape)`, `dL/dsecond = sumToShape(-g, second.shape)`.

```java
Tensor x = new Tensor(new double[]{5, 7, 9}, new int[]{3}, null, "x", DataType.FLOAT64);
// x = [5, 7, 9]

Tensor bias = Tensor.scalar(2.0, DataType.FLOAT64);
// bias = 2

Tensor y = x.sub(bias);
// formula: y = x - broadcast(bias)
// broadcast(bias) = [2, 2, 2]
// y = [3, 5, 7]
// if g = [1, 1, 1], dL/dbias = [-3]
```

### `mul`

```java
Tensor mul(Tensor second)
static Tensor TensorOps.mul(Tensor first, Tensor second)
```

Forward: `out = first * second`.

Gradient: `dL/dfirst = sumToShape(g * second, first.shape)`, `dL/dsecond = sumToShape(g * first, second.shape)`.

```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "x", DataType.FLOAT64);
// x = [
//   [1, 2],
//   [3, 4]
// ]

Tensor scale = new Tensor(new double[]{2, 10}, new int[]{1, 2}, null, "scale", DataType.FLOAT64);
// scale = [[2, 10]]

Tensor y = x.mul(scale);
// formula: y = x * broadcast(scale)
// y = [
//   [2, 20],
//   [6, 40]
// ]
// if g is all ones, dL/dscale = [[4, 6]]
```

### `div`

```java
Tensor div(Tensor second)
static Tensor TensorOps.div(Tensor first, Tensor second)
```

Forward: `out = first / second`.

Gradient: `dL/dfirst = sumToShape(g / second, first.shape)`, `dL/dsecond = sumToShape(-g * first / second^2, second.shape)`.

```java
Tensor numerator = new Tensor(new double[]{10, 20, 30, 40}, new int[]{2, 2}, null, "numerator", DataType.FLOAT64);
// numerator = [
//   [10, 20],
//   [30, 40]
// ]

Tensor denominator = new Tensor(new double[]{10, 20}, new int[]{1, 2}, null, "denominator", DataType.FLOAT64);
// denominator = [[10, 20]]

Tensor y = numerator.div(denominator);
// formula: y = numerator / broadcast(denominator)
// broadcast(denominator) = [
//   [10, 20],
//   [10, 20]
// ]
// y = [
//   [1, 1],
//   [3, 2]
// ]
// if g is all ones, dL/ddenominator = [[-0.4, -0.15]]
```

### `min`

```java
Tensor min(Tensor second)
static Tensor TensorOps.min(Tensor first, Tensor second)
```

Forward: elementwise minimum with broadcasting.

Gradient: upstream gradient routes to the operand that supplied the smaller value. If values are tied, binary `min` splits the gradient equally: each side receives `0.5 * g`, then broadcast reduction is applied.

```java
Tensor a = new Tensor(new double[]{1, 5, 5}, new int[]{3}, null, "a", DataType.FLOAT64);
// a = [1, 5, 5]

Tensor b = new Tensor(new double[]{2, 4, 5}, new int[]{3}, null, "b", DataType.FLOAT64);
// b = [2, 4, 5]

Tensor y = a.min(b);
// formula: y[i] = min(a[i], b[i])
// y = [1, 4, 5]
// gradient routing for g = [1, 1, 1]:
// dL/da = [1, 0, 0.5]
// dL/db = [0, 1, 0.5]
```

### `max`

```java
Tensor max(Tensor second)
static Tensor TensorOps.max(Tensor first, Tensor second)
```

Forward: elementwise maximum with broadcasting.

Gradient: upstream gradient routes to the operand that supplied the larger value. If values are tied, binary `max` splits the gradient equally: each side receives `0.5 * g`, then broadcast reduction is applied.

```java
Tensor a = new Tensor(new double[]{1, 5, 5}, new int[]{3}, null, "a", DataType.FLOAT64);
// a = [1, 5, 5]

Tensor b = new Tensor(new double[]{2, 4, 5}, new int[]{3}, null, "b", DataType.FLOAT64);
// b = [2, 4, 5]

Tensor y = a.max(b);
// y = [2, 5, 5]
// gradient routing for g = [1, 1, 1]:
// dL/da = [0, 1, 0.5]
// dL/db = [1, 0, 0.5]
```

### `minimum`

```java
Tensor minimum(Tensor second)
static Tensor TensorOps.minimum(Tensor first, Tensor second)
```

Builds `where(first.lessThan(second), first, second)`. This is a piecewise selection, not the `MIN` primitive.

Gradient: routes through `where`. Ties route to the second branch because the condition is strict `<`.

```java
Tensor a = new Tensor(new double[]{1, 5, 5}, new int[]{3}, null, "a", DataType.FLOAT64);
// a = [1, 5, 5]

Tensor b = new Tensor(new double[]{2, 4, 5}, new int[]{3}, null, "b", DataType.FLOAT64);
// b = [2, 4, 5]

Tensor y = a.minimum(b);
// formula: y = where(a < b, a, b)
// y = [1, 4, 5]
// tie at index 2 uses b because a < b is false
// for g = [1, 1, 1], dL/da = [1, 0, 0] and dL/db = [0, 1, 1]
```

### `maximum`

```java
Tensor maximum(Tensor second)
static Tensor TensorOps.maximum(Tensor first, Tensor second)
```

Builds `where(first.greaterThan(second), first, second)`. This is a piecewise selection, not the `MAX` primitive.

Gradient: routes through `where`. Ties route to the second branch because the condition is strict `>`.

```java
Tensor a = new Tensor(new double[]{1, 5, 5}, new int[]{3}, null, "a", DataType.FLOAT64);
// a = [1, 5, 5]

Tensor b = new Tensor(new double[]{2, 4, 5}, new int[]{3}, null, "b", DataType.FLOAT64);
// b = [2, 4, 5]

Tensor y = a.maximum(b);
// formula: y = where(a > b, a, b)
// y = [2, 5, 5]
// tie at index 2 uses b because a > b is false
// for g = [1, 1, 1], dL/da = [0, 1, 0] and dL/db = [1, 0, 1]
```

### `mul(double)` / `mulScalar`

```java
Tensor mul(double scalar)
static Tensor TensorOps.mulScalar(Tensor input, double scalar)
```

Forward: `out = input * scalar`.

Gradient: `dL/dinput = g * scalar`. For `FLOAT32`, the scalar is rounded to `float` for the operation and gradient constant. Multiplication by `0`, `1`, or `-1` may be simplified to `zerosLike`, identity, or `neg`.

```java
Tensor x = new Tensor(new double[]{1, -2, 3}, new int[]{3}, null, "x", DataType.FLOAT64);
// x = [1, -2, 3]

Tensor y = x.mul(2.5);
// formula: y = x * 2.5
// y = [2.5, -5.0, 7.5]
// if g = [1, 1, 1], dL/dx = [2.5, 2.5, 2.5]
```

## Comparisons, Boolean Logic, And Selection

Implementation: `src/main/java/tensor/ops/compare/TensorCompareOps.java`, `src/main/java/tensor/ops/bool/TensorBoolOps.java`, `src/main/java/tensor/ops/select/TensorSelectOps.java`

Mechanism: comparisons are broadcast-planned elementwise graph ops. They require floating numeric inputs, produce `BOOL` tensors, and are explicitly no-grad nodes. Boolean logic ops are similar, but they require `BOOL` inputs. `where` is different: it uses a three-input broadcast plan and remains differentiable with respect to the numeric branches.

### `greaterThan`

```java
Tensor greaterThan(Tensor second)
static Tensor TensorOps.greaterThan(Tensor first, Tensor second)
```

Returns `BOOL` where `first > second`. Floating numeric inputs only. No gradient.

```java
Tensor x = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "x", DataType.FLOAT64);
// x = [1, 2, 3]

Tensor threshold = Tensor.scalar(2.0, DataType.FLOAT64);
// threshold = 2

Tensor mask = x.greaterThan(threshold);
// formula: mask[i] = x[i] > 2
// broadcast(threshold) = [2, 2, 2]
// mask = [false, false, true]
// mask dtype = BOOL
// mask has no gradient
```

### `lessThan`

```java
Tensor lessThan(Tensor second)
static Tensor TensorOps.lessThan(Tensor first, Tensor second)
```

Returns `BOOL` where `first < second`. Floating numeric inputs only. No gradient.

```java
Tensor x = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "x", DataType.FLOAT64);
// x = [1, 2, 3]

Tensor threshold = Tensor.scalar(2.0, DataType.FLOAT64);
// threshold = 2

Tensor mask = x.lessThan(threshold);
// formula: mask[i] = x[i] < 2
// mask = [true, false, false]
```

### `greaterOrEqual`

```java
Tensor greaterOrEqual(Tensor second)
static Tensor TensorOps.greaterOrEqual(Tensor first, Tensor second)
```

Returns `BOOL` where `first >= second`. Floating numeric inputs only. No gradient.

```java
Tensor x = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "x", DataType.FLOAT64);
// x = [1, 2, 3]

Tensor threshold = Tensor.scalar(2.0, DataType.FLOAT64);
// threshold = 2

Tensor mask = x.greaterOrEqual(threshold);
// formula: mask[i] = x[i] >= 2
// mask = [false, true, true]
```

### `lessOrEqual`

```java
Tensor lessOrEqual(Tensor second)
static Tensor TensorOps.lessOrEqual(Tensor first, Tensor second)
```

Returns `BOOL` where `first <= second`. Floating numeric inputs only. No gradient.

```java
Tensor x = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "x", DataType.FLOAT64);
// x = [1, 2, 3]

Tensor threshold = Tensor.scalar(2.0, DataType.FLOAT64);
// threshold = 2

Tensor mask = x.lessOrEqual(threshold);
// formula: mask[i] = x[i] <= 2
// mask = [true, true, false]
```

### `equalTo`

```java
Tensor equalTo(Tensor second)
static Tensor TensorOps.equalTo(Tensor first, Tensor second)
```

Returns `BOOL` where `first == second`. Floating numeric inputs only. No gradient.

```java
Tensor x = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "x", DataType.FLOAT64);
// x = [1, 2, 3]

Tensor y = new Tensor(new double[]{1, 0, 3}, new int[]{3}, null, "y", DataType.FLOAT64);
// y = [1, 0, 3]

Tensor mask = x.equalTo(y);
// formula: mask[i] = x[i] == y[i]
// mask = [true, false, true]
```

### `notEqualTo`

```java
Tensor notEqualTo(Tensor second)
static Tensor TensorOps.notEqualTo(Tensor first, Tensor second)
```

Returns `BOOL` where `first != second`. Floating numeric inputs only. No gradient.

```java
Tensor x = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "x", DataType.FLOAT64);
// x = [1, 2, 3]

Tensor y = new Tensor(new double[]{1, 0, 3}, new int[]{3}, null, "y", DataType.FLOAT64);
// y = [1, 0, 3]

Tensor mask = x.notEqualTo(y);
// formula: mask[i] = x[i] != y[i]
// mask = [false, true, false]
```

### `logicalAnd`

```java
Tensor logicalAnd(Tensor second)
static Tensor TensorOps.logicalAnd(Tensor first, Tensor second)
```

Broadcasts two `BOOL` tensors and returns logical AND. No gradient.

```java
Tensor a = new Tensor(new byte[]{1, 0, 1}, new int[]{3}, null, "a", DataType.BOOL);
// a = [true, false, true]

Tensor b = new Tensor(new byte[]{1}, new int[]{1}, null, "b", DataType.BOOL);
// b = [true]

Tensor y = a.logicalAnd(b);
// formula: y = a AND broadcast(b)
// broadcast(b) = [true, true, true]
// y = [true, false, true]
```

### `logicalOr`

```java
Tensor logicalOr(Tensor second)
static Tensor TensorOps.logicalOr(Tensor first, Tensor second)
```

Broadcasts two `BOOL` tensors and returns logical OR. No gradient.

```java
Tensor a = new Tensor(new byte[]{1, 0, 0}, new int[]{3}, null, "a", DataType.BOOL);
// a = [true, false, false]

Tensor b = new Tensor(new byte[]{0, 1, 0}, new int[]{3}, null, "b", DataType.BOOL);
// b = [false, true, false]

Tensor y = a.logicalOr(b);
// formula: y[i] = a[i] OR b[i]
// y = [true, true, false]
```

### `logicalNot`

```java
Tensor logicalNot()
static Tensor TensorOps.logicalNot(Tensor input)
```

Returns logical NOT for a `BOOL` tensor. No gradient.

```java
Tensor mask = new Tensor(new byte[]{1, 0, 1}, new int[]{3}, null, "mask", DataType.BOOL);
// mask = [true, false, true]

Tensor inverse = mask.logicalNot();
// formula: inverse[i] = NOT mask[i]
// inverse = [false, true, false]
```

### `where`

```java
static Tensor Tensor.where(Tensor condition, Tensor ifTrue, Tensor ifFalse)
static Tensor TensorOps.where(Tensor condition, Tensor ifTrue, Tensor ifFalse)
```

Broadcasts `condition`, `ifTrue`, and `ifFalse`; returns values from `ifTrue` where `condition` is true and from `ifFalse` otherwise. `condition` must be `BOOL`; branches must be floating numeric.

Implementation mechanism: `WhereBroadcastPlanner.plan(...)` computes one common output shape for all three inputs. The operation node stores the original three tensors; at execution, each output coordinate reads one condition value and then reads either the true branch or false branch at the broadcast-compatible coordinate.

Gradient: no gradient for `condition`. `ifTrue` receives `where(condition, g, 0)` reduced to its original shape. `ifFalse` receives `where(condition, 0, g)` reduced to its original shape.

```java
Tensor condition = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "condition", DataType.BOOL);
// condition = [
//   [true, false],
//   [true, false]
// ]

Tensor a = Tensor.scalar(10.0, DataType.FLOAT64);
// a = 10

Tensor b = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "b", DataType.FLOAT64);
// b = [
//   [1, 2],
//   [3, 4]
// ]

Tensor y = Tensor.where(condition, a, b);
// formula: y = condition ? broadcast(a) : b
// y = [
//   [10, 2],
//   [10, 4]
// ]
// if g is all ones, dL/da = [2] and dL/db = [
//   [0, 1],
//   [0, 1]
// ]
```

## Indexing, Gather, Scatter, And Take Along Axis

Implementation: `src/main/java/tensor/ops/index/TensorIndexOps.java`, `src/main/java/operations/index/*.java`

Index tensors may be `INT32` or floating numeric tensors containing integral values. `BOOL` indices are rejected.

Mental model: `select` is a metadata alias view, while `gather`, `scatterAdd`, and `takeAlongAxis` are index kernels that materialize values based on an index tensor.

- `select` picks one fixed index for the whole slice and can be represented as shape/stride/storage-offset metadata.
- `gather` picks one index per output coordinate after removing the gathered axis.
- `takeAlongAxis` keeps the same rank as the input and lets the index tensor choose positions along one axis.
- `scatterAdd` is the reverse direction: values from `src` are added into a base-shaped output at indexed positions.

### `gather`

```java
Tensor gather(Tensor indices, int dimension)
static Tensor TensorOps.gather(Tensor input, Tensor indices, int dimension)
```

Selects one element along `dimension` for each coordinate in the reduced output shape. `indices.shape` must equal `input.shape` with `dimension` removed.

Implementation mechanism: output coordinates are coordinates in the input with one axis removed. For each output coordinate, the index tensor supplies the missing coordinate on `dimension`.

Gradient: only `input` receives gradients, through a `gatherGrad` operation that scatters upstream values back to gathered positions. `indices` has no gradient.

```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
// x = [
//   [1, 2, 3],
//   [4, 5, 6]
// ]

Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", DataType.INT32);
// indices = [2, 0]

Tensor y = x.gather(indices, 1);
// formula: y[row] = x[row, indices[row]]
// y = [3, 4]
```

### `scatterAdd`

```java
Tensor scatterAdd(Tensor indices, Tensor src, int dimension)
static Tensor TensorOps.scatterAdd(Tensor base, Tensor indices, Tensor src, int dimension)
```

Returns a tensor shaped like `base` with `src` values added at positions selected by `indices` along `dimension`. `base` and `src` must be floating tensors with matching dtype. `indices.shape` and `src.shape` must equal `base.shape` with `dimension` removed.

Implementation mechanism: the output starts from `base`; each coordinate in `src` maps to one coordinate in the output by inserting `indices[srcCoord]` into `dimension`. If multiple source elements target the same output element, their contributions are added.

Gradient: `base` receives `g`. `src` receives `g.gather(indices, dimension)`. `indices` has no gradient.

```java
Tensor base = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "base", DataType.FLOAT64);
// base = [10, 20, 30]

Tensor indices = new Tensor(new int[]{1}, new int[]{1}, null, "indices", DataType.INT32);
// indices = [1]

Tensor src = new Tensor(new double[]{5}, new int[]{1}, null, "src", DataType.FLOAT64);
// src = [5]

Tensor y = base.scatterAdd(indices, src, 0);
// formula: y = base; y[1] += 5
// y = [10, 25, 30]
```

### `takeAlongAxis`

```java
Tensor takeAlongAxis(Tensor indices, int dimension)
static Tensor TensorOps.takeAlongAxis(Tensor input, Tensor indices, int dimension)
```

Selects values from `input` along `dimension` using an index tensor with a shape validated against the input rank. The output shape equals `indices.shape`.

Implementation mechanism: unlike `gather`, `takeAlongAxis` does not remove the selected axis. `indices.rank` must equal `input.rank`, and every non-selected dimension must match the input. The selected dimension may have a different size, allowing the caller to choose a custom number of values per row, column, or higher-rank slice.

Gradient: only `input` receives gradients, through a `takeAlongAxisGrad` operation that scatters upstream values back to selected positions. `indices` has no gradient.

```java
Tensor x = new Tensor(new double[]{
        10, 20, 30,
        40, 50, 60
}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
// x = [
//   [10, 20, 30],
//   [40, 50, 60]
// ]

Tensor indices = new Tensor(new int[]{
        2, 1,
        0, 0
}, new int[]{2, 2}, null, "indices", DataType.INT32);
// indices = [
//   [2, 1],
//   [0, 0]
// ]

Tensor y = x.takeAlongAxis(indices, 1);
// formula: y[row, outCol] = x[row, indices[row, outCol]]
// y = [
//   [30, 20],
//   [40, 40]
// ]
// if g is all ones, dL/dx = [
//   [0, 1, 1],
//   [2, 0, 0]
// ]
```

## Unary Math

Implementation: `src/main/java/tensor/ops/unary/TensorUnaryOps.java`, `src/main/java/operations/elementwise/unary/*.java`

Unary math operations reject `BOOL` and `INT32` unless noted otherwise.

Mechanism: unary math operations create one-input graph nodes with the same shape as the input. Most preserve the input floating dtype through `TensorDataTypeUtil.unary(...)`. Their backward functions compose other tensor operations, so gradients remain part of the graph instead of being computed as detached Java arrays.

### `neg`

```java
Tensor neg()
static Tensor TensorOps.neg(Tensor input)
```

Forward: `out = -input`.

Gradient: `dL/dinput = -g`.

```java
Tensor x = new Tensor(new double[]{-2, 0, 3}, new int[]{3}, null, "x", DataType.FLOAT64);
// x = [-2, 0, 3]

Tensor y = x.neg();
// formula: y = -x
// y = [2, -0, -3]
// if g = [1, 1, 1], dL/dx = [-1, -1, -1]
```

### `abs`

```java
Tensor abs()
static Tensor TensorOps.abs(Tensor input)
```

Forward: absolute value.

Gradient: `g * sign(input)`, with sign `1` for positive values, `-1` for negative values, and `0` at exactly zero.

```java
Tensor x = new Tensor(new double[]{-2, 0, 3}, new int[]{3}, null, "x", DataType.FLOAT64);
// x = [-2, 0, 3]

Tensor y = x.abs();
// formula: y = |x|
// y = [2, 0, 3]
// sign(x) = [-1, 0, 1]
// if g = [1, 1, 1], dL/dx = [-1, 0, 1]
```

### `log`

```java
Tensor log()
static Tensor TensorOps.log(Tensor input)
```

Forward: natural logarithm.

Gradient: `dL/dinput = g / input`.

```java
Tensor x = new Tensor(new double[]{1, 2, 4}, new int[]{3}, null, "x", DataType.FLOAT64);
// x = [1, 2, 4]

Tensor y = x.log();
// formula: y = ln(x)
// y approximately = [0.0, 0.693147, 1.386294]
// if g = [1, 1, 1], dL/dx = [1.0, 0.5, 0.25]
```

### `exp`

```java
Tensor exp()
static Tensor TensorOps.exp(Tensor input)
```

Forward: exponential.

Gradient: `dL/dinput = g * exp(input)`, implemented as `g * out`.

```java
Tensor x = new Tensor(new double[]{0, 1}, new int[]{2}, null, "x", DataType.FLOAT64);
// x = [0, 1]

Tensor y = x.exp();
// formula: y = e^x
// y approximately = [1.0, 2.718282]
// if g = [1, 1], dL/dx approximately = [1.0, 2.718282]
```

### `fastExp`

```java
Tensor fastExp()
static Tensor TensorOps.fastExp(Tensor input)
```

Forward: approximate exponential operation.

Gradient: `dL/dinput = g * out`, so the gradient follows the approximate forward output.

```java
Tensor x = new Tensor(new double[]{0, 1}, new int[]{2}, null, "x", DataType.FLOAT64);
// x = [0, 1]

Tensor y = x.fastExp();
// formula: y = approximateExp(x)
// y is close to [1.0, 2.718282], but uses FastTranscendentals instead of Math.exp
// if g = [1, 1], dL/dx = y, so the gradient follows the approximation
```

### `pow`

```java
Tensor pow(double exponent)
static Tensor TensorOps.pow(Tensor input, double exponent)
```

Forward: `out = input ^ exponent`.

Gradient: `dL/dinput = g * exponent * input^(exponent - 1)`. For `FLOAT32`, the exponent is rounded to `float` for the operation and gradient. Exponents `0`, `1`, `-1`, and `2` are simplified to `onesLike`, identity, `inv`, and `mul(input)` respectively.

```java
Tensor x = new Tensor(new double[]{2, 3}, new int[]{2}, null, "x", DataType.FLOAT64);
// x = [2, 3]

Tensor y = x.pow(3.0);
// formula: y = x^3
// y = [8, 27]
// if g = [1, 1], dL/dx = [12, 27]
```

### `mul(double)`

```java
Tensor mul(double scalar)
static Tensor TensorOps.mulScalar(Tensor input, double scalar)
```

This is the same scalar multiplication overload documented in [Binary Broadcasting And Scalar Arithmetic](#binary-broadcasting-and-scalar-arithmetic). It appears in unary-style code because it has one tensor input and one Java scalar constant.

Forward: `out = input * scalar`.

Gradient: `dL/dinput = g * scalar`.

```java
Tensor x = new Tensor(new double[]{-1, 0, 4}, new int[]{3}, null, "x", DataType.FLOAT64);
// x = [-1, 0, 4]

Tensor y = x.mul(-0.5);
// formula: y = x * -0.5
// y = [0.5, -0.0, -2.0]
// if g = [1, 1, 1], dL/dx = [-0.5, -0.5, -0.5]
```

### `inv`

```java
Tensor inv()
static Tensor TensorOps.inv(Tensor input)
```

Forward: reciprocal, `out = 1 / input`.

Gradient: `dL/dinput = -g * out^2`.

```java
Tensor x = new Tensor(new double[]{2, 4}, new int[]{2}, null, "x", DataType.FLOAT64);
// x = [2, 4]

Tensor y = x.inv();
// formula: y = 1 / x
// y = [0.5, 0.25]
// if g = [1, 1], dL/dx = [-0.25, -0.0625]
```

### `sqrt`

```java
Tensor sqrt()
static Tensor TensorOps.sqrt(Tensor input)
```

Forward: square root.

Gradient: `dL/dinput = g * 0.5 / sqrt(input)`, implemented as `g * 0.5 * out.inv()`.

```java
Tensor x = new Tensor(new double[]{4, 9}, new int[]{2}, null, "x", DataType.FLOAT64);
// x = [4, 9]

Tensor y = x.sqrt();
// formula: y = sqrt(x)
// y = [2, 3]
// if g = [1, 1], dL/dx = [0.25, 0.1666666667]
```

### `sigmoid`

```java
Tensor sigmoid()
static Tensor TensorOps.sigmoid(Tensor input)
```

Forward: logistic sigmoid.

Gradient: `dL/dinput = g * out * (1 - out)`.

```java
Tensor x = new Tensor(new double[]{0, 2}, new int[]{2}, null, "x", DataType.FLOAT64);
// x = [0, 2]

Tensor y = x.sigmoid();
// formula: y = 1 / (1 + exp(-x))
// y approximately = [0.5, 0.880797]
// if g = [1, 1], dL/dx approximately = [0.25, 0.104994]
```

### `tanh`

```java
Tensor tanh()
static Tensor TensorOps.tanh(Tensor input)
```

Forward: hyperbolic tangent.

Gradient: `dL/dinput = g * (1 - out^2)`.

```java
Tensor x = new Tensor(new double[]{0, 1}, new int[]{2}, null, "x", DataType.FLOAT64);
// x = [0, 1]

Tensor y = x.tanh();
// formula: y = tanh(x)
// y approximately = [0.0, 0.761594]
// if g = [1, 1], dL/dx approximately = [1.0, 0.419974]
```

### `fastTanh`

```java
Tensor fastTanh()
static Tensor TensorOps.fastTanh(Tensor input)
```

Forward: approximate hyperbolic tangent.

Gradient: `dL/dinput = g * (1 - out^2)`, so the gradient follows the approximate forward output.

```java
Tensor x = new Tensor(new double[]{0, 1}, new int[]{2}, null, "x", DataType.FLOAT64);
// x = [0, 1]

Tensor y = x.fastTanh();
// formula: y = approximateTanh(x)
// y is close to [0.0, 0.761594], but uses FastTranscendentals
// if g = [1, 1], dL/dx = [1 - y[0]^2, 1 - y[1]^2]
```

### `relu`

```java
Tensor relu()
static Tensor TensorOps.relu(Tensor input)
```

Forward: `max(input, 0)`.

Gradient: `g` where `input > 0`, otherwise `0`. At exactly zero the gradient is `0`.

```java
Tensor x = new Tensor(new double[]{-2, 0, 3}, new int[]{3}, null, "x", DataType.FLOAT64);
// x = [-2, 0, 3]

Tensor y = x.relu();
// formula: y = max(x, 0)
// y = [0, 0, 3]
// if g = [1, 1, 1], dL/dx = [0, 0, 1]
```

### `clamp`

```java
Tensor clamp(double minValue, double maxValue)
static Tensor TensorOps.clamp(Tensor input, double minValue, double maxValue)
```

Composes `clampMax(maxValue).clampMin(minValue)` and requires `minValue <= maxValue`.

Gradient: product of the two clamp masks. Values inside the interval receive `g`; values outside receive `0`. Boundaries are inclusive because `clampMin` uses `>= minValue` and `clampMax` uses `<= maxValue`.

```java
Tensor x = new Tensor(new double[]{-2, -1, 0, 1, 2}, new int[]{5}, null, "x", DataType.FLOAT64);
// x = [-2, -1, 0, 1, 2]

Tensor y = x.clamp(-1.0, 1.0);
// y = [-1, -1, 0, 1, 1]
// if g is all ones, dL/dx = [0, 1, 1, 1, 0]
```

### `clampMin`

```java
Tensor clampMin(double minValue)
static Tensor TensorOps.clampMin(Tensor input, double minValue)
```

Forward: `max(input, minValue)`. `-Infinity` returns the input. Nested `clampMin` operations are collapsed to the stricter minimum.

Gradient: `g` where `input >= minValue`, otherwise `0`.

```java
Tensor x = new Tensor(new double[]{-2, -1, 0, 1}, new int[]{4}, null, "x", DataType.FLOAT64);
// x = [-2, -1, 0, 1]

Tensor y = x.clampMin(-1.0);
// formula: y = max(x, -1)
// y = [-1, -1, 0, 1]
// if g = [1, 1, 1, 1], dL/dx = [0, 1, 1, 1]
```

### `clampMax`

```java
Tensor clampMax(double maxValue)
static Tensor TensorOps.clampMax(Tensor input, double maxValue)
```

Forward: `min(input, maxValue)`. `+Infinity` returns the input. Nested `clampMax` operations are collapsed to the stricter maximum.

Gradient: `g` where `input <= maxValue`, otherwise `0`.

```java
Tensor x = new Tensor(new double[]{-1, 0, 1, 2}, new int[]{4}, null, "x", DataType.FLOAT64);
// x = [-1, 0, 1, 2]

Tensor y = x.clampMax(1.0);
// formula: y = min(x, 1)
// y = [-1, 0, 1, 1]
// if g = [1, 1, 1, 1], dL/dx = [1, 1, 1, 0]
```

## Reductions, Softmax, And LogSoftmax

Implementation: `src/main/java/tensor/ops/reduction/TensorReduceOps.java`, `src/main/java/operations/reduction/*.java`

Floating reductions reject `BOOL` and `INT32`. Boolean reductions require `BOOL` input and do not carry gradients.

Mechanism: a reduction replaces one axis with either nothing (`keepDims=false`) or size `1` (`keepDims=true`). Backward reverses that shape change by inserting the reduced axis when needed and then broadcasting the upstream gradient back to the original input shape.

### `sum`

```java
Tensor sum(int dimension)
Tensor sum(int dimension, boolean keepDims)
Tensor sum()
static Tensor TensorOps.sum(Tensor input, int dimension)
static Tensor TensorOps.sum(Tensor input, int dimension, boolean keepDims)
static Tensor TensorOps.sumAll(Tensor input)
```

Forward: sum along one axis or all elements. `keepDims=false` removes the reduced axis. `sum()` and `sumAll` return shape `[1]`.

Gradient: expand `g` back to the input shape.

```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
// x = [
//   [1, 2, 3],
//   [4, 5, 6]
// ]

Tensor y = x.sum(1);
// formula: y[row] = sum(x[row, :])
// y = [6, 15]
// if g = [1, 1], aligned g = [
//   [1],
//   [1]
// ]
// dL/dx = broadcast(aligned g) = [
//   [1, 1, 1],
//   [1, 1, 1]
// ]
```

### `mean`

```java
Tensor mean(int dimension)
Tensor mean(int dimension, boolean keepDims)
Tensor mean()
static Tensor TensorOps.mean(Tensor input, int dimension)
static Tensor TensorOps.mean(Tensor input, int dimension, boolean keepDims)
static Tensor TensorOps.meanAll(Tensor input)
```

Forward: arithmetic mean along one axis or all elements.

Gradient: expand `g` back to input shape and divide by the number of reduced elements.

```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
// x = [
//   [1, 2, 3],
//   [4, 5, 6]
// ]

Tensor y = x.mean(1);
// formula: y[row] = mean(x[row, :])
// y = [2, 5]
// if g = [1, 1], dL/dx = [
//   [0.3333333333, 0.3333333333, 0.3333333333],
//   [0.3333333333, 0.3333333333, 0.3333333333]
// ]
```

### `min` Reduction

```java
Tensor min(int dimension)
Tensor min(int dimension, boolean keepDims)
Tensor min()
static Tensor TensorOps.min(Tensor input, int dimension)
static Tensor TensorOps.min(Tensor input, int dimension, boolean keepDims)
static Tensor TensorOps.minAll(Tensor input)
```

Forward: minimum along one axis or all elements.

Gradient: routes `g` to all elements equal to the reduced minimum. Ties split the gradient equally across all minima in each reduction group.

```java
Tensor x = new Tensor(new double[]{1, 1, 3, 4, 0, 0}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
// x = [
//   [1, 1, 3],
//   [4, 0, 0]
// ]

Tensor y = x.min(1);
// formula: y[row] = min(x[row, :])
// y = [1, 0]
// row 0 has two minima; row 1 has two minima
// if g = [1, 1], dL/dx = [
//   [0.5, 0.5, 0],
//   [0, 0.5, 0.5]
// ]
```

### `max` Reduction

```java
Tensor max(int dimension)
Tensor max(int dimension, boolean keepDims)
Tensor max()
static Tensor TensorOps.max(Tensor input, int dimension)
static Tensor TensorOps.max(Tensor input, int dimension, boolean keepDims)
static Tensor TensorOps.maxAll(Tensor input)
```

Forward: maximum along one axis or all elements.

Gradient: routes `g` to all elements equal to the reduced maximum. Ties split the gradient equally across all maxima in each reduction group.

```java
Tensor x = new Tensor(new double[]{1, 5, 5, 2}, new int[]{4}, null, "x", DataType.FLOAT64);
// x = [1, 5, 5, 2]

Tensor y = x.max();
// y = [5]
// if g = [1], dL/dx = [0, 0.5, 0.5, 0]
```

### `all`

```java
Tensor all(int dimension)
Tensor all(int dimension, boolean keepDims)
Tensor all()
static Tensor TensorOps.all(Tensor input, int dimension)
static Tensor TensorOps.all(Tensor input, int dimension, boolean keepDims)
static Tensor TensorOps.allAll(Tensor input)
```

Boolean reduction that returns `true` only when all values in the reduction group are true. No gradient.

```java
Tensor mask = new Tensor(new byte[]{1, 1, 0, 1}, new int[]{2, 2}, null, "mask", DataType.BOOL);
// mask = [
//   [true, true],
//   [false, true]
// ]

Tensor y = mask.all(1);
// formula: y[row] = mask[row, 0] AND mask[row, 1]
// y = [true, false]
// y dtype = BOOL
// y has no gradient
```

### `any`

```java
Tensor any(int dimension)
Tensor any(int dimension, boolean keepDims)
Tensor any()
static Tensor TensorOps.any(Tensor input, int dimension)
static Tensor TensorOps.any(Tensor input, int dimension, boolean keepDims)
static Tensor TensorOps.anyAll(Tensor input)
```

Boolean reduction that returns `true` when any value in the reduction group is true. No gradient.

```java
Tensor mask = new Tensor(new byte[]{0, 0, 0, 1}, new int[]{2, 2}, null, "mask", DataType.BOOL);
// mask = [
//   [false, false],
//   [false, true]
// ]

Tensor y = mask.any(1);
// formula: y[row] = mask[row, 0] OR mask[row, 1]
// y = [false, true]
// y dtype = BOOL
// y has no gradient
```

### `softmax`

```java
Tensor softmax(int dimension)
static Tensor TensorOps.softmax(Tensor input, int dimension)
```

Forward: normalized exponentials along `dimension`; output shape equals input shape.

Gradient: for softmax output `s`, `dL/dinput = s * (g - sum(g * s, dimension, keepDims=true))`.

```java
Tensor logits = new Tensor(new double[]{0, 0}, new int[]{2}, null, "logits", DataType.FLOAT64);
// logits = [0, 0]

Tensor probs = logits.softmax(0);
// probs = [0.5, 0.5]
// Jacobian = [
//   [0.25, -0.25],
//   [-0.25, 0.25]
// ]
```

### `logSoftmax`

```java
Tensor logSoftmax(int dimension)
static Tensor TensorOps.logSoftmax(Tensor input, int dimension)
```

Forward: log of softmax along `dimension`; output shape equals input shape.

Gradient: for probabilities `p = exp(out)`, `dL/dinput = g - p * sum(g, dimension, keepDims=true)`.

```java
Tensor logits = new Tensor(new double[]{0, 0}, new int[]{2}, null, "logits", DataType.FLOAT64);
// logits = [0, 0]

Tensor logProbs = logits.logSoftmax(0);
// formula: logProbs = log(softmax(logits))
// softmax(logits) = [0.5, 0.5]
// logProbs approximately = [-0.693147, -0.693147]
// if g = [1, 0], sum(g) = [1]
// dL/dlogits = g - softmax(logits) * sum(g) = [0.5, -0.5]
```

## Matrix, Linear, And Attention Operations

Implementation: `src/main/java/tensor/ops/linalg/*.java`, `src/main/java/operations/linalg/*.java`

### `matmul`

```java
Tensor matmul(Tensor second)
static Tensor TensorOps.matmul(Tensor first, Tensor second)
```

Requires rank at least `2` on both inputs. The last two dimensions are matrix dimensions: `first[..., M, K] @ second[..., K, N] -> out[..., M, N]`. Leading batch dimensions are broadcast.

Gradient: `dL/dfirst = sumToShape(g @ transposeLastTwo(second), first.shape)`, `dL/dsecond = sumToShape(transposeLastTwo(first) @ g, second.shape)`.

```java
Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "a", DataType.FLOAT64);
// a = [
//   [1, 2],
//   [3, 4]
// ]

Tensor b = new Tensor(new double[]{10, 20, 30, 40}, new int[]{2, 2}, null, "b", DataType.FLOAT64);
// b = [
//   [10, 20],
//   [30, 40]
// ]

Tensor y = a.matmul(b);
// formula: y = a @ b
// y = [
//   [70, 100],
//   [150, 220]
// ]
```

### `linear`

```java
Tensor linear(Tensor weight)
Tensor linear(Tensor weight, Tensor bias)
static Tensor TensorOps.linear(Tensor input, Tensor weight)
static Tensor TensorOps.linear(Tensor input, Tensor weight, Tensor bias)
```

Requires `input.rank >= 2`, `weight.shape = [inFeatures, outFeatures]`, and optional `bias.shape = [outFeatures]`. Output shape equals `input.shape` with the last dimension replaced by `outFeatures`.

Gradient: `dL/dinput = g @ weight^T`, `dL/dweight = input^T @ g` summed to `weight.shape`, and `dL/dbias = sumToShape(g, bias.shape)`.

```java
Tensor input = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "input", DataType.FLOAT64);
// input = [
//   [1, 2],
//   [3, 4]
// ]

Tensor weight = new Tensor(new double[]{10, 20, 30, 40}, new int[]{2, 2}, null, "weight", DataType.FLOAT64);
// weight = [
//   [10, 20],
//   [30, 40]
// ]

Tensor bias = new Tensor(new double[]{1, 2}, new int[]{2}, null, "bias", DataType.FLOAT64);
// bias = [1, 2]

Tensor y = input.linear(weight, bias);
// formula: y = input @ weight + bias
// y = [
//   [71, 102],
//   [151, 222]
// ]
```

### `scaledDotProductAttention`

```java
Tensor scaledDotProductAttention(Tensor key, Tensor value, AttentionOptions options)
Tensor scaledDotProductAttention(Tensor key, Tensor value, Tensor mask, AttentionOptions options)
static Tensor TensorOps.scaledDotProductAttention(Tensor query, Tensor key, Tensor value, AttentionOptions options)
static Tensor TensorOps.scaledDotProductAttention(Tensor query, Tensor key, Tensor value, Tensor mask, AttentionOptions options)
```

Requires floating query, key, and value tensors with rank at least `2`. Query/key last dimensions must match. Key/value sequence dimensions must match. Leading batch dimensions are broadcast. A mask must be `BOOL`; causal masks can be requested through `AttentionOptions`.

Forward: compute scores `query @ key^T * scale`, apply mask if present, softmax over the key axis, then multiply by `value`.

Gradient: value receives `weights^T @ g`; query and key receive gradients through the softmax Jacobian and the score matmul. Mask and causal mask tensors have no gradient. Lowered backward operations exist for `FLOAT64`, `FLOAT32`, and `BFLOAT16`.

```java
Tensor query = new Tensor(new double[]{1, 0}, new int[]{1, 2}, null, "query", DataType.FLOAT64);
// query = [[1, 0]]

Tensor key = new Tensor(new double[]{1, 0, 0, 1}, new int[]{2, 2}, null, "key", DataType.FLOAT64);
// key = [
//   [1, 0],
//   [0, 1]
// ]

Tensor value = new Tensor(new double[]{10, 20}, new int[]{2, 1}, null, "value", DataType.FLOAT64);
// value = [
//   [10],
//   [20]
// ]

Tensor y = query.scaledDotProductAttention(key, value, AttentionOptions.defaults().withScale(1.0));
// formula: scores = query @ key^T * 1.0 = [[1, 0]]
// weights = softmax(scores, keyAxis) approximately = [[0.731059, 0.268941]]
// y = weights @ value approximately = [[12.68941]]
```

## Convolution And Pooling

Implementation: `src/main/java/tensor/ops/conv/TensorConvOps.java`, `src/main/java/tensor/ops/pool/TensorPoolOps.java`, `src/main/java/tensor/options/*.java`

### `conv2d`

```java
Tensor conv2d(Tensor weight, Conv2dOptions options)
Tensor conv2d(Tensor weight, Tensor bias, Conv2dOptions options)
static Tensor TensorOps.conv2d(Tensor input, Tensor weight, Conv2dOptions options)
static Tensor TensorOps.conv2d(Tensor input, Tensor weight, Tensor bias, Conv2dOptions options)
```

Requires rank-4 NCHW input and rank-4 weight. Input shape is `[N, inChannels, H, W]`. Weight shape is `[outChannels, inChannels / groups, kernelH, kernelW]`. Optional bias shape is `[outChannels]`.

`Conv2dOptions.defaults()` uses stride `1x1`, padding `0x0`, dilation `1x1`, and `groups=1`.

Gradient: input and weight use dedicated `conv2dBackwardInput` and `conv2dBackwardWeight` primitives. Bias gradient sums `outGrad` over batch, height, and width: `outGrad.sum(0).sum(1).sum(1)`.

```java
Tensor input = new Tensor(new double[]{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
}, new int[]{1, 1, 3, 3}, null, "input", DataType.FLOAT64);
// input[0, 0] = [
//   [1, 2, 3],
//   [4, 5, 6],
//   [7, 8, 9]
// ]

Tensor weight = new Tensor(new double[]{
        1, 1,
        1, 1
}, new int[]{1, 1, 2, 2}, null, "weight", DataType.FLOAT64);
// weight[0, 0] = [
//   [1, 1],
//   [1, 1]
// ]

Tensor y = input.conv2d(weight, Conv2dOptions.defaults());
// formula: each output cell is a 2x2 window sum
// y[0, 0] = [
//   [12, 16],
//   [24, 28]
// ]
```

### `maxPool2d`

```java
Tensor maxPool2d(Pool2dOptions options)
static Tensor TensorOps.maxPool2d(Tensor input, Pool2dOptions options)
```

Requires rank-4 NCHW floating input. `Pool2dOptions.square(k)` creates a `kxk` pool with stride `kxk`, no padding, and `countIncludePad=false`.

Gradient: upstream gradients route to the argmax position recorded for each output window. The CPU direct backend keeps the first maximum found in scan order when a window has ties.

```java
Tensor x = new Tensor(new double[]{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
}, new int[]{1, 1, 4, 4}, null, "x", DataType.FLOAT64);
// x[0, 0] = [
//   [1, 2, 3, 4],
//   [5, 6, 7, 8],
//   [9, 10, 11, 12],
//   [13, 14, 15, 16]
// ]

Tensor y = x.maxPool2d(Pool2dOptions.square(2));
// formula: 2x2 max pool with stride 2
// y[0, 0] = [
//   [6, 8],
//   [14, 16]
// ]
```

### `avgPool2d`

```java
Tensor avgPool2d(Pool2dOptions options)
static Tensor TensorOps.avgPool2d(Tensor input, Pool2dOptions options)
```

Requires rank-4 NCHW floating input. Forward averages each pooling window. If `countIncludePad=false`, padded cells are excluded from the divisor; if `true`, the full kernel area is used.

Gradient: each output gradient is distributed evenly over the contributing input cells, using the same divisor rule as the forward pass.

```java
Tensor x = new Tensor(new double[]{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
}, new int[]{1, 1, 4, 4}, null, "x", DataType.FLOAT64);
// x[0, 0] = [
//   [1, 2, 3, 4],
//   [5, 6, 7, 8],
//   [9, 10, 11, 12],
//   [13, 14, 15, 16]
// ]

Tensor y = x.avgPool2d(Pool2dOptions.square(2));
// formula: 2x2 average pool with stride 2
// y[0, 0] = [
//   [(1 + 2 + 5 + 6) / 4, (3 + 4 + 7 + 8) / 4],
//   [(9 + 10 + 13 + 14) / 4, (11 + 12 + 15 + 16) / 4]
// ]
// y[0, 0] = [
//   [3.5, 5.5],
//   [11.5, 13.5]
// ]
```

## Normalization

Implementation: `src/main/java/tensor/ops/normalization/TensorNormalizationOps.java`, `src/main/java/operations/normalization/*.java`

### `batchNorm`

```java
Tensor batchNorm(Tensor gamma, Tensor beta, int channelDimension, double epsilon)
Tensor batchNorm(Tensor gamma, Tensor beta, Tensor mean, Tensor variance, int channelDimension, double epsilon)
static Tensor TensorOps.batchNorm(Tensor input, Tensor gamma, Tensor beta, int channelDimension, double epsilon)
static Tensor TensorOps.batchNorm(Tensor input, Tensor gamma, Tensor beta, Tensor mean, Tensor variance, int channelDimension, double epsilon)
```

Requires floating input, gamma, beta, and optional mean/variance. Gamma, beta, mean, and variance are channel vectors with length equal to `input.shape[channelDimension]`.

The training-style overload computes mean and variance from the input over all axes except the channel axis. The stats overload uses supplied mean and variance. The output is built from primitive tensor operations:

`(input - mean) / sqrt(variance + epsilon) * gamma + beta`

Gradient: because the output is composed from differentiable primitive operations, gradients follow the sub/add/div/sqrt/mul chain, including broadcast reductions for channel parameters.

```java
Tensor input = new Tensor(new double[]{1, 3}, new int[]{2, 1}, null, "input", DataType.FLOAT64);
// input = [
//   [1],
//   [3]
// ]

Tensor gamma = new Tensor(new double[]{1}, new int[]{1}, null, "gamma", DataType.FLOAT64);
// gamma = [1]

Tensor beta = new Tensor(new double[]{0}, new int[]{1}, null, "beta", DataType.FLOAT64);
// beta = [0]

Tensor y = input.batchNorm(gamma, beta, 1, 1e-5);
// channelDimension = 1, so stats are computed over the batch axis
// mean = [2]
// variance = [1]
// formula: y = (input - mean) / sqrt(variance + 1e-5) * gamma + beta
// y approximately = [
//   [-0.999995],
//   [0.999995]
// ]
```

### `layerNorm`

```java
Tensor layerNorm(Tensor gamma, Tensor beta, double epsilon)
static Tensor TensorOps.layerNorm(Tensor input, Tensor gamma, Tensor beta, double epsilon)
```

Normalizes over the trailing dimensions covered by `gamma.shape`; `beta.shape` must match `gamma.shape`.

Gradient: implementation uses the standard layer-norm formula with `xHat`, `invStd`, reduction over normalized dimensions for input, and reduction over leading dimensions for gamma/beta.

```java
Tensor input = new Tensor(new double[]{1, 3}, new int[]{1, 2}, null, "input", DataType.FLOAT64);
// input = [[1, 3]]

Tensor gamma = new Tensor(new double[]{1, 1}, new int[]{2}, null, "gamma", DataType.FLOAT64);
// gamma = [1, 1]

Tensor beta = new Tensor(new double[]{0, 0}, new int[]{2}, null, "beta", DataType.FLOAT64);
// beta = [0, 0]

Tensor y = input.layerNorm(gamma, beta, 1e-5);
// normalized trailing shape = [2]
// mean = [[2]]
// variance = [[1]]
// formula: y = (input - mean) / sqrt(variance + 1e-5) * gamma + beta
// y approximately = [[-0.999995, 0.999995]]
```

### `rmsNorm`

```java
Tensor rmsNorm(Tensor gamma, double epsilon)
static Tensor TensorOps.rmsNorm(Tensor input, Tensor gamma, double epsilon)
```

Normalizes by root mean square over the trailing dimensions covered by `gamma.shape`.

Gradient: input gradient uses `weighted * invRms - input * mean(weighted * input) * invRms^3`; gamma gradient reduces `outGrad * input * invRms` over leading dimensions.

```java
Tensor input = new Tensor(new double[]{1, 2}, new int[]{1, 2}, null, "input", DataType.FLOAT64);
// input = [[1, 2]]

Tensor gamma = new Tensor(new double[]{1, 1}, new int[]{2}, null, "gamma", DataType.FLOAT64);
// gamma = [1, 1]

Tensor y = input.rmsNorm(gamma, 1e-5);
// meanSquares = [[(1^2 + 2^2) / 2]] = [[2.5]]
// rms = sqrt(2.5 + 1e-5) approximately = [[1.581142]]
// formula: y = input / rms * gamma
// y approximately = [[0.632454, 1.264908]]
```

## Loss Functions

Implementation: `src/main/java/tensor/ops/loss/TensorLossOps.java`, `src/main/java/tensor/ops/loss/LossSupport.java`, `src/main/java/tensor/loss/LossReduction.java`

`LossReduction` values are `MEAN`, `SUM`, and `NONE`. Index target losses use target tensors with shape equal to the input shape with the class dimension removed.

### `nllLoss`

```java
Tensor nllLoss(Tensor targets, int classDimension)
static Tensor TensorOps.nllLoss(Tensor logProbs, Tensor targets, int classDimension)
```

Requires floating `logProbs` and floating one-hot or distribution-like `targets` with the same shape.

Forward: negative mean of `targets * logProbs` over samples outside the class axis.

Gradient: `dL/dlogProbs = -targets * scale`; `dL/dtargets = -logProbs * scale`, where `scale = g / sampleCount`.

```java
Tensor logProbs = new Tensor(new double[]{-0.1, -2.0}, new int[]{2}, null, "logProbs", DataType.FLOAT64);
// logProbs = [-0.1, -2.0]
// interpreted as log-probabilities for two classes

Tensor targets = new Tensor(new double[]{1, 0}, new int[]{2}, null, "targets", DataType.FLOAT64);
// targets = [1, 0]
// one-hot target selects class 0

Tensor loss = logProbs.nllLoss(targets, 0);
// formula: loss = -mean_over_samples(sum_over_classes(targets * logProbs))
// sampleCount outside class axis = 1
// loss = -((1 * -0.1) + (0 * -2.0)) = [0.1]
// if g = [1], dL/dlogProbs = [-1, 0]
```

### `crossEntropyLoss`

```java
Tensor crossEntropyLoss(Tensor targets, int classDimension)
static Tensor TensorOps.crossEntropyLoss(Tensor logits, Tensor targets, int classDimension)
```

Requires floating logits and floating targets with matching shape.

Forward: cross entropy from logits and distribution-like targets.

Gradient: `dL/dlogits = (softmax(logits) - targets) * scale`; `dL/dtargets = -logSoftmax(logits) * scale`.

```java
Tensor logits = new Tensor(new double[]{2, 0}, new int[]{2}, null, "logits", DataType.FLOAT64);
// logits = [2, 0]

Tensor targets = new Tensor(new double[]{1, 0}, new int[]{2}, null, "targets", DataType.FLOAT64);
// targets = [1, 0]

Tensor loss = logits.crossEntropyLoss(targets, 0);
// formula: loss = -sum(targets * logSoftmax(logits))
// softmax(logits) approximately = [0.880797, 0.119203]
// logSoftmax(logits) approximately = [-0.126928, -2.126928]
// loss approximately = [0.126928]
// if g = [1], dL/dlogits approximately = [-0.119203, 0.119203]
```

### `nllLossFromIndices`

```java
Tensor nllLossFromIndices(Tensor targetIndices, int classDimension)
Tensor nllLossFromIndices(Tensor targetIndices, int classDimension, LossReduction reduction)
Tensor nllLossFromIndices(Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction)
Tensor nllLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex)
Tensor nllLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction)
Tensor nllLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction)
```

Forward: gather `logProbs` at target class indices, negate, optionally multiply by class weights, optionally ignore samples, then apply reduction.

Gradient: unweighted forms are built from `gather`, `neg`, `where`, and reduction primitives, so gradients scatter back to selected log-probability positions. Weighted forms multiply selected gradients by gathered class weights. Target indices have no gradient.

```java
Tensor logProbs = new Tensor(new double[]{
        -0.1, -2.0,
        -3.0, -0.2
}, new int[]{2, 2}, null, "logProbs", DataType.FLOAT64);
// logProbs = [
//   [-0.1, -2.0],
//   [-3.0, -0.2]
// ]

Tensor target = new Tensor(new int[]{0, 1}, new int[]{2}, null, "target", DataType.INT32);
// target = [0, 1]

Tensor loss = logProbs.nllLossFromIndices(target, 1);
// formula: perSampleLoss[row] = -logProbs[row, target[row]]
// perSampleLoss = [0.1, 0.2]
// default reduction = MEAN
// loss = [(0.1 + 0.2) / 2] = [0.15]
// if g = [1], dL/dlogProbs = [
//   [-0.5, 0],
//   [0, -0.5]
// ]
```

### `crossEntropyLossFromIndices`

```java
Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension)
Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, LossReduction reduction)
Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction)
Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex)
Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction)
Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction)
```

Unweighted index cross entropy uses a primitive `crossEntropyLossFromIndices` operation with output shape `[1]` for `MEAN`/`SUM` and reduced target shape for `NONE`.

Gradient for logits: `softmax(logits) * sampleScale - oneHot(target) * sampleScale`, with ignored samples scaled to zero. For `MEAN`, `sampleScale` divides by sample count or valid non-ignored count. Weighted overloads lower through `logSoftmax(...).nllLossFromIndices(...)`, so gradients include class-weight scaling through that composition.

```java
Tensor logits = new Tensor(new double[]{2, 0, 0}, new int[]{1, 3}, null, "logits", DataType.FLOAT64);
// logits = [[2, 0, 0]]

Tensor target = new Tensor(new int[]{0}, new int[]{1}, null, "target", DataType.INT32);
// target = [0]

Tensor loss = logits.crossEntropyLossFromIndices(target, 1);
// formula: loss = -logSoftmax(logits)[0, 0]
// softmax(logits) approximately = [[0.786986, 0.106507, 0.106507]]
// loss approximately = [0.239545]
// dL/dlogits approximately = [[-0.213014, 0.106507, 0.106507]]
```

## Dtype, Shape, And Edge-Case Rules

- Floating math accepts `FLOAT64`, `FLOAT32`, and `BFLOAT16`; result dtype is promoted by `TensorDataTypeUtil`.
- Comparisons reject `BOOL` and `INT32` inputs and return `BOOL`.
- Boolean logic and `all`/`any` require `BOOL` input and do not produce gradients.
- Index tensors cannot be `BOOL`; index values must be integral at runtime.
- Broadcasting gradients always reduce back to the original operand shape.
- Binary `min`/`max` split ties equally. Piecewise `minimum`/`maximum` route ties to the second branch. Reduction `min`/`max` split ties across every winner in the reduction group.
- `reshape`, `expand`, `permute`, `expandDims`, `squeeze`, and `select` are view-like where possible and define explicit backward behavior except `contiguous`, which currently has no attached backward lambda.
- `maxPool2d` backward routes to recorded argmax positions; tied windows use the first maximum selected by the forward scan in the CPU direct backend.
- `setDataType` does not permit implicit conversion to or from `INT32`, or between `BOOL` and numeric dtypes.

## Implementation Source Map

| Area | Public methods | Primary implementation | Runtime/tests |
|---|---|---|---|
| Constructors/storage/dtype | constructors, `scalar`, `onesLike`, `zerosLike`, `TensorDataFactory.*` | `src/main/java/tensor/Tensor.java`, `src/main/java/tensor/TensorDataFactory.java`, `src/main/java/tensor/TensorStorageSupport.java` | `src/test/java/TensorConstructorDataTypeTest.java`, `src/test/java/TensorStorageDataTypeTest.java`, `src/test/java/TensorDataFactoryTest.java` |
| Graph lifecycle | `compile`, `prepare`, `compute`, `topologicalSort`, `forwardOutput` | `src/main/java/tensor/TensorExecutionSupport.java`, `src/main/java/graph/CompiledGraph.java`, `src/main/java/graph/execution/PreparedExecution.java` | `src/test/java/TensorComputeConvenienceApiTest.java`, `src/test/java/PreparedExecutionBuildTest.java`, `src/test/java/PreparedExecutionTrainingCapabilityTest.java` |
| Layout/views | `contiguous`, `reshape`, `expand`, `permute`, `transpose`, `expandDims`, `squeeze`, `select` | `src/main/java/tensor/ops/layout/TensorLayoutOps.java`, `src/main/java/tensor/ops/index/TensorIndexOps.java` | `src/test/java/TransformOpsTest.java`, `src/test/java/NonContiguousExecutionTest.java`, `src/test/java/SelectExecutionTest.java` |
| Binary broadcasting | `add`, `sub`, `mul`, `div`, `min`, `max`, `minimum`, `maximum`, `mul(double)` | `src/main/java/tensor/ops/binary/TensorBinaryOps.java`, `src/main/java/tensor/TensorBroadcastOps.java`, `src/main/java/tensor/TensorPiecewiseOps.java` | `src/test/java/BroadcastBinaryOpsTest.java`, `src/test/java/AddBroadcastTest.java`, `src/test/java/BroadcastContractMatrixTest.java`, `src/test/java/CompareSelectExecutionTest.java` |
| Comparisons/logic/select | `greaterThan`, `lessThan`, `greaterOrEqual`, `lessOrEqual`, `equalTo`, `notEqualTo`, `logicalAnd`, `logicalOr`, `logicalNot`, `where` | `src/main/java/tensor/ops/compare/TensorCompareOps.java`, `src/main/java/tensor/ops/bool/TensorBoolOps.java`, `src/main/java/tensor/ops/select/TensorSelectOps.java` | `src/test/java/CompareSelectExecutionTest.java`, `src/test/java/BoolTensorInfrastructureTest.java`, `src/test/java/SelectExecutionTest.java` |
| Indexing | `gather`, `scatterAdd`, `takeAlongAxis` | `src/main/java/tensor/ops/index/TensorIndexOps.java`, `src/main/java/operations/index/*.java` | `src/test/java/GatherExecutionTest.java`, `src/test/java/ScatterAddExecutionTest.java`, `src/test/java/TakeAlongAxisExecutionTest.java` |
| Unary math | `neg`, `abs`, `log`, `exp`, `fastExp`, `pow`, `inv`, `sqrt`, `sigmoid`, `tanh`, `fastTanh`, `relu`, `clamp*` | `src/main/java/tensor/ops/unary/TensorUnaryOps.java`, `src/main/java/operations/elementwise/unary/*.java` | `src/test/java/AbsExecutionTest.java`, `src/test/java/ClampExecutionTest.java`, `src/test/java/FastExpTest.java`, `src/test/java/FastTanhTest.java` |
| Reductions/softmax | `sum`, `mean`, `min`, `max`, `all`, `any`, `softmax`, `logSoftmax` | `src/main/java/tensor/ops/reduction/TensorReduceOps.java`, `src/main/java/operations/reduction/*.java` | `src/test/java/SumExecutionModesTest.java`, `src/test/java/MeanPrimitiveTest.java`, `src/test/java/ReductionBroadcastContractTest.java`, `src/test/java/SoftmaxExecutionTest.java`, `src/test/java/LogSoftmaxExecutionTest.java`, `src/test/java/MinMaxReductionExecutionTest.java` |
| Matrix/linear/attention | `matmul`, `linear`, `scaledDotProductAttention` | `src/main/java/tensor/ops/linalg/*.java`, `src/main/java/operations/linalg/*.java` | `src/test/java/MatMulTest.java`, `src/test/java/LinearExecutionTest.java`, `src/test/java/AttentionExecutionTest.java` |
| Conv/pool | `conv2d`, `maxPool2d`, `avgPool2d` | `src/main/java/tensor/ops/conv/*.java`, `src/main/java/tensor/ops/pool/*.java`, `src/main/java/tensor/options/*.java` | `src/test/java/Conv2dExecutionTest.java`, `src/test/java/Conv2dLoweringRuleTest.java`, `src/test/java/Pool2dExecutionTest.java` |
| Normalization | `batchNorm`, `layerNorm`, `rmsNorm` | `src/main/java/tensor/ops/normalization/*.java`, `src/main/java/operations/normalization/*.java` | `src/test/java/NormalizationExecutionTest.java` |
| Losses | `nllLoss`, `crossEntropyLoss`, `nllLossFromIndices`, `crossEntropyLossFromIndices` | `src/main/java/tensor/ops/loss/*.java`, `src/main/java/operations/loss/*.java`, `src/main/java/tensor/loss/LossReduction.java` | `src/test/java/NllLossExecutionTest.java`, `src/test/java/CrossEntropyLossExecutionTest.java`, `src/test/java/IndexTargetNllLossExecutionTest.java`, `src/test/java/IndexTargetCrossEntropyLossExecutionTest.java`, `src/test/java/IndexLossReductionExecutionTest.java`, `src/test/java/IgnoreIndexLossExecutionTest.java`, `src/test/java/WeightedIndexLossExecutionTest.java` |
