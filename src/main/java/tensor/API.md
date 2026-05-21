# Tensor API Reference

This document describes the current public surface of `tensor.Tensor`.

Primary class:
- [src/main/java/tensor/Tensor.java](../tensor/Tensor.java)

Related public semantic types used by this API live in:

- [src/main/java/tensor/options/AttentionOptions.java](../tensor/options/AttentionOptions.java)
- [src/main/java/tensor/options/Conv2dOptions.java](../tensor/options/Conv2dOptions.java)
- [src/main/java/tensor/options/Pool2dOptions.java](../tensor/options/Pool2dOptions.java)
- [src/main/java/tensor/loss/LossReduction.java](../tensor/loss/LossReduction.java)

The focus here is practical API usage:
- what each public operation does
- what parameters it accepts
- what it returns
- short commented examples

Where it helps readability, examples use explicit shapes and values instead of abstract tensors like `A` and `B`.

Important scope note:

- the operation sections in this document describe the preferred modeling surface
- the later metadata/storage/graph-wiring sections also include low-level and runtime-oriented methods that still exist on `Tensor`
- those low-level methods are real and supported, but they are not the recommended starting point for adding new user-facing tensor operations
- low-level storage allocation/conversion and graph traversal internals are increasingly being pushed into package-private helpers such as:
  - [src/main/java/tensor/storage/TensorStorageSupport.java](../tensor/storage/TensorStorageSupport.java)
  - [src/main/java/tensor/internal/TensorGraphTraversal.java](../tensor/internal/TensorGraphTraversal.java)
  - [src/main/java/tensor/TensorDebugSupport.java](../tensor/TensorDebugSupport.java)
  - [src/main/java/tensor/internal/TensorExecutionSupport.java](../tensor/internal/TensorExecutionSupport.java)

Related construction helpers:

- [src/main/java/tensor/factory/TensorDataFactory.java](../tensor/factory/TensorDataFactory.java)
  - leaf constants and convenience tensor factories
- [src/main/java/tensor/internal/TensorPrimitiveBuilder.java](../tensor/internal/TensorPrimitiveBuilder.java)
  - shared primitive, no-grad, and alias/view node construction

## Reading Guide

This reference intentionally mixes two different kinds of public methods:

- modeling surface
  - the methods you normally use to build graphs
  - mostly the operation sections below
- low-level surface
  - methods used by runtime, graph compilation, rewrites, and tests
  - mostly the later metadata/storage/graph-wiring sections

When adding new tensor semantics:

- start in `tensor.ops.*`
- add or update `Operation` only when the surface is truly primitive-backed
- treat the low-level `Tensor` methods as infrastructure surface, not as the place where new modeling semantics should live

## Contents

- [Conventions](#conventions)
- [Construction Surface](#construction-surface)
  - [Recommended construction paths](#recommended-construction-paths)
  - [Low-level public constructors](#low-level-public-constructors)
- [Static Facade](#static-facade)
- [Execution API](#execution-api)
  - [`compile()`](#compile)
  - [`compile(CompileMode compileMode)`](#compilecompilemode-compilemode)
  - [`prepare(ExecutionProfile profile)`](#prepareexecutionprofile-profile)
  - [`compute()`](#compute)
  - [`compute(CompileMode compileMode)`](#computecompilemode-compilemode)
  - [`compute(ComputeOptions options)`](#computecomputeoptions-options)
  - [`compute(ExecutionProfile profile)`](#computeexecutionprofile-profile)
  - [`compute(PreparedExecution execution, ExecutionMode mode)`](#computepreparedexecution-execution-executionmode-mode)
- [Static Factories](#static-factories)
  - [`scalar(double value)` / `scalar(double value, DataType dataType)`](#scalardouble-value)
  - [`zeros(...)` / `ones(...)`](#zeros--ones)
  - [`randn(...)`](#randn)
  - [`arange(...)`](#arange)
  - [`onesLike(Tensor other)`](#onesliketensor-other)
  - [`zerosLike(Tensor other)`](#zerosliketensor-other)
- [Broadcasting Contract](#broadcasting-contract)
- [View Semantics](#view-semantics)
- [Contiguous and Materialization Contract](#contiguous-and-materialization-contract)
- [Layout / Shape Operations](#layout--shape-operations)
  - [`contiguous()`](#contiguous)
  - [`reshape(int... newShape)`](#reshapeint-newshape)
  - [`expand(int... newShape)`](#expandint-newshape)
  - [`permute(int... axes)`](#permuteint-axes)
  - [`transpose()`](#transpose)
  - [`expandDims(int axis)`](#expanddimsint-axis)
  - [`squeeze(int axis)`](#squeezeint-axis)
  - [`sliceAxis(int axis, int fromInclusive, int toExclusive)`](#sliceaxisint-axis-int-frominclusive-int-toexclusive)
  - [`stack(int axis, Tensor... inputs)` / `unstack(int axis)`](#stackint-axis-tensor-inputs--unstackint-axis)
- [Binary Arithmetic Operations](#binary-arithmetic-operations)
  - [`add(Tensor second)`](#addtensor-second)
  - [`sub(Tensor second)`](#subtensor-second)
  - [`mul(Tensor second)`](#multensor-second)
  - [`div(Tensor second)`](#divtensor-second)
  - [`min(Tensor second)`](#mintensor-second)
  - [`max(Tensor second)`](#maxtensor-second)
  - [`matmul(Tensor second)`](#matmultensor-second)
  - [`linear(Tensor weight)`](#lineartensor-weight)
  - [`linear(Tensor weight, Tensor bias)`](#lineartensor-weight-tensor-bias)
- [Attention Operations](#attention-operations)
  - [`scaledDotProductAttention(Tensor key, Tensor value, AttentionOptions options)`](#scaleddotproductattentiontensor-key-tensor-value-attentionoptions-options)
  - [`scaledDotProductAttention(Tensor key, Tensor value, Tensor mask, AttentionOptions options)`](#scaleddotproductattentiontensor-key-tensor-value-tensor-mask-attentionoptions-options)
- [Spatial Operations](#spatial-operations)
  - [`conv2d(Tensor weight, Conv2dOptions options)`](#conv2dtensor-weight-conv2doptions-options)
  - [`conv2d(Tensor weight, Tensor bias, Conv2dOptions options)`](#conv2dtensor-weight-tensor-bias-conv2doptions-options)
  - [`maxPool2d(Pool2dOptions options)`](#maxpool2dpool2doptions-options)
  - [`avgPool2d(Pool2dOptions options)`](#avgpool2dpool2doptions-options)
- [Normalization Operations](#normalization-operations)
  - [`batchNorm(Tensor gamma, Tensor beta, int channelDimension, double epsilon)`](#batchnormtensor-gamma-tensor-beta-int-channeldimension-double-epsilon)
  - [`batchNorm(Tensor gamma, Tensor beta, Tensor mean, Tensor variance, int channelDimension, double epsilon)`](#batchnormtensor-gamma-tensor-beta-tensor-mean-tensor-variance-int-channeldimension-double-epsilon)
  - [`layerNorm(Tensor gamma, Tensor beta, double epsilon)`](#layernormtensor-gamma-tensor-beta-double-epsilon)
  - [`rmsNorm(Tensor gamma, double epsilon)`](#rmsnormtensor-gamma-double-epsilon)
- [Comparison Operations](#comparison-operations)
  - [`greaterThan(Tensor second)`](#greaterthantensor-second)
  - [`greaterOrEqual(Tensor second)`](#greaterorequaltensor-second)
  - [`lessThan(Tensor second)`](#lessthantensor-second)
  - [`lessOrEqual(Tensor second)`](#lessorequaltensor-second)
  - [`equalTo(Tensor second)`](#equaltotensor-second)
  - [`notEqualTo(Tensor second)`](#notequaltotensor-second)
- [Select Operation](#select-operation)
  - [`where(Tensor condition, Tensor ifTrue, Tensor ifFalse)`](#wheretensor-condition-tensor-iftrue-tensor-iffalse-static)
  - [`minimum(Tensor second)`](#minimumtensor-second)
  - [`maximum(Tensor second)`](#maximumtensor-second)
- [Indexing Operations](#indexing-operations)
  - [`select(int dimension, int index)`](#selectint-dimension-int-index)
  - [`gather(Tensor indices, int dimension)`](#gathertensor-indices-int-dimension)
  - [`gatherAxis(Tensor indices, int axis)` / `take(int axis, ...)`](#gatheraxistensor-indices-int-axis--takeint-axis-)
  - [`gatherNd(Tensor indices[, int batchDims])`](#gatherndtensor-indices-int-batchdims)
  - [`takeAlongAxis(Tensor indices, int dimension)`](#takealongaxistensor-indices-int-dimension)
  - [`scatterAdd(Tensor indices, Tensor src, int dimension)`](#scatteraddtensor-indices-tensor-src-int-dimension)
  - [`scatterElements(Tensor indices, Tensor updates, int axis[, ScatterReduction reduction])`](#scatterelementstensor-indices-tensor-updates-int-axis-scatterreduction-reduction)
  - [`scatterNd(Tensor indices, Tensor updates[, ScatterReduction reduction])`](#scatterndtensor-indices-tensor-updates-scatterreduction-reduction)
- [Logical Bool Operations](#logical-bool-operations)
  - [`logicalAnd(Tensor second)`](#logicalandtensor-second)
  - [`logicalOr(Tensor second)`](#logicalortensor-second)
  - [`logicalNot()`](#logicalnot)
- [Unary / Scalar Operations](#unary--scalar-operations)
  - [`relu()`](#relu)
  - [`abs()`](#abs)
  - [`neg()`](#neg)
  - [`log()`](#log)
  - [`exp()`](#exp)
  - [`fastExp()`](#fastexp)
  - [`tanh()`](#tanh)
  - [`fastTanh()`](#fasttanh)
  - [`pow(double exponent)`](#powdouble-exponent)
  - [`mul(double scalar)`](#muldouble-scalar)
  - [`inv()`](#inv)
  - [`sqrt()`](#sqrt)
  - [`sigmoid()`](#sigmoid)
  - [`clamp(double minValue, double maxValue)`](#clampdouble-minvalue-double-maxvalue)
  - [`clampMin(double minValue)`](#clampmindouble-minvalue)
  - [`clampMax(double maxValue)`](#clampmaxdouble-maxvalue)
- [Reduction Operations](#reduction-operations)
  - [`sum()`](#sum)
  - [`sum(int dimension)`](#sumint-dimension)
  - [`sum(int dimension, boolean keepDims)`](#sumint-dimension-boolean-keepdims)
  - [`mean()`](#mean)
  - [`mean(int dimension)`](#meanint-dimension)
  - [`mean(int dimension, boolean keepDims)`](#meanint-dimension-boolean-keepdims)
  - [`softmax(int dimension)`](#softmaxint-dimension)
  - [`logSoftmax(int dimension)`](#logsoftmaxint-dimension)
  - [`min()`](#min)
  - [`min(int dimension)`](#minint-dimension)
  - [`min(int dimension, boolean keepDims)`](#minint-dimension-boolean-keepdims)
  - [`max()`](#max)
  - [`max(int dimension)`](#maxint-dimension)
  - [`max(int dimension, boolean keepDims)`](#maxint-dimension-boolean-keepdims)
  - [`all()`](#all)
  - [`all(int dimension)`](#allint-dimension)
  - [`all(int dimension, boolean keepDims)`](#allint-dimension-boolean-keepdims)
  - [`any()`](#any)
  - [`any(int dimension)`](#anyint-dimension)
  - [`any(int dimension, boolean keepDims)`](#anyint-dimension-boolean-keepdims)
- [Loss / N-ary Operations](#loss--n-ary-operations)
  - [`nllLoss(Tensor targets, int classDimension)`](#nlllosstensor-targets-int-classdimension)
  - [`crossEntropyLoss(Tensor targets, int classDimension)`](#crossentropylosstensor-targets-int-classdimension)
  - [`nllLossFromIndices(Tensor targetIndices, int classDimension)`](#nlllossfromindicestensor-targetindices-int-classdimension)
  - [`nllLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex)`](#nlllossfromindicestensor-targetindices-int-classdimension-int-ignoreindex)
  - [`nllLossFromIndices(Tensor targetIndices, int classDimension, LossReduction reduction)`](#nlllossfromindicestensor-targetindices-int-classdimension-lossreduction-reduction)
  - [`nllLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction)`](#nlllossfromindicestensor-targetindices-int-classdimension-int-ignoreindex-lossreduction-reduction)
  - [`nllLossFromIndices(Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction)`](#nlllossfromindicestensor-targetindices-int-classdimension-tensor-classweights-lossreduction-reduction)
  - [`nllLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction)`](#nlllossfromindicestensor-targetindices-int-classdimension-int-ignoreindex-tensor-classweights-lossreduction-reduction)
  - [`crossEntropyLossFromIndices(Tensor targetIndices, int classDimension)`](#crossentropylossfromindicestensor-targetindices-int-classdimension)
  - [`crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex)`](#crossentropylossfromindicestensor-targetindices-int-classdimension-int-ignoreindex)
  - [`crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, LossReduction reduction)`](#crossentropylossfromindicestensor-targetindices-int-classdimension-lossreduction-reduction)
  - [`crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction)`](#crossentropylossfromindicestensor-targetindices-int-classdimension-int-ignoreindex-lossreduction-reduction)
  - [`crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction)`](#crossentropylossfromindicestensor-targetindices-int-classdimension-tensor-classweights-lossreduction-reduction)
  - [`crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction)`](#crossentropylossfromindicestensor-targetindices-int-classdimension-int-ignoreindex-tensor-classweights-lossreduction-reduction)
- [Execution Anchor / Autodiff Helpers](#execution-anchor--autodiff-helpers)
  - [`forwardOutput()`](#forwardoutput)
- [Metadata and Data Access](#metadata-and-data-access)
  - [Shape / layout accessors](#shape--layout)
  - [`rank()`](#rank)
  - [`size()`](#size)
  - [`lastDim()`](#lastdim)
  - [`shapeEquals(int... shape)`](#shapeequalsint-shape)
  - [`shapeCopy()`](#shapecopy)
  - [Data / dtype / storage accessors](#data--dtype--storage)
  - [Labels / grad / graph wiring](#labels--grad--graph-wiring)
- [Debug Formatting](#debug-formatting)
  - [`toStructString()`](#tostructstring)

## Conventions

- Axis indices are zero-based.
- Shapes are written as arrays such as `[2, 3, 4]`.
- `BOOL` tensors are logical tensors.
- `INT32` and `INT64` are supported integer dtypes for indexing-style tensors such as gather/scatter targets. `INT64` exists primarily for ONNX-compatible index and shape-like values.
- Comparison ops and logical bool ops are nondifferentiable.
- `where(condition, x, y)` is differentiable only in the data branches.
- `all` / `any` are `BOOL`-only reductions and are nondifferentiable.

## Construction Surface

The public `Tensor` surface exposes more constructors than ordinary modeling code usually needs.

That is intentional:

- user-facing code needs ergonomic leaf tensor creation
- runtime/rewrite/tests sometimes need low-level tensor or primitive-node construction
- `Tensor` is still the public anchor type, so both layers are visible here

### Recommended construction paths

For ordinary modeling code, prefer these paths:

- `Tensor.scalar(...)`
- `Tensor.zeros(...)`
- `Tensor.ones(...)`
- `Tensor.randn(...)`
- `Tensor.arange(...)`
- `Tensor.onesLike(...)`
- `Tensor.zerosLike(...)`
- array-backed leaf constructors such as:
  - `new Tensor(double[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)`
  - `new Tensor(float[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)`
  - `new Tensor(short[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)`
  - `new Tensor(byte[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)`
  - `new Tensor(int[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)`
  - `new Tensor(long[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)`
- multidimensional Java-array construction:
  - `new Tensor(Object multiDimArray, List<Tensor> previous, String label, DataType dataType)`

Practical guidance:

- for leaf inputs, `previous` is usually `null` or an empty list
- for user data, prefer typed flat-array constructors when shape is already known
- use `Object multiDimArray` only as a convenience path; it infers shape and flattens nested Java arrays

Example:

```java
Tensor x = new Tensor(
        new double[]{1, 2, 3, 4, 5, 6},
        new int[]{2, 3},
        null,
        "x",
        DataType.FLOAT64
);

Tensor y = Tensor.scalar(2.0, DataType.FLOAT64);

Tensor zeros = Tensor.zeros(new int[]{2, 3}, DataType.FLOAT64, "zeros");
Tensor valid = Tensor.ones(new int[]{2, 3}, DataType.BOOL, "valid");
Tensor ids = Tensor.arange(0, 3, 1, DataType.INT32);
```

The factory helpers avoid constructor boilerplate when all values follow a simple pattern. They still create ordinary leaf tensors: `valid` is a `[2, 3]` BOOL mask, and `ids` is a rank-1 INT32 tensor with logical values `[0, 1, 2]`.

### Low-level public constructors

The following constructor families are public, but they are primarily infrastructure surface:

- shape-only allocation:
  - `Tensor(int[] dimensions, List<Tensor> previous, String label, DataType dataType)`
- primitive-node construction:
  - `Tensor(int[] shape, List<Tensor> previous, Operation operation, String label, DataType dataType)`
  - `Tensor(int[] shape, int[] strides, List<Tensor> previous, Operation operation, String label, DataType dataType)`
  - `Tensor(int[] shape, int[] strides, int storageOffset, List<Tensor> previous, Operation operation, String label, DataType dataType)`
- typed storage constructors with explicit strides:
  - `Tensor(double[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType)`
  - `Tensor(float[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType)`
  - `Tensor(short[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType)`
  - `Tensor(byte[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType)`
  - `Tensor(int[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType)`
  - `Tensor(long[] data, int[] shape, int[] strides, List<Tensor> previous, String label, DataType dataType)`

These constructors are mainly for:

- tests
- graph/runtime/rewrite infrastructure
- explicit alias/layout setup
- primitive-backed node creation

They are valid public API, but they are not the recommended starting point for ordinary graph modeling.

## Static Facade

The graph-building API is intentionally exposed in two equivalent forms:

- instance methods on `Tensor`
- static methods on [TensorOps.java](../tensor/TensorOps.java)

Examples:

```java
Tensor z1 = x.add(y);
Tensor z2 = TensorOps.add(x, y);

Tensor p1 = x.permute(1, 0);
Tensor p2 = TensorOps.permute(x, new int[]{1, 0});
```

Practical rule:

- prefer instance methods in ordinary modeling code
- prefer `TensorOps` when writing generic helpers, static builders, or code that should read like explicit n-ary graph assembly

This reference documents the semantic contract once, using the `Tensor` instance form where possible.
The corresponding `TensorOps` methods delegate to the same family builders.

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

Worked value example:

```text
left  = [[1, 2, 3],
         [4, 5, 6]]      shape [2, 3]
right = [10, 20, 30]     shape [3]
```

`left.add(right)` produces:

```text
[[11, 22, 33],
 [14, 25, 36]]
```

because `right` is broadcast across the leading dimension.

## View Semantics

There is intentionally no public `view()` method on `Tensor`.

Instead, the public layout/indexing surface exposes a set of operations with explicit semantics:

- `reshape(...)`
- `permute(...)`
- `transpose()`
- `expand(...)`
- `expandDims(...)`
- `squeeze(...)`
- `select(...)`

Why the API is shaped this way:

- the framework wants layout intent to be explicit
- not every layout transform has the same aliasing rules
- some transforms are pure metadata rewrites
- some are broadcast alias views
- some may require later materialization depending on execution constraints

Current practical contract:

- `permute(...)` creates a stride-reordered alias view
- `transpose()` is the rank-2 special case of `permute(...)`
- `expand(...)` creates a zero-stride broadcast alias view
- `select(...)` creates a storage-offset alias view
- `expandDims(...)` and `squeeze(...)` rewrite logical shape/strides without copying storage
- `reshape(...)` preserves element count and is the public reshape operation; for contiguous layouts it aliases the same storage, while non-contiguous cases may require later materialization or remapping in execution paths
- `contiguous()` is the explicit operation that asks for dense row-major materialization

So if you are looking for a PyTorch-style `view(...)`, the nearest public equivalent is usually:

- `reshape(...)` when you want a different logical shape
- `permute(...)` / `transpose()` when you want a different axis order
- `expand(...)` when you want broadcast aliasing
- `contiguous()` when you want to force dense materialization after a view-like transform

Example:

```java
Tensor x = new Tensor(new double[]{
        1, 2, 3,
        4, 5, 6
}, new int[]{2, 3}, null, "x", DataType.FLOAT64);

Tensor y = x.reshape(3, 2);      // reshape-style view/transform
Tensor z = x.permute(1, 0);      // stride-reordered alias view
Tensor dense = z.contiguous();   // explicit dense materialization
```

Concrete values:

```text
x = [[1, 2, 3],
     [4, 5, 6]]

x.transpose() = [[1, 4],
                 [2, 5],
                 [3, 6]]

x.reshape(3, 2) = [[1, 2],
                   [3, 4],
                   [5, 6]]
```

The identical output shape `[3, 2]` does not mean identical semantics.
`transpose()` changes axis order, while `reshape(...)` changes logical shape.

## Reduction Interoperability

Axis reductions and broadcasting are designed to compose directly.

Supported `keepDims` reductions are:

- `sum(int dimension, boolean keepDims)`
- `mean(int dimension, boolean keepDims)`
- `min(int dimension, boolean keepDims)`
- `max(int dimension, boolean keepDims)`
- `all(int dimension, boolean keepDims)`
- `any(int dimension, boolean keepDims)`

Shape rules:

- `keepDims=false` removes the reduced axis
- `keepDims=true` keeps the reduced axis with size `1`

This matters because `keepDims=true` makes the result immediately reusable in the next broadcast-aware op without an explicit `expandDims(...)`.

Example:

```java
Tensor x = new Tensor(new double[]{
        1, 2, 3,
        4, 5, 6
}, new int[]{2, 3}, null, "x", DataType.FLOAT64);

Tensor rowMean = x.mean(1, true);
Tensor centered = x.sub(rowMean);

// rowMean has shape [2, 1].
// centered has shape [2, 3].
// centered values are [-1, 0, 1, -1, 0, 1].
```

Backward reduction notes:

- `sum` and `mean` expand the upstream gradient back to input shape
- `mean` also scales by the reciprocal of the reduced extent
- `min` and `max` route gradients only to winner positions
- ties in `min` / `max` split gradient evenly across winners
- if a reduction result is later broadcast in forward execution, the downstream gradient is reduced back through the same broadcast contract

## Contiguous and Materialization Contract

`Tensor` supports both:

- dense contiguous tensors
- non-contiguous or view-like tensors based on shape/stride metadata

Important behavior:

- `permute(...)` creates a view-like tensor with reordered strides
- `expand(...)` creates a zero-stride broadcast alias view
- `select(...)` creates a storage-offset alias view
- `expandDims(...)` creates a stride-preserving alias view with one inserted size-`1` axis
- `squeeze(...)` creates a stride-preserving alias view with one removed size-`1` axis
- `reshape(...)` is a layout-level transform that preserves element count
  - for contiguous inputs it aliases the same storage as a reshape view
  - for non-contiguous inputs it may materialize a dense reshaped result
- `contiguous()` is the canonical explicit materialization path
- CPU execution keeps offset views as views through layout ops
  - strided element-wise, compare, logical, `where`, reduction, softmax/logSoftmax, dense-target loss, indexing kernels, and min/max reduction-grad kernels can consume them natively
  - when a downstream kernel does not natively support non-zero base offsets, planner-side prepared inputs materialize them first

`contiguous()` is useful when:

- a later kernel performs better on dense layout
- you want to materialize an expanded zero-stride broadcast view
- you want a stable dense tensor independent of the source view layout

Example:
```java
// expanded is a zero-stride broadcast view, not a dense copy.
Tensor expanded = bias.expand(batch, features);
// contiguous() materializes it into dense storage.
Tensor dense = expanded.contiguous();
// Returns: a dense tensor with the same logical values as expanded.
```

## Execution API

### `compile()`

Builds a compiled graph using the convenience default:

- `CompileMode.INFERENCE_ONLY`
- `CompileConfig.inference()`

Returns:
- `CompiledGraph`

Example:
```java
Tensor y = logits.softmax(-1);
CompiledGraph graph = y.compile();
PreparedExecution prepared = graph.prepare();
prepared.execute(ExecutionMode.FORWARD);
```

### `compile(CompileMode compileMode)`

Builds a compiled graph with an explicit compile intent.

Supported intents:
- `INFERENCE_ONLY`
  - always compile forward only
- `TRAINING`
  - compile forward + backward when the graph has trainable leaf inputs
- `AUTO`
  - preserve historical behavior; compile backward only when trainable leaf inputs exist

Returns:
- `CompiledGraph`

Example:
```java
Tensor loss = logits.logSoftmax(-1).mean();
CompiledGraph graph = loss.compile(CompileMode.TRAINING);
PreparedExecution prepared = graph.prepare();
prepared.execute(ExecutionMode.FORWARD_BACKWARD);
```

### `prepare(ExecutionProfile profile)`

Builds a prepared execution for the graph rooted at this tensor.

Parameters:
- `profile`: execution profile containing compile policy, runtime config, and execution mode defaults

Returns:
- `PreparedExecution`

Example:
```java
// Builds a prepared execution object for the graph rooted at y.
ExecutionProfile profile = new ExecutionProfile(
        "manual-infer",
        "manual-infer",
        y.getDataType(),
        ExecutionMode.FORWARD,
        CompileConfig.inference(),
        RuntimeConfig.inferenceDefaults()
);
PreparedExecution execution = y.prepare(profile);
// Returns: a PreparedExecution bound to the selected compile/runtime profile.
```

### `compute()`

Convenience one-shot execution.

Semantics:
- compile with `CompileMode.INFERENCE_ONLY`
- use inference optimizer defaults
- use inference runtime defaults
- execute `FORWARD`

Returns:
- the same tensor instance after execution

Example:
```java
Tensor y = logits.softmax(-1).compute();
// Returns: the same tensor instance, now holding computed forward data.
```

### `compute(CompileMode compileMode)`

Convenience one-shot execution with explicit compile intent.

Semantics:
- resolves default compile/runtime presets from the compile mode
- executes `FORWARD` for `INFERENCE_ONLY`
- executes `FORWARD_BACKWARD` for `TRAINING` when the graph has trainable leaves
- `AUTO` chooses between those two behaviors from the graph

Returns:
- the same tensor instance after execution

Example:
```java
Tensor loss = logits.mean();
loss.compute(CompileMode.TRAINING);
```

### `compute(ComputeOptions options)`

Configurable convenience execution.

Supported options today:
- `compileMode(...)`
- `autotune(...)`
- `compile(...)`
- `runtime(...)`

`AutotunePolicy` semantics:
- `NEVER`
  - use the resolved compile/runtime profile directly
- `IF_MISSING`
  - for this tensor graph, dtype, mode, and hardware fingerprint:
    - reuse a cached generic best-profile from `build/tuning/tensor/<platform-id>/<graph-signature>/<seed-signature>/...` if present
    - otherwise run generic stage-order autotune once and persist the winner
- `FORCE`
  - rerun generic stage-order autotune and overwrite the cached winner

The generic autotune used here is intentionally lightweight:
- it autotunes the current tensor root as a generic workload
- it searches constrained stage-order candidates
- it persists the winner under the `build/tuning/tensor/...` tree together with matching history

In practical terms, a winner path looks like:

```text
build/tuning/tensor/<platform-id>/<graph-signature>/<seed-signature>/<dtype>-<mode>-best-profile.json
```

Returns:
- the same tensor instance after execution

Example:
```java
Tensor loss = logits.mean().compute(
        new ComputeOptions()
                .compileMode(CompileMode.TRAINING)
                .autotune(AutotunePolicy.IF_MISSING)
);
```

This single call may:

1. derive a default training profile
2. resolve or create a generic autotuned best profile for the current graph
3. prepare runtime metadata
4. execute `FORWARD_BACKWARD`

### `compute(ExecutionProfile profile)`

Builds and executes the graph using the supplied profile.

Parameters:
- `profile`: execution profile to use

Returns:
- nothing; the graph is executed in place

Example:
```java
// Compiles and executes the graph rooted at loss using the training profile.
ExecutionProfile profile = new ExecutionProfile(
        "manual-train",
        "manual-train",
        loss.getDataType(),
        ExecutionMode.FORWARD_BACKWARD,
        CompileConfig.training(),
        RuntimeConfig.trainingDefaults()
);
loss.compute(profile);
// Returns: nothing; loss and all dependent tensors are executed in place.
```

### `compute(PreparedExecution execution, ExecutionMode mode)`

Runs a previously prepared execution.

Parameters:
- `execution`: prepared execution object
- `mode`: `FORWARD` or `FORWARD_BACKWARD`

Returns:
- nothing; the prepared execution is run in place

Example:
```java
// Reuses a prepared execution and runs only the forward pass.
PreparedExecution execution = y.prepare(profile);
y.compute(execution, ExecutionMode.FORWARD);
// Returns: nothing; the prepared execution is run in place.
```

## Static Factories

### `scalar(double value)`
### `scalar(double value, DataType dataType)`

Creates a scalar tensor with shape `[1]`.

Parameters:
- `value`: scalar value
- `dataType`: optional dtype

Returns:
- scalar tensor

Example:
```java
// Creates a default scalar tensor with value 3.0 and shape [1].
Tensor a = Tensor.scalar(3.0);
// Creates a FLOAT64 scalar tensor with value 1.5 and shape [1].
Tensor b = Tensor.scalar(1.5, DataType.FLOAT64);
// Returns: scalar tensors.
```

### `zeros(...)` / `ones(...)`

Creates a dense tensor filled with a constant.

Signatures:
```java
Tensor.zeros(int[] shape)
Tensor.zeros(int[] shape, DataType dataType)
Tensor.zeros(int[] shape, DataType dataType, String label)
Tensor.ones(int[] shape)
Tensor.ones(int[] shape, DataType dataType)
Tensor.ones(int[] shape, DataType dataType, String label)
```

Parameters:
- `shape`: output shape
- `dataType`: optional output dtype
- `label`: optional graph/debug label

Returns:
- newly allocated tensor with the requested shape and dtype

Example:
```java
Tensor weights = Tensor.zeros(new int[]{4, 8}, DataType.FLOAT64, "weights");
Tensor mask = Tensor.ones(new int[]{2, 3}, DataType.BOOL, "validMask");
// Returns: constant tensors with explicit shape and dtype.
```

### `randn(...)`

Creates a floating tensor filled with normally distributed values.

Signatures:
```java
Tensor.randn(int[] shape)
Tensor.randn(int[] shape, double mean, double stdDev, DataType dataType, String label)
```

Contract:
- supports floating dtypes only: `FLOAT64`, `FLOAT32`, and `BFLOAT16`
- `stdDev` must be finite and non-negative

Example:
```java
Tensor x = Tensor.randn(new int[]{2, 3, 4}, 0.0, 1.0, DataType.FLOAT64, "x");
// Returns: shape [2, 3, 4].
```

### `arange(...)`

Creates a rank-1 numeric tensor from an integer range.

Signature:
```java
Tensor.arange(int start, int end, int step, DataType dataType)
```

Contract:
- `start` is inclusive
- `end` is exclusive
- `step` cannot be zero
- empty ranges are rejected
- `BOOL` dtype is rejected

Example:
```java
Tensor indices = Tensor.arange(0, 6, 2, DataType.INT32);
// indices values = [0, 2, 4]
```

### `onesLike(Tensor other)`

Creates a dense tensor of ones matching `other.shape` and `other.dtype`.

Parameters:
- `other`: reference tensor

Returns:
- tensor filled with ones

Example:
```java
// Creates a tensor of ones with the same shape and dtype as x.
Tensor mask = Tensor.onesLike(x);
// Returns: a tensor full of ones.
```

### `zerosLike(Tensor other)`

Creates a dense tensor of zeros matching `other.shape` and `other.dtype`.

Parameters:
- `other`: reference tensor

Returns:
- tensor filled with zeros

Example:
```java
// Creates a tensor of zeros with the same shape and dtype as x.
Tensor zeros = Tensor.zerosLike(x);
// Returns: a tensor full of zeros.
```

## Layout / Shape Operations

### `contiguous()`

Materializes the tensor into a dense contiguous layout.

Use this when you have a view-like tensor whose logical values are correct, but whose storage layout is non-contiguous or zero-stride.
`contiguous()` creates a dense tensor with the same logical values in row-major order.

Parameters:
- none

Returns:
- tensor with the same logical shape and values, but contiguous storage

Example:
```java
Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base");
// base logically represents:
// [[1, 2, 3],
//  [4, 5, 6]]

Tensor permuted = base.permute(1, 0);
// permuted is a view of shape [3, 2] with logical values:
// [[1, 4],
//  [2, 5],
//  [3, 6]]

Tensor dense = permuted.contiguous();
// dense keeps the same logical shape [3, 2]
// and the same logical values:
// [[1, 4],
//  [2, 5],
//  [3, 6]]
// but stores them in contiguous row-major order.
//
// Returns: a dense materialized tensor with the same logical values as permuted.
```

### `reshape(int... newShape)`

Changes the logical shape while preserving element count.

`reshape(...)` does not change values, only how the same flat element sequence is interpreted as dimensions.
The total number of elements before and after reshape must match.

Parameters:
- `newShape`: requested output shape; may include one inferred `-1`

Returns:
- reshaped tensor

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
// x logically represents:
// [[1, 2, 3],
//  [4, 5, 6]]

Tensor y = x.reshape(3, 2);
// y logically represents:
// [[1, 2],
//  [3, 4],
//  [5, 6]]

Tensor z = x.reshape(3, -1);
// -1 is inferred, so z is also shape [3, 2]
// with the same logical values as y.
//
// Returns: tensors with the same elements but a different shape interpretation.
```

### `expand(int... newShape)`

Expands singleton dimensions using broadcast semantics.
Current implementation is a zero-stride alias view, not a dense materialization.

`expand(...)` is valid only when every expanded axis was size `1` in the source or already matches the target size.
It does not create new independent values; it creates a logical broadcast view over the original storage.

Parameters:
- `newShape`: target shape; rank may stay the same or increase

Returns:
- expanded view tensor

Example:
```java
Tensor bias = new Tensor(new double[]{10, 20, 30}, new int[]{1, 3}, null, "bias");
// bias logically represents:
// [[10, 20, 30]]

Tensor y = bias.expand(2, 3);
// y logically represents:
// [[10, 20, 30],
//  [10, 20, 30]]
//
// Important:
// y is not a dense copy yet
// it is a zero-stride broadcast view over bias

Tensor dense = y.contiguous();
// dense has the same logical values as y
// but is now explicitly materialized.
//
// Returns: zero-stride broadcast views, optionally materialized later via contiguous().
```

### `permute(int... axes)`

Reorders tensor axes.

`permute(...)` changes axis order.
It does not change numeric values, but it changes which coordinate maps to which value.

Parameters:
- `axes`: permutation of all axes

Returns:
- permuted view tensor

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
// x logically represents:
// [[1, 2, 3],
//  [4, 5, 6]]

Tensor y = x.permute(1, 0);
// y has shape [3, 2]
// and logically represents:
// [[1, 4],
//  [2, 5],
//  [3, 6]]

Tensor z = y.contiguous();
// z keeps the same logical values as y
// but stores them densely in row-major order.
//
// Returns: permuted view tensors.
```

### `transpose()`

Convenience wrapper for rank-2 transpose.

For rank-2 tensors, `transpose()` is equivalent to `permute(1, 0)`.

Parameters:
- none

Returns:
- transposed rank-2 tensor

Example:
```java
Tensor matrix = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "matrix");
// matrix logically represents:
// [[1, 2, 3],
//  [4, 5, 6]]

Tensor y = matrix.transpose();
// y has shape [3, 2]
// and logically represents:
// [[1, 4],
//  [2, 5],
//  [3, 6]]
//
// Returns: a rank-2 transposed view tensor.
```

### `expandDims(int axis)`

Inserts a singleton dimension at `axis`.

This is a pure shape transform and a true alias view.
It does not change values and does not materialize new dense storage; it only adds an axis of size `1` and updates shape/stride metadata.

Parameters:
- `axis`: insertion position

Returns:
- tensor with one additional dimension of size `1`

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "x");
// x logically represents:
// [[1, 2],
//  [3, 4]]

Tensor y = x.expandDims(0);
// y has shape [1, 2, 2]
// and logically represents:
// [[[1, 2],
//   [3, 4]]]

Tensor z = x.expandDims(2);
// z has shape [2, 2, 1]
// and logically represents:
// [[[1], [2]],
//  [[3], [4]]]
//
// Returns: alias-view tensors with one extra size-1 axis.
```

### `squeeze(int axis)`

Removes a singleton dimension at `axis`.

This is the inverse of `expandDims(...)` when the chosen axis has size `1`.
It is also a true alias view and only rewrites shape/stride metadata.

Parameters:
- `axis`: axis that must have size `1`

Returns:
- tensor with that dimension removed

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4}, new int[]{1, 2, 2}, null, "x");
// x logically represents:
// [[[1, 2],
//   [3, 4]]]

Tensor y = x.squeeze(0);
// y has shape [2, 2]
// and logically represents:
// [[1, 2],
//  [3, 4]]

Tensor expanded = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2, 1}, null, "expanded");
Tensor z = expanded.squeeze(2);
// z has shape [2, 2]
// and keeps the same logical values.
//
// Returns: alias-view tensors with one fewer size-1 axis.
```

### `sliceAxis(int axis, int fromInclusive, int toExclusive)`

Slices one axis with positive step `1`.

Parameters:
- `axis`: axis to slice
- `fromInclusive`: first included index
- `toExclusive`: first excluded index

Returns:
- tensor with the same rank as the input and a shorter selected axis

Example:
```java
Tensor x = new Tensor(new double[]{
        1, 2, 3, 4,
        5, 6, 7, 8
}, new int[]{2, 4}, null, "x");

Tensor y = x.sliceAxis(1, 1, 3);
// y shape = [2, 2]
// y values = [[2, 3], [6, 7]]
```

### `stack(int axis, Tensor... inputs)` / `unstack(int axis)`

`Tensor.stack(axis, inputs...)` inserts a new dimension and concatenates same-shaped tensors along that new dimension. `unstack(axis)` splits one tensor along an axis and removes that axis from each returned tensor.

Contract:
- all stacked inputs must have the same shape and dtype
- `axis` for `stack` is an insertion axis in `[0, rank]`
- `axis` for `unstack` is an existing axis in `[0, rank)`
- both operations are gradient-safe compositions of layout/index primitives

Example:
```java
Tensor t0 = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "t0");
Tensor t1 = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "t1");

Tensor seq = Tensor.stack(1, t0, t1);
// seq shape = [2, 2, 2]

Tensor[] timesteps = seq.unstack(1);
// timesteps.length = 2
// timesteps[0].shape = [2, 2]
```

## Binary Arithmetic Operations

These operations:
- use standard binary broadcasting
- align ranks from the right
- treat missing leading dimensions as `1`
- reduce broadcasted gradients back to original operand shapes during backward execution

### `add(Tensor second)`

Element-wise addition.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise sum

Example:
```java
Tensor a = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "b");

Tensor y = a.add(b);
// y has shape [3] and values [11, 22, 33].

Tensor matrix = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "matrix");
Tensor bias = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "bias");
Tensor z = matrix.add(bias);
// z has shape [2, 3] and values:
// [[11, 22, 33],
//  [14, 25, 36]]
//
// Returns: element-wise sums with optional broadcasting.
```

### `sub(Tensor second)`

Element-wise subtraction.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise difference

Example:
```java
Tensor a = new Tensor(new double[]{5, 7, 9}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "b");

Tensor y = a.sub(b);
// y has shape [3] and values [4, 5, 6].

Tensor prediction = new Tensor(new double[]{0.9, 0.1}, new int[]{2}, null, "prediction");
Tensor target = new Tensor(new double[]{1.0, 0.0}, new int[]{2}, null, "target");
Tensor z = prediction.sub(target);
// z has shape [2] and values [-0.1, 0.1].
//
// Returns: element-wise differences.
```

### `mul(Tensor second)`

Element-wise multiplication.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise product

Example:
```java
Tensor a = new Tensor(new double[]{2, 3, 4}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "b");

Tensor y = a.mul(b);
// y has shape [3] and values [20, 60, 120].

Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor mask = new Tensor(new double[]{1, 0, 1}, new int[]{3}, null, "mask");
Tensor z = x.mul(mask);
// z has shape [2, 3] and values:
// [[1, 0, 3],
//  [4, 0, 6]]
//
// Returns: element-wise products with optional broadcasting.
```

### `div(Tensor second)`

Element-wise division.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise quotient

Example:
```java
Tensor a = new Tensor(new double[]{8, 9, 10}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{2, 3, 5}, new int[]{3}, null, "b");

Tensor y = a.div(b);
// y has shape [3] and values [4, 3, 2].

Tensor x = new Tensor(new double[]{2, 4, 6, 8, 10, 12}, new int[]{2, 3}, null, "x");
Tensor scale = new Tensor(new double[]{2, 2, 3}, new int[]{3}, null, "scale");
Tensor z = x.div(scale);
// z has shape [2, 3] and values:
// [[1, 2, 2],
//  [4, 5, 4]]
//
// Returns: element-wise quotients with optional broadcasting.
```

### `min(Tensor second)`

Element-wise minimum.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise minimum

Example:
```java
Tensor a = new Tensor(new double[]{1, 5, 3}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{2, 4, 3}, new int[]{3}, null, "b");

Tensor y = a.min(b);
// y has shape [3] and values [1, 4, 3].

Tensor logits = new Tensor(new double[]{1, 9, 3, 7}, new int[]{2, 2}, null, "logits");
Tensor cap = new Tensor(new double[]{4}, new int[]{1}, null, "cap");
Tensor z = logits.min(cap);
// z has shape [2, 2] and values:
// [[1, 4],
//  [3, 4]]
//
// Returns: element-wise minima with optional broadcasting.
```

### `max(Tensor second)`

Element-wise maximum.

Parameters:
- `second`: right-hand operand

Returns:
- broadcasted element-wise maximum

Example:
```java
Tensor a = new Tensor(new double[]{1, 5, 3}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{2, 4, 3}, new int[]{3}, null, "b");

Tensor y = a.max(b);
// y has shape [3] and values [2, 5, 3].

Tensor x = new Tensor(new double[]{-2, 1, -1, 3}, new int[]{2, 2}, null, "x");
Tensor floor = new Tensor(new double[]{0}, new int[]{1}, null, "floor");
Tensor z = x.max(floor);
// z has shape [2, 2] and values:
// [[0, 1],
//  [0, 3]]
//
// Returns: element-wise maxima with optional broadcasting.
```

### `matmul(Tensor second)`

Matrix multiplication over the last two dimensions, with broadcast over leading batch dimensions.

Parameters:
- `second`: right-hand matrix or batched matrix tensor

Returns:
- tensor with broadcasted batch shape and matrix product in the last two dimensions

Contract:
- both inputs must have rank `>= 2`
- for `a[..., m, k]` and `b[..., k, n]`, the result has shape `broadcast(aBatch, bBatch) + [m, n]`
- leading dimensions follow standard broadcasting rules
- rank-2 inputs remain the ordinary matrix-multiplication case

Example:
```java
Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "a");
Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "b");

Tensor y = a.matmul(b);
// y has shape [2, 2] and values:
// [[19, 22],
//  [43, 50]]
//
// Returns: matrix products for rank-2 tensors.
```

Batch example:
```java
Tensor a = new Tensor(new double[]{
        1, 2, 3, 4,
        5, 6, 7, 8
}, new int[]{2, 2, 2}, null, "a");
Tensor b = new Tensor(new double[]{
        1, 2,
        3, 4
}, new int[]{2, 2, 1}, null, "b");

Tensor y = a.matmul(b);
// a has shape [2, 2, 2]
// b has shape [2, 2, 1]
// y has shape [2, 2, 1] and values [5, 11, 39, 53].
//
// Returns: batched matrix products over the last two axes.
```

### `linear(Tensor weight)`

Applies a linear projection without bias.

Parameters:
- `weight`: rank-2 tensor with shape `[inFeatures, outFeatures]`

Returns:
- tensor with the same leading dimensions as the input and last dimension `outFeatures`

Contract:
- receiver tensor must have rank `>= 2`
- receiver shape is `[..., inFeatures]`
- `weight.shape[0]` must equal `inFeatures`
- this surface is intentionally defined over the existing batched `matmul` contract

Example:
```java
Tensor x = new Tensor(new double[]{
        1, 2,
        3, 4
}, new int[]{2, 2}, null, "x");
Tensor w = new Tensor(new double[]{
        5, 6,
        7, 8
}, new int[]{2, 2}, null, "w");

Tensor y = x.linear(w);
// x shape = [2, 2]
// w shape = [2, 2]
// y shape = [2, 2] and values [19, 22, 43, 50].
//
// Returns: x @ w.
```

### `linear(Tensor weight, Tensor bias)`

Applies a linear projection with bias.

Parameters:
- `weight`: rank-2 tensor with shape `[inFeatures, outFeatures]`
- `bias`: tensor with shape `[outFeatures]` or `[1, outFeatures]`

Returns:
- tensor with the same leading dimensions as the input and last dimension `outFeatures`

Contract:
- receiver tensor must have shape `[..., inFeatures]`
- `weight` must have shape `[inFeatures, outFeatures]`
- `bias` must have shape `[outFeatures]` or `[1, outFeatures]`
- bias is broadcast across all leading dimensions

Example:
```java
Tensor x = new Tensor(new double[]{
        1, 2,
        3, 4,
        5, 6,
        7, 8
}, new int[]{2, 2, 2}, null, "x");
Tensor w = new Tensor(new double[]{
        1, 10,
        100, 1000
}, new int[]{2, 2}, null, "w");
Tensor b = new Tensor(new double[]{0.5, -0.5}, new int[]{2}, null, "b");

Tensor y = x.linear(w, b);
// x shape = [2, 2, 2]
// w shape = [2, 2]
// b shape = [2]
// y shape = [2, 2, 2]
//
// Returns: x @ w + b with bias broadcast over the leading dimensions.
```

## Attention Operations

Attention is exposed as a dedicated primitive-backed surface.

That matters because it gives the graph/compiler/backend a stable semantic contract for:

- fused attention lowering
- attention-specific backward handling
- runtime caching of auxiliary state when needed

### `scaledDotProductAttention(Tensor key, Tensor value, AttentionOptions options)`

Builds standard scaled dot-product attention without an explicit external mask.

Parameters:
- `key`: tensor with shape `[..., keyLen, headDim]`
- `value`: tensor with shape `[..., keyLen, valueDim]`
- `options`: attention configuration covering `causal` and optional scale override

Returns:
- tensor with shape `[..., queryLen, valueDim]`

Contract:
- query tensor is the receiver and has shape `[..., queryLen, headDim]`
- `query.headDim == key.headDim`
- `key.keyLen == value.keyLen`
- leading dimensions use batched `matmul` broadcasting rules
- default scale is `1 / sqrt(headDim)` unless explicitly overridden

Example:
```java
Tensor q = new Tensor(new double[]{
        1, 0,
        0, 1
}, new int[]{1, 2, 2}, null, "q");
Tensor k = new Tensor(new double[]{
        1, 0,
        0, 1
}, new int[]{1, 2, 2}, null, "k");
Tensor v = new Tensor(new double[]{
        10, 1,
        1, 10
}, new int[]{1, 2, 2}, null, "v");

Tensor y = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(1.0));
// q shape = [1, 2, 2]
// k shape = [1, 2, 2]
// v shape = [1, 2, 2]
// y shape = [1, 2, 2]
//
// Returns: standard attention output computed from q, k, v.
```

### `scaledDotProductAttention(Tensor key, Tensor value, Tensor mask, AttentionOptions options)`

Builds scaled dot-product attention with an explicit boolean mask.

Parameters:
- `key`: tensor with shape `[..., keyLen, headDim]`
- `value`: tensor with shape `[..., keyLen, valueDim]`
- `mask`: `BOOL` tensor broadcastable to the attention score shape `[..., queryLen, keyLen]`
- `options`: attention configuration covering `causal` and optional scale override

Returns:
- tensor with shape `[..., queryLen, valueDim]`

Contract:
- `mask == true` means “this score is allowed”
- `mask == false` means “this score is suppressed before softmax”
- if `options.causal()` is also enabled, the external mask is combined with the causal mask through logical `and`

Example:
```java
Tensor q = new Tensor(new double[]{1, 0}, new int[]{1, 1, 2}, null, "q");
Tensor k = new Tensor(new double[]{
        1, 0,
        0, 1
}, new int[]{1, 2, 2}, null, "k");
Tensor v = new Tensor(new double[]{
        10, 1,
        1, 10
}, new int[]{1, 2, 2}, null, "v");
Tensor mask = new Tensor(new byte[]{0, 1}, new int[]{1, 1, 2}, null, "mask", DataType.BOOL);

Tensor y = q.scaledDotProductAttention(k, v, mask, AttentionOptions.defaults().withScale(1.0));
// Only the second key/value position is allowed by the mask.
// y has shape [1, 1, 2] and values [1, 10].
//
// Returns: masked attention output.
```

## Spatial Operations

### `conv2d(Tensor weight, Conv2dOptions options)`

Runs 2D convolution without bias.

Parameters:
- `weight`: convolution kernel tensor with shape `[outChannels, inChannelsPerGroup, kernelH, kernelW]`
- `options`: convolution configuration carrying stride, padding, dilation, and `groups`

Returns:
- floating tensor with shape `[batch, outChannels, outH, outW]`

Contract:
- input tensor shape must be `[batch, inChannels, inH, inW]`
- input layout is `NCHW`
- weight layout is `OIHW`
- only floating dtypes are accepted
- grouped convolution is supported through `options.groups()`
- output size uses floor semantics:
  - `out = floor((input + 2 * pad - effectiveKernel) / stride) + 1`
  - `effectiveKernel = dilation * (kernel - 1) + 1`

Backward:
- input gradient is produced through dedicated `CONV2D_BACKWARD_INPUT`
- weight gradient is produced through dedicated `CONV2D_BACKWARD_WEIGHT`

Example:
```java
Tensor input = new Tensor(new double[]{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
}, new int[]{1, 1, 3, 3}, null, "input");
Tensor weight = new Tensor(new double[]{
        1, 0,
        0, -1
}, new int[]{1, 1, 2, 2}, null, "weight");

Tensor y = input.conv2d(weight, Conv2dOptions.defaults());
// input shape = [1, 1, 3, 3]
// weight shape = [1, 1, 2, 2]
// y has shape [1, 1, 2, 2] and values:
// [[[-4, -4],
//   [-4, -4]]]
//
// Returns: the direct 2D convolution result.
```

### `conv2d(Tensor weight, Tensor bias, Conv2dOptions options)`

Runs 2D convolution with per-output-channel bias.

Parameters:
- `weight`: convolution kernel tensor with shape `[outChannels, inChannelsPerGroup, kernelH, kernelW]`
- `bias`: bias tensor with shape `[outChannels]`
- `options`: convolution configuration carrying stride, padding, dilation, and `groups`

Returns:
- floating tensor with shape `[batch, outChannels, outH, outW]`

Contract:
- same input and weight contract as the bias-free overload
- `bias` must be rank-1 with one value per output channel
- output dtype is promoted from input, weight, and bias dtypes

Example:
```java
Tensor input = new Tensor(new double[]{
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
}, new int[]{1, 1, 3, 3}, null, "input");
Tensor weight = new Tensor(new double[]{
        1, 1,
        1, 1
}, new int[]{1, 1, 2, 2}, null, "weight");
Tensor bias = new Tensor(new double[]{0.5}, new int[]{1}, null, "bias");

Tensor y = input.conv2d(
        weight,
        bias,
        Conv2dOptions.defaults().withStride(2, 2).withPadding(1, 1)
);
// input shape = [1, 1, 3, 3]
// weight shape = [1, 1, 2, 2]
// bias shape = [1]
// y has shape [1, 1, 2, 2] and values:
// [[[1.5, 5.5],
//   [11.5, 28.5]]]
//
// Returns: convolution output with channel bias added after accumulation.
```

### `maxPool2d(Pool2dOptions options)`

Runs 2D max pooling over a rank-4 `NCHW` tensor.

Parameters:
- `options`: pooling window configuration carrying kernel, stride, and padding

Returns:
- floating tensor with shape `[batch, channels, outH, outW]`

Contract:
- input tensor shape must be `[batch, channels, inH, inW]`
- input layout is `NCHW`
- only floating dtypes are accepted
- output size uses floor semantics:
  - `out = floor((input + 2 * pad - kernel) / stride) + 1`

Backward:
- gradient is routed to the first maximal element encountered in scan order inside each pooling window
- prepared execution stores per-output argmax indices in runtime workspace and reuses them in backward

Example:
```java
Tensor input = new Tensor(new double[]{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
}, new int[]{1, 1, 4, 4}, null, "input");

Tensor y = input.maxPool2d(Pool2dOptions.square(2));
// input shape = [1, 1, 4, 4]
// 2x2 pooling with default stride 2 produces shape [1, 1, 2, 2]
// y values:
// [[[6, 8],
//   [14, 16]]]
//
// Returns: window-wise maxima over the spatial axes.
```

### `avgPool2d(Pool2dOptions options)`

Runs 2D average pooling over a rank-4 `NCHW` tensor.

Parameters:
- `options`: pooling window configuration carrying kernel, stride, padding, and `countIncludePad`

Returns:
- floating tensor with shape `[batch, channels, outH, outW]`

Contract:
- same rank/layout contract as `maxPool2d`
- by default `countIncludePad = false`
  - denominator counts only valid in-bounds input elements
- if `countIncludePad = true`
  - denominator is always `kernelH * kernelW`
- pooling runs through dense prepared inputs
  - non-contiguous or offset inputs are materialized before execution

Backward:
- upstream gradient is distributed uniformly over the contributing window according to the same divisor that forward used

Example:
```java
Tensor input = new Tensor(new double[]{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
}, new int[]{1, 1, 4, 4}, null, "input");

Tensor y = input.avgPool2d(Pool2dOptions.square(2));
// input shape = [1, 1, 4, 4]
// 2x2 pooling with default stride 2 produces shape [1, 1, 2, 2]
// y values:
// [[[3.5, 5.5],
//   [11.5, 13.5]]]
//
// Returns: window-wise averages over the spatial axes.
```

## Normalization Operations

Normalization surfaces are:
- stateless
- composition-based over existing tensor primitives
- free of hidden running-stat mutations

### `batchNorm(Tensor gamma, Tensor beta, int channelDimension, double epsilon)`

Applies stateless batch normalization using batch statistics computed from the input being normalized.

Parameters:
- `gamma`: rank-1 scale tensor with shape `[channels]`
- `beta`: rank-1 bias tensor with shape `[channels]`
- `channelDimension`: axis treated as the channel axis
- `epsilon`: positive numerical stabilizer

Returns:
- tensor with the same shape as the input

Contract:
- input must use a floating dtype
- input must have at least one non-channel axis
- `gamma` and `beta` must both have shape `[input.shape[channelDimension]]`
- mean and variance are reduced over all axes except the channel axis

Example:
```java
Tensor x = new Tensor(new double[]{
        1, 2,
        3, 4
}, new int[]{2, 2, 1, 1}, null, "x");
Tensor gamma = new Tensor(new double[]{1, 1}, new int[]{2}, null, "gamma");
Tensor beta = new Tensor(new double[]{0, 0}, new int[]{2}, null, "beta");

Tensor y = x.batchNorm(gamma, beta, 1, 1e-12);
// Channel 0 sees values [1, 3], so mean=2 and variance=1.
// Channel 1 sees values [2, 4], so mean=3 and variance=1.
// y keeps shape [2, 2, 1, 1] and values:
// [[[-1], [-1]],
//  [[ 1], [ 1]]]
//
// Returns: stateless batch-normalized output using batch statistics from this input.
```

### `batchNorm(Tensor gamma, Tensor beta, Tensor mean, Tensor variance, int channelDimension, double epsilon)`

Applies batch normalization using externally provided channel statistics.

Parameters:
- `gamma`: rank-1 scale tensor with shape `[channels]`
- `beta`: rank-1 bias tensor with shape `[channels]`
- `mean`: rank-1 mean tensor with shape `[channels]`
- `variance`: rank-1 variance tensor with shape `[channels]`
- `channelDimension`: axis treated as the channel axis
- `epsilon`: positive numerical stabilizer

Returns:
- tensor with the same shape as the input

Contract:
- same floating/channel-shape contract as the batch-stat overload
- `mean` and `variance` must also have shape `[channels]`
- this overload is the clean inference-style surface because it has no hidden state updates

Example:
```java
Tensor x = new Tensor(new double[]{5, 8}, new int[]{1, 2, 1, 1}, null, "x");
Tensor gamma = new Tensor(new double[]{2, 3}, new int[]{2}, null, "gamma");
Tensor beta = new Tensor(new double[]{10, 20}, new int[]{2}, null, "beta");
Tensor mean = new Tensor(new double[]{1, 2}, new int[]{2}, null, "mean");
Tensor variance = new Tensor(new double[]{4, 9}, new int[]{2}, null, "variance");

Tensor y = x.batchNorm(gamma, beta, mean, variance, 1, 1e-12);
// Channel 0: ((5 - 1) / sqrt(4)) * 2 + 10 = 14
// Channel 1: ((8 - 2) / sqrt(9)) * 3 + 20 = 26
// y has shape [1, 2, 1, 1] and values [14, 26].
//
// Returns: batch-normalized output using provided statistics.
```

### `layerNorm(Tensor gamma, Tensor beta, double epsilon)`

Applies layer normalization over the trailing dimensions described by `gamma` / `beta`.

Parameters:
- `gamma`: floating tensor whose shape must match the trailing normalized input shape
- `beta`: floating tensor with the same shape as `gamma`
- `epsilon`: positive numerical stabilizer

Returns:
- tensor with the same shape as the input

Contract:
- `gamma.shape` and `beta.shape` must be identical
- those shapes must exactly match the trailing input dimensions
- mean and variance are reduced over the normalized trailing block

Example:
```java
Tensor x = new Tensor(new double[]{
        1, 3,
        2, 4
}, new int[]{2, 2}, null, "x");
Tensor gamma = new Tensor(new double[]{2, 3}, new int[]{2}, null, "gamma");
Tensor beta = new Tensor(new double[]{10, 20}, new int[]{2}, null, "beta");

Tensor y = x.layerNorm(gamma, beta, 1e-12);
// Each row is normalized over its last dimension:
// [1, 3] -> [-1, 1]
// [2, 4] -> [-1, 1]
// Then affine transform:
// [-1, 1] * [2, 3] + [10, 20] = [8, 23]
// y has shape [2, 2] and values [8, 23, 8, 23].
//
// Returns: layer-normalized output over the trailing feature shape.
```

### `rmsNorm(Tensor gamma, double epsilon)`

Applies RMS normalization over the trailing dimensions described by `gamma`.

Parameters:
- `gamma`: floating tensor whose shape must match the trailing normalized input shape
- `epsilon`: positive numerical stabilizer

Returns:
- tensor with the same shape as the input

Contract:
- `gamma.shape` must exactly match the trailing input dimensions
- normalization uses `sqrt(mean(x^2) + epsilon)`
- there is no bias term in this surface

Example:
```java
Tensor x = new Tensor(new double[]{3, 4}, new int[]{1, 2}, null, "x");
Tensor gamma = new Tensor(new double[]{1, 1}, new int[]{2}, null, "gamma");

Tensor y = x.rmsNorm(gamma, 1e-12);
// mean(x^2) = (9 + 16) / 2 = 12.5
// rms = sqrt(12.5)
// y = x / rms
// y has shape [1, 2] and values approximately [0.8485, 1.1314].
//
// Returns: RMS-normalized output over the trailing feature shape.
```

## Comparison Operations

Comparison ops:
- require numeric inputs
- use the same binary broadcasting contract as arithmetic binary ops
- return `BOOL` tensors
- are nondifferentiable

### `greaterThan(Tensor second)`

Element-wise `left > right`.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this > second`

Example:
```java
Tensor a = new Tensor(new double[]{1, 5, 3}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{4, 4, 4}, new int[]{3}, null, "b");

Tensor mask = a.greaterThan(b);
// mask has shape [3] and values [false, true, false].

Tensor scores = new Tensor(new double[]{0.2, 0.8, 0.4}, new int[]{3}, null, "scores");
Tensor threshold = new Tensor(new double[]{0.5}, new int[]{1}, null, "threshold");
Tensor active = scores.greaterThan(threshold);
// active has shape [3] and values [false, true, false].
//
// Returns: BOOL tensors.
```

### `greaterOrEqual(Tensor second)`

Element-wise `left >= right`.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this >= second`

Example:
```java
Tensor a = new Tensor(new double[]{1, 4, 4}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{1, 5, 4}, new int[]{3}, null, "b");

Tensor mask = a.greaterOrEqual(b);
// mask has shape [3] and values [true, false, true].
//
// Returns: a BOOL tensor.
```

### `lessThan(Tensor second)`

Element-wise `left < right`.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this < second`

Example:
```java
Tensor a = new Tensor(new double[]{1, 5, 3}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{4, 4, 4}, new int[]{3}, null, "b");

Tensor mask = a.lessThan(b);
// mask has shape [3] and values [true, false, true].
//
// Returns: a BOOL tensor.
```

### `lessOrEqual(Tensor second)`

Element-wise `left <= right`.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this <= second`

Example:
```java
Tensor a = new Tensor(new double[]{1, 4, 4}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{1, 5, 4}, new int[]{3}, null, "b");

Tensor mask = a.lessOrEqual(b);
// mask has shape [3] and values [true, true, true].
//
// Returns: a BOOL tensor.
```

### `equalTo(Tensor second)`

Element-wise exact equality.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this == second`

Example:
```java
Tensor a = new Tensor(new double[]{1, 2, 2}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{1, 0, 2}, new int[]{3}, null, "b");

Tensor mask = a.equalTo(b);
// mask has shape [3] and values [true, false, true].
//
// Returns: a BOOL tensor.
```

### `notEqualTo(Tensor second)`

Element-wise exact inequality.

Parameters:
- `second`: right-hand operand

Returns:
- `BOOL` tensor with `true` where `this != second`

Example:
```java
Tensor a = new Tensor(new double[]{1, 2, 2}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{1, 0, 2}, new int[]{3}, null, "b");

Tensor mask = a.notEqualTo(b);
// mask has shape [3] and values [false, true, false].
//
// Returns: a BOOL tensor.
```

## Select Operation

### `where(Tensor condition, Tensor ifTrue, Tensor ifFalse)` (static)

Element-wise branch selection.

Parameters:
- `condition`: `BOOL` tensor
- `ifTrue`: numeric tensor used where condition is true
- `ifFalse`: numeric tensor used where condition is false

Returns:
- numeric tensor with common broadcasted output shape and promoted branch dtype

Behavior:
- `condition`, `ifTrue`, and `ifFalse` are broadcast to a common output shape
- output dtype is the promoted numeric dtype of `ifTrue` and `ifFalse`
- gradient flows only through selected data branches
- `condition` itself has no gradient

Example:
```java
Tensor mask = new Tensor(new byte[]{1, 0, 1}, new int[]{3}, null, "mask", DataType.BOOL);
Tensor x = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "x");
Tensor yFallback = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "yFallback");

Tensor y = Tensor.where(mask, x, yFallback);
// y has shape [3] and values [10, 2, 30].

Tensor cap = new Tensor(new double[]{15}, new int[]{1}, null, "cap");
Tensor clipped = Tensor.where(x.greaterThan(cap), cap, x);
// clipped has shape [3] and values [10, 15, 15].
//
// Returns: numeric tensors with the promoted branch dtype.
```

### `minimum(Tensor second)`

Piecewise minimum surface built on compare/select semantics.

Parameters:
- `second`: right-hand numeric operand

Returns:
- numeric tensor selecting the smaller value element-wise

Behavior:
- effectively `where(this < second, this, second)`
- uses where-style branch semantics on ties
- this is intentionally different from the specialized `min(Tensor second)` op, which keeps its own tie-gradient contract

Example:
```java
Tensor a = new Tensor(new double[]{1, 5, 3}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{2, 4, 3}, new int[]{3}, null, "b");

Tensor y = a.minimum(b);
// y has shape [3] and values [1, 4, 3].
// On the tie at the last element, the where-style branch selects the second operand.
//
// Returns: a compare/select-based numeric minimum tensor.
```

### `maximum(Tensor second)`

Piecewise maximum surface built on compare/select semantics.

Parameters:
- `second`: right-hand numeric operand

Returns:
- numeric tensor selecting the larger value element-wise

Behavior:
- effectively `where(this > second, this, second)`
- uses where-style branch semantics on ties
- this is intentionally different from the specialized `max(Tensor second)` op, which keeps its own tie-gradient contract

Example:
```java
Tensor a = new Tensor(new double[]{1, 5, 3}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{2, 4, 3}, new int[]{3}, null, "b");

Tensor y = a.maximum(b);
// y has shape [3] and values [2, 5, 3].
// On the tie at the last element, the where-style branch selects the second operand.
//
// Returns: a compare/select-based numeric maximum tensor.
```

## Indexing Operations

### `select(int dimension, int index)`

Selects one slice along a chosen axis.

Parameters:
- `dimension`: axis to select from
- `index`: position inside that axis; negative indexing is accepted and counts from the end

Returns:
- tensor whose shape equals the input shape with the selected axis removed

Behavior:
- semantically this is single-index read indexing
- implementation is a true alias view:
  - shape/stride metadata is rewritten
  - backing storage is shared
  - selected slice position is represented through `storageOffset`
- this makes it the natural ergonomic companion to:
  - `takeAlongAxis` for axis-wise materialized read indexing
  - `scatterAdd` for write/update indexing

Example:
```java
Tensor x = new Tensor(new double[]{
        1, 2, 3,
        4, 5, 6
}, new int[]{2, 3}, null, "x");
Tensor y = x.select(1, 2);
// x has shape [2, 3]:
// [[1, 2, 3],
//  [4, 5, 6]]
//
// Selecting axis 1, index 2 picks the last column.
// y has shape [2] and values [3, 6].
//
// Returns: a tensor containing one slice extracted along the chosen axis.
```

### `gather(Tensor indices, int dimension)`

Gathers one value per logical sample position along the chosen axis.

Parameters:
- `indices`: tensor whose shape equals the input shape with the gathered axis removed
- `dimension`: axis to gather from

Returns:
- tensor whose shape equals `indices.shape`

Behavior:
- `INT32` and `INT64` are supported index dtypes
- numeric floating tensors with integral values are still accepted as a compatibility mode
- this surface is intentionally narrow and acts as the minimal gather primitive for index-style losses and related indexing contracts
- backward scatters upstream gradient back into the selected input positions

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor indices = new Tensor(new double[]{2, 0}, new int[]{2}, null, "indices");
Tensor y = x.gather(indices, 1);
// y has shape [2] and values [3, 4].
//
// Returns: a gathered tensor with one selected value per row.
```

### `gatherAxis(Tensor indices, int axis)` / `take(int axis, ...)`

ONNX-style axis gather. The index tensor shape is inserted at the gathered axis.

Signatures:
```java
Tensor gatherAxis(Tensor indices, int axis)
Tensor take(int axis, Tensor indices)
Tensor take(int axis, int[] indices)
```

Parameters:
- `indices`: numeric integral indices
- `axis`: source axis

Returns:
- tensor with shape `dataShape[:axis] + indicesShape + dataShape[axis + 1:]`

Example:
```java
Tensor x = new Tensor(new double[]{
        1, 2, 3, 4,
        5, 6, 7, 8
}, new int[]{2, 4}, null, "x");

Tensor y = x.take(1, new int[]{3, 1});
// y shape = [2, 2]
// y values = [[4, 2], [8, 6]]
```

### `takeAlongAxis(Tensor indices, int dimension)`

Gathers values along one axis while preserving rank.

Parameters:
- `indices`: tensor with the same rank as the input; all non-axis dimensions must match the input
- `dimension`: axis to index along

Returns:
- tensor whose shape equals `indices.shape`

Behavior:
- unlike narrow `gather`, this op keeps rank
- the selected axis length is driven by `indices.shape[dimension]`
- backward scatters upstream gradient back into the selected input positions

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor indices = new Tensor(new int[]{2, 1, 0, 0}, new int[]{2, 2}, null, "indices", DataType.INT32);
Tensor y = x.takeAlongAxis(indices, 1);
// y has shape [2, 2] and values:
// [[3, 2],
//  [4, 4]]
//
// Returns: an axis-wise gathered tensor with the same rank as indices.
```

### `scatterAdd(Tensor indices, Tensor src, int dimension)`

Adds source values into positions selected by indices along one axis.

Parameters:
- `indices`: tensor whose shape equals the base tensor shape with the scattered axis removed
- `src`: tensor with the same shape as `indices`
- `dimension`: axis to scatter into

Returns:
- tensor with the same shape as the base tensor

Behavior:
- starts from the base tensor values
- adds `src` contributions into positions selected by `indices`
- is the write/update-side companion to `gather`

Example:
```java
Tensor base = new Tensor(new double[]{10, 20, 30, 40, 50, 60}, new int[]{2, 3}, null, "base");
Tensor indices = new Tensor(new double[]{2, 0}, new int[]{2}, null, "indices");
Tensor src = new Tensor(new double[]{1, 5}, new int[]{2}, null, "src");
Tensor y = base.scatterAdd(indices, src, 1);
// y has shape [2, 3] and values:
// [[10, 20, 31],
//  [45, 50, 60]]
//
// Returns: a tensor with scattered additions applied.
```

### `scatterElements(Tensor indices, Tensor updates, int axis[, ScatterReduction reduction])`

Writes same-rank updates into positions selected along one axis.

Parameters:
- `indices`: integral index tensor with the same rank as the base tensor
- `updates`: tensor with the same shape as `indices` and the same dtype as the base tensor
- `axis`: axis whose coordinate is read from `indices`
- `reduction`: optional write policy; defaults to `NONE`

Returns:
- tensor with the same shape as the base tensor

Behavior:
- starts from a copy of the base tensor
- `NONE` overwrites and rejects duplicate target elements
- `ADD`, `MUL`, `MAX`, and `MIN` reduce repeated writes
- backward is supported only for floating `NONE` and `ADD`

Example:
```java
Tensor data = new Tensor(new double[]{10, 20, 30, 40, 50, 60}, new int[]{2, 3}, null, "data");
Tensor indices = new Tensor(new int[]{2, 0, 0, 2}, new int[]{2, 2}, null, "indices", DataType.INT32);
Tensor updates = new Tensor(new double[]{1, 5, 7, 9}, new int[]{2, 2}, null, "updates");
Tensor y = data.scatterElements(indices, updates, 1);
// y has shape [2, 3] and values:
// [[5, 20, 1],
//  [7, 50, 9]]
```

### `gatherNd(Tensor indices[, int batchDims])`

Gathers values or slices using tuple indices.

Parameters:
- `indices`: integral index tensor; its final dimension is the coordinate tuple length
- `batchDims`: optional number of leading dimensions shared by input and indices; defaults to `0`

Returns:
- tensor with shape `indices.shape[:batchDims] + indices.shape[batchDims:-1] + input.shape[batchDims + indices.shape[-1]:]`

Behavior:
- leading batch dimensions select matching input/index slices and are not stored inside each coordinate tuple
- if the tuple length covers the non-batch input rank, each tuple reads one element; Synaptik uses its internal scalar shape `[1]` for the one-index scalar case
- if the tuple length is smaller than the non-batch input rank, each tuple reads a slice
- backward scatters upstream gradients back with duplicate-index accumulation

Example:
```java
Tensor data = new Tensor(new double[]{10, 20, 30, 40, 50, 60}, new int[]{2, 3}, null, "data");
Tensor indices = new Tensor(new int[]{0, 2, 1, 0}, new int[]{2, 2}, null, "indices", DataType.INT32);
Tensor y = data.gatherNd(indices);
// y has shape [2] and values [30, 40].
```

```java
Tensor batched = new Tensor(new double[12], new int[]{2, 3, 2}, null, "batched");
Tensor idx = new Tensor(new int[]{2, 0, 1, 0}, new int[]{2, 2, 1}, null, "idx", DataType.INT32);
Tensor y = batched.gatherNd(idx, 1);
// y has shape [2, 2, 2]; y[b, i, :] = batched[b, idx[b, i, 0], :].
```

### `scatterNd(Tensor indices, Tensor updates[, ScatterReduction reduction])`

Writes updates into tuple-indexed positions.

Parameters:
- `indices`: integral index tensor; its final dimension is the coordinate tuple length
- `updates`: tensor with shape `indices.shape[:-1] + base.shape[indices.shape[-1]:]`
- `reduction`: optional write policy; defaults to `NONE`

Returns:
- tensor with the same shape as the base tensor

Behavior:
- starts from a copy of the base tensor
- if the tuple length equals the base rank, each update writes one element; Synaptik also accepts its internal scalar shape `[1]` for the one-index case
- if the tuple length is smaller than the base rank, each update writes a slice
- backward is supported for floating `NONE` and `ADD`; `MUL`, `MAX`, and `MIN` remain inference-only

Example:
```java
Tensor data = new Tensor(new double[]{10, 20, 30, 40, 50, 60}, new int[]{2, 3}, null, "data");
Tensor indices = new Tensor(new int[]{0, 2, 1, 0}, new int[]{2, 2}, null, "indices", DataType.INT32);
Tensor updates = new Tensor(new double[]{1, 7}, new int[]{2}, null, "updates");
Tensor y = data.scatterNd(indices, updates);
// y has shape [2, 3] and values:
// [[10, 20, 1],
//  [7, 50, 60]]
```

## Logical Bool Operations

Logical bool ops:
- require `BOOL` tensors
- return `BOOL` tensors
- are nondifferentiable

### `logicalAnd(Tensor second)`

Element-wise logical conjunction.

Parameters:
- `second`: right-hand `BOOL` operand

Returns:
- `BOOL` tensor with `true` where both operands are true

Example:
```java
Tensor a = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "a", DataType.BOOL);
Tensor b = new Tensor(new byte[]{1, 1, 0, 0}, new int[]{2, 2}, null, "b", DataType.BOOL);

Tensor mask = a.logicalAnd(b);
// mask has shape [2, 2] and values:
// [[true,  false],
//  [false, false]]
//
// Returns: a BOOL tensor.
```

### `logicalOr(Tensor second)`

Element-wise logical disjunction.

Parameters:
- `second`: right-hand `BOOL` operand

Returns:
- `BOOL` tensor with `true` where at least one operand is true

Example:
```java
Tensor a = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "a", DataType.BOOL);
Tensor b = new Tensor(new byte[]{1, 1, 0, 0}, new int[]{2, 2}, null, "b", DataType.BOOL);

Tensor mask = a.logicalOr(b);
// mask has shape [2, 2] and values:
// [[true,  true],
//  [true,  false]]
//
// Returns: a BOOL tensor.
```

### `logicalNot()`

Element-wise logical negation.

Parameters:
- none

Returns:
- `BOOL` tensor with inverted logical values

Example:
```java
Tensor mask = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "mask", DataType.BOOL);

Tensor inverted = mask.logicalNot();
// inverted has shape [2, 2] and values:
// [[false, true],
//  [false, true]]
//
// Returns: a BOOL tensor.
```

## Unary / Scalar Operations

### `relu()`

Element-wise rectified linear unit.

Parameters:
- none

Returns:
- tensor of `max(x, 0)`

Example:
```java
Tensor x = new Tensor(new double[]{-2, 0, 3}, new int[]{3}, null, "x");
Tensor y = x.relu();
// y has shape [3] and values [0, 0, 3].
//
// Returns: a tensor after element-wise ReLU.
```

### `abs()`

Element-wise absolute value.

Parameters:
- none

Returns:
- tensor of `|x|`

Behavior:
- uses the specialized `ABS` primitive
- backward uses `sign(x)` with gradient `0` at `x == 0`

Example:
```java
Tensor x = new Tensor(new double[]{-3.0, -0.5, 0.0, 2.0}, new int[]{4}, null, "x");
Tensor y = x.abs();
// y has shape [4] and values [3.0, 0.5, 0.0, 2.0].
//
// Returns: a tensor after element-wise absolute value.
```

### `neg()`

Arithmetic negation.

Parameters:
- none

Returns:
- tensor with all elements multiplied by `-1`

Example:
```java
Tensor x = new Tensor(new double[]{1, -2, 3}, new int[]{3}, null, "x");
Tensor y = x.neg();
// y has shape [3] and values [-1, 2, -3].
//
// Returns: a tensor with all values multiplied by -1.
```

### `log()`

Natural logarithm applied element-wise.

Parameters:
- none

Returns:
- tensor of `ln(x)` values

Example:
```java
Tensor x = new Tensor(new double[]{1, Math.E, Math.E * Math.E}, new int[]{3}, null, "x");
Tensor y = x.log();
// y has shape [3] and values [0, 1, 2].
//
// Returns: a tensor of ln(x) values.
```

### `exp()`

Natural exponential applied element-wise.

Parameters:
- none

Returns:
- tensor of `e^x` values

Example:
```java
Tensor x = new Tensor(new double[]{0, 1, 2}, new int[]{3}, null, "x");
Tensor y = x.exp();
// y has shape [3] and values [1, e, e^2].
//
// Returns: a tensor of exp(x) values.
```

### `fastExp()`

Approximation-oriented exponential op.

Parameters:
- none

Returns:
- tensor with fast approximate `exp`

Example:
```java
Tensor x = new Tensor(new double[]{0, 1, 2}, new int[]{3}, null, "x");
Tensor y = x.fastExp();
// y has shape [3] and returns an approximate exp(x) result.
//
// Returns: a tensor of approximate exp(x) values.
```

### `tanh()`

Hyperbolic tangent applied element-wise.

Parameters:
- none

Returns:
- tensor of `tanh(x)` values

Example:
```java
Tensor x = new Tensor(new double[]{-1, 0, 1}, new int[]{3}, null, "x");
Tensor y = x.tanh();
// y has shape [3] and values [tanh(-1), 0, tanh(1)].
//
// Returns: a tensor of tanh(x) values.
```

### `fastTanh()`

Approximation-oriented hyperbolic tangent op.

Parameters:
- none

Returns:
- tensor with fast approximate `tanh`

Example:
```java
Tensor x = new Tensor(new double[]{-1, 0, 1}, new int[]{3}, null, "x");
Tensor y = x.fastTanh();
// y has shape [3] and returns an approximate tanh(x) result.
//
// Returns: a tensor of approximate tanh(x) values.
```

### `pow(double exponent)`

Raises each element to a scalar exponent.

Parameters:
- `exponent`: scalar exponent

Returns:
- tensor of `x^exponent`

Example:
```java
Tensor x = new Tensor(new double[]{4, 9, 16}, new int[]{3}, null, "x");
Tensor y = x.pow(2.0);
// y has values [16, 81, 256].
Tensor z = x.pow(0.5);
// z has values [2, 3, 4].
//
// Returns: tensors of x^exponent.
```

### `mul(double scalar)`

Multiplies each element by a scalar.

Parameters:
- `scalar`: scalar multiplier

Returns:
- scaled tensor

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "x");
Tensor y = x.mul(0.5);
// y has values [0.5, 1.0, 1.5].
Tensor z = x.mul(2.0);
// z has values [2, 4, 6].
//
// Returns: scaled tensors.
```

### `inv()`

Element-wise reciprocal.

Parameters:
- none

Returns:
- tensor of `1 / x`

Example:
```java
Tensor x = new Tensor(new double[]{2, 4, 8}, new int[]{3}, null, "x");
Tensor y = x.inv();
// y has values [0.5, 0.25, 0.125].
//
// Returns: a tensor of 1 / x values.
```

### `sqrt()`

Element-wise square root.

Parameters:
- none

Returns:
- tensor of `sqrt(x)`

Example:
```java
Tensor x = new Tensor(new double[]{4, 9, 16}, new int[]{3}, null, "x");
Tensor y = x.sqrt();
// y has values [2, 3, 4].
//
// Returns: a tensor of sqrt(x) values.
```

### `sigmoid()`

Element-wise logistic sigmoid.

Parameters:
- none

Returns:
- tensor of `1 / (1 + exp(-x))`

Example:
```java
Tensor logits = new Tensor(new double[]{-2, 0, 2}, new int[]{3}, null, "logits");
Tensor y = logits.sigmoid();
// y has values [sigmoid(-2), 0.5, sigmoid(2)].
//
// Returns: a tensor of sigmoid(logits) values.
```

### `clamp(double minValue, double maxValue)`

Clamps every numeric element into the closed interval `[minValue, maxValue]`.

Parameters:
- `minValue`: lower bound
- `maxValue`: upper bound

Returns:
- clamped numeric tensor

Behavior:
- values below `minValue` become `minValue`
- values above `maxValue` become `maxValue`
- values already inside the interval stay unchanged
- implemented as composition over specialized `clampMax(...)` and `clampMin(...)`

Example:
```java
Tensor x = new Tensor(new double[]{-2.0, -0.5, 0.5, 3.0}, new int[]{4}, null, "x");
Tensor y = x.clamp(0.0, 1.0);
// y has values [0.0, 0.0, 0.5, 1.0].
//
// Returns: a tensor whose values are limited to [0.0, 1.0].
```

### `clampMin(double minValue)`

Raises only values below the lower bound.

Parameters:
- `minValue`: lower bound

Returns:
- numeric tensor where values below `minValue` are replaced by `minValue`

Notes:
- uses the specialized one-sided `CLAMP_MIN` primitive

Example:
```java
Tensor x = new Tensor(new double[]{-2.0, -0.5, 0.5, 3.0}, new int[]{4}, null, "x");
Tensor y = x.clampMin(0.0);
// y has values [0.0, 0.0, 0.5, 3.0].
//
// Returns: a tensor clamped only from below.
```

### `clampMax(double maxValue)`

Lowers only values above the upper bound.

Parameters:
- `maxValue`: upper bound

Returns:
- numeric tensor where values above `maxValue` are replaced by `maxValue`

Notes:
- uses the specialized one-sided `CLAMP_MAX` primitive

Example:
```java
Tensor x = new Tensor(new double[]{-2.0, -0.5, 0.5, 3.0}, new int[]{4}, null, "x");
Tensor y = x.clampMax(1.0);
// y has values [-2.0, -0.5, 0.5, 1.0].
//
// Returns: a tensor clamped only from above.
```

## Reduction Operations

Reductions:
- reduce one axis or all elements
- optionally preserve the reduced axis via `keepDims`

### `sum()`

Reduces all elements to a tensor of shape `[1]`.

Parameters:
- none

Returns:
- scalar-shaped sum tensor

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "x");
Tensor y = x.sum();
// y has shape [1] and value [10].
//
// Returns: a scalar-shaped sum tensor.
```

### `sum(int dimension)`

Reduces one axis and removes it from the output shape.

Parameters:
- `dimension`: axis to reduce

Returns:
- reduced tensor with one fewer dimension

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor y = x.sum(1);
// y has shape [2] and values [6, 15].
//
// Returns: an axis-reduced tensor.
```

### `sum(int dimension, boolean keepDims)`

Reduces one axis and optionally preserves it as size `1`.

Parameters:
- `dimension`: axis to reduce
- `keepDims`: whether to keep the reduced axis

Returns:
- reduced tensor

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor y = x.sum(1, true);
// y has shape [2, 1] and values [6, 15].
Tensor z = x.sum(0, false);
// z has shape [3] and values [5, 7, 9].
//
// Returns: reduced tensors.
```

### `mean()`

Reduces all elements to shape `[1]` and divides by element count.

Parameters:
- none

Returns:
- scalar-shaped mean tensor

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "x");
Tensor y = x.mean();
// y has shape [1] and value [2.5].
//
// Returns: a scalar-shaped mean tensor.
```

### `mean(int dimension)`

Reduces one axis by arithmetic mean.

Parameters:
- `dimension`: axis to reduce

Returns:
- reduced mean tensor

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor y = x.mean(1);
// y has shape [2] and values [2.0, 5.0].
//
// Returns: an axis-reduced mean tensor.
```

### `mean(int dimension, boolean keepDims)`

Axis mean with optional reduced-axis preservation.

Parameters:
- `dimension`: axis to reduce
- `keepDims`: whether to keep the reduced axis as size `1`

Returns:
- reduced mean tensor

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor y = x.mean(1, true);
// y has shape [2, 1] and values [2.0, 5.0].
//
// Returns: a reduced mean tensor.
```

### `sum(int dimension, Tensor mask)` / `mean(int dimension, Tensor mask)`

Masked reductions ignore positions where a BOOL mask is false.

Parameters:
- `dimension`: axis to reduce
- `mask`: BOOL mask broadcastable to the input tensor

Returns:
- reduced tensor with masked-out positions excluded

Behavior:
- `sum` treats masked-out values as zero
- `mean` divides by the number of valid positions, not by the full padded axis length
- if every position for an output element is masked out, the denominator is clamped to `1`, so the result is `0`

Example:
```java
Tensor x = new Tensor(new double[]{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12
}, new int[]{2, 3, 2}, null, "x");
Tensor mask = new Tensor(new byte[]{
        1, 1, 0,
        1, 0, 0
}, new int[]{2, 3}, null, "mask", DataType.BOOL);

Tensor mean = x.mean(1, mask);
// x shape = [batch, time, features] = [2, 3, 2]
// mask shape = [batch, time] = [2, 3]
// mean shape = [2, 2]
// mean values = [[2, 3], [7, 8]]
```

### `softmax(int dimension)`

Applies numerically stable softmax along one axis.

Parameters:
- `dimension`: axis to normalize

Returns:
- tensor of the same shape where values along the chosen axis sum to `1`

Behavior:
- subtracts the per-axis maximum before exponentiation for numerical stability
- keeps the original shape
- builds a canonical primitive DAG: max-shift, exponentiation, sum, and division

Example:
```java
Tensor logits = new Tensor(new double[]{1, 2, 3, 0, 0, 0}, new int[]{2, 3}, null, "logits");
Tensor probs = logits.softmax(1);
// probs has shape [2, 3].
// Row 0 becomes softmax([1, 2, 3]).
// Row 1 becomes [1/3, 1/3, 1/3].
//
// Returns: an axis-normalized probability tensor.
```

### `logSoftmax(int dimension)`

Applies numerically stable log-softmax along one axis.

Parameters:
- `dimension`: axis to normalize

Returns:
- tensor of the same shape containing log-probabilities along the chosen axis

Behavior:
- computes `x - logsumexp(x)` along the selected axis
- keeps the original shape
- builds a canonical primitive DAG: max-shift, exponentiation, sum, log, and subtraction

Example:
```java
Tensor logits = new Tensor(new double[]{1, 2, 3, 0, 0, 0}, new int[]{2, 3}, null, "logits");
Tensor logProbs = logits.logSoftmax(1);
// logProbs has shape [2, 3].
// exp(logProbs) matches logits.softmax(1).
//
// Returns: an axis-normalized log-probability tensor.
```

### `min()`

Reduces all elements to the global minimum.

Parameters:
- none

Returns:
- scalar-shaped minimum tensor

Example:
```java
Tensor x = new Tensor(new double[]{4, 1, 9, 2}, new int[]{4}, null, "x");
Tensor y = x.min();
// y has shape [1] and value [1].
//
// Returns: a scalar-shaped minimum tensor.
```

### `min(int dimension)`

Reduces one axis by minimum.

Parameters:
- `dimension`: axis to reduce

Returns:
- reduced minimum tensor

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor y = x.min(1);
// y has shape [2] and values [1, 4].
//
// Returns: an axis-reduced minimum tensor.
```

### `min(int dimension, boolean keepDims)`

Axis minimum with optional reduced-axis preservation.

Parameters:
- `dimension`: axis to reduce
- `keepDims`: whether to keep the reduced axis as size `1`

Returns:
- reduced minimum tensor

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor y = x.min(1, true);
// y has shape [2, 1] and values [1, 4].
//
// Returns: a reduced minimum tensor.
```

### `max()`

Reduces all elements to the global maximum.

Parameters:
- none

Returns:
- scalar-shaped maximum tensor

Example:
```java
Tensor x = new Tensor(new double[]{4, 1, 9, 2}, new int[]{4}, null, "x");
Tensor y = x.max();
// y has shape [1] and value [9].
//
// Returns: a scalar-shaped maximum tensor.
```

### `max(int dimension)`

Reduces one axis by maximum.

Parameters:
- `dimension`: axis to reduce

Returns:
- reduced maximum tensor

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor y = x.max(1);
// y has shape [2] and values [3, 6].
//
// Returns: an axis-reduced maximum tensor.
```

### `max(int dimension, boolean keepDims)`

Axis maximum with optional reduced-axis preservation.

Parameters:
- `dimension`: axis to reduce
- `keepDims`: whether to keep the reduced axis as size `1`

Returns:
- reduced maximum tensor

Example:
```java
Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x");
Tensor y = x.max(1, true);
// y has shape [2, 1] and values [3, 6].
//
// Returns: a reduced maximum tensor.
```

### `all()`

Reduces all elements of a `BOOL` tensor with logical conjunction.

Parameters:
- none

Returns:
- scalar-shaped `BOOL` tensor that is `true` only if all elements are `true`

Example:
```java
Tensor mask = new Tensor(new byte[]{1, 1, 0, 1}, new int[]{4}, null, "mask", DataType.BOOL);
Tensor y = mask.all();
// y has shape [1] and value [false].
//
// Returns: a scalar-shaped BOOL tensor.
```

### `all(int dimension)`

Reduces one axis of a `BOOL` tensor with logical conjunction.

Parameters:
- `dimension`: axis to reduce

Returns:
- reduced `BOOL` tensor

Example:
```java
Tensor mask = new Tensor(new byte[]{1, 1, 0, 1, 1, 1}, new int[]{2, 3}, null, "mask", DataType.BOOL);
Tensor y = mask.all(1);
// y has shape [2] and values [false, true].
//
// Returns: an axis-reduced BOOL tensor.
```

### `all(int dimension, boolean keepDims)`

Reduces one axis of a `BOOL` tensor with logical conjunction and optional axis preservation.

Parameters:
- `dimension`: axis to reduce
- `keepDims`: whether to keep the reduced axis as size `1`

Returns:
- reduced `BOOL` tensor

Example:
```java
Tensor mask = new Tensor(new byte[]{1, 1, 0, 1, 1, 1}, new int[]{2, 3}, null, "mask", DataType.BOOL);
Tensor y = mask.all(1, true);
// y has shape [2, 1] and values [false, true].
//
// Returns: a reduced BOOL tensor.
```

### `any()`

Reduces all elements of a `BOOL` tensor with logical disjunction.

Parameters:
- none

Returns:
- scalar-shaped `BOOL` tensor that is `true` if any element is `true`

Example:
```java
Tensor mask = new Tensor(new byte[]{0, 0, 0, 1}, new int[]{4}, null, "mask", DataType.BOOL);
Tensor y = mask.any();
// y has shape [1] and value [true].
//
// Returns: a scalar-shaped BOOL tensor.
```

### `any(int dimension)`

Reduces one axis of a `BOOL` tensor with logical disjunction.

Parameters:
- `dimension`: axis to reduce

Returns:
- reduced `BOOL` tensor

Example:
```java
Tensor mask = new Tensor(new byte[]{1, 1, 0, 1, 1, 1}, new int[]{2, 3}, null, "mask", DataType.BOOL);
Tensor y = mask.any(1);
// y has shape [2] and values [true, true].
//
// Returns: an axis-reduced BOOL tensor.
```

### `any(int dimension, boolean keepDims)`

Reduces one axis of a `BOOL` tensor with logical disjunction and optional axis preservation.

Parameters:
- `dimension`: axis to reduce
- `keepDims`: whether to keep the reduced axis as size `1`

Returns:
- reduced `BOOL` tensor

Example:
```java
Tensor mask = new Tensor(new byte[]{1, 1, 0, 1, 1, 1}, new int[]{2, 3}, null, "mask", DataType.BOOL);
Tensor y = mask.any(1, true);
// y has shape [2, 1] and values [true, true].
//
// Returns: a reduced BOOL tensor.
```

Reduction gradient notes:
- `sum` and `mean` broadcast gradients back to input shape
- `mean` additionally scales by reduced-size reciprocal
- `min` and `max` route gradients only to winning elements
- ties split gradient evenly across winners
- `all` and `any` are nondifferentiable

## Loss / N-ary Operations

### `nllLoss(Tensor targets, int classDimension)`

Computes mean negative log-likelihood from log-probabilities and same-shape target distributions.

Parameters:
- `targets`: tensor with the same shape as `logProbs`; typically one-hot labels or a target distribution
- `classDimension`: axis containing class probabilities

Returns:
- scalar-shaped loss tensor

Behavior:
- expects the source tensor to contain log-probabilities, typically from `logSoftmax(...)`
- reduces over the class axis and then averages over all remaining sample positions
- uses the dedicated `NLL_LOSS` primitive

Example:
```java
Tensor logProbs = logits.logSoftmax(1);
Tensor targets = new Tensor(new double[]{
        0, 0, 1,
        1, 0, 0
}, new int[]{2, 3}, null, "targets");
Tensor loss = logProbs.nllLoss(targets, 1);
// loss has shape [1] and contains the mean negative log-likelihood
// for the two samples in the batch.
//
// Returns: a scalar mean NLL loss tensor.
```

### `crossEntropyLoss(Tensor targets, int classDimension)`

Computes mean cross-entropy loss directly from logits.

Parameters:
- `targets`: tensor with the same shape as `logits`; typically one-hot labels or a target distribution
- `classDimension`: axis containing class logits

Returns:
- scalar-shaped loss tensor

Behavior:
- this is the logits-facing ergonomic surface
- semantic reference is `logits.logSoftmax(classDimension).nllLoss(targets, classDimension)`
- runtime uses the dedicated `CROSS_ENTROPY_LOSS` primitive for the dense-target mean-reduction contract described here

Example:
```java
Tensor loss = logits.crossEntropyLoss(targets, 1);
// Equivalent to:
// Tensor loss = logits.logSoftmax(1).nllLoss(targets, 1);
//
// Returns: a scalar mean cross-entropy loss tensor.
```

### `crossEntropyLoss(Tensor targets, int classDimension, Tensor mask)`

Computes dense-target cross-entropy from logits while ignoring masked-out samples.

Parameters:
- `targets`: tensor with the same shape as `logits`
- `classDimension`: axis containing class logits
- `mask`: BOOL mask broadcastable to `logits.shape` with the class axis removed

Returns:
- scalar-shaped mean loss normalized by the number of valid mask positions

Example:
```java
Tensor logits = Tensor.randn(new int[]{2, 3, 10}, 0.0, 1.0, DataType.FLOAT64, "logits");
Tensor targets = Tensor.zeros(new int[]{2, 3, 10}, DataType.FLOAT64, "targets");
Tensor mask = new Tensor(new byte[]{
        1, 1, 0,
        1, 0, 0
}, new int[]{2, 3}, null, "mask", DataType.BOOL);

Tensor loss = logits.crossEntropyLoss(targets, 2, mask);
// classDimension = 2, so mask shape [2, 3] matches the sample shape.
```

### `nllLossFromIndices(Tensor targetIndices, int classDimension)`

Computes mean negative log-likelihood from log-probabilities and class-id targets.

Parameters:
- `targetIndices`: tensor whose shape equals `logProbs.shape` without the class axis
- `classDimension`: axis containing class log-probabilities

Returns:
- scalar-shaped loss tensor

Behavior:
- implementation is composition-first:
  - `logProbs.gather(targetIndices, classDimension).neg().mean()`
- `INT32` and `INT64` are supported target-index dtypes
- this is the first index-target loss surface built on top of the new indexing primitive

Example:
```java
Tensor logProbs = logits.logSoftmax(1);
Tensor targetIndices = new Tensor(new double[]{2, 0}, new int[]{2}, null, "targetIndices");
Tensor loss = logProbs.nllLossFromIndices(targetIndices, 1);
// Returns: a scalar mean NLL loss for class-id targets.
```

### `nllLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex)`

Computes mean negative log-likelihood from log-probabilities and class-id targets, while ignoring one target id.

Parameters:
- `targetIndices`: tensor whose shape equals `logProbs.shape` without the class axis
- `classDimension`: axis containing class log-probabilities
- `ignoreIndex`: target id that should be excluded from loss and gradient

Returns:
- scalar-shaped loss tensor

Behavior:
- ignored samples do not contribute to loss
- ignored samples have zero gradient
- `mean` denominator uses only non-ignored samples
- if all samples are ignored, the result is zero

Example:
```java
Tensor loss = logProbs.nllLossFromIndices(targetIndices, 1, -1);
// Returns: a scalar mean NLL loss over non-ignored class-id targets.
```

### `nllLossFromIndices(Tensor targetIndices, int classDimension, LossReduction reduction)`

Computes index-target NLL with explicit reduction mode.

Parameters:
- `targetIndices`: tensor whose shape equals `logProbs.shape` without the class axis
- `classDimension`: axis containing class log-probabilities
- `reduction`: `MEAN`, `SUM`, or `NONE`

Returns:
- scalar tensor for `MEAN` / `SUM`
- unreduced per-sample tensor for `NONE`

Example:
```java
Tensor perSample = logProbs.nllLossFromIndices(targetIndices, 1, LossReduction.NONE);
// perSample shape equals targetIndices.shape.
```

### `nllLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction)`

Computes index-target NLL with both `ignoreIndex` and explicit reduction mode.

Parameters:
- `targetIndices`: tensor whose shape equals `logProbs.shape` without the class axis
- `classDimension`: axis containing class log-probabilities
- `ignoreIndex`: target id that should be excluded
- `reduction`: `MEAN`, `SUM`, or `NONE`

Returns:
- scalar tensor for `MEAN` / `SUM`
- unreduced per-sample tensor for `NONE`

Example:
```java
Tensor loss = logProbs.nllLossFromIndices(targetIndices, 1, -1, LossReduction.SUM);
```

### `nllLossFromIndices(Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction)`

Computes weighted index-target NLL with explicit reduction mode.

Parameters:
- `targetIndices`: tensor whose shape equals `logProbs.shape` without the class axis
- `classDimension`: axis containing class log-probabilities
- `classWeights`: rank-1 tensor with shape `[numClasses]`
- `reduction`: `MEAN`, `SUM`, or `NONE`

Returns:
- scalar tensor for `MEAN` / `SUM`
- weighted unreduced per-sample tensor for `NONE`

Behavior:
- per-sample loss is multiplied by the weight of the selected target class
- class weight lookup is built through tensor indexing composition:
  - `reshape -> expand -> takeAlongAxis -> squeeze`
- `MEAN` uses weighted mean:
  - `sum(weightedLoss) / sum(appliedWeights)`

Example:
```java
Tensor logProbs = logits.logSoftmax(1);
Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "targetIndices", DataType.INT32);
Tensor classWeights = new Tensor(new double[]{0.5, 1.0, 2.0}, new int[]{3}, null, "classWeights");
Tensor loss = logProbs.nllLossFromIndices(targetIndices, 1, classWeights, LossReduction.MEAN);
// The first sample uses weight 2.0, the second sample uses weight 0.5.
// Per-sample losses are weighted before reduction.
// Returns: a scalar weighted mean NLL loss.
```

### `nllLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction)`

Computes weighted index-target NLL with both `ignoreIndex` and explicit reduction mode.

Parameters:
- `targetIndices`: tensor whose shape equals `logProbs.shape` without the class axis
- `classDimension`: axis containing class log-probabilities
- `ignoreIndex`: target id that should be excluded
- `classWeights`: rank-1 tensor with shape `[numClasses]`
- `reduction`: `MEAN`, `SUM`, or `NONE`

Returns:
- scalar tensor for `MEAN` / `SUM`
- weighted unreduced per-sample tensor for `NONE`

Behavior:
- ignored samples contribute neither weighted loss nor weight denominator
- `MEAN` divides by the sum of weights of non-ignored samples

### `crossEntropyLossFromIndices(Tensor targetIndices, int classDimension)`

Computes mean cross-entropy loss directly from logits and class-id targets.

Parameters:
- `targetIndices`: tensor whose shape equals `logits.shape` without the class axis
- `classDimension`: axis containing class logits

Returns:
- scalar-shaped loss tensor

Behavior:
- implementation is composition-first:
  - `logits.logSoftmax(classDimension).nllLossFromIndices(targetIndices, classDimension)`
- this is the logits-facing ergonomic surface for class-id targets

Example:
```java
Tensor loss = logits.crossEntropyLossFromIndices(targetIndices, 1);
// Equivalent to:
// Tensor loss = logits.logSoftmax(1).nllLossFromIndices(targetIndices, 1);
//
// Returns: a scalar mean cross-entropy loss tensor for class-id targets.
```

### `crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex)`

Computes mean cross-entropy loss from logits and class-id targets, while ignoring one target id.

Parameters:
- `targetIndices`: tensor whose shape equals `logits.shape` without the class axis
- `classDimension`: axis containing class logits
- `ignoreIndex`: target id that should be excluded from loss and gradient

Returns:
- scalar-shaped loss tensor

Behavior:
- implementation is composition-first:
  - `logits.logSoftmax(classDimension).nllLossFromIndices(targetIndices, classDimension, ignoreIndex)`
- ignored samples do not contribute to loss or gradient

Example:
```java
Tensor loss = logits.crossEntropyLossFromIndices(targetIndices, 1, -1);
// Returns: a scalar mean cross-entropy loss tensor over non-ignored class-id targets.
```

### `crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, LossReduction reduction)`

Computes logits-facing index-target cross-entropy with explicit reduction mode.

Parameters:
- `targetIndices`: tensor whose shape equals `logits.shape` without the class axis
- `classDimension`: axis containing class logits
- `reduction`: `MEAN`, `SUM`, or `NONE`

Returns:
- scalar tensor for `MEAN` / `SUM`
- unreduced per-sample tensor for `NONE`

Example:
```java
Tensor loss = logits.crossEntropyLossFromIndices(targetIndices, 1, LossReduction.NONE);
```

### `crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, LossReduction reduction)`

Computes logits-facing index-target cross-entropy with both `ignoreIndex` and explicit reduction mode.

Parameters:
- `targetIndices`: tensor whose shape equals `logits.shape` without the class axis
- `classDimension`: axis containing class logits
- `ignoreIndex`: target id that should be excluded
- `reduction`: `MEAN`, `SUM`, or `NONE`

Returns:
- scalar tensor for `MEAN` / `SUM`
- unreduced per-sample tensor for `NONE`

### `crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, Tensor classWeights, LossReduction reduction)`

Computes weighted logits-facing index-target cross-entropy with explicit reduction mode.

Parameters:
- `targetIndices`: tensor whose shape equals `logits.shape` without the class axis
- `classDimension`: axis containing class logits
- `classWeights`: rank-1 tensor with shape `[numClasses]`
- `reduction`: `MEAN`, `SUM`, or `NONE`

Returns:
- scalar tensor for `MEAN` / `SUM`
- weighted unreduced per-sample tensor for `NONE`

Behavior:
- equivalent to `logits.logSoftmax(classDimension)` followed by weighted `nllLossFromIndices(...)`
- class weights are selected per sample from the target class id

### `crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex, Tensor classWeights, LossReduction reduction)`

Computes weighted logits-facing index-target cross-entropy with both `ignoreIndex` and explicit reduction mode.

Parameters:
- `targetIndices`: tensor whose shape equals `logits.shape` without the class axis
- `classDimension`: axis containing class logits
- `ignoreIndex`: target id that should be excluded
- `classWeights`: rank-1 tensor with shape `[numClasses]`
- `reduction`: `MEAN`, `SUM`, or `NONE`

Returns:
- scalar tensor for `MEAN` / `SUM`
- weighted unreduced per-sample tensor for `NONE`

Behavior:
- ignored samples contribute neither to weighted loss nor to the weighted `MEAN` denominator

## Execution Anchor / Autodiff Helpers

### `forwardOutput()`

Creates a forward anchor node used as the execution sink of the graph rooted at this tensor.

Parameters:
- none

Returns:
- tensor that aliases this tensor as forward output anchor

Example:
```java
// Creates the explicit forward sink node for y.
Tensor out = y.forwardOutput();
// Returns: a tensor used as the forward execution anchor.
```

Internal backward wiring is no longer part of the public `Tensor` API surface.
Runtime/rewrite/autograd infrastructure uses `tensor.TensorInternalAccess` for that plumbing.

## Metadata and Data Access

This section deliberately separates:

- safe inspection helpers that are fine in ordinary modeling/debug code
- low-level hooks used mainly by runtime, graph infrastructure, and tests

If you are building a model, you will usually only need the safe inspection helpers.
If you are implementing execution, layout, or rewrite infrastructure, the rest of this section is the relevant contract.

### Shape / layout inspection

#### `rank()`
Returns the number of logical axes.

Example:
```java
Tensor x = Tensor.randn(new int[]{2, 3, 4}, 0.0, 1.0, DataType.FLOAT64, "x");
int rank = x.rank();
// rank = 3
```

#### `size()`
Returns the logical element count. This is the product of all shape dimensions and is equivalent to `getFlatDataSize()`.

Example:
```java
Tensor x = Tensor.randn(new int[]{2, 3, 4}, 0.0, 1.0, DataType.FLOAT64, "x");
int elements = x.size();
// elements = 24
```

#### `lastDim()`
Returns the final logical dimension.

This is the common helper for last-dimension APIs such as N-D `linear`, where an input shape `[..., inFeatures]` must match weight shape `[inFeatures, outFeatures]`.

Example:
```java
Tensor x = Tensor.randn(new int[]{2, 3, 4}, 0.0, 1.0, DataType.FLOAT64, "x");
int features = x.lastDim();
// features = 4
```

#### `shapeEquals(int... shape)`
Compares this tensor's logical shape with an expected shape exactly.

Example:
```java
Tensor x = Tensor.randn(new int[]{2, 3, 4}, 0.0, 1.0, DataType.FLOAT64, "x");
boolean ok = x.shapeEquals(2, 3, 4);
// ok = true
```

#### `shapeCopy()`
Returns a defensive copy of the logical shape.

Use this in public or consumer code when the caller may store or mutate the array. Use `getShapeUnsafe()` only in low-level infrastructure that already owns layout invariants.

Example:
```java
Tensor x = Tensor.randn(new int[]{2, 3, 4}, 0.0, 1.0, DataType.FLOAT64, "x");
int[] copy = x.shapeCopy();
copy[0] = 99;
// x.shapeEquals(2, 3, 4) is still true
```

#### `getShape()`
Returns a defensive copy of the logical shape.

Example:
```java
int[] shape = x.getShape();
// Returns: a copy such as [2, 3, 4].
```

#### `getStrides()`
Returns a defensive copy of the logical strides.

#### `getStride(int index)`
Returns stride for one axis.

#### `getDimensionAt(int index)`
Returns size of one logical axis.

#### `getFlatDataSize()`
Returns the logical number of elements.

#### `isContiguous()`
Returns whether the tensor layout is dense contiguous.

#### `getFlatIndex(int[] indices)`
Maps logical coordinates to logical flat row-major index.

#### `getSpatialIndex(int index)`
Maps a logical flat row-major index back to logical coordinates.

### Low-level layout / storage view access

#### `getShapeUnsafe()`
Returns the internal shape reference.
Use this only in runtime, planner, backend, or other low-level infrastructure code.

#### `getStridesUnsafe()`
Returns the internal stride reference.
Use this only in low-level code that already owns layout invariants.

#### `getStorageOffsetUnsafe()`
Returns the base storage offset of this logical tensor view.
Primarily relevant for offset views such as `select(...)`.

#### `hasStorageOffset()`
Returns whether the tensor starts at a non-zero base storage offset.

### DType / storage access

#### `getDataType()`
Returns tensor dtype.

#### `setDataType(DataType dataType)`
Changes dtype within compatible numeric families by rebuilding storage from the current logical values.

Contract:
- `BOOL <-> numeric` implicit conversion is not supported
- `INT32/INT64 <-> other dtype` implicit conversion is not supported

#### `storageVersion()`
Returns the backing storage version counter.
This is mainly useful for runtime caches and mutation tracking.

#### `toFloat32ArrayCopy()`
Returns a logical row-major `float[]` copy for `FLOAT32` tensors.

#### `toFloat64ArrayCopy()`
Returns a logical row-major `double[]` copy for `FLOAT64` tensors.

#### `toBFloat16BitsArrayCopy()`
Returns a logical row-major `short[]` copy of raw BFLOAT16 bit patterns.

#### `toInt32ArrayCopy()`
Returns a logical row-major `int[]` copy for `INT32` tensors.

#### `toInt64ArrayCopy()`
Returns a logical row-major `long[]` copy for `INT64` tensors.

#### `toBoolByteArrayCopy()`
Returns a logical row-major `byte[]` copy for `BOOL` tensors.

Raw storage objects and mutable backing arrays are not public `Tensor` API.
Runtime, backend, and storage-level tests use `TensorInternalAccess`; model code should use logical copy/read methods.

### Data mutation / logical reads

#### `setData(double[] data)`
Replaces tensor contents using numeric `double[]` input.

#### `setData(float[] data)`
Replaces tensor contents using `float[]` input.

#### `setData(short[] data)`
Replaces tensor contents using raw `BFLOAT16` storage values.

#### `setData(byte[] data)`
Replaces tensor contents using raw `BOOL` storage values.

#### `setData(int[] data)`
Replaces tensor contents using raw `INT32` storage values.

#### `setData(long[] data)`
Replaces tensor contents using raw `INT64` storage values.

#### `setFloat32Data(float[] data)`
`FLOAT32`-only convenience setter.

#### `copyDataFrom(Tensor source)`
Copies typed tensor contents from another tensor of the same shape and dtype.
This is mainly an internal/runtime-oriented helper.

#### `getByFlatIndex(int index)`
Reads one logical element as `double`.

Example:
```java
double v = x.getByFlatIndex(3);
// Returns: the logical value at flat index 3.
```

#### `setDataAt(int flatIndex, double value)`
Writes one logical element for numeric tensors.
Broadcast views are not writable through this API.

#### `toDoubleArrayCopy()`
Returns logical tensor contents as `double[]`.

#### `toBooleanArrayCopy()`
Returns logical tensor contents as `boolean[]` for `BOOL` tensors.

#### `scalarAsDouble()`
Returns scalar value for numeric scalar tensors.

### Labels / autograd / graph plumbing

#### `getLabel()`
Returns tensor label.

#### `setLabel(String label)`
Sets tensor label.

#### `getRequiresGrad()`
Returns whether this tensor participates in gradient tracking.

#### `setRequiresGrad(boolean requiresGrad)`
Enables or disables gradient tracking.

#### `getGradient()`
Returns gradient tensor if present.

#### `getPrevTensors()`
Returns an unmodifiable view of predecessor tensors in the graph.

#### `getOperation()`
Returns operation descriptor for this node.

#### `isBackward()`
Returns whether this node is marked as part of backward-stage graph execution.

#### `topologicalSort()`
Returns topological order rooted at this tensor.

Internal structural graph mutation, backward wiring, and runtime aliasing helpers are no longer public `Tensor` methods.
They now live behind `tensor.TensorInternalAccess`.

## Debug Formatting

### `toStructString()`

Returns a string representation of shape, strides, and logical values.

Example:
```java
// Formats x for debugging, including shape, strides, and logical values.
System.out.println(x.toStructString());
// Returns: a human-readable debug string.
```
