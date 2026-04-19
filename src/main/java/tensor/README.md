# Tensor Package

The `tensor` package is the public graph-building surface of Synaptik.

Its core responsibilities are:

- represent tensor values and semantic graph nodes
- expose the user-facing API used by workloads, tests, and models
- bridge ergonomic tensor calls to primitive graph nodes and backward builders

It is not the whole execution pipeline.
Compilation, runtime preparation, and execution live in:

- [../graph/CompiledGraph.java](../graph/CompiledGraph.java)
- [../graph/execution/PreparedExecution.java](../graph/execution/PreparedExecution.java)

## Layering

The intended layering is:

1. `Tensor` / `TensorOps`
2. `tensor.ops.*`
3. `operations.*`
4. `graph/*`
5. `backend/*`

Meaning:

- `Tensor` is the ergonomic public anchor
- `tensor.ops.*` owns family-specific validation and graph building
- `operations.*` owns immutable primitive descriptors
- `graph/*` owns compile-time transformation and preparation
- `backend/*` owns execution

## Main Public Entry Points

- [Tensor.java](../tensor/Tensor.java)
  - main user-facing tensor/node type
- [TensorOps.java](../tensor/TensorOps.java)
  - static facade for the same semantic families

Ordinary modeling code usually starts there.

## Internal Family Split

The public surface is implemented through family-specific builders under `tensor.ops.*`.

Current families:

- unary:
  - [ops/unary/TensorUnaryOps.java](../tensor/ops/unary/TensorUnaryOps.java)
- binary:
  - [ops/binary/TensorBinaryOps.java](../tensor/ops/binary/TensorBinaryOps.java)
- compare:
  - [ops/compare/TensorCompareOps.java](../tensor/ops/compare/TensorCompareOps.java)
- bool:
  - [ops/bool/TensorBoolOps.java](../tensor/ops/bool/TensorBoolOps.java)
- select:
  - [ops/select/TensorSelectOps.java](../tensor/ops/select/TensorSelectOps.java)
- layout:
  - [ops/layout/TensorLayoutOps.java](../tensor/ops/layout/TensorLayoutOps.java)
- index:
  - [ops/index/TensorIndexOps.java](../tensor/ops/index/TensorIndexOps.java)
- reduction:
  - [ops/reduction/TensorReduceOps.java](../tensor/ops/reduction/TensorReduceOps.java)
- linalg:
  - [ops/linalg/TensorMatMulOps.java](../tensor/ops/linalg/TensorMatMulOps.java)
  - [ops/linalg/TensorLinearOps.java](../tensor/ops/linalg/TensorLinearOps.java)
  - [ops/linalg/TensorAttentionOps.java](../tensor/ops/linalg/TensorAttentionOps.java)
- conv / pool / normalization / loss:
  - [ops/conv/TensorConvOps.java](../tensor/ops/conv/TensorConvOps.java)
  - [ops/pool/TensorPoolOps.java](../tensor/ops/pool/TensorPoolOps.java)
  - [ops/normalization/TensorNormalizationOps.java](../tensor/ops/normalization/TensorNormalizationOps.java)
  - [ops/loss/TensorLossOps.java](../tensor/ops/loss/TensorLossOps.java)

This split matters because Synaptik does not want one giant mixed `Tensor` implementation file to own:

- validation
- forward construction
- backward graph formulas
- runtime helpers
- storage helpers

all at once.

## What `Tensor` Represents

`Tensor` is intentionally a hybrid of:

- logical tensor value
- semantic graph node
- publication target for computed forward outputs and gradients
- thin convenience gateway into compile/prepare/compute

Today it owns:

- dtype
- shape
- strides
- storage offset
- backing storage
- predecessor edges
- operation descriptor
- gradient reference
- backward lambda
- optional forced backend override

That is acceptable as long as new semantics are pushed into `tensor.ops.*` instead of being added inline everywhere inside `Tensor`.

## Public Semantic Option Types

Public semantic option/config value types live in dedicated subpackages:

- [options/AttentionOptions.java](../tensor/options/AttentionOptions.java)
- [options/Conv2dOptions.java](../tensor/options/Conv2dOptions.java)
- [options/Pool2dOptions.java](../tensor/options/Pool2dOptions.java)
- [loss/LossReduction.java](../tensor/loss/LossReduction.java)
- [CompileMode.java](../tensor/CompileMode.java)
- [ComputeOptions.java](../tensor/ComputeOptions.java)
- [AutotunePolicy.java](../tensor/AutotunePolicy.java)

These are genuine public API types.
They are intentionally not hidden inside private support classes.

## Construction Surface

The public `Tensor` surface includes both:

- ergonomic user-facing construction helpers
- lower-level constructors used by tests, runtime, and optimizer code

### Recommended public construction paths

For ordinary modeling code prefer:

- `Tensor.scalar(...)`
- `Tensor.onesLike(...)`
- `Tensor.zerosLike(...)`
- typed flat-array constructors
- nested-array convenience constructor when readability matters more than strict control

Example:

```java
Tensor x = new Tensor(
        new double[]{1.0, 2.0, 3.0, 4.0},
        new int[]{2, 2},
        null,
        "x",
        DataType.FLOAT64
);

Tensor bias = Tensor.scalar(10.0, DataType.FLOAT64);
Tensor y = x.add(bias);
```

Values:

- `x` = `[[1, 2], [3, 4]]`
- `bias` = `10`
- `y` = `[[11, 12], [13, 14]]`

### Why low-level constructors are still public

They are still needed by:

- tests
- graph rewrites
- runtime/layout helpers
- explicit alias/view setup

So not every public constructor is equally "high level".

## Static vs Instance API

Synaptik exposes the public API in two equivalent styles:

- instance methods on `Tensor`
- static methods on `TensorOps`

Example:

```java
Tensor z1 = x.add(y);
Tensor z2 = TensorOps.add(x, y);
```

Use instance methods in ordinary modeling code.
Use `TensorOps` when writing generic builders or code that reads more clearly in static form.

## Layout and View Semantics

There is intentionally no single generic public `view(...)` method.

Instead, the API exposes explicit layout operations:

- `reshape(...)`
- `permute(...)`
- `transpose()`
- `expand(...)`
- `expandDims(...)`
- `squeeze(...)`
- `select(...)`
- `contiguous()`

Why this matters:

- `reshape` changes logical shape
- `permute` changes axis order
- `expand` introduces broadcast-style repeated views
- `select` slices one axis
- `contiguous` materializes dense storage if needed

Those behaviors are semantically different enough that one generic `view(...)` name would hide important meaning.

## Broadcasting Contract

Broadcasting follows the usual trailing-axis rule:

- align ranks from the right
- missing leading axes behave as `1`
- dimensions are compatible when equal or when one side is `1`

Example:

- shape `[2, 3, 4]`
- plus shape `[3, 4]`
- result shape `[2, 3, 4]`

Concrete value example:

```text
left  = [[[1,2],[3,4],[5,6]],
         [[7,8],[9,10],[11,12]]]   shape [2,3,2]
right = [[10,20],[30,40],[50,60]]  shape [3,2]
```

Broadcasted result:

```text
[[[11,22],[33,44],[55,66]],
 [[17,28],[39,50],[61,72]]]
```

Gradients of broadcast operands are reduced back to original shapes in backward.

## Autodiff Model

Synaptik uses reverse-mode autodiff over semantic tensor graphs.

Current high-level model:

- forward builders create primitive or composed graph nodes
- each builder also attaches backward logic
- compile can then build a joint forward/backward graph

Example:

```java
Tensor x = new Tensor(new double[]{1.0, -2.0, 3.0}, new int[]{3}, null, "x", DataType.FLOAT64);
x.setRequiresGrad(true);

Tensor loss = x.relu().sum();
loss.compute(CompileMode.TRAINING);
```

Forward values:

- `relu([1, -2, 3]) = [1, 0, 3]`
- `sum = 4`

Gradient:

- derivative of `relu` is `[1, 0, 1]`
- derivative of `sum` distributes ones
- so `x.grad = [1, 0, 1]`

## Convenience Execution API

`Tensor` intentionally exposes a thin execution convenience layer:

- `compile()`
- `compile(CompileMode)`
- `prepare(ExecutionProfile)`
- `compute()`
- `compute(CompileMode)`
- `compute(ComputeOptions)`
- `compute(ExecutionProfile)`

These are convenience entry points, not an excuse to move compile/runtime logic back into `tensor`.

Current defaults:

- `compute()`:
  - inference compile
  - inference optimizer/runtime defaults
  - `FORWARD`
- `compute(CompileMode.TRAINING)`:
  - training defaults
  - `FORWARD_BACKWARD` only when trainable leaves exist
- `compute(ComputeOptions)` can optionally trigger generic graph autotune

## Mutation Contract

`Tensor` remains mutable because it is:

- the semantic modeling surface
- the publication target for execution results

But there is an important lifecycle rule:

- before compile/prepare:
  - ordinary graph mutation is expected
- after compile/prepare:
  - treat the semantic graph as frozen if you want to reuse the compiled/prepared artifact

Why:

- compiled/prepared artifacts snapshot graph structure
- they do not track arbitrary later user-side graph mutations

## What Should Not Be Added Directly To `Tensor`

New code should avoid putting these concerns directly into `Tensor`:

- family-specific validation rules
- large backward formulas
- optimizer logic
- backend dispatch logic
- deep runtime caches

Those belong respectively in:

- `tensor.ops.*`
- `graph/*`
- `backend/*`

## Supporting Internal Helpers

Several supporting classes exist to keep `Tensor` itself smaller:

- [TensorPrimitiveBuilder.java](../tensor/TensorPrimitiveBuilder.java)
- [TensorDataFactory.java](../tensor/TensorDataFactory.java)
- [TensorBroadcastOps.java](../tensor/TensorBroadcastOps.java)
- [TensorLayoutTransform.java](../tensor/TensorLayoutTransform.java)
- [TensorGraphTraversal.java](../tensor/TensorGraphTraversal.java)
- [TensorExecutionSupport.java](../tensor/TensorExecutionSupport.java)
- [TensorStorageSupport.java](../tensor/TensorStorageSupport.java)
- [factory/TensorArrayData.java](../tensor/factory/TensorArrayData.java)

These are part of the package design, but they are not the first place to look when using the public API.

For method-by-method API details, examples, and signatures, see:

- [API.md](./API.md)
