# Tensor Package

## Purpose

The `tensor` package is the public graph-building surface of the project.
It is responsible for three things:

- representing tensor values and tensor graph nodes
- exposing the modeling API used by workloads, tests, and higher-level code
- providing the immediate bridge from user-facing tensor calls to graph primitives

The package deliberately does **not** own the full execution pipeline.
Compilation, planning, and prepared runtime execution live in:

- [graph/CompiledGraph.java](../graph/CompiledGraph.java)
- [graph/execution/PreparedExecution.java](../graph/execution/PreparedExecution.java)

The key design rule is:

- `tensor` builds the graph
- `operations` describes graph primitives
- `graph/*` compiles and rewrites the graph
- `backend/*` executes the graph

## Layering Overview

The package stack looks like this:

1. Public modeling surface
   - [tensor/Tensor.java](../tensor/Tensor.java)
   - [tensor/TensorOps.java](../tensor/TensorOps.java)
2. Family-specific graph builders
   - [tensor/ops/](../tensor/ops)
3. Primitive descriptors
   - [operations/](../operations)
4. Graph compilation and runtime execution
   - [graph/](../graph)
   - [backend/](../backend)

In practical terms:

- `Tensor` instance methods are the ergonomic entry point for most code
- `TensorOps` is the static facade over the same family builders
- `tensor.ops.*` owns validation, primitive creation, and backward wiring
- `operations/**` owns immutable operation descriptors only
- those descriptors are grouped by the same broad families used in `tensor.ops.*` and backend dispatch

Public construction is also intentionally split:

- ordinary modeling code should prefer `Tensor.scalar(...)`, `Tensor.onesLike(...)`, `Tensor.zerosLike(...)`, and typed leaf constructors
- low-level primitive-node and explicit-stride constructors remain public because runtime, rewrites, and tests still need them
- this means not every public `Tensor` constructor is equally "high-level"; some are infrastructure surface by design

## How To Read This Package

There are three different audiences, and the package should read clearly for all of them:

1. Modeling code
   - uses `Tensor` instance methods or `TensorOps`
   - should think in terms of graph construction, not storage internals
2. Tensor family implementers
   - work mostly in `tensor.ops.*`
   - decide validation rules, primitive-vs-composed surface, and backward wiring
3. Runtime / optimizer / backend code
   - may touch metadata, typed storage, prepared execution, or low-level helpers
   - should avoid leaking those concerns back into the modeling surface

That distinction matters because `tensor` intentionally exposes both:

- a clean modeling surface for graph construction
- a narrower low-level surface used by planners, executors, rewrites, and tests

## Package Map

### Root `tensor/`

The root package contains the core public type plus shared graph-building and layout/data infrastructure:

- [tensor/Tensor.java](../tensor/Tensor.java)
  - the main tensor node/value type
- [tensor/TensorOps.java](../tensor/TensorOps.java)
  - static public facade over the family builders in `tensor.ops.*`
- [tensor/TensorDataFactory.java](../tensor/TensorDataFactory.java)
  - convenience constructors, leaf constant helpers, and small tensor factory utilities
- [tensor/TensorPrimitiveBuilder.java](../tensor/TensorPrimitiveBuilder.java)
  - shared helpers for creating primitive nodes, no-grad nodes, and alias/view nodes
- [tensor/TensorPiecewiseOps.java](../tensor/TensorPiecewiseOps.java)
  - small composed helpers such as compare/select-based piecewise building
- [tensor/TensorBroadcastOps.java](../tensor/TensorBroadcastOps.java)
  - broadcast-shape helpers used by graph construction
- [tensor/TensorLayoutTransform.java](../tensor/TensorLayoutTransform.java)
  - axis normalization and layout utility logic
- [tensor/TensorGraphTraversal.java](../tensor/TensorGraphTraversal.java)
  - internal DAG traversal helpers such as topological sort
- [tensor/TensorDebugSupport.java](../tensor/TensorDebugSupport.java)
  - internal debug/inspection helpers such as structured printing and logical array snapshots
- [tensor/TensorExecutionSupport.java](../tensor/TensorExecutionSupport.java)
  - internal compile/prepare/compute and backend-resolution helpers
- [tensor/TensorRemap.java](../tensor/TensorRemap.java)
  - low-level layout remap support
- [tensor/TensorMetadata.java](../tensor/TensorMetadata.java)
  - logical shape/stride/offset metadata
- [tensor/TensorStorage.java](../tensor/TensorStorage.java)
  - typed storage abstraction
- [tensor/TensorStorageSupport.java](../tensor/TensorStorageSupport.java)
  - internal typed-storage allocation, conversion, and storage-offset access helpers
- dtype-specific storage implementations
  - [tensor/Float64Storage.java](../tensor/Float64Storage.java)
  - [tensor/Float32Storage.java](../tensor/Float32Storage.java)
  - [tensor/BFloat16Storage.java](../tensor/BFloat16Storage.java)
  - [tensor/BoolStorage.java](../tensor/BoolStorage.java)
  - [tensor/Int32Storage.java](../tensor/Int32Storage.java)

The root package still contains some low-level/runtime-oriented pieces because `Tensor` is still the public anchor type.
The preferred direction is:

- public modeling entry points stay on `Tensor` and `TensorOps`
- primitive construction stays in `tensor.ops.*`
- storage, traversal, debug inspection, execution glue, and array-introspection mechanics move into small internal helpers

### Public semantic value types

Semantic option/config types are now kept in dedicated public subpackages instead of the root:

- [tensor/options/AttentionOptions.java](../tensor/options/AttentionOptions.java)
- [tensor/options/Conv2dOptions.java](../tensor/options/Conv2dOptions.java)
- [tensor/options/Pool2dOptions.java](../tensor/options/Pool2dOptions.java)
- [tensor/loss/LossReduction.java](../tensor/loss/LossReduction.java)

These are public API types.
They are intentionally **not** hidden inside support helpers because they are used across:

- tensor API surface
- operation builders
- rewrites
- backend/runtime code
- workloads and tests

### Graph-building families in `tensor.ops.*`

The public modeling surface is internally split by operation family:

- [tensor/ops/unary/TensorUnaryOps.java](../tensor/ops/unary/TensorUnaryOps.java)
- [tensor/ops/binary/TensorBinaryOps.java](../tensor/ops/binary/TensorBinaryOps.java)
- [tensor/ops/compare/TensorCompareOps.java](../tensor/ops/compare/TensorCompareOps.java)
- [tensor/ops/select/TensorSelectOps.java](../tensor/ops/select/TensorSelectOps.java)
- [tensor/ops/bool/TensorBoolOps.java](../tensor/ops/bool/TensorBoolOps.java)
- [tensor/ops/layout/TensorLayoutOps.java](../tensor/ops/layout/TensorLayoutOps.java)
- [tensor/ops/index/TensorIndexOps.java](../tensor/ops/index/TensorIndexOps.java)
- [tensor/ops/reduction/TensorReduceOps.java](../tensor/ops/reduction/TensorReduceOps.java)
- [tensor/ops/linalg/TensorMatMulOps.java](../tensor/ops/linalg/TensorMatMulOps.java)
- [tensor/ops/linalg/TensorLinearOps.java](../tensor/ops/linalg/TensorLinearOps.java)
- [tensor/ops/linalg/TensorAttentionOps.java](../tensor/ops/linalg/TensorAttentionOps.java)
- [tensor/ops/conv/TensorConvOps.java](../tensor/ops/conv/TensorConvOps.java)
- [tensor/ops/pool/TensorPoolOps.java](../tensor/ops/pool/TensorPoolOps.java)
- [tensor/ops/normalization/TensorNormalizationOps.java](../tensor/ops/normalization/TensorNormalizationOps.java)
- [tensor/ops/loss/TensorLossOps.java](../tensor/ops/loss/TensorLossOps.java)

Each family also has local support helpers such as `UnarySupport`, `BinarySupport`, `LossSupport`, and so on.

This split is important because it keeps:

- validation rules near the operation family
- backward graph formulas near the forward builder
- cross-family composition explicit instead of being hidden in one giant helper class

### Data-shape helper extraction

Array-shape inference and flattening logic used by `Tensor` constructors now lives in:

- [tensor/factory/TensorArrayData.java](../tensor/factory/TensorArrayData.java)

This is a small but important cleanup step:

- `Tensor` still owns tensor semantics
- array-introspection glue no longer bloats the core class unnecessarily

## What `Tensor` Is Responsible For

`Tensor` is intentionally a hybrid of:

- logical tensor value
- graph node
- lightweight execution facade

Today it owns:

- dtype, shape, strides, and storage offset metadata
- backing typed storage
- producing `Operation`
- predecessor links
- gradient reference
- backward lambda
- optional forced backend override
- thin convenience entry points into compile/prepare/compute

This means `Tensor` is still the main user-facing anchor.
That is acceptable as long as new semantics are pushed into family builders instead of being added inline to `Tensor` itself.

In other words:

- `Tensor` is allowed to be the public anchor
- `Tensor` is not allowed to become the implementation dumping ground

## What Should Not Be Added To `Tensor`

New code should avoid putting these concerns directly into `Tensor`:

- family-specific forward math construction
- family-specific validation rules
- large backward formulas
- backend dispatch logic
- optimizer/rewrite policy
- execution caches with deep runtime meaning

Those belong respectively in:

- `tensor.ops.*`
- `operations/**`
- `graph/*`
- `backend/*`

## Public API Shape

The public ergonomic surface is intentionally available in two forms:

- instance methods on `Tensor`
- static helpers on [tensor/TensorOps.java](../tensor/TensorOps.java)

They lead to the same internal builders.

Current public operation families are:

- layout/shape
  - `contiguous`, `reshape`, `expand`, `permute`, `transpose`, `expandDims`, `squeeze`
- elementwise arithmetic and compare
  - `add`, `sub`, `mul`, `div`, `min`, `max`
  - `greaterThan`, `greaterOrEqual`, `lessThan`, `lessOrEqual`, `equalTo`, `notEqualTo`
- piecewise/select/bool
  - `where`, `minimum`, `maximum`, `logicalAnd`, `logicalOr`, `logicalNot`
- unary/scalar
  - `relu`, `abs`, `neg`, `log`, `exp`, `fastExp`, `tanh`, `fastTanh`, `pow`, `mul(double)`, `inv`, `sqrt`, `sigmoid`, `clamp*`
- reduction
  - `sum`, `mean`, `softmax`, `logSoftmax`, `min`, `max`, `all`, `any`
- linalg
  - `matmul`, `linear`, `scaledDotProductAttention`
- spatial
  - `conv2d`, `maxPool2d`, `avgPool2d`
- normalization
  - `batchNorm`, `layerNorm`, `rmsNorm`
- indexing and losses
  - `select`, `gather`, `takeAlongAxis`, `scatterAdd`
  - `nllLoss*`, `crossEntropyLoss*`

The detailed method-level surface remains documented in:

- [tensor/API.md](../tensor/API.md)

Important layout-surface note:

- there is intentionally no public `view()` method
- view semantics are expressed through the explicit layout/indexing operations instead
- `reshape`, `permute`, `transpose`, `expand`, `expandDims`, `squeeze`, and `select` are the public view-like transforms
- `contiguous()` is the explicit "materialize this into dense storage" operation

That API shape is deliberate:

- it keeps aliasing intent explicit
- it avoids overloading one generic `view()` name for materially different layout behaviors
- it makes broadcast aliasing (`expand`) and dense materialization (`contiguous`) visible in the graph surface

## Primitive-Backed vs Composed Surface

Not every public tensor call is implemented the same way.

There are two intentionally different patterns.

### Primitive-backed operations

These create a dedicated graph primitive with a matching `Operation` descriptor.

Examples:

- `add`, `sub`, `mul`, `div`
- `relu`, `sigmoid`, `exp`, `tanh`, `log`
- `softmax`, `logSoftmax`
- `matmul`, `linear`
- `conv2d`, `pool2d`
- `scaledDotProductAttention`
- indexed cross-entropy and related specialized loss paths

Typical flow:

1. validate inputs
2. create `Operation` descriptor
3. build a primitive tensor node through `TensorPrimitiveBuilder`
4. attach backward lambda

`TensorPrimitiveBuilder` now explicitly supports three cases:

- regular primitive nodes
- `no-grad` helper nodes used inside backward graphs
- alias/view nodes used by layout/indexing paths

### Composed operations

These are still public API surface, but they are built by composing other tensor operations instead of creating a new canonical primitive.

Examples:

- `minimum` / `maximum`
- `clamp`
- the `batchNorm(...)` composition path
- many backward formulas inside unary/binary/reduction families

This is intentional when:

- the semantic contract is naturally expressed in existing primitives
- a new primitive would only add API clutter
- the backend does not need a distinct runtime contract

## Autograd Contract

The autograd design used here is:

- forward builders live in `tensor.ops.*`
- backward formulas stay close to those forward builders as lambdas
- gradient graphs are built primarily from other `Tensor` operations whenever practical

This is the important rule:

- prefer building backward graphs from existing tensor operations
- only introduce specialized backward primitives where there is a real semantic or performance reason

Examples:

- `relu` backward is composed from `where(...)`
- many binary/unary gradients are composed from ordinary tensor ops
- attention and softmax families use dedicated backward primitives where that is the canonical contract

This keeps most gradients readable as graph algebra while still allowing specialized primitives in known hot paths.

## Broadcasting Contract

Broadcasting follows standard right-aligned semantics:

- ranks align from the right
- missing leading axes behave as size `1`
- two dimensions are compatible when they are equal or one side is `1`
- output shape takes the per-axis maximum of the compatible dimensions

This contract is shared by:

- binary arithmetic
- comparison ops
- logical binary ops
- `where(condition, ifTrue, ifFalse)`

Backward rule:

- if a forward operand was broadcast, its gradient is reduced back to the original operand shape before accumulation

## Views, Layout, and Materialization

The tensor layer supports both dense tensors and logical views.

Important cases:

- `permute(...)` creates a stride-reordered view
- `expand(...)` creates a zero-stride broadcast alias view
- `select(...)` creates an offset alias view
- `expandDims(...)` and `squeeze(...)` adjust logical shape/strides without copying data
- `reshape(...)` preserves element count and may stay as a view or materialize later depending on layout constraints
- `contiguous()` is the explicit “make this dense” operation

This matters because downstream runtime kernels may:

- consume the view directly
- use specialized strided execution
- or materialize before a faster dense path

The tensor layer therefore has to preserve enough metadata for the planner/runtime to make that decision later.

## Execution Flow

The intended execution flow is:

1. Build a graph from `Tensor` operations.
2. Compile it into a [graph/CompiledGraph.java](../graph/CompiledGraph.java).
3. Bind runtime state into a [graph/execution/PreparedExecution.java](../graph/execution/PreparedExecution.java).
4. Execute in `FORWARD` or `FORWARD_BACKWARD`.

Convenience entry points still exist on `Tensor`:

- `prepare(ExecutionProfile profile)`
- `compute(ExecutionProfile profile)`
- `compute(PreparedExecution execution, ExecutionMode mode)`

These are intentionally thin facades.
They are not a substitute for the actual compile/runtime layers.

## DType and Storage Model

Current public dtypes are:

- `FLOAT64`
- `FLOAT32`
- `BFLOAT16`
- `INT32`
- `BOOL`

The tensor package keeps typed storage explicit instead of erasing everything to one generic buffer.
That is important for:

- backend code generation
- dtype-specific kernels
- indexing and bool semantics
- avoiding accidental implicit conversions

`INT32` now matters especially for indexing-style workloads and index-target losses.

## Runtime-Oriented Methods Still Present On `Tensor`

`Tensor` still exposes several methods that are useful mainly to runtime, graph infrastructure, or tests, for example:

- `getShapeUnsafe`
- `getStridesUnsafe`
- `getStorage`
- typed raw storage getters such as `getFloat32Data()` and `getBFloat16Data()`
- `getStorageOffsetUnsafe`
- `storageVersion`
- `markStorageModified`
- `setPrevTensors`
- `setOperation`
- `setGradient`
- `setBackward`
- `getRuntimeCache` / `setRuntimeCache`
- `copyDataFrom`

These methods are real and supported, but they should not be treated as the preferred high-level modeling API.
If you are adding a new user-facing operation, start in `tensor.ops.*`, not by expanding this low-level surface.

## How To Add Or Change Tensor Functionality

For a new operation or a major semantic change, the usual path is:

1. Decide whether the operation is primitive-backed or composed.
2. Put validation and graph construction into the correct `tensor.ops.*` family.
3. Add or update the `Operation` descriptor only if the operation is a real primitive.
4. Keep backward logic near the forward builder.
5. Add optimizer/backend support only if the primitive needs runtime meaning.
6. Update [operations/README.md](../operations/README.md) and [tensor/API.md](../tensor/API.md) when the public contract changes.

That keeps the API coherent and prevents `Tensor.java` from becoming the dumping ground for unrelated logic.
