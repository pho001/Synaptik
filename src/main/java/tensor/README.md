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
  - [src/main/java/tensor/TensorConvOps.java](../tensor/TensorConvOps.java)
  - [src/main/java/tensor/TensorPoolOps.java](../tensor/TensorPoolOps.java)
- Operation descriptors and addition guide:
  - [src/main/java/operations/README.md](../operations/README.md)
- Layout remap utility:
  - [src/main/java/tensor/TensorRemap.java](../tensor/TensorRemap.java)
- Storage/type abstractions:
  - [src/main/java/tensor/TensorStorage.java](../tensor/TensorStorage.java)
  - [src/main/java/tensor/DataType.java](../tensor/DataType.java)
  - [src/main/java/tensor/BFloat16Storage.java](../tensor/BFloat16Storage.java)
  - [src/main/java/tensor/Float32Storage.java](../tensor/Float32Storage.java)
  - [src/main/java/tensor/Float64Storage.java](../tensor/Float64Storage.java)

## Data Model

`Tensor` stores:

- tensor storage (`BFLOAT16`, `FLOAT32`, `FLOAT64`, `BOOL`)
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
- `linear(Tensor weight)`
- `linear(Tensor weight, Tensor bias)`
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

### Attention

- `scaledDotProductAttention(Tensor key, Tensor value, AttentionOptions options)`
- `scaledDotProductAttention(Tensor key, Tensor value, Tensor mask, AttentionOptions options)`

### Spatial

- `conv2d(Tensor weight, Conv2dOptions options)`
- `conv2d(Tensor weight, Tensor bias, Conv2dOptions options)`
- `maxPool2d(Pool2dOptions options)`
- `avgPool2d(Pool2dOptions options)`

### Normalization

- `batchNorm(Tensor gamma, Tensor beta, int channelDimension, double epsilon)`
- `batchNorm(Tensor gamma, Tensor beta, Tensor mean, Tensor variance, int channelDimension, double epsilon)`
- `layerNorm(Tensor gamma, Tensor beta, double epsilon)`
- `rmsNorm(Tensor gamma, double epsilon)`

### Unary / Scalar

- `relu()`
- `abs()`
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
- `softmax(int dimension)`
- `logSoftmax(int dimension)`
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

### Loss / N-ary

- `nllLoss(Tensor targets, int classDimension)`
- `crossEntropyLoss(Tensor targets, int classDimension)`
- `nllLossFromIndices(Tensor targetIndices, int classDimension)`
- `nllLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex)`
- `crossEntropyLossFromIndices(Tensor targetIndices, int classDimension)`
- `crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, int ignoreIndex)`

### Indexing

- `select(int dimension, int index)`
- `gather(Tensor indices, int dimension)`
- `takeAlongAxis(Tensor indices, int dimension)`
- `scatterAdd(Tensor indices, Tensor src, int dimension)`

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
- `select(...)` creates a storage-offset alias view
- `expandDims(...)` creates a stride-preserving alias view with one inserted size-`1` axis
- `squeeze(...)` creates a stride-preserving alias view with one removed size-`1` axis
- `reshape(...)` changes shape interpretation while preserving element count
  - for contiguous inputs it aliases the same storage as a reshape view
  - for non-contiguous inputs it may materialize a dense reshaped result at runtime
- `contiguous()` is the canonical explicit materialization path
- current CPU execution keeps offset views as views through layout ops
  - strided element-wise, compare, logical, `where`, reduction, softmax/logSoftmax, dense-target loss, indexing kernels, and min/max reduction-grad kernels can consume offset views directly
  - if a downstream kernel does not support non-zero base offsets directly, planner-side prepared inputs materialize it first

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

## Reduction Interoperability

Reduction and broadcasting are intended to compose directly.

Axis reductions support `keepDims` where it makes semantic sense:

- `sum(int dimension, boolean keepDims)`
- `mean(int dimension, boolean keepDims)`
- `min(int dimension, boolean keepDims)`
- `max(int dimension, boolean keepDims)`
- `all(int dimension, boolean keepDims)`
- `any(int dimension, boolean keepDims)`

Shape contract:

- `keepDims=false` removes the reduced axis
- `keepDims=true` preserves the reduced axis with size `1`

Why this matters:

- `keepDims=true` allows the reduction result to feed the next broadcast-aware op without an explicit `expandDims(...)`
- this is the intended path for expressions like centering, normalization, and mask pipelines

Example:

```java
Tensor x = new Tensor(new double[]{
        1, 2, 3,
        4, 5, 6
}, new int[]{2, 3}, null, "x", DataType.FLOAT64);

Tensor rowMean = x.mean(1, true);
Tensor centered = x.sub(rowMean);

// rowMean shape: [2, 1]
// centered shape: [2, 3]
// centered values: [-1, 0, 1, -1, 0, 1]
```

Backward contract:

- `sum` and `mean` expand their upstream gradient back to input shape
- `mean` also scales by the reciprocal of the reduced axis size
- `min` and `max` route gradients only to winning elements
- ties in `min` / `max` split gradient evenly across winners
- if a reduction result is later broadcast in the forward path, the downstream gradient is reduced back through the same broadcast contract

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

`softmax(int dimension)` is an axis-wise normalization primitive:

- output shape matches input shape
- values along the chosen axis are exponentiated in a numerically stable way and normalized to sum to `1`
- backward uses the standard softmax Jacobian-vector product form

`logSoftmax(int dimension)` is the logarithmic counterpart:

- output shape matches input shape
- values along the chosen axis are computed as `x - logsumexp(x)` in a numerically stable way
- backward uses `g - softmax(x) * sum(g)` along the chosen axis

`nllLoss(targets, classDimension)` is the first loss primitive built on top of log-probabilities:

- input is expected to be log-probabilities
- `targets` must have the same shape as `logProbs`
- `targets` are interpreted as one-hot labels or more general target distributions along the class axis
- output is a scalar mean loss across all sample positions outside the class axis

`crossEntropyLoss(targets, classDimension)` is currently the ergonomic logits-facing surface:

- input is raw logits
- internally it now maps to a dedicated `CROSS_ENTROPY_LOSS` primitive
- semantic reference remains `logSoftmax(classDimension).nllLoss(targets, classDimension)`
- current contract is still the narrow dense-target mean-reduction variant

`select(dimension, index)` is the single-slice indexing surface:

- output shape is input shape without the selected axis
- negative indices count from the end of the selected axis
- current implementation is a true alias view with shared storage
- the selected slice is represented by rewritten strides plus non-zero `storageOffset` when needed
- semantically this is the ergonomic one-index read operation

`gather(indices, dimension)` is the first narrow indexing primitive:

- output shape is input shape with the selected axis removed
- `indices` must have exactly that output shape
- `INT32` is the preferred index dtype
- numeric floating tensors with integral values are still accepted as a compatibility mode
- backward scatters upstream gradient back into selected input positions
- this is intentionally a minimal first step before a broader indexing/gather family

`takeAlongAxis(indices, dimension)` je obecnejsi axis-wise read primitive:

- output shape = `indices.shape`
- rank se zachovava
- vsechny ne-indexovane osy musi sedet se vstupem
- indexovana osa se ridi shape tensoru `indices`
- backward scatteruje gradient zpet do vstupu podél stejne osy

`scatterAdd(indices, src, dimension)` is the first explicit write/update indexing primitive:

- output shape matches the base tensor shape
- `indices` and `src` must have shape equal to the base shape without the scattered axis
- values from `src` are added into positions selected by `indices`
- this is the natural write-side pair to `gather`

## Spatial / Convolution Contract

`conv2d(...)` is the first spatial primitive in the tensor surface.

Current contract:

- input layout is `NCHW`
- weight layout is `OIHW`
- optional bias shape is `[outChannels]`
- options are explicit through `Conv2dOptions`
  - `strideH`, `strideW`
  - `padH`, `padW`
  - `dilationH`, `dilationW`
  - `groups`
- only floating dtypes are accepted
- grouped convolution is supported

Output shape:

- input: `[batch, inChannels, inH, inW]`
- weight: `[outChannels, inChannelsPerGroup, kernelH, kernelW]`
- output: `[batch, outChannels, outH, outW]`

Where:

- `inChannelsPerGroup * groups == inChannels`
- `outChannels % groups == 0`
- `outH = floor((inH + 2 * padH - effectiveKernelH) / strideH) + 1`
- `outW = floor((inW + 2 * padW - effectiveKernelW) / strideW) + 1`
- `effectiveKernelH = dilationH * (kernelH - 1) + 1`
- `effectiveKernelW = dilationW * (kernelW - 1) + 1`

Backward contract:

- input gradient is produced by a dedicated `conv2dBackwardInput` primitive
- weight gradient is produced by a dedicated `conv2dBackwardWeight` primitive
- bias gradient is currently expressed compositionally as reduction over:
  - batch axis
  - spatial height axis
  - spatial width axis

Architectural note:

- this is a first-class runtime primitive, not a public `im2col`-style helper
- the current CPU backend uses direct loop kernels
- that keeps the public tensor algebra clean while leaving later room for backend-specific lowering or autotuned specialization

## Pooling Contract

`maxPool2d(...)` and `avgPool2d(...)` are the first spatial reduction primitives in the tensor surface.

Current contract:

- input layout is `NCHW`
- input shape must be `[batch, channels, inH, inW]`
- output shape is `[batch, channels, outH, outW]`
- options are explicit through `Pool2dOptions`
  - `kernelH`, `kernelW`
  - `strideH`, `strideW`
  - `padH`, `padW`
  - `countIncludePad`
- only floating dtypes are accepted

Output shape:

- `outH = floor((inH + 2 * padH - kernelH) / strideH) + 1`
- `outW = floor((inW + 2 * padW - kernelW) / strideW) + 1`

Behavior:

- `maxPool2d`
  - returns the maximum value inside each pooling window
  - backward routes gradient to the first maximal element in scan order
- `avgPool2d`
  - returns the arithmetic mean inside each pooling window
  - default `countIncludePad = false`
    - border windows divide only by valid in-bounds elements
  - if `countIncludePad = true`
    - border windows divide by the full kernel area including padded positions

Architectural note:

- pooling is modeled as first-class runtime primitive, not as composition over gather/select helpers
- current CPU backend again uses direct loops
- backward is also specialized:
  - `maxPool2dBackwardInput`
  - `avgPool2dBackwardInput`
- `maxPool2d` forward stores argmax workspace in prepared runtime metadata
  - backward reads that workspace instead of rescanning the input windows
- pooling is currently an explicit dense-first backend boundary
  - non-contiguous or offset inputs are materialized by prepared-input planning before kernel execution

## Linear Surface

`linear(...)` is the first explicit neural-network projection surface in the tensor API.

Current contract:

- input shape is `[..., inFeatures]`
- weight shape is `[inFeatures, outFeatures]`
- optional bias shape is `[outFeatures]`
- output shape is `[..., outFeatures]`

Architectural point:

- `linear` is now a public surface backed by a canonical internal `LINEAR` primitive
- direct `linear(...)` API builds `LINEAR` nodes
- optimizer can also lower `matmul + rank-1 bias add` into `LINEAR`
- CPU execution reuses the BLAS-backed matmul family and adds fused bias handling in the linear kernel

Why this still matters:

- forward and backward both map naturally to matmul family hot paths
- introducing the surface now gives model-building ergonomics immediately
- future deeper specialization can still happen later without changing the public API

## Normalization Contract

Current normalization surface is intentionally stateless and explicit.

Supported surfaces:

- `batchNorm(gamma, beta, channelDimension, epsilon)`
  - computes batch statistics from the current input
- `batchNorm(gamma, beta, mean, variance, channelDimension, epsilon)`
  - uses externally supplied channel statistics
- `layerNorm(gamma, beta, epsilon)`
  - normalizes over trailing feature dimensions
- `rmsNorm(gamma, epsilon)`
  - normalizes over trailing feature dimensions without bias

Architectural rules:

- no hidden running-stat mutation inside tensor execution
- no implicit framework-owned state buffers
- normalization surfaces are currently expressed compositionally over:
  - `mean`
  - `sub`
  - `pow`
  - `sqrt`
  - `div`
  - `mul`
  - `add`

That means:

- public API is already clean and semantically stable
- autodiff works immediately through existing graph primitives
- future specialized lowering remains possible later if profiling justifies it

Shape contracts:

- batch norm:
  - `gamma`, `beta`, and optional `mean` / `variance` have shape `[channels]`
  - `channelDimension` selects the preserved axis
  - all remaining axes are reduced for statistics
- layer norm:
  - `gamma.shape == beta.shape`
  - that shape must equal the trailing normalized input block
- RMS norm:
  - `gamma.shape` must equal the trailing normalized input block

## Attention Contract

Current attention surface is intentionally composition-first.

Supported surface:

- `scaledDotProductAttention(key, value, options)`
- `scaledDotProductAttention(key, value, mask, options)`

Current tensor contract:

- query is the receiver tensor with shape `[..., queryLen, headDim]`
- key has shape `[..., keyLen, headDim]`
- value has shape `[..., keyLen, valueDim]`
- output has shape `[..., queryLen, valueDim]`
- leading dimensions follow batched `matmul` broadcasting rules

Mask contract:

- external mask must be `BOOL`
- mask must be broadcastable to score shape `[..., queryLen, keyLen]`
- `true` means the score is kept
- `false` means the score is suppressed before softmax
- causal masking can be enabled through `AttentionOptions`

Architectural note:

- attention is not a dedicated backend primitive yet
- current implementation is pure graph composition over:
  - batched `matmul`
  - scaling
  - `where`
  - `softmax`
  - batched `matmul`
- this keeps the public API clean while postponing fused/specialized lowering until profiling justifies it

`nllLossFromIndices(targetIndices, classDimension)` is the first index-target loss surface:

- input is `logProbs`
- `targetIndices` shape equals `logProbs.shape` without the class axis
- `INT32` is the preferred target-index dtype
- current implementation is composition-first over `gather(...).neg().mean()`
- this is the bridge from dense-target loss family toward classic class-id target workflows

`crossEntropyLossFromIndices(targetIndices, classDimension)` is the logits-facing ergonomic counterpart:

- input is raw logits
- `targetIndices` shape equals `logits.shape` without the class axis
- current implementation is composition-first over:
  - `logits.logSoftmax(classDimension).nllLossFromIndices(targetIndices, classDimension)`

`ignoreIndex` overloady existuji zatim jen pro index-target loss family:

- ignorovane sample neprispivaji do loss
- ignorovane sample maji nulovy gradient
- denominator `mean` redukce se pocita jen z neignorovanych sample
- kdyz jsou ignorovane vsechny sample, loss i gradient jsou nulove

`classWeights` jsou zatim podporovane nejdriv pro index-target loss family:

- `classWeights` musi mit shape `[numClasses]`
- dtype musi odpovidat floating dtype logits / logProbs
- lookup vah je ted vyjadreny ciste pres tensor indexing composition:
  - `reshape -> expand -> takeAlongAxis -> squeeze`
- `MEAN` u weighted losses dela weighted mean:
  - `sum(weightedLoss) / sum(appliedWeights)`
- pri `ignoreIndex` ignorovane sample neprispivaji ani do citatele, ani do jmenovatele
- weight se vybira podle target class id
- `NONE`: vraci weighted per-sample loss
- `SUM`: vraci soucet weighted per-sample loss
- `MEAN`: vraci weighted mean:
  - `sum(weightedLoss) / sum(appliedWeights)`

`LossReduction` je zatim zavedena nejdriv pro index-target loss family:

- `MEAN`
  - scalar mean loss
- `SUM`
  - scalar soucet per-sample loss hodnot
- `NONE`
  - bez redukce
  - vystup shape = `targetIndices.shape`

`min(...)` and `max(...)` follow the same shape policy as `sum(...)`.
Their backward semantics route gradient only to winning elements; if multiple values tie for the extremum, the gradient is split evenly across the winners.

`all(...)` and `any(...)` are `BOOL`-only reductions.
They follow the same shape policy as other reductions, but they are nondifferentiable.

`clampMin(minValue)` and `clampMax(maxValue)` are specialized one-sided unary clamp primitives:

- `clampMin` only raises values below the lower bound
- `clampMax` only lowers values above the upper bound

`clamp(minValue, maxValue)` is the two-sided convenience surface built on top of those one-sided clamp operations.
It keeps values inside the interval and replaces values below/above the interval bounds by the corresponding boundary value.

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
- `BFloat16Storage`
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
