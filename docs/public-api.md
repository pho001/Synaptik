<!-- generated-by: gsd-doc-writer -->
# Public API

Navigation: [Index](index.md#recommended-reading-paths) | [Quickstart](quickstart.md#what-synaptik-is) | [Tensor API](tensor-api.md#api-surface-and-conventions) | [Sequence Tensor Primitives](sequence-tensor-primitives.md#scope) | [Examples](examples.md#running-examples) | [Configuration](configuration.md#runtimeconfig) | [Compute Flow](compute-flow.md#tensor-compute-api) | [ONNX](onnx.md#onnx-import-and-export) | [Native Bridges & BLAS](native-bridges-and-blas.md#configuration-and-library-lookup) | [Metal Backend](metal-backend.md#buffer-residency-and-materialization) | [Troubleshooting](troubleshooting.md#unsupported-dtype-in-a-kernel)

Chapters: [Stability Map](#stability-map) | [Tensor](#tensor) | [ComputeOptions, CompileMode, And AutotunePolicy](#computeoptions-compilemode-and-autotunepolicy) | [CompiledGraph](#compiledgraph) | [PreparedExecution](#preparedexecution) | [PublicationPolicy](#publicationpolicy) | [Configuration APIs](#configuration-apis) | [ONNX APIs](#onnx-apis) | [Tuning Fluent API](#tuning-fluent-api) | [CLI Entry Point](#cli-entry-point) | [Probably Internal APIs](#probably-internal-apis) | [Verification Notes](#verification-notes)

This document describes the Java API surfaces that are usable from application code today and separates them from public Java types that are probably internal implementation hooks.

## Table Of Contents

- [Stability Map](#stability-map)
- [Tensor](#tensor)
- [ComputeOptions, CompileMode, And AutotunePolicy](#computeoptions-compilemode-and-autotunepolicy)
- [CompiledGraph](#compiledgraph)
- [PreparedExecution](#preparedexecution)
- [PublicationPolicy](#publicationpolicy)
- [Configuration APIs](#configuration-apis)
- [ONNX APIs](#onnx-apis)
- [Tuning Fluent API](#tuning-fluent-api)
- [CLI Entry Point](#cli-entry-point)
- [Probably Internal APIs](#probably-internal-apis)
- [Verification Notes](#verification-notes)

## Stability Map

| Status | API surface | Source path | Notes |
|---|---|---|---|
| Public | `tensor.Tensor` | `src/main/java/tensor/Tensor.java` | Main modeling, graph construction, execution, and inspection API. |
| Public | `tensor.ComputeOptions` | `src/main/java/tensor/ComputeOptions.java` | Mutable convenience options for `Tensor.compute(...)`. |
| Public | `tensor.CompileMode` | `src/main/java/tensor/CompileMode.java` | Compile intent for convenience execution. |
| Public | `tensor.AutotunePolicy` | `src/main/java/tensor/AutotunePolicy.java` | Autotune behavior for `ComputeOptions`. |
| Public | `graph.CompiledGraph` | `src/main/java/graph/CompiledGraph.java` | Compiled graph artifact and explicit prepare/execute entry point. |
| Public | `graph.execution.PreparedExecution` | `src/main/java/graph/execution/PreparedExecution.java` | Prepared runtime plan that can be executed repeatedly. |
| Public | `graph.execution.PublicationPolicy` | `src/main/java/graph/execution/PublicationPolicy.java` | Execution side-effect policy for output/intermediate/gradient publication. |
| Public | `config.compile.*` | `src/main/java/config/compile/*.java` | Compile-time semantic, graph optimization, backend planning, region optimization, and memory planning records. |
| Public | `config.runtime.*` | `src/main/java/config/runtime/*.java` | Runtime backend, BLAS, approximation, fused, and accelerator configuration records. |
| Public | `config.profile.*` | `src/main/java/config/profile/*.java` | Persistable execution and platform runtime profile records plus profile IO. |
| Public | `onnx.Onnx`, `OnnxModel`, `ImportedOnnxModel`, ONNX option records | `src/main/java/onnx/*.java` | Static dense ONNX import/export facade, in-memory model wrapper, imported model wrapper, and interchange policies. |
| Public | `tuning.api.*` | `src/main/java/tuning/api/*.java` | Fluent Java API for calibration, execution-profile construction, benchmark workflows, and report policy configuration. |
| Public | `synaptik.app.TuningCli` | `src/main/java/synaptik/app/TuningCli.java` | Gradle application CLI entry point for tuning workflows. |
| Public | `synaptik.app.Main` | `src/main/java/synaptik/app/Main.java` | Programmatic calibration and benchmark entry point using regular Java calls. |
| Probably internal | `tensor.TensorOps`, `tensor.TensorPrimitiveBuilder`, `tensor.TensorStorage*`, `tensor.TensorInternalAccess` | `src/main/java/tensor/*.java` | Public or package-visible support for operation construction and storage plumbing; prefer `Tensor` methods. |
| Probably internal | `config.optimizer.*` | `src/main/java/config/optimizer/*.java` | Legacy and implementation-adjacent optimizer/region helper records that are currently composed through `config.compile.*`; prefer `CompileConfig` for application code. |
| Probably internal | `backend.ComputeEngine`, backend kernel classes, backend bridge classes | `src/main/java/backend/**/*.java` | Runtime dispatch and kernel implementation details. |
| Probably internal | Most `tuning.*` candidate/search/measurement classes | `src/main/java/tuning/**/*.java` | Useful for extending the tuning system, but not the shortest supported application API. Prefer CLI/profile APIs first. |

## Tensor

**Source:** `src/main/java/tensor/Tensor.java`

**Purpose:** `Tensor` is the primary API for creating tensors, composing semantic graph operations, compiling and executing graphs, and reading outputs or gradients.

### Constructors And Factories

Common signatures:

```java
new Tensor(double[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)
new Tensor(float[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)
new Tensor(short[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)
new Tensor(int[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)
new Tensor(long[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)
new Tensor(byte[] data, int[] shape, List<Tensor> previous, String label, DataType dataType)
new Tensor(int[] dimensions, List<Tensor> previous, String label, DataType dataType)
new Tensor(Object multiDimArray, List<Tensor> previous, String label, DataType dataType)
Tensor.scalar(double value)
Tensor.scalar(double value, DataType dataType)
Tensor.zeros(int[] shape)
Tensor.zeros(int[] shape, DataType dataType)
Tensor.zeros(int[] shape, DataType dataType, String label)
Tensor.ones(int[] shape)
Tensor.ones(int[] shape, DataType dataType)
Tensor.ones(int[] shape, DataType dataType, String label)
Tensor.randn(int[] shape)
Tensor.randn(int[] shape, double mean, double stdDev, DataType dataType, String label)
Tensor.arange(int start, int end, int step, DataType dataType)
Tensor.onesLike(Tensor other)
Tensor.zerosLike(Tensor other)
```

Parameters:

- `data`: flat primitive storage in row-major logical order.
- `shape` or `dimensions`: logical tensor dimensions.
- `previous`: predecessor tensors when constructing graph nodes manually; application code usually passes `null` for leaves.
- `label`: optional debug label.
- `dataType`: one of `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, or `BOOL`.

Returns: a `Tensor` with shape, dtype, strides, and storage initialized.

Factory rules:

- `zeros` and `ones` support floating, integer, and `BOOL` dtypes.
- `randn` supports floating dtypes only because a normal distribution is not meaningful for `BOOL` or integer storage.
- `arange` returns a rank-1 integer-like or floating numeric tensor and rejects `BOOL`.
- `onesLike` and `zerosLike` preserve both shape and dtype, including `BOOL`.

Failures:

- Shape/data length mismatch throws `IllegalArgumentException`.
- Implicit numeric conversion to `BOOL`, `INT32`, or `INT64` storage throws `UnsupportedOperationException`.
- Invalid multidimensional array shape can fail during shape inference.

Side effects:

- Most matching-dtype primitive array constructors keep the supplied array as storage. This applies to `double[]`/`FLOAT64`, `float[]`/`FLOAT32`, `short[]`/`BFLOAT16`, `int[]`/`INT32`, and `long[]`/`INT64`.
- `byte[]` boolean input is copied and normalized to `0` or `1`; mutating the original byte array after construction does not update the tensor.
- Mutating an aliased numeric input array or the typed storage getter result can change the tensor. Use `toDoubleArrayCopy()` when a detached snapshot is needed.

Performance and concurrency notes:

- Shape arrays returned by `getShape()` and `getStrides()` are defensive copies; `getShapeUnsafe()` and `getStridesUnsafe()` return internal references.
- Tensor storage has a version counter, but tensors are mutable and should not be treated as thread-safe data structures.

Example:

```java
import tensor.DataType;
import tensor.Tensor;

Tensor x = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{2, 2}, null, "x", DataType.FLOAT64);
System.out.println(java.util.Arrays.toString(x.getShape()));
System.out.println(java.util.Arrays.toString(x.toDoubleArrayCopy()));

Tensor mask = Tensor.ones(new int[]{2, 3}, DataType.BOOL, "validMask");
Tensor time = Tensor.arange(0, 3, 1, DataType.INT32);
```

Expected output:

```text
[2, 2]
[1.0, 2.0, 3.0, 4.0]
```

The two factory tensors in the example are ordinary leaf tensors:

```text
mask shape = [2, 3], dtype = BOOL, values = true everywhere
time shape = [3], dtype = INT32, values = [0, 1, 2]
```

### Graph Operations

Common operation signatures:

```java
Tensor add(Tensor second)
Tensor sub(Tensor second)
Tensor mul(Tensor second)
Tensor div(Tensor second)
Tensor mul(double scalar)
Tensor neg()
Tensor exp()
Tensor log()
Tensor sqrt()
Tensor sigmoid()
Tensor tanh()
Tensor relu()
Tensor pow(double exp)
Tensor clamp(double minValue, double maxValue)
Tensor reshape(int... newShape)
Tensor expand(int... newShape)
Tensor permute(int... axes)
Tensor transpose()
Tensor sliceAxis(int axis, int fromInclusive, int toExclusive)
static Tensor stack(int axis, Tensor... inputs)
Tensor[] unstack(int axis)
Tensor take(int axis, Tensor indices)
Tensor take(int axis, int[] indices)
Tensor matmul(Tensor second)
Tensor linear(Tensor weight)
Tensor linear(Tensor weight, Tensor bias)
Tensor conv2d(Tensor weight, Conv2dOptions options)
Tensor conv2d(Tensor weight, Tensor bias, Conv2dOptions options)
Tensor maxPool2d(Pool2dOptions options)
Tensor avgPool2d(Pool2dOptions options)
Tensor scaledDotProductAttention(Tensor key, Tensor value, AttentionOptions options)
Tensor sum()
Tensor sum(int dimension)
Tensor sum(int dimension, boolean keepDims)
Tensor sum(int dimension, Tensor mask)
Tensor mean()
Tensor mean(int dimension, Tensor mask)
Tensor softmax(int dimension)
Tensor logSoftmax(int dimension)
Tensor crossEntropyLoss(Tensor targets, int classDimension, Tensor mask)
Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension)
Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, Tensor mask)
Tensor nllLossFromIndices(Tensor targetIndices, int classDimension)
static Tensor where(Tensor condition, Tensor ifTrue, Tensor ifFalse)
```

Parameters:

- Tensor operands must have compatible dtype, shape, and broadcast semantics for the specific operation.
- Dimension parameters are logical axis indices. Negative dimensions are accepted for some operations where the delegated operation support handles them; use non-negative dimensions for portable code.
- Option records such as `Conv2dOptions`, `Pool2dOptions`, and `AttentionOptions` validate their own numeric constraints.

Returns: a new graph `Tensor`. The operation is not executed until `compute(...)`, `CompiledGraph.execute(...)`, or `PreparedExecution.execute(...)` runs.

Failures:

- Invalid shapes, axes, groups, strides, padding, dilation, or dtype combinations throw `IllegalArgumentException`, `IllegalStateException`, or `UnsupportedOperationException` depending on the operation.
- `transpose()` requires a rank-2 tensor and throws `IllegalStateException` otherwise.

Side effects: graph construction does not execute kernels or overwrite input data.

Performance and concurrency notes:

- Operation chains build a semantic graph. Compilation can fuse or rewrite the graph before runtime execution.
- View-like operations such as reshape, expand, permute, and select can produce non-contiguous tensors. Call `contiguous()` when a dense materialized layout is required.

Example:

```java
import tensor.DataType;
import tensor.Tensor;

Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{2, 2}, null, "a", DataType.FLOAT64);
Tensor b = new Tensor(new double[]{10.0, 20.0}, new int[]{2}, null, "b", DataType.FLOAT64);
Tensor y = a.add(b).relu().compute();

System.out.println(java.util.Arrays.toString(y.toDoubleArrayCopy()));
```

Expected output:

```text
[11.0, 22.0, 13.0, 24.0]
```

### Data And Metadata Access

Important signatures:

```java
int[] getShape()
int[] getShapeUnsafe()
int[] getStrides()
int[] getStridesUnsafe()
int getFlatDataSize()
int rank()
int size()
int lastDim()
boolean shapeEquals(int... shape)
int[] shapeCopy()
DataType getDataType()
void setDataType(DataType dataType)
double getByFlatIndex(int index)
void setDataAt(int flatIndex, double value)
double[] toDoubleArrayCopy()
boolean[] toBooleanArrayCopy()
double scalarAsDouble()
float[] getFloat32Data()
double[] getFloat64Data()
short[] getBFloat16Data()
int[] getInt32Data()
long[] getInt64Data()
byte[] getBoolData()
Tensor getGradient()
boolean getRequiresGrad()
void setRequiresGrad(boolean requiresGrad)
```

Returns:

- `toDoubleArrayCopy()` and `toBooleanArrayCopy()` return logical-order copies.
- Typed storage getters return the backing array for matching storage types or `null` otherwise.
- `getGradient()` returns the gradient tensor after backward execution, or `null` if no gradient has been produced.
- `rank()` returns the number of logical axes.
- `size()` returns the logical element count and is equivalent to `getFlatDataSize()`.
- `lastDim()` returns the final dimension, which is useful for last-dimension projection helpers such as `linear`.
- `shapeEquals(...)` is a concise exact-shape check for public API validation in consumer code.
- `shapeCopy()` is the safe public alias for `getShape()`.

Failures:

- `getByFlatIndex` throws `IndexOutOfBoundsException` for invalid logical indices.
- `setDataAt` rejects broadcast views with `UnsupportedOperationException`.
- `getData()` only supports `FLOAT64`; for other dtypes use typed storage getters or `toDoubleArrayCopy()`.
- `setDataType` rejects implicit `INT32`, `INT64`, and `BOOL` conversions.
- `lastDim()` throws if called on an empty-rank tensor. Public scalar-like tensors normally use shape `[1]`, so this is mostly a guard for low-level infrastructure shapes.

Side effects:

- `setDataAt`, `setData(...)`, `copyDataFrom(...)`, `setDataType(...)`, and `setRequiresGrad(...)` mutate the tensor.
- Direct edits through typed backing arrays are possible; call `markStorageModified()` if downstream cache invalidation depends on the storage version.

Example:

```java
Tensor seq = Tensor.randn(new int[]{2, 3, 4}, 0.0, 1.0, DataType.FLOAT64, "seq");

System.out.println(seq.rank());                  // 3
System.out.println(seq.size());                  // 24
System.out.println(seq.lastDim());               // 4
System.out.println(seq.shapeEquals(2, 3, 4));    // true

int[] safe = seq.shapeCopy();
safe[0] = 999;                                   // does not mutate seq
System.out.println(seq.shapeEquals(2, 3, 4));    // still true
```

### Sequence-Friendly Tensor Surface

The public `Tensor` API includes general N-D primitives that let consumer frameworks keep sequence data in one tensor instead of in `Tensor[]`.

Common signatures:

```java
Tensor linear(Tensor weight)
Tensor linear(Tensor weight, Tensor bias)
Tensor sliceAxis(int axis, int fromInclusive, int toExclusive)
static Tensor stack(int axis, Tensor... inputs)
Tensor[] unstack(int axis)
Tensor take(int axis, Tensor indices)
Tensor take(int axis, int[] indices)
Tensor sum(int dimension, Tensor mask)
Tensor mean(int dimension, Tensor mask)
Tensor crossEntropyLoss(Tensor targets, int classDimension, Tensor mask)
Tensor crossEntropyLossFromIndices(Tensor targetIndices, int classDimension, Tensor mask)
```

Shape model:

- `linear` treats the last axis as features: input shape `[..., inFeatures]`, weight shape `[inFeatures, outFeatures]`, output shape `[..., outFeatures]`.
- `stack(axis, ...)` inserts a new axis, so three `[batch, features]` tensors can become `[batch, time, features]`.
- `unstack(axis)` removes one axis and returns one tensor per position on that axis.
- `sliceAxis` is a one-axis positive-step slice helper for contiguous ranges.
- `take` is the ergonomic ONNX Gather-style helper for explicit positions along one axis.
- Masked reductions and masked losses use `BOOL` masks where `true` means valid data and `false` means padding or ignored data.

Example:

```java
Tensor x = Tensor.randn(new int[]{2, 3, 4}, 0.0, 1.0, DataType.FLOAT64, "x");
Tensor w = Tensor.randn(new int[]{4, 5}, 0.0, 0.02, DataType.FLOAT64, "w");
Tensor b = Tensor.zeros(new int[]{5}, DataType.FLOAT64, "b");

Tensor projected = x.linear(w, b);
// x shape         = [batch, time, inFeatures]  = [2, 3, 4]
// projected shape = [batch, time, outFeatures] = [2, 3, 5]

Tensor mask = new Tensor(new byte[]{
        1, 1, 0,
        1, 0, 0
}, new int[]{2, 3}, null, "valid", DataType.BOOL);

Tensor pooled = projected.mean(1, mask);
// pooled shape = [2, 5]
// denominator is the number of true mask entries per batch, not the padded length 3
```

These methods are still tensor primitives. They do not add `Layer`, `Model`, `RNN`, `LSTM`, `GRU`, or any Neurotik-specific abstraction to Synaptik core. For the full contract with value examples and gradient notes, see [Sequence Tensor Primitives](sequence-tensor-primitives.md#scope).

### Compile And Execute From Tensor

Signatures:

```java
CompiledGraph compile()
CompiledGraph compile(CompileMode compileMode)
PreparedExecution prepare(ExecutionProfile profile)
Tensor compute()
Tensor compute(CompileMode compileMode)
Tensor compute(ComputeOptions options)
void compute(ExecutionProfile profile)
void compute(PreparedExecution execution, ExecutionMode mode)
```

Parameters:

- `CompileMode.INFERENCE_ONLY`: forward graph, inference defaults.
- `CompileMode.TRAINING`: forward/backward when the graph has trainable leaf tensors; otherwise forward only.
- `CompileMode.AUTO`: training if a leaf has `requiresGrad=true`, otherwise inference.
- `ExecutionProfile`: explicit optimizer, runtime, dtype, and execution mode.

Returns:

- `compute(...)` returning `Tensor` returns the root tensor after execution.
- `prepare(...)` returns a reusable `PreparedExecution`.

Failures:

- Null profile or execution objects throw `IllegalArgumentException`.
- Backward execution without backward support throws from `PreparedExecution`.
- Runtime backend failures propagate as unchecked exceptions.

Side effects:

- Execution writes output storage into graph tensors and publishes gradients for trainable leaves during `FORWARD_BACKWARD`.
- Convenience autotune can persist profile/history files under `build/tuning/tensor/...` when enabled.

Example:

```java
import tensor.CompileMode;
import tensor.DataType;
import tensor.Tensor;

Tensor x = new Tensor(new double[]{1.0, -2.0, 3.0}, new int[]{3}, null, "x", DataType.FLOAT64);
x.setRequiresGrad(true);

Tensor loss = x.mul(x).sum();
loss.compute(CompileMode.TRAINING);

System.out.println(loss.scalarAsDouble());
System.out.println(java.util.Arrays.toString(x.getGradient().toDoubleArrayCopy()));
```

Expected output:

```text
14.0
[2.0, -4.0, 6.0]
```

## ComputeOptions, CompileMode, And AutotunePolicy

**Sources:**

- `src/main/java/tensor/ComputeOptions.java`
- `src/main/java/tensor/CompileMode.java`
- `src/main/java/tensor/AutotunePolicy.java`

Signatures:

```java
CompileMode compileMode()
ComputeOptions compileMode(CompileMode compileMode)
AutotunePolicy autotunePolicy()
ComputeOptions autotune(AutotunePolicy autotunePolicy)
CompileConfig compile()
ComputeOptions compile(CompileConfig compile)
RuntimeConfig runtime()
ComputeOptions runtime(RuntimeConfig runtime)
```

Purpose: customize `Tensor.compute(ComputeOptions)` without manually building an `ExecutionProfile`.

For the full execution walkthrough, including code examples with concrete values and comments, see [Tensor API: Compute Convenience API](tensor-api.md#compute-convenience-api) and [Compute Flow: Tensor Compute API](compute-flow.md#tensor-compute-api).

Defaults:

- `compileMode`: `INFERENCE_ONLY`
- `autotunePolicy`: `NEVER`
- `compile`: inferred from compile mode unless set
- `runtime`: inferred from compile mode unless set

Option behavior:

| Option | Values | Meaning |
|---|---|---|
| `compileMode(CompileMode.INFERENCE_ONLY)` | forward-only | Uses inference compile/runtime defaults and executes `ExecutionMode.FORWARD`. |
| `compileMode(CompileMode.TRAINING)` | training intent | Uses training compile/runtime defaults; executes `FORWARD_BACKWARD` only if trainable leaf tensors exist. |
| `compileMode(CompileMode.AUTO)` | graph-sensitive | Chooses training behavior when a trainable leaf exists, otherwise inference behavior. |
| `autotune(AutotunePolicy.NEVER)` | default | Executes the resolved profile directly. |
| `autotune(AutotunePolicy.IF_MISSING)` | cache first | Reuses a matching generic best profile from `build/tuning/tensor/...` or runs one standard graph-autotune pass and persists the winner. |
| `autotune(AutotunePolicy.FORCE)` | always measure | Reruns generic graph autotune before execution and persists the new evidence. |
| `compile(CompileConfig)` | explicit config | Overrides semantic canonicalization, graph optimization, backend planning, region optimization, and memory planning in the generated execution profile. |
| `runtime(RuntimeConfig)` | explicit config | Overrides the runtime config used during prepare. |

Failure modes:

- Null compile mode resets to `INFERENCE_ONLY`.
- Null autotune policy resets to `NEVER`.
- Invalid compile/runtime objects fail when they are constructed or when compilation/preparation uses them.

Side effects:

- `ComputeOptions` is mutable and each setter returns `this`.
- `AutotunePolicy.IF_MISSING` reuses an existing generic best profile or persists a new winner.
- `AutotunePolicy.FORCE` always reruns generic graph autotune before execution.

Performance and concurrency notes:

- Autotune uses `GraphAutotuneMode.STANDARD`, `TuningPreset.BALANCED`, and a one-candidate search policy in the convenience path.
- `ComputeOptions` is not immutable; do not share and mutate one instance across concurrent executions.

Example:

```java
import tensor.AutotunePolicy;
import tensor.CompileMode;
import tensor.ComputeOptions;
import tensor.Tensor;

Tensor result = Tensor.scalar(2.0)
        .mul(3.0)
        .compute(new ComputeOptions()
                .compileMode(CompileMode.INFERENCE_ONLY)
                .autotune(AutotunePolicy.NEVER));

System.out.println(result.scalarAsDouble());
```

Expected output:

```text
6.0
```

## CompiledGraph

**Source:** `src/main/java/graph/CompiledGraph.java`

**Purpose:** compile a semantic tensor graph into stable compile artifacts, then prepare or execute it with explicit runtime configuration.

Signatures:

```java
static CompiledGraph compile(Tensor rootTensor, CompileConfig compileConfig)
static CompiledGraph compile(Tensor rootTensor, CompileConfig compileConfig, CompileMode compileMode)
static CompiledGraph compile(Tensor rootTensor, GraphOptimizer optimizer)
static CompiledGraph compile(Tensor rootTensor, GraphOptimizer optimizer, CompileMode compileMode)
void compile()
boolean supportsBackward()
CompileMode compileMode()
PreparedExecution prepare()
PreparedExecution prepare(RuntimeConfig runtimeConfig)
PreparedExecution prepare(ExecutionProfile profile)
void execute(RuntimeConfig runtimeConfig, ExecutionMode mode)
void execute(ExecutionProfile profile)
RunTrace executeTraced(RuntimeConfig runtimeConfig, ExecutionMode mode)
RunTrace executeTraced(ExecutionProfile profile)
void executePrepared(PreparedExecution execution, ExecutionMode mode)
void zeroGrad()
Tensor getRootTensor()
List<Tensor> getCompiledGraphAsList()
CompileTrace compileTrace()
CompileArtifacts compileArtifacts()
```

Parameters:

- `rootTensor`: graph root to compile.
- `compileConfig`: semantic canonicalization, graph optimization, backend planning, region optimization, and memory planning settings.
- `runtimeConfig`: kernel/backend runtime choices; `null` selects training or inference defaults based on backward support.
- `profile`: combines compile policy, runtime policy, dtype, and execution mode.

Returns:

- `compile(...)` returns a compiled graph and runs compilation during construction.
- `prepare(...)` returns a `PreparedExecution`.
- `executeTraced(...)` returns a `RunTrace`.

Failures:

- Null root, compile config, optimizer, or profile throws `IllegalArgumentException`.
- `compileArtifacts()` throws `IllegalStateException` if artifacts are absent.
- Invalid compile sub-configs throw when their records are constructed or used.

Side effects:

- `compile()` replaces the current compile artifacts and trace.
- `execute(...)` writes tensor output and gradient storage.
- `zeroGrad()` fills existing gradient tensors with zero.

Performance and concurrency notes:

- Compile and prepare are distinct. Reuse a `PreparedExecution` when running the same compiled graph repeatedly with the same runtime config.
- `CompiledGraph` holds mutable compile artifacts; avoid recompiling the same instance concurrently.

Example:

```java
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import tensor.DataType;
import tensor.Tensor;

Tensor x = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "x", DataType.FLOAT64);
Tensor y = x.mul(4.0);

CompiledGraph compiled = CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline());
compiled.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

System.out.println(java.util.Arrays.toString(y.toDoubleArrayCopy()));
```

Expected output:

```text
[4.0, 8.0]
```

## PreparedExecution

**Source:** `src/main/java/graph/execution/PreparedExecution.java`

**Purpose:** represent an executable runtime plan built from compiled graph artifacts and a `RuntimeConfig`.

Signatures:

```java
RuntimeConfig runtimeConfig()
boolean supportsBackward()
List<PreparedNodeExecution> forwardSteps()
List<PreparedNodeExecution> backwardSteps()
List<PreparedNodeExecution> executionSteps()
PrepareTrace prepareTrace()
void execute(ExecutionMode mode)
void execute(ExecutionMode mode, PublicationPolicy publicationPolicy)
RunTrace executeTraced(ExecutionMode mode)
RunTrace executeTraced(ExecutionMode mode, PublicationPolicy publicationPolicy)
void executeOptimizerStep(TrainingOptimizer optimizer)
void executeOptimizerStep(TrainingOptimizer optimizer, PublicationPolicy publicationPolicy)
RunTrace executeOptimizerStepTraced(TrainingOptimizer optimizer)
RunTrace executeOptimizerStepTraced(TrainingOptimizer optimizer, PublicationPolicy publicationPolicy)
void backward()
```

Parameters:

- `ExecutionMode.FORWARD`: executes forward steps.
- `ExecutionMode.FORWARD_BACKWARD`: seeds the root gradient, runs forward and backward steps, syncs root data, and publishes compiled gradients.

Returns:

- Step accessors return immutable copies created during construction.
- `executeTraced` returns a `RunTrace` with per-step metadata.

Failures:

- Null execution mode throws `NullPointerException`.
- `FORWARD_BACKWARD` on a forward-only prepared execution throws `IllegalStateException`.
- `backward()` prints `Info: No gradients to compute.` and returns when backward is unsupported.

Side effects:

- Execution binds runtime memory, executes backend kernels, and publishes values according to `PublicationPolicy`.

Performance and concurrency notes:

- A prepared execution avoids repeating compile and prepare work.
- Prepared execution instances reference mutable tensors and runtime memory state during execution; do not execute the same instance concurrently unless the call sites coordinate access.

## PublicationPolicy

**Source:** `src/main/java/graph/execution/PublicationPolicy.java`

Purpose: control which run-scoped values are synchronized back to user-visible tensors after execution.

Publication is an execution side-effect policy. It is not graph optimization, backend planning, memory planning, or runtime dispatch. The graph has already executed; publication only decides which computed values are copied back to public `Tensor` storage and gradient fields.

Values:

```java
PublicationPolicy.ALL
PublicationPolicy.OUTPUT_AND_GRADIENTS
PublicationPolicy.OUTPUT_ONLY
PublicationPolicy.NONE
```

Semantics:

| Policy | Publishes all forward values | Publishes root output | Publishes gradients |
|---|---:|---:|---:|
| `ALL` | yes | yes | yes |
| `OUTPUT_AND_GRADIENTS` | no | yes | yes |
| `OUTPUT_ONLY` | no | yes | no |
| `NONE` | no | no | no |

Defaults:

```java
PublicationPolicy.defaultExecution()      // OUTPUT_AND_GRADIENTS
PublicationPolicy.defaultOptimizerStep()  // OUTPUT_ONLY
```

Example:

```java
PreparedExecution prepared = CompiledGraph.compile(loss, CompileConfig.training())
        .prepare(RuntimeConfig.trainingDefaults());

prepared.execute(ExecutionMode.FORWARD_BACKWARD, PublicationPolicy.OUTPUT_AND_GRADIENTS);
```

Use `ALL` when a test or debug session needs intermediate tensor values. Use `OUTPUT_AND_GRADIENTS` for normal training-style execution. Use `OUTPUT_ONLY` when the output is needed but gradients should not be eagerly attached to public tensors. Use `NONE` for measurements where publication overhead would distort the thing being measured.

## Configuration APIs

### CompileConfig

**Source:** `src/main/java/config/compile/CompileConfig.java`

Purpose: configure compile-time semantic canonicalization, graph optimization, backend planning, region optimization, and memory planning.

Signatures:

```java
static CompileConfig training()
static CompileConfig inference()
static CompileConfig trainingAutoAccelerator()
static CompileConfig inferenceAutoAccelerator()
static CompileConfig trainingExplicitAccelerator()
static CompileConfig inferenceExplicitAccelerator()
static CompileConfig requireExplicitAccelerator()
static CompileConfig noGraphOptimization()
static CompileConfig noGraphOptimizationBaseline()
static CompileConfig cpuOnlyBaseline()
CompileConfig withSemanticCanonicalization(SemanticCanonicalizationConfig newConfig)
CompileConfig withGraphOptimization(GraphOptimizationConfig newConfig)
CompileConfig withBackendPlanning(BackendPlanningConfig newConfig)
CompileConfig withRegionOptimization(RegionOptimizationConfig newConfig)
CompileConfig withMemoryPlanning(MemoryPlanningConfig newConfig)
```

Failure modes:

- Null sub-configs are normalized to defaults by the record constructor.
- Backend planning rejects impossible combinations such as `CPU_ONLY` plus required accelerator regions.
- Lower-level configs such as `FuseConfig` and `MemoryConfig` validate their own numeric constraints.

Side effects: records and `with...` methods create new config objects.

Performance notes:

- `training()` and `inference()` both use backend-neutral graph optimization plus explicit backend intent planning.
- `noGraphOptimization()` disables graph cleanup/lowering only; it does not disable backend planning or runtime backend selection.
- `cpuOnlyBaseline()` disables graph optimization, accelerator planning, CPU natural regions, region optimization, and optional memory reuse for strict baseline comparisons.

### RuntimeConfig

**Source:** `src/main/java/config/runtime/RuntimeConfig.java`

Purpose: configure runtime kernel, BLAS, convolution, fused execution, approximation, and accelerator behavior.

Signatures:

```java
static RuntimeConfig trainingDefaults()
static RuntimeConfig inferenceDefaults()
static RuntimeConfig noOptNoVecNoPar()
CpuKernelConfig cpuKernelConfig()
RuntimeConfig withAccelerator(AcceleratorConfig newAccelerator)
```

Failure modes:

- `kernel` is required.
- Constructor overloads requiring `CpuKernelConfig` reject null CPU config.
- Sub-config records normalize nulls and non-positive thresholds as defined in their constructors.

Side effects: none; `RuntimeConfig` is a record and `withAccelerator` returns a new record.

Performance notes:

- `trainingDefaults()` uses CPU kernel training defaults, disabled BLAS, disabled conv2d BLAS, ASM fused execution with fallback, and default accelerator config.
- `inferenceDefaults()` uses CPU inference defaults and inference fused/accelerator defaults.
- `noOptNoVecNoPar()` disables practical vectorization, parallelism, and BLAS by using very high thresholds.
- `BlasConfig` can make OpenBLAS eligible through `BlasProvider.OPENBLAS_FFM`, but node-level BLAS dispatch still depends on dtype, shape, contiguity, and work gates. See [Native Bridges & BLAS: Dispatch Terms](native-bridges-and-blas.md#dispatch-terms).

### ExecutionProfile And PlatformRuntimeProfile

**Sources:**

- `src/main/java/config/profile/ExecutionProfile.java`
- `src/main/java/config/profile/PlatformRuntimeProfile.java`
- `src/main/java/config/profile/PlatformRuntimeProfileIO.java`

Purpose: carry compile/runtime policy and persist calibrated platform runtime values.

Signatures:

```java
new ExecutionProfile(String profileName, String candidateName, DataType dataType, ExecutionMode mode, CompileConfig compile, RuntimeConfig runtime)
new ExecutionProfile(String profileName, String candidateName, DataType dataType, ExecutionMode mode, CompileConfig compile, RuntimeConfig runtime, WorkloadProfile workload)
static PlatformRuntimeProfile fromExecutionProfile(String platformProfileId, String hardwareKey, String calibrationPreset, ExecutionProfile profile)
RuntimeConfig PlatformRuntimeProfile.toRuntimeConfig()
DataType PlatformRuntimeProfile.dataType()
static void PlatformRuntimeProfileIO.save(Path path, PlatformRuntimeProfile profile)
static String PlatformRuntimeProfileIO.toJson(PlatformRuntimeProfile profile)
static PlatformRuntimeProfile PlatformRuntimeProfileIO.loadOrDefault(Path path, PlatformRuntimeProfile fallback)
static PlatformRuntimeProfile PlatformRuntimeProfileIO.fromJsonOrDefault(String json, PlatformRuntimeProfile fallback)
```

Failure modes:

- `ExecutionProfile` requires non-null dtype, mode, compile, and runtime.
- `PlatformRuntimeProfile` requires metadata, matmul, fused, elementwise, reduction, scheduler, materialization, and numerics records.
- Profile save failures throw `IllegalStateException`.
- Load/parse failures return the provided fallback.

Side effects:

- `PlatformRuntimeProfileIO.save(...)` creates parent directories and writes UTF-8 JSON.
- `loadOrDefault(...)` reads from disk when the path exists.

Performance notes:

- `PlatformRuntimeProfile.toRuntimeConfig()` maps persisted thresholds and backend choices back into runtime config for prepared execution.
- `MatmulPlatformProfile` and `BlasConfig` currently normalize `blasThreads` to `0`.

## ONNX APIs

**Sources:**

- `src/main/java/onnx/Onnx.java`
- `src/main/java/onnx/OnnxModel.java`
- `src/main/java/onnx/ImportedOnnxModel.java`
- `src/main/java/onnx/OnnxExportOptions.java`
- `src/main/java/onnx/OnnxImportOptions.java`

Purpose: import and export the supported static dense ONNX inference subset.

Primary facade signatures:

```java
static OnnxModel Onnx.exportModel(Tensor output)
static OnnxModel Onnx.exportModel(Tensor output, OnnxExportOptions options)
static OnnxModel Onnx.exportModel(List<Tensor> outputs)
static OnnxModel Onnx.exportModel(List<Tensor> outputs, OnnxExportOptions options)
static void Onnx.write(Tensor output, Path path)
static void Onnx.write(Tensor output, Path path, OnnxExportOptions options)
static ImportedOnnxModel Onnx.read(Path path)
static ImportedOnnxModel Onnx.read(Path path, OnnxImportOptions options)
static ImportedOnnxModel Onnx.importModel(OnnxProto.ModelProto model)
static ImportedOnnxModel Onnx.importModel(OnnxProto.ModelProto model, OnnxImportOptions options)
```

Model wrapper signatures:

```java
void OnnxModel.write(Path path)
byte[] OnnxModel.toByteArray()

Map<String, Tensor> ImportedOnnxModel.inputs()
Map<String, Tensor> ImportedOnnxModel.outputs()
Tensor ImportedOnnxModel.input(String name)
Tensor ImportedOnnxModel.output(String name)
CompiledGraph ImportedOnnxModel.compile(CompileConfig config)
CompiledGraph ImportedOnnxModel.compile(String outputName, CompileConfig config)
```

Example:

```java
Tensor y = a.add(b).relu();
y.setLabel("y");

OnnxModel model = Onnx.exportModel(
        y,
        OnnxExportOptions.defaults()
                .withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS)
);
model.write(Path.of("build/quickstart/model.onnx"));

ImportedOnnxModel imported = Onnx.read(Path.of("build/quickstart/model.onnx"));
imported.input("a").setData(new float[]{1f, 2f, 3f, 4f});
imported.input("b").setData(new float[]{10f, 20f});
imported.compile("y", CompileConfig.inference())
        .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
```

Failure modes:

- Unsupported ONNX ops, custom domains, dynamic-shape runtime forms, unsupported dtypes, and unsupported attributes throw `OnnxUnsupportedException`.
- File IO failures are wrapped in `OnnxException`.
- `ImportedOnnxModel.compile(config)` requires exactly one ONNX output. Use `compile(outputName, config)` for multi-output imported models.

Boundary:

ONNX import/export support is not the same as accelerator-native support. A model can be importable, exportable, and CPU-executable while still falling back or being unsupported on a specific GPU backend. See [ONNX](onnx.md) and [ONNX Coverage](onnx-coverage.md).

## Tuning Fluent API

**Source:** `src/main/java/tuning/api/*.java`

Purpose: configure calibration, execution profiles, benchmark workflows, and benchmark report policy
from Java code without parsing CLI tokens or repeating long record constructors.

Primary entry point:

```java
Synaptik.tuning()
```

Implemented workflows:

- `Synaptik.tuning().calibration()` builds a `CalibrationCommand` and runs it with `CalibrationRunner`.
- `Synaptik.tuning().profile()` builds an immutable `ExecutionProfile`.
- `Synaptik.tuning().benchmark()` builds a `BenchmarkRequest` and runs it with `BenchmarkSession`.

Concrete calibration example:

```java
List<PlatformCalibrationResult> results = Synaptik.tuning()
        .calibration()
        .dtypes().single(DataType.FLOAT64)
        .families().all()
        .quick()
        .mode().training()
        .measurement().iterations(1, 3, 1)
        .progress().lines()
        .color().auto()
        .outputRoot(Path.of("profiles"))
        .run();
```

Concrete benchmark example:

```java
ExecutionProfile baselineProfile = Synaptik.tuning()
        .profile()
        .name("main-baseline-no-opt-f64")
        .candidate("baseline-no-opt")
        .dtype(DataType.FLOAT64)
        .mode().training()
        .compile().noGraphOptimization()
        .runtime().noOptNoVecNoPar()
        .build();

ExecutionProfile calibratedProfile = Synaptik.tuning()
        .profile()
        .name("main-calibrated-runtime-f64")
        .candidate("calibrated-runtime")
        .dtype(DataType.FLOAT64)
        .mode().training()
        .compile().trainingDefaults()
        .runtime().fromPlatformProfile(calibratedRuntime)
        .build();

BenchmarkReport report = Synaptik.tuning()
        .benchmark()
        .workload(workload)
        .quick()
        .report()
                .hotStepLimit(5)
                .includeTrace()
                .includeFailedCandidates()
                .done()
        .compare()
        .baseline("baseline-no-opt", baselineProfile)
        .candidate("calibrated-runtime", calibratedProfile)
        .run();

// baselineProfile.compile() = CompileConfig.noGraphOptimizationBaseline()
// baselineProfile.runtime() = RuntimeConfig.noOptNoVecNoPar()
// calibratedProfile.compile() = CompileConfig.training()
// calibratedProfile.runtime() = calibratedRuntime.toRuntimeConfig()
```

Execution-profile builder catalog:

| API | Purpose | Required? | Output |
|---|---|---:|---|
| `.name(String)` | Sets `ExecutionProfile.profileName`; null becomes `"default"` inside the record. | No | Parent builder |
| `.candidate(String)` | Sets benchmark/autotune candidate display name; blank/null falls back to profile name. | No | Parent builder |
| `.dtype(DataType)` | Selects dtype for the runnable profile. | Yes | Parent builder |
| `.mode().forward()` | Selects `ExecutionMode.FORWARD`. | No | Parent builder |
| `.mode().training()` / `.mode().forwardBackward()` | Selects `ExecutionMode.FORWARD_BACKWARD`. | No | Parent builder |
| `.compile().noGraphOptimization()` | Uses `CompileConfig.noGraphOptimizationBaseline()` for baseline comparison. | Yes, unless explicit compile policy is supplied | Parent builder |
| `.compile().inferenceDefaults()` | Uses inference compile defaults. | Yes, unless explicit compile policy is supplied | Parent builder |
| `.compile().trainingDefaults()` | Uses training compile defaults. | Yes, unless explicit compile policy is supplied | Parent builder |
| `.compile().trainingAutoAccelerator()` | Uses training compile defaults with automatic accelerator discovery. | Yes, unless explicit compile policy is supplied | Parent builder |
| `.compile().inferenceAutoAccelerator()` | Uses inference compile defaults with automatic accelerator discovery. | Yes, unless explicit compile policy is supplied | Parent builder |
| `.compile(CompileConfig)` / `.compile().config(...)` | Uses an explicit compile config. | Yes, unless selector is used | Parent builder |
| `.runtime().noOptNoVecNoPar()` | Uses the conservative runtime baseline with practical vector/parallel/BLAS paths disabled. | Yes, unless explicit runtime is supplied | Parent builder |
| `.runtime().inferenceDefaults()` | Uses inference runtime defaults. | Yes, unless explicit runtime is supplied | Parent builder |
| `.runtime().trainingDefaults()` | Uses training runtime defaults. | Yes, unless explicit runtime is supplied | Parent builder |
| `.runtime().fromPlatformProfile(PlatformRuntimeProfile)` | Converts a calibrated platform profile through `toRuntimeConfig()`. | Yes, unless explicit runtime is supplied | Parent builder |
| `.runtime(RuntimeConfig)` / `.runtime().config(...)` | Uses an explicit runtime config. | Yes, unless selector is used | Parent builder |
| `.workload(WorkloadProfile)` | Adds optional workload metadata; null becomes `WorkloadProfile.none()`. | No | Parent builder |
| `.build()` / `.toExecutionProfile()` | Creates the immutable `ExecutionProfile`. | n/a | `ExecutionProfile` |

Failure modes:

- `.build()` throws `IllegalStateException` if dtype, compile, or runtime has not been selected.
- `.dtype(null)`, `.compile(null)`, `.runtime(null)`, and
  `.runtime().fromPlatformProfile(null)` throw `NullPointerException`.
- The builder is mutable and intended for one thread while assembling one profile; the built
  `ExecutionProfile` is immutable.

## CLI Entry Point

**Source:** `src/main/java/synaptik/app/TuningCli.java`

Purpose: run calibration, graph autotune, and benchmark flows from Gradle.

Signatures:

```java
public static void main(String[] args)
```

Supported commands from `TuningCli.java`:

```bash
./gradlew run
./gradlew run --args="full <f64|f32|bf16>"
./gradlew run --args="calibrate --dtype <f64|f32|bf16> --family <family-id>"
./gradlew run --args="calibrate --dtype <f64|f32|bf16> --families all"
./gradlew run --args="calibrate --dtypes all --families all"
./gradlew run --args="autotune <f64|f32|bf16>"
./gradlew run --args="benchmark-winner <f64|f32|bf16>"
./gradlew run --args="benchmark-graph-space <f64|f32|bf16>"
```

Parameters:

- Dtypes accepted by `TuningCli`: `f64`, `f32`, `bf16`.
- Calibration options are parsed by `CalibrationCommand` and include `--preset`, `--mode`, `--measurement warmup:measure:repeats`, `--color`, `--progress`, `--output-root`, and `--include-accelerators`.

Failures:

- Unknown commands print usage and return.
- Wrong argument counts and unknown dtype values throw `IllegalArgumentException` after usage output.
- `autotune` requires an existing calibration profile and throws `IllegalStateException` if it is missing.
- `benchmark-winner` requires an existing best-profile artifact and throws `IllegalStateException` if it is missing.

Side effects:

- Calibration writes profile, manifest, history, and report artifacts under `profiles/platform/<platform-id>/calibration/schema-v2/...` by default.
- `autotune` writes `profiles/platform/<platform-id>/tuning/<workload-namespace>/<dtype>-best-profile.json`
  and `<dtype>-history.jsonl`. The default namespace is `abc`; transformer shapes such as
  `--workload transformer-block --shape large` use shape-specific namespaces such as
  `transformer_block_hot_path_large`.
- Benchmarks print text reports to standard output.

## Probably Internal APIs

These APIs are visible in Java but are implementation-oriented. Prefer the public surfaces above unless changing Synaptik itself.

| API | Source path | Why probably internal |
|---|---|---|
| `tensor.TensorOps` | `src/main/java/tensor/TensorOps.java` | Static delegation layer behind `Tensor` instance methods. |
| `tensor.TensorPrimitiveBuilder` | `src/main/java/tensor/TensorPrimitiveBuilder.java` | Constructs primitive graph nodes and operation descriptors directly. |
| `tensor.TensorStorage`, `Float64Storage`, `Float32Storage`, `BFloat16Storage`, `Int32Storage`, `BoolStorage` | `src/main/java/tensor/*.java` | Backing storage implementation, exposes mutable arrays. |
| `tensor.TensorInternalAccess` | `src/main/java/tensor/TensorInternalAccess.java` | Internal access helper for backend/graph packages. |
| `backend.ComputeEngine` | `src/main/java/backend/ComputeEngine.java` | Dispatches prepared compiled nodes; callers should use `Tensor`, `CompiledGraph`, or `PreparedExecution`. |
| `backend.memory.StorageResidency`, `backend.memory.TensorResidencyState`, `backend.memory.CpuMaterializationReason`, `backend.memory.CpuMaterializationResult`, `backend.memory.DeviceBufferBinding`, `backend.memory.DeviceToCpuMaterializer`, `backend.memory.ExecutionResource` | `src/main/java/backend/memory/*.java` | Public Java types because execution/trace code crosses packages, but they describe per-run runtime storage state rather than an application-facing tensor storage API. `CpuMaterializationReason` is diagnostic; it names why CPU storage is required. `DeviceBufferBinding` is the backend-neutral handle contract used by shared-buffer execution paths. `DeviceToCpuMaterializer` synchronizes an active device binding into CPU storage; `ExecutionResource` scopes native resource cleanup to one execution run. |
| `graph.execution.trace.CpuMaterializationTrace` | `src/main/java/graph/execution/trace/CpuMaterializationTrace.java` | Run-trace record for CPU-readable storage requests. It is observability for residency/materialization decisions, not a public tensor materialization API. |
| `backend.metal.buffer.MetalBufferAccess`, `MetalBufferHandle`, `MetalBufferBinding`, `MetalBufferAllocator`, `MetalDeviceToCpuMaterializer`, `MetalBufferResource` | `src/main/java/backend/metal/buffer/*.java` | Java-side contract for native shared-buffer Metal execution. These remain internal/SPI-oriented rather than application APIs: they are tied to compiled node ids, run-scoped native handles, and execution-state ownership. See [Metal Backend: Buffer Residency And Materialization](metal-backend.md#buffer-residency-and-materialization). |
| `backend.metal.bridge.MetalMpsBridgeExecutionStats`, `MetalMpsBridgeExecutionPath` | `src/main/java/backend/metal/bridge/MetalMpsBridgeExecutionStats.java`, `src/main/java/backend/metal/bridge/MetalMpsBridgeExecutionPath.java` | Trace/report diagnostics for Metal bridge executions and fallbacks, surfaced through run trace attributes and benchmark reports rather than through normal tensor APIs. |
| `backend.cpu.*`, `backend.metal.*`, `backend.cuda.*` | `src/main/java/backend/**/*.java` | Kernel implementations and native bridge plumbing. |
| `graph.compile.*`, `graph.optimizer.*` | `src/main/java/graph/**/*.java` | Compile pipeline internals; only `CompiledGraph` is the general entry point. |
| `tuning.candidate.*`, `tuning.search.*`, `tuning.measure.*` | `src/main/java/tuning/**/*.java` | Extensible tuning machinery; stable application workflows should use execution profiles, platform profiles, or the CLI. |

## Verification Notes

The examples in this file were checked against compiled classes using the Gradle build output and targeted tests:

```bash
./gradlew classes
./gradlew test --tests BroadcastBinaryOpsTest --tests CompiledGraphIdempotencyTest --tests PreparedExecutionBuildTest --tests ExecutionProfileIoTest
```
