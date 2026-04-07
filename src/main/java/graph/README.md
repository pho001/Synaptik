# Graph (src/main/java/graph)

## Contents

- [Purpose](#purpose)
- [Main Components](#main-components)
- [Compile Pipeline](#compile-pipeline)
- [Runtime Preparation](#runtime-preparation)
- [Execution Modes](#execution-modes)
- [Execution Pipeline](#execution-pipeline)
- [Fused Codegen Path](#fused-codegen-path)
- [Backward Graph Notes](#backward-graph-notes)
- [Canonicalization Notes](#canonicalization-notes)
- [Related Modules](#related-modules)

## Purpose

The `graph` package turns a tensor expression DAG into explicit execution artifacts:

- `CompiledGraph`: compile-time artifact
- `PreparedExecution`: runtime-bound artifact

This layer owns:

- forward/backward graph assembly
- optimizer application
- per-node prepared execution metadata
- fused-kernel preparation

## Main Components

- Compile artifact:
  - [src/main/java/graph/CompiledGraph.java](../graph/CompiledGraph.java)
- Runtime artifact:
  - [src/main/java/graph/execution/PreparedExecution.java](../graph/execution/PreparedExecution.java)
  - [src/main/java/graph/execution/PreparedNodeExecution.java](../graph/execution/PreparedNodeExecution.java)
  - [src/main/java/graph/execution/CompiledNodeExecutionMetadata.java](../graph/execution/CompiledNodeExecutionMetadata.java)
- Fused runtime:
  - [src/main/java/graph/codegen/FusedExpressionPlan.java](../graph/codegen/FusedExpressionPlan.java)
  - [src/main/java/graph/codegen/FusedKernelGeneratorRouter.java](../graph/codegen/FusedKernelGeneratorRouter.java)
  - [src/main/java/graph/codegen/FusedOperationGenerator.java](../graph/codegen/FusedOperationGenerator.java)
  - [src/main/java/graph/fused/FusedExecutionBackendResolver.java](../graph/fused/FusedExecutionBackendResolver.java)
  - [src/main/java/graph/fused/PreparedFusedExecutable.java](../graph/fused/PreparedFusedExecutable.java)
- Optimizer module:
  - [src/main/java/graph/optimizer/README.md](../graph/optimizer/README.md)

## Compile Pipeline

`CompiledGraph` performs the following high-level steps:

1. create a forward execution anchor via `rootTensor.forwardOutput()`
2. topologically sort the forward graph
3. if the graph has differentiable leaf inputs:
   - seed root gradient
   - build backward graph nodes
   - collect backward targets
   - create a temporary super-root to unify forward and backward sinks
4. run the configured optimizer over the assembled graph
5. record the boundary between forward and backward execution sections

The core public entry points are:

- `CompiledGraph.compile(Tensor root, OptimizerConfig optimizerConfig)`

Internal / lower-level tooling may still construct a `GraphOptimizer` and pass it into package-local compile paths, but the supported public compile contract is config-based:

- `CompiledGraph.compile(Tensor root, OptimizerConfig optimizerConfig)`

## Runtime Preparation

`CompiledGraph.prepare(...)` converts the compiled graph into a runtime-bound `PreparedExecution`.

Preparation builds:

- ordered forward steps
- ordered backward steps
- per-node backend metadata
- pre-resolved CPU execution plans
- prepared fused runtime executables where the resolved compute mode needs them

For fused nodes this now means:

- prepare resolves a fused execution backend through `FusedExecutionBackendResolver`
- resolver reads fused backend policy from `RuntimeConfig.fused()`
- `FLOAT64` currently resolves to direct fused backend when `Vector API` is available
- `FLOAT32` currently resolves to direct fused backend when `Vector API` is available
- `BF16_F32_COMPUTE` currently resolves to direct fused backend when `Vector API` is available
- ASM remains the fused fallback backend when direct fused backend is unavailable or does not support the runtime combination
- prepared metadata stores one unified `PreparedFusedExecutable`, regardless of which backend produced it

Current fused policy knobs are:

- primary backend:
  - `DIRECT_VECTOR`
  - `ASM`
- whether backend fallback is allowed
- whether compare/select chains should still prefer direct backend

The vector-length threshold for direct fused backend selection is now unified under:

- `cpu.vectorMinSize`

That threshold therefore has one owner and is shared with the CPU planner instead of being duplicated inside fused policy.

That means fused backend selection is no longer a hardcoded router rule.
It is part of the execution profile surface and can therefore be autotuned.

This preparation is runtime-specific because it depends on:

- `RuntimeConfig`
- backend planner thresholds
- approximation policy
- BLAS policy

## Execution Modes

The execution engine now uses engine-oriented names:

- `ExecutionMode.FORWARD`
- `ExecutionMode.FORWARD_BACKWARD`

Semantics:

- `FORWARD`
  - execute only prepared forward steps
- `FORWARD_BACKWARD`
  - execute prepared forward steps
  - synchronize output back to the root tensor
  - seed root gradient
  - execute prepared backward steps

Backward capability is exposed via:

- `CompiledGraph.supportsBackward()`
- `PreparedExecution.supportsBackward()`

## Execution Pipeline

`PreparedExecution.execute(mode)`:

1. creates `ExecutionContext`
2. iterates prepared forward steps in topological order
3. syncs root tensor data from the optimized result node
4. if mode is `FORWARD_BACKWARD`, zeroes gradients, seeds the root gradient, and runs backward steps

Actual kernel dispatch then goes through:

- [src/main/java/backend/ComputeEngine.java](../backend/ComputeEngine.java)

## Fused Codegen Path

The fused path exists to turn a small compute-only subgraph into one prepared runtime kernel.

The key idea is:

- keep the graph-level semantic structure
- but remove unnecessary intermediate tensor materialization for hot arithmetic chains

### What "fused" means here

Given a graph like:

```java
Tensor out = a.add(b).relu().exp();
```

the non-fused execution model is conceptually:

1. compute `tmp0 = a + b`
2. materialize/store `tmp0`
3. compute `tmp1 = relu(tmp0)`
4. materialize/store `tmp1`
5. compute `out = exp(tmp1)`

The fused execution model instead generates one runtime kernel that computes:

```java
// For each logical output element i:
double v0 = a[i] + b[i];
double v1 = Math.max(v0, 0.0);
out[i] = Math.exp(v1);
```

That is the entire purpose of the fused optimizer/codegen path:

- one output-space loop
- no intermediate tensor materialization
- explicit prepared runtime kernel

### Architectural split

The current fused architecture is intentionally split into two layers:

1. **graph descriptor layer**
   - `FusedOperation`
   - `FusedExpressionPlan`
   - `FusedNodePlan`
   - `FusedExternalInputPlan`
2. **prepared runtime layer**
   - `FusedExecutionBackendResolver`
   - `PreparedFusedExecutable`
   - `vector` backend
   - `asm` backend
   - `CpuFusedKernel`

That split matters because:

- the graph should keep semantic meaning
- prepared runtime state is runtime-specific and must not leak back into the graph descriptor

So:

- `FusedOperation` is only a descriptor
- the generated runtime executable is created later during `CompiledGraph.prepare(...)`
- runtime compiled code is stored in prepared metadata, not on the descriptor itself

### End-to-end fused flow

When `FuseElementWiseRule` collapses a cluster:

1. the optimizer identifies a fused-compute cluster
2. `FusedOperationFactory` validates its external input access chains
3. `FusedOperationFactory` resolves backing runtime inputs and builds `FusedExpressionPlan`
4. the optimized graph keeps a single `FusedOperation` descriptor node
5. `CompiledGraph.prepare(...)` resolves a CPU node-level compute mode
6. `FusedExecutionBackendResolver` selects the execution backend for that fused node
7. the `vector` backend builds a direct prepared executable, while the `asm` backend goes through `FusedKernelGeneratorRouter`
8. the resulting `PreparedFusedExecutable` is stored in `CompiledNodeExecutionMetadata`
9. `CpuFusedKernel` executes that prepared executable without knowing whether it came from direct vector runtime or ASM codegen

Important note:

- tensor storage dtype and compute mode are now distinct concepts
- example: a `BFLOAT16` tensor may execute as `BF16_F32_COMPUTE` or `BF16_BLAS`
- the mode is resolved during prepare and stored in the execution recipe
- F32/F64/BF16 direct fused execution no longer pays prepare-time ASM compilation cost when the resolver selects the direct fused backend
- direct fused execution may use vector fast paths for contiguous numeric chains, including arithmetic, clamp, abs, sqrt, exp/log/tanh family, sigmoid and selected `pow` exponents
- mixed-bool/broadcast-heavy chains still stay on scalar direct fallback
- ASM now exists as a fused execution backend fallback, not as the only prepared runtime model

Related BF16 GEMM note:

- compound kernels such as `LINEAR` and `CONV2D_GEMM` may keep intermediate GEMM output in `float[]` workspace through their internal post-processing phase
- that is an intra-kernel continuation optimization, not yet a cross-node prepared execution contract

There is one first prepared-execution continuation contract already active:

- inference-only `BF16_BLAS` `MATMUL` / `LINEAR`
- exactly one supported BF16 consumer
- producer publishes float continuation into prepared workspace
- consumer reads that float continuation directly and only then materializes to `BFLOAT16`

That consumer subset is currently:

- unary BF16 post-ops
- binary no-broadcast BF16 ops (`ADD/SUB/MUL/DIV/MIN/MAX`)
- numeric contiguous BF16 fused consumer chains

So the first deferred-materialization path exists, but it is intentionally constrained to a small safe subset instead of being exposed as a general tensor runtime contract.

### Example: simple numeric fusion

Graph:

```java
Tensor a = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "a", DataType.FLOAT64);
Tensor b = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "b", DataType.FLOAT64);
Tensor out = a.add(b).relu().exp();
```

Logical fused interpretation:

```java
// Inputs:
// a = [1, 2, 3]
// b = [10, 20, 30]
//
// Elementwise:
// v0 = a + b      = [11, 22, 33]
// v1 = relu(v0)   = [11, 22, 33]
// out = exp(v1)
//
// Returns:
// out = [exp(11), exp(22), exp(33)]
```

No intermediate `Tensor tmp0` or `Tensor tmp1` is needed at runtime.

### Example: fused compare/select flow

Graph:

```java
Tensor cond = a.greaterThan(b);
Tensor out = Tensor.where(cond, x, y).mul(x);
```

Logical fused interpretation:

```java
// Inputs:
// a = [1, 5, 3, 8]
// b = [2, 4, 3, 1]
// x = [10, 20, 30, 40]
// y = [100, 200, 300, 400]
//
// Step 1: cond = a > b
// cond = [false, true, false, true]
//
// Step 2: select branch
// where(cond, x, y) = [100, 20, 300, 40]
//
// Step 3: multiply by x
// out = [1000, 400, 9000, 1600]
```

Here the fused kernel internally carries:

- numeric intermediates
- bool intermediates

without materializing a separate bool tensor for `cond` inside the runtime kernel body.

### Fused compute scope

The fused compute algebra is intentionally narrow.

Current fused compute nodes:

- unary/binary numeric ops
- compare ops
- logical ops
- `where`

That means the fused compiler is designed for:

- one logical output space
- one loop nest over that output space
- local per-element compute

### Access algebra vs compute algebra

This is the most important design rule.

Operations such as:

- `select`
- `permute`
- `expand`
- `reshape`
- `expandDims`
- `squeeze`

are **not** fused compute nodes.

They do not compute a new arithmetic value.
They only change:

- where the value is read from
- how logical coordinates map into backing storage

So they are absorbed into fused external input access metadata.

### Example: absorbed view chain

Graph:

```java
Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base", DataType.FLOAT64);
Tensor view = base.select(0, 1);
Tensor out = view.relu().exp();
```

Logical meaning:

```java
// base as [2, 3]:
// [[1, 2, 3],
//  [4, 5, 6]]
//
// view = base.select(0, 1)
// view = [4, 5, 6]
//
// out = exp(relu(view))
```

In the fused runtime path:

- `view` is not passed as a separate runtime tensor input
- the backing tensor `base` is passed instead
- the fused external input descriptor carries:
  - logical output shape
  - effective strides
  - storage offset
  - access kind

So the fused runtime kernel still computes the correct `view` semantics,
but without treating `select(...)` as a fused compute node.

### Access metadata

The key fused input descriptor is:

- [src/main/java/graph/codegen/FusedExternalInputPlan.java](../graph/codegen/FusedExternalInputPlan.java)

It carries:

- `inputIndex`
  - which external input this is
- `dataType`
  - input dtype (`FLOAT64`, `FLOAT32`, `BFLOAT16`, `BOOL`)
- `logicalOutputShape`
  - fused kernel output logical space
- `logicalOutputDenseStrides`
  - dense row-major strides for that logical output space
- `storageOffset`
  - where the backing view starts in storage
- `effectiveStrides`
  - how logical output coordinates map to storage
- `accessKind`
  - coarse access topology classification

Current access kinds:

- `DIRECT_CONTIGUOUS`
- `DIRECT_STRIDED`
- `OFFSET_CONTIGUOUS`
- `BROADCAST_STRIDED`
- `OFFSET_STRIDED`

Those access kinds are used for:

- scheduler signature
- cost model
- later autotune-oriented policy decisions

### Fusion barriers

The fused compiler intentionally stops at operations that no longer fit
the "one output-space loop + local element compute" model.

Current barriers:

- indexing ops:
  - `gather`
  - `takeAlongAxis`
  - `scatterAdd`
- reductions:
  - `sum`
  - `mean`
  - `reduceMin`
  - `reduceMax`
  - `reduceAll`
  - `reduceAny`
- special reductions / losses:
  - `softmax`
  - `logSoftmax`
  - `nllLoss`
  - `crossEntropyLoss`
- linear algebra:
  - `matmul`
- special grad kernels

### Example: barrier behavior

Graph:

```java
Tensor out = base.gather(indices, 1).relu().exp();
```

Behavior:

- `gather(...)` remains outside the fused cluster
- only `relu().exp()` may fuse on top of the gather result

That is intentional.
`gather` is not treated as a fused access transform or a fused compute node.

### Scalar vs vector fused execution

Prepared fused executables expose two methods:

- `applyRangeScalar(...)`
- `applyRangeVector(...)`

Vector execution is used when:

- dtype and fused node mix support it
- the planner recommends vector mode

Scalar execution is used when:

- vector mode is not profitable
- or the fused cluster contains semantics that currently stay scalar-only

Important current rule:

- scalar path is the correctness baseline
- vector path is enabled only where the runtime contract is explicit
- the direct fused backend and the ASM backend both implement the same prepared executable contract
- unsupported vector cases may fall back either to scalar direct execution or to the ASM backend, depending on resolver support

### Example: vector-friendly fused cluster

```java
Tensor out = a.add(b).mul(c).relu();
```

This is vector-friendly because:

- all inputs are numeric
- all ops are numeric compute nodes
- there is no bool output
- there is no indexing/reduction barrier

### Example: scalar fallback fused cluster

```java
Tensor out = Tensor.where(a.greaterThan(b), x, y).relu();
```

This may still fuse,
but if a specific dtype/path is not fully vectorized,
the fused runtime can fall back to scalar fused execution while still keeping:

- one fused node
- one prepared kernel
- one output-space loop

That is still correct and still materially better than leaving the graph fully unfused.

## Backward Graph Notes

Backward nodes are built from gradients attached to forward nodes and marked with:

- `tensor.setBackward(true)`

This marker lets optimizer rules preserve phase boundaries between forward and backward sections.

## Canonicalization Notes

Algebraic rewriting includes canonical sigmoid recognition in forward-only inference-style graphs:

- `1 / (1 + exp(-x)) -> sigmoid(x)`
- also recognizes the `mulScalar(-1)` form

For graphs that require gradients, this rewrite is intentionally skipped to preserve the current backward-graph construction semantics.

## Related Modules

- Tensor front-end:
  - [src/main/java/tensor/README.md](../tensor/README.md)
- Backend execution:
  - [src/main/java/backend/README.md](../backend/README.md)
- Optimizer:
  - [src/main/java/graph/optimizer/README.md](../graph/optimizer/README.md)
- Numerics tooling:
  - [src/main/java/numerics/README.md](../numerics/README.md)
