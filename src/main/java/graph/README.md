# Graph

The `graph` layer turns a tensor expression DAG into an explicit runnable artifact. That is its single main job. It is not a backend and it is not the public tensor surface.

The current contract is:

- `Tensor` builds the semantic DAG
- `CompiledGraph` turns it into an optimized execution graph
- `PreparedExecution` attaches runtime-specific metadata
- the backend then executes prepared node steps

## Reading Guide

Go here if you need to understand:

- what `compile(...)` actually does
- what `prepare(...)` actually does
- where the forward/backward boundary is created
- how fused executables are prepared
- how hot-path trace data flow back into benchmarks and debug tooling

Related documentation:

- tensor/public API: [../tensor/README.md](../tensor/README.md)
- operation descriptors: [../operations/README.md](../operations/README.md)
- optimizer pipeline: [../graph/optimizer/README.md](../graph/optimizer/README.md)
- backend execution: [../backend/README.md](../backend/README.md)

## Main Components

- compile artifact
  - [CompiledGraph.java](../graph/CompiledGraph.java)
- prepared runtime artifact
  - [PreparedExecution.java](../graph/execution/PreparedExecution.java)
  - [PreparedNodeExecution.java](../graph/execution/PreparedNodeExecution.java)
  - [CompiledNodeExecutionMetadata.java](../graph/execution/CompiledNodeExecutionMetadata.java)
- tracing
  - [CompileTrace.java](../graph/execution/trace/CompileTrace.java)
  - [PrepareTrace.java](../graph/execution/trace/PrepareTrace.java)
  - [RunTrace.java](../graph/execution/trace/RunTrace.java)
- fused preparation
  - [FusedExecutionPlan.java](../graph/fused/FusedExecutionPlan.java)
  - [FusedExecutionBackendResolver.java](../graph/fused/FusedExecutionBackendResolver.java)
  - [PreparedFusedExecutable.java](../graph/fused/PreparedFusedExecutable.java)

## Lifecycle

The most important thing is to keep three different artifacts clearly separated:

### 1. `Tensor` graph

A semantic graph built from public tensor operations.

It contains:

- operation descriptors
- input dependencies
- gradient references
- metadata and runtime data storage

It does not contain:

- prepared backend metadata
- runtime dispatch hints
- a prepared fused executable

### 2. `CompiledGraph`

A compile-time artifact.

It contains:

- final topological node order
- forward/backward section separation
- optimizer output
- compile trace

### 3. `PreparedExecution`

A runtime-bound artifact.

It contains:

- ordered prepared forward steps
- ordered prepared backward steps
- per-node prepared metadata
- the runtime config with which the graph was prepared
- prepare trace

`PreparedExecution` is the artifact you should keep for repeated hot execution over the same graph.

Just as importantly, `PreparedExecution` is where backend policy becomes concrete:

- layout materialization decisions are already fixed
- elementwise dispatch hints are already fixed
- reduction chunking/vectorization hints are already fixed
- matmul / conv2d / attention backend choices are already fixed

Executors should not "look back up" and re-run planner logic at runtime.

## Compile Pipeline

`CompiledGraph.compile(root, optimizerConfig)` currently performs this flow:

1. takes `rootTensor.forwardOutput()`
2. performs topological sort of the forward closure
3. if the graph has no trainable leaf inputs:
   - optimizes only the forward graph
   - stores the boundary at the forward output
4. if the graph supports backward:
   - seeds the root gradient
   - calls `buildBackwardGraph()` in reverse forward order
   - collects backward targets
   - creates a temporary `noop` super-root to unify sinks
   - optimizes the whole combined graph
5. stores the forward-section end index

This means the optimizer runs over one larger graph that may contain both forward and backward sections.

## Forward/Backward Boundary

The boundary is not inferred later at runtime. It is stored explicitly inside `CompiledGraph`.

That has several consequences:

- optimizer rules must respect forward/backward phase boundaries
- tracing can separate forward and backward steps
- `PreparedExecution` does not need to guess which section is which

Backward support is visible through:

- `CompiledGraph.supportsBackward()`
- `PreparedExecution.supportsBackward()`

## Prepare Pipeline

`CompiledGraph.prepare(runtimeConfig)` is a runtime-specific step. It is not just "copy `finalGraph` into another object".

It actually does:

1. chooses the effective runtime config
   - either the explicit input
   - or inference/training defaults depending on backward support
2. creates `CpuExecutionPlanner`
3. iterates `finalGraph`
4. prepares `CompiledNodeExecutionMetadata` for each executable node
5. splits prepared steps into forward/backward
6. returns `PreparedExecution`

Prepared metadata contain, depending on node type:

- the resolved backend
- `CpuKernel`
- `CpuNodeExecutionPlan`
- `PreparedFusedExecutable`
- `CpuNodeWorkspace`

## What Prepare Resolves

This is the key point: `prepare(...)` resolves decisions that we do not want to make inside the hot inner loop.

Typical examples:

- input materialization / prepared inputs
- broadcast plan
- `where` broadcast plan
- compute contract
- dispatch hints
- reduction hints
- matmul hints
- fused executable generation
- workspace allocation
- selected BF16 continuation policies

## Execution Flow

`PreparedExecution.execute(mode)` does the following:

1. creates `ExecutionContext`
2. runs prepared forward steps in topological order
3. synchronizes data from the optimized forward output back into the original root tensor
4. if the mode is `FORWARD_BACKWARD`:
   - zeros gradients
   - seeds the root gradient with ones
   - runs prepared backward steps

This matters for correct benchmarking:

- compile overhead does not belong in steady-state numbers
- prepare overhead does not belong in steady-state numbers
- repeated execution should run over one `PreparedExecution`

## Traced Execution

`PreparedExecution.executeTraced(...)` returns `RunTrace`.

Each step trace contains:

- node label
- `Operation.OpType`
- shape
- dtype
- backend
- kernel class
- step duration
- structured metadata

Structured metadata include, for example:

- compute metadata
- layout metadata
- dispatch metadata
- reduction metadata
- matmul metadata
- fused metadata

Practical value:

- you can verify that a benchmark really ran on the expected path
- you can inspect `vectorWidth`, worker count, tile sizes, and BLAS use
- you can find scalar fallbacks or unexpected strided paths

## Fused Preparation

Fused execution has two distinct layers:

### Graph descriptor layer

- `FusedOperation`
- `FusedExpressionPlan`
- `FusedExternalInputPlan`
- other codegen/fusion helper descriptors

This is still graph-level representation.

### Prepared runtime layer

- `FusedExecutionPlan`
- `PreparedFusedExecutable`
- `FusedExecutionBackendResolver`

This is already runtime-specific executable state.

## Current Fused Reality

This area historically had the most drift between documentation and code, so explicitly:

- the optimizer may create a fused node
- `prepare(...)` computes a fused execution plan for it
- `FusedExecutionBackendResolver` currently uses the ASM fused backend
- if the ASM backend does not support the plan, prepare fails

So:

- the fused path is no longer a direct/vector hybrid backend with fallback
- the prepared fused executable is currently an ASM-generated executable
- runtime scheduling above it can still be scalar/vector/parallel according to prepared dispatch hints

`PreparedFusedExecutable` exposes the contract:

- `applyRangeScalar(...)`
- `applyRangeVector(...)`

The default vector implementation on the interface falls back to scalar. Real vectorization therefore depends on the concrete prepared implementation.

## Fused Access Model

The fused compiler distinguishes not only "which operation is computed" but also "how input tensors are accessed".

This is why it separates:

- compute algebra
- access algebra

### Compute algebra

This is fused per-element computation:

- unary numeric ops
- binary numeric ops
- compare ops
- logical ops
- `where`

### Access algebra

This is view/layout transformation on the input:

- `SELECT`
- `PERMUTE`
- `EXPAND`
- `RESHAPE`
- `EXPAND_DIMS`
- `SQUEEZE`

These are not treated as fused compute nodes. They are absorbed into `FusedExternalInputPlan`.

That enables:

- one output-space loop
- no intermediate tensors
- but still correct stride/offset/broadcast mapping to backing storage

## Example: Fused Arithmetic Chain

```java
Tensor out = a.add(b).relu().exp();
```

The non-fused runtime model conceptually does:

1. `tmp0 = a + b`
2. materialize `tmp0`
3. `tmp1 = relu(tmp0)`
4. materialize `tmp1`
5. `out = exp(tmp1)`

The fused runtime model instead does:

```java
for (int i = 0; i < out.numel(); i++) {
    double v0 = a[i] + b[i];
    double v1 = Math.max(v0, 0.0);
    out[i] = Math.exp(v1);
}
```

That is the whole point of the fused path:

- remove intermediate materialization
- reduce dispatch overhead
- keep the computation inside one output-space loop

## Example: Access Chain Absorption

```java
Tensor base = ...;
Tensor view = base.select(0, 1).permute(1, 0);
Tensor out = view.relu().exp();
```

What happens:

- `relu` and `exp` are fused compute nodes
- `select` and `permute` are not fused compute nodes
- the fused node receives the backing tensor `base` as an external input
- access metadata describe offset/strides/logical mapping

This matters architecturally as well:

- the graph still carries the semantics of the view operations
- the runtime fused executable does not receive the whole graph
- it only receives the already resolved prepared access contract

## Barriers

The fused cluster cannot absorb everything. Today the following typically act as barriers:

- indexing
  - `GATHER`
  - `TAKE_ALONG_AXIS`
  - `SCATTER_ADD`
- reductions
  - `SUM`
  - `MEAN`
  - `REDUCE_MIN`
  - `REDUCE_MAX`
  - `REDUCE_ALL`
  - `REDUCE_ANY`
  - `SOFTMAX`
  - `LOG_SOFTMAX`
- linear algebra
  - `MATMUL`
- losses and special structured kernels
- special gradient kernels

That is intentional. Those families have their own traversal/kernel logic and are not merely local per-element algebra.

## Relationship To Optimizer Rewrites

The graph layer itself does not rewrite anything. It only applies the optimizer pipeline. But it is important to understand that after optimization the graph may already contain specialized primitives instead of decomposed patterns.

For example:

- `matmul + bias` may be rewritten into `LINEAR`
- an attention pattern may be rewritten into `SCALED_DOT_PRODUCT_ATTENTION`
- a backward softmax pattern may be rewritten into `SOFTMAX_GRAD`
- a cross-entropy-from-indices pattern may be rewritten into `CROSS_ENTROPY_LOSS_INDICES`

The graph layer then treats that as the compile-time reality and prepares metadata for the resulting descriptor.

## Workspaces

Some nodes require extra prepared workspace. `CompiledGraph` allocates it during the prepare phase.

Examples:

- max-pool argmax buffer
- BF16 float workspace for `MATMUL`
- packed weights workspace for `LINEAR`
- float workspace for selected continuation paths

Why this matters:

- workspace is not allocated ad hoc inside every hot kernel call
- prepared metadata explicitly tell you which node has which workspace

## Example: Explicit Compile / Prepare Reuse

```java
Tensor out = logits.logSoftmax(-1).sum();

CompiledGraph graph = CompiledGraph.compile(out, OptimizerConfig.trainingDefaults());
PreparedExecution prepared = graph.prepare(RuntimeConfig.trainingDefaults());

prepared.execute(ExecutionMode.FORWARD_BACKWARD);
prepared.execute(ExecutionMode.FORWARD_BACKWARD);
```

Use this for:

- performance measurements
- trace collection
- repeated inference/training runs

## Example: Trace Audit

A typical performance audit looks like this:

1. build the graph through the `Tensor` API
2. call `CompiledGraph.compile(...)`
3. `PreparedExecution prepared = graph.prepare(...)`
4. `RunTrace trace = prepared.executeTraced(...)`
5. analyze step metadata

Pay special attention to:

- kernel class
- `compute.backend`
- dispatch mode
- `vectorWidth`
- `plannedWorkers`
- matmul `useBlas`
- fused executable class

## Public Entry Points

Publicly relevant entry points:

- `CompiledGraph.compile(Tensor root, OptimizerConfig optimizerConfig)`
- `CompiledGraph.prepare(RuntimeConfig runtimeConfig)`
- `CompiledGraph.execute(...)`
- `CompiledGraph.executeTraced(...)`
- `PreparedExecution.execute(...)`
- `PreparedExecution.executeTraced(...)`

Lower-level `GraphOptimizer` injection still exists, but it is not the preferred public compile contract.

## Common Mistakes

- benchmarking `Tensor.compute(profile)` instead of reusing `PreparedExecution`
- treating an `Operation` descriptor as if it were the hot executable
- assuming `prepare(...)` is just a cheap wrapper with no runtime decisions
- thinking a fused node is already a compiled ASM class inside the optimizer
- mixing graph policy and runtime policy in the same conceptual layer

## Related Modules

- tensor: [../tensor/README.md](../tensor/README.md)
- operations: [../operations/README.md](../operations/README.md)
- optimizer: [../graph/optimizer/README.md](../graph/optimizer/README.md)
- backend: [../backend/README.md](../backend/README.md)
- numerics: [../numerics/README.md](../numerics/README.md)
