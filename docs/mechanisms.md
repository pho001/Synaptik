<!-- generated-by: gsd-doc-writer -->
# Mechanisms

Navigation: [Index](index.md#recommended-reading-paths) | [Compute Flow](compute-flow.md#lifecycle-map) | [Graph Optimizer](graph-optimizer.md#graph-optimizer) | [Backend Planning](backend-planning-and-regions.md#backend-planning-and-regions) | [Metal Backend](metal-backend.md#buffer-residency-and-materialization) | [Tensor API](tensor-api.md#graph-lifecycle-and-execution) | [Architecture](architecture.md#core-artifact-boundaries) | [Modules](modules.md#package-map)

Chapters: [Graph Construction](#graph-construction) | [Broadcasting](#broadcasting) | [Autodiff / Backward Graph](#autodiff-backward-graph) | [Compile Pipeline](#compile-pipeline) | [Semantic Canonicalization](#semantic-canonicalization) | [Graph Optimization And Compile Planning](#graph-optimization-and-compile-planning) | [Prepared Execution](#prepared-execution) | [Memory Planning / Runtime Binding](#memory-planning-runtime-binding) | [CPU Dispatch](#cpu-dispatch) | [Fused ASM Execution](#fused-asm-execution) | [Tuning / Calibration / Persistence](#tuning-calibration-persistence)

This document explains the major mechanisms in Synaptik using the same structure for each one: problem, mental model, key concepts, where it lives, step-by-step, worked example, internals, edge cases, misconceptions, and related mechanisms.

## Table Of Contents

- [Graph Construction](#graph-construction)
- [Broadcasting](#broadcasting)
- [Autodiff / Backward Graph](#autodiff-backward-graph)
- [Compile Pipeline](#compile-pipeline)
- [Semantic Canonicalization](#semantic-canonicalization)
- [Graph Optimization And Compile Planning](#graph-optimization-and-compile-planning)
- [Prepared Execution](#prepared-execution)
- [Memory Planning / Runtime Binding](#memory-planning-runtime-binding)
- [CPU Dispatch](#cpu-dispatch)
- [Fused ASM Execution](#fused-asm-execution)
- [Tuning / Calibration / Persistence](#tuning-calibration-persistence)

## Graph Construction

**Problem**

User-facing tensor calls need to build an executable mathematical graph while still feeling like ordinary tensor operations.

**Mental Model**

Each operation call returns a new `Tensor` node. Leaf tensors hold user data; derived tensors hold an `Operation` descriptor plus references to predecessor tensors.

**Key Concepts**

- Leaf tensor: `operation == null`.
- Derived tensor: `operation != null` and `prevTensors` records inputs.
- `requiresGrad` propagates from inputs unless a builder forces no-grad.
- Backward lambdas are attached during construction for differentiable operations.

**Where It Lives**

- [`Tensor.java`](../src/main/java/tensor/Tensor.java)
- [`TensorPrimitiveBuilder.java`](../src/main/java/tensor/internal/TensorPrimitiveBuilder.java)
- [`tensor.ops.*`](../src/main/java/tensor/ops)
- [`operations/Operation.java`](../src/main/java/operations/Operation.java)

**Step-By-Step**

1. User creates leaf tensors with data, shape, label, and dtype.
2. User calls an operation such as `a.add(b)`.
3. The relevant `tensor.ops.*` builder validates inputs and derives output shape/dtype.
4. The builder creates an immutable operation descriptor.
5. `TensorPrimitiveBuilder` creates a new tensor node with predecessor references.
6. If differentiable, the builder attaches a backward lambda.
7. Later, compile obtains the reachable DAG with `topologicalSort()`.

**Worked Example**

```java
Tensor a = new Tensor(new double[]{1, 2}, new int[]{2}, null, "a", DataType.FLOAT64);
Tensor b = new Tensor(new double[]{10, 20}, new int[]{2}, null, "b", DataType.FLOAT64);
Tensor y = a.add(b).relu();
```

Concrete graph:

```text
a --\
     ADD -> RELU
b --/
```

Values after execution are `[11.0, 22.0]`.

**Internals**

`AddOp.build(...)` creates an `operations.elementwise.binary.add` descriptor containing a `BroadcastPlan`, then uses `TensorPrimitiveBuilder.binary(...)` to create the output node. The builder does not execute the add. It only records semantics and graph edges.

**Edge Cases**

- Some arithmetic identities are simplified during construction for scalar constants, such as `x + 0 -> x` and `x * 1 -> x`.
- No-grad builder variants exist for nondifferentiable or infrastructure nodes.
- The public `Tensor` class still contains low-level constructors used by tests, rewrites, runtime, and view setup.

**Misconceptions**

- A graph node is not a separate class from `Tensor`; `Tensor` is the graph node.
- An `Operation` does not contain the CPU loop.

**Related Mechanisms**

- Broadcasting
- Autodiff / Backward Graph
- Compile Pipeline

## Broadcasting

**Problem**

Elementwise operations need to combine tensors with compatible but different shapes, and backward must reduce gradients to the original operand shapes.

**Mental Model**

Shapes align from the right. Any axis with dimension `1` can repeat across the other operand. Repetition is represented by effective stride `0`.

**Key Concepts**

- `BroadcastPlan`
- output shape
- effective input strides
- reduce axes for each operand
- `TensorBroadcastOps.sumToShape(...)`

**Where It Lives**

- [`BroadcastPlanner.java`](../src/main/java/tensor/layout/BroadcastPlanner.java)
- [`BroadcastPlan.java`](../src/main/java/tensor/layout/BroadcastPlan.java)
- [`TensorBroadcastOps.java`](../src/main/java/tensor/TensorBroadcastOps.java)
- [`BroadcastContractMatrixTest.java`](../src/test/java/BroadcastContractMatrixTest.java)

**Step-By-Step**

1. Determine output rank as `max(leftRank, rightRank)`.
2. Pad missing leading shape dimensions with `1`.
3. Pad missing leading stride dimensions with `0`.
4. For each axis, require dimensions to be equal or one side to be `1`.
5. Set output dimension to the max of both dimensions.
6. Set an input effective stride to `0` when that input broadcasts on the axis.
7. Record that axis as a gradient reduction axis for the broadcast input.

**Worked Example**

```text
left  shape [2, 1, 4]
right shape [3, 4]
right aligned as [1, 3, 4]
out   shape [2, 3, 4]
```

For `left + right`, a left value at `[row, 0, col]` is reused for all `3` middle-axis positions. During backward, the left gradient sums over axis `1`; the right gradient sums over axis `0`.

**Internals**

`BroadcastPlanner.plan(...)` returns effective strides and reduce axes. Binary operation builders such as `AddOp`, `MulOp`, `MinOp`, and `MaxOp` store that plan in operation descriptors. The backward lambdas call `TensorBroadcastOps.sumToShape(outGrad, originalShape)`.

**Edge Cases**

- A mismatch such as `[2, 3]` and `[2, 4]` throws `IllegalArgumentException`.
- Scalar shape normalization uses `[1]`.
- `min` and `max` tie gradients split equal values through specialized gradient operations.

**Misconceptions**

- Broadcasting does not copy repeated input data during graph construction.
- Backward cannot simply pass `outGrad` through; it must reduce broadcast axes.

**Related Mechanisms**

- Graph Construction
- CPU Dispatch
- Fused ASM Execution

## Autodiff / Backward Graph

**Problem**

Training needs gradients for trainable leaves while preserving the same compiled/optimized execution lifecycle as forward-only inference.

**Mental Model**

Forward builders attach recipes for how to build gradient nodes. Training compile runs those recipes in reverse topological order to create a backward graph.

**Key Concepts**

- `requiresGrad`
- backward lambda
- seed gradient of ones
- backward-only node marker
- compiled gradient binding
- detached gradient publication

**Where It Lives**

- [`BackwardGraphBuilder.java`](../src/main/java/graph/compile/session/BackwardGraphBuilder.java)
- [`GradientBindingCollector.java`](../src/main/java/graph/compile/session/GradientBindingCollector.java)
- [`PreparedExecution.java`](../src/main/java/graph/execution/PreparedExecution.java)
- [`GradientEngineRegressionTest.java`](../src/test/java/GradientEngineRegressionTest.java)

**Step-By-Step**

1. Compile decides backward support from `CompileMode` and trainable leaf presence.
2. The forward root gradient is initialized to `onesLike(forwardRoot)`.
3. Forward nodes are visited from output back to leaves.
4. Each node's backward builder creates gradient graph nodes for its inputs.
5. Gradients for trainable leaves become backward targets.
6. Compile creates a temporary `noop` super-root over the forward output plus backward targets.
7. The joint graph is optimized and compiled.
8. Execution seeds the compiled root gradient and runs forward/backward steps.
9. Gradients are published as detached tensor copies.

**Worked Example**

```java
Tensor x = new Tensor(new double[]{1, -2, 3}, new int[]{3}, null, "x", DataType.FLOAT64);
x.setRequiresGrad(true);
Tensor loss = x.mul(x).sum();
```

Forward value: `1 + 4 + 9 = 14`.

Backward value: `d(sum(x*x))/dx = 2x = [2, -4, 6]`.

**Internals**

`BackwardGraphBuilder.build(...)` clears stale state before compile, seeds the root, invokes `TensorInternalAccess.buildBackwardGraph(...)`, collects trainable-leaf gradients, and marks backward nodes that are not part of the original forward set. `PreparedExecution.publishCompiledGradients(...)` maps compiled gradient bindings back to original source tensors.

**Edge Cases**

- `INFERENCE_ONLY` never compiles backward.
- `TRAINING` and `AUTO` compile backward only when trainable leaves exist.
- `BOOL` and `INT32` roots do not support backward execution.
- Recompiling a graph after a training run ignores previously published semantic gradients; tests verify graph size does not grow across repeated compiles.

**Misconceptions**

- Calling training compile does not guarantee backward steps if no leaf requires gradients.
- Published gradients are not aliases of internal runtime buffers.

**Related Mechanisms**

- Compile Pipeline
- Prepared Execution
- Memory Planning / Runtime Binding

## Compile Pipeline

**Problem**

The live semantic graph is mutable and user-owned, but the optimizer and runtime need a stable compile artifact.

**Mental Model**

Compile creates a snapshot, optionally expands it with backward nodes, optimizes the snapshot, and records enough metadata for prepare to bind runtime policy.

**Key Concepts**

- `CompiledGraph`
- `GraphCompiler`
- `CompileArtifacts`
- `CompiledNode`
- forward boundary
- source tensor mapping
- compile trace

**Where It Lives**

- [`CompiledGraph.java`](../src/main/java/graph/CompiledGraph.java)
- [`GraphCompiler.java`](../src/main/java/graph/compile/GraphCompiler.java)
- [`CompileArtifacts.java`](../src/main/java/graph/compile/CompileArtifacts.java)
- [`CompiledNode.java`](../src/main/java/graph/CompiledNode.java)
- [`CompiledGraphIdempotencyTest.java`](../src/test/java/CompiledGraphIdempotencyTest.java)

**Step-By-Step**

1. Normalize the requested root with `rootTensor.forwardOutput()`.
2. Optionally canonicalize the forward graph.
3. Reset autograd build state.
4. Decide backward support.
5. Build only forward graph for inference, or joint forward/backward graph for training.
6. Capture an optimizer snapshot.
7. Run configured backend-neutral graph optimization.
8. Rebuild `CompiledNode` snapshots and source mappings.
9. Collect gradient bindings and forward seed binding.
10. Build partition planning and lowering-ready memory state.
11. Return `CompileArtifacts` plus `CompileTrace`.

**Worked Example**

For:

```java
Tensor y = a.add(b).relu();
CompiledGraph compiled = CompiledGraph.compile(y, CompileConfig.inference(), CompileMode.INFERENCE_ONLY);
```

Compile sees a forward DAG with leaf `a`, leaf `b`, `ADD`, `RELU`, and a system forward-output `NOOP`. The compile artifact supports forward execution and has no backward steps.

**Internals**

`CompiledNode.snapshot(...)` records node id, publication tensor, operation, resolved backend, input ids, storage owner id, shape, strides, storage offset, dtype, backward marker, leaf flag, gradient/publication flags, contiguity, flat size, label, and static data snapshot. Compile topology and input relationships are value-based, so prepare/lowering does not depend on mutable topology in the original tensor objects.

**Edge Cases**

- Compile maps canonicalized or lowered forward roots back to the original root so execution can publish results to the expected semantic tensor.
- If the optimized graph no longer contains the forward output, compile fails.
- Compile finalizes region and memory planning artifacts when partition planning discovers planned partitions.

**Misconceptions**

- Compile does not execute kernels.
- Compile is not allowed to mutate the original inference graph; tests verify original operation links remain unchanged.

**Related Mechanisms**

- Semantic Canonicalization
- Optimizer Stages
- Prepared Execution

## Semantic Canonicalization

**Problem**

Some user graphs are semantically equivalent to higher-level primitives, and optimizing them before backward construction can produce cleaner forward/backward graphs.

**Mental Model**

Semantic canonicalization rebuilds a forward-only equivalent graph without mutating the original user graph, then records source mappings for publication.

**Key Concepts**

- `SemanticForwardCanonicalizer`
- forward-only rebuild
- source tensor map
- semantic lowering
- fallback to original graph on unsupported rebuild

**Where It Lives**

- [`SemanticForwardCanonicalizer.java`](../src/main/java/graph/SemanticForwardCanonicalizer.java)
- [`RewriteConfig.java`](../src/main/java/config/optimizer/RewriteConfig.java)
- [`SemanticForwardCanonicalizationCompileTest.java`](../src/test/java/graph/SemanticForwardCanonicalizationCompileTest.java)

**Step-By-Step**

1. Walk the forward graph recursively.
2. Rewrite inputs first.
3. Try semantic lowerings for the current node.
4. If a lowering matches, build the canonical tensor node.
5. If inputs changed but no lowering matched, rebuild an equivalent node.
6. If rebuild is unsupported, return the original graph.
7. Return the canonical graph, canonical forward output, and source tensor map.

**Worked Example**

```java
Tensor manual = input.matmul(weight).add(bias).sum();
```

When shapes match linear semantics, canonicalization can lower `matmul + bias` to `input.linear(weight, bias)` before autograd. Tests compare forward and gradients against the direct `linear(...)` graph and assert the compiled graph contains `Operation.OpType.LINEAR`.

**Internals**

Lowerings include piecewise patterns like sigmoid/relu/clamp, linear lowering, indexed cross-entropy lowering, and attention lowering. The implementation matches operation descriptors and input structure rather than string expressions.

**Edge Cases**

- If canonicalization cannot safely rebuild a node, it returns the original graph.
- Canonicalization runs before backward graph construction, so lowered primitives must still carry valid backward behavior.
- Source mapping is required when the canonical forward root differs from the original root.

**Misconceptions**

- Semantic canonicalization is not the same as the `AR` optimizer stage, although both perform rewrites.
- It does not mutate the user's graph in place.

**Related Mechanisms**

- Compile Pipeline
- Optimizer Stages
- Autodiff / Backward Graph

## Graph Optimization And Compile Planning

**Problem**

The compiled graph should be simpler and deduplicated before backend ownership, region optimization, and memory planning prepare it for runtime.

**Mental Model**

`GraphOptimizer` owns backend-neutral graph simplification. Backend planning, region optimization, and memory planning are later compile phases.

**Key Concepts**

- `GraphOptimizationConfig`
- `CompileConfig`
- `BackendPlanningConfig`
- `OptimizerFactory`
- `OptimizerState`
- simplification fixpoint
- backend ownership planning
- region optimization
- memory planning

**Where It Lives**

- [`CompileConfig.java`](../src/main/java/config/compile/CompileConfig.java)
- [`GraphOptimizationConfig.java`](../src/main/java/config/compile/GraphOptimizationConfig.java)
- [`BackendPlanningConfig.java`](../src/main/java/config/compile/BackendPlanningConfig.java)
- [`OptimizerFactory.java`](../src/main/java/graph/optimizer/OptimizerFactory.java)
- [`BackendPlanningService.java`](../src/main/java/graph/compile/planning/BackendPlanningService.java)
- [`graph/optimizer`](../src/main/java/graph/optimizer)

**Step-By-Step**

1. `OptimizerFactory.create(config.graphOptimization())` builds graph simplification and optional lowering rules.
2. `AR`, `CF`, `CSE`, and `DCE` run inside a simplification fixpoint.
3. `LOWER` optionally creates backend-neutral specialized operation surfaces.
4. `BackendPlanningService` discovers CPU and accelerator ownership regions from `BackendPlanningConfig`.
5. Region optimization creates execution units inside owned regions.
6. Memory planning builds lifetime and handoff plans.
7. The final `OptimizerState` and compile artifacts are stored for prepare.

**Worked Example**

For:

```text
z = exp(log(x)).add(0)
```

`AR` can simplify redundant algebraic structure. If another identical expression remains, `CSE` can point both uses at one representative. Backend planning then groups backend-compatible structure, region optimization may convert an elementwise chain into a fused execution unit, and memory planning assigns reusable slots for temporaries.

**Internals**

`GraphOptimizationConfig` currently defines training and inference graph optimization as:

```text
CLEANUP_FIXPOINT(AR -> CF -> CSE -> DCE) -> LOWER
```

`CompileConfig.training()` and `CompileConfig.inference()` then add explicit backend planning, region optimization, and memory planning.

**Edge Cases**

- `CompileConfig.noGraphOptimization()` disables graph optimization only.
- Explicit backend planning still has to work when graph optimization is disabled.
- There is no outer fixpoint loop around backend planning, region optimization, and memory planning.

**Misconceptions**

- Runtime thresholds such as vector width or BLAS work cutoffs are not optimizer responsibilities.
- Region optimization does not directly execute fused code; it shapes optimized regions and descriptors for later prepare/runtime work.

**Related Mechanisms**

- Compile Pipeline
- Prepared Execution
- CPU Dispatch
- Fused ASM Execution

## Prepared Execution

**Problem**

Compiled graph semantics need to become a reusable executable plan with concrete backend metadata.

**Mental Model**

Prepare binds policy; execute consumes the prepared recipe.

**Key Concepts**

- `PreparedExecution`
- `PreparedExecutionStep`
- `CompiledNodeExecutionMetadata`
- backend prepare dispatcher
- forward steps
- backward steps
- prepare trace

**Where It Lives**

- [`PreparedExecutionBuilder.java`](../src/main/java/backend/prepare/PreparedExecutionBuilder.java)
- [`PreparedExecution.java`](../src/main/java/graph/execution/PreparedExecution.java)
- [`PreparedExecutionStep.java`](../src/main/java/graph/execution/PreparedExecutionStep.java)
- [`PreparedExecutionBuildTest.java`](../src/test/java/PreparedExecutionBuildTest.java)

**Step-By-Step**

1. Build a consumer map from compiled nodes.
2. Create `BackendPrepareContext` with runtime config and backward support.
3. Select backend partition plans.
4. Lower optimized regions for supported backends.
5. Dispatch each non-leaf node to a backend preparer.
6. Publish metadata into the prepare context.
7. Skip partition-interior nodes as executable steps.
8. Split executable steps by forward boundary.
9. Create `PreparedExecution`.

**Worked Example**

For `a.add(b).mul(a).sigmoid()` with inference defaults, prepare creates forward steps for the executable operations. With fusion enabled, the final executable step may be a partition anchor whose metadata contains a fused execution operation, fused executable, CPU plan, and execution input node ids.

**Internals**

`PreparedExecutionBuilder` uses `DefaultBackendSelectionPolicy`, `LoweringPipeline`, and `BackendPrepareDispatcher`. `CpuNodePreparer` resolves CPU kernels and CPU node plans, prepares fused executables when needed, and creates workspaces for operations such as max-pool backward, matmul, linear, log-softmax, and BFLOAT16 continuation paths.

**Edge Cases**

- Partition interior nodes produce metadata but are not executed directly.
- Repeated prepare on the same compiled graph returns independent prepared executions.
- Prepared execution throws if asked for `FORWARD_BACKWARD` when compile did not support backward.

**Misconceptions**

- Prepare does not rerun graph optimization or backend planning.
- Prepare does not change tensor formulas.

**Related Mechanisms**

- Compile Pipeline
- Memory Planning / Runtime Binding
- CPU Dispatch
- [Metal Backend: End-To-End Flow](metal-backend.md#end-to-end-flow)

## Memory Planning / Runtime Binding

**Problem**

Execution creates many temporary tensors. Reusing compatible buffers reduces allocation and peak memory, but cannot break saved forward values, gradients, views, or workspace-sensitive operations.

**Mental Model**

Compile-time memory planning computes lifetimes, reusable slots, region value flow, region bindings, handoff requirements, and runtime binding policy. Runtime binding assigns actual arrays to eligible runtime tensors before execution.

**Key Concepts**

- `MemoryPlanner`
- `MemoryPlan`
- tensor lifetime planning
- reusable intervals
- slot assignment
- region value lifetimes
- region memory bindings
- region handoff requirements
- runtime binding policy
- `RuntimeMemoryBinder`

**Where It Lives**

- [`MemoryPlanner.java`](../src/main/java/graph/compile/planning/memory/MemoryPlanner.java)
- [`TensorLifetimePlanner.java`](../src/main/java/graph/compile/planning/memory/TensorLifetimePlanner.java)
- [`ReusableSlotAllocator.java`](../src/main/java/graph/compile/planning/memory/ReusableSlotAllocator.java)
- [`RegionValueFlowPlanner.java`](../src/main/java/graph/compile/planning/memory/RegionValueFlowPlanner.java)
- [`RegionBindingAllocator.java`](../src/main/java/graph/compile/planning/memory/RegionBindingAllocator.java)
- [`RegionHandoffPlanner.java`](../src/main/java/graph/compile/planning/memory/RegionHandoffPlanner.java)
- [`MemoryPlanningInput.java`](../src/main/java/graph/compile/planning/memory/MemoryPlanningInput.java)
- [`RuntimeMemoryBinder.java`](../src/main/java/graph/execution/residency/RuntimeMemoryBinder.java)
- [`MemoryPlannerSummaryTest.java`](../src/test/java/MemoryPlannerSummaryTest.java)
- [`RuntimeMemoryBinderTest.java`](../src/test/java/graph/execution/RuntimeMemoryBinderTest.java)

**Step-By-Step**

1. Build index positions for the sorted graph.
2. Resolve storage owners for alias/view nodes.
3. Count consumers and last-read indices.
4. Mark saved forward owners crossing the forward/backward boundary.
5. Build lifetimes and reusable intervals.
6. Assign compatible intervals to slots.
7. Build region-value flow artifacts for optimized regions.
8. Allocate region memory bindings for materialized and continuation values.
9. Build cross-region handoff requirements.
10. During execution, bind eligible runtime tensors to shared slot arrays.

**Worked Example**

```java
Tensor t1 = a.add(b);          // size 4
Tensor t2 = t1.sum(1, true);   // size 2
Tensor out = t2.add(c.reshape(2, 1));
```

If `t1` is no longer needed after `t2`, its buffer may become reusable. A strict policy keeps same-size slots separate, while a larger-buffer reuse policy can assign smaller later intervals to larger earlier slots. Tests verify flexible policy can reduce slot count and increase reuse.

**Internals**

`MemoryPlanner` is the public assembler for `MemoryPlan`. Tensor lifetime analysis, interval filtering, slot allocation, region value flow, region binding allocation, handoff planning, runtime binding policy, and summary reporting live in separate package-local classes with no public compatibility layer. `RuntimeMemoryBindingPolicyPlanner` marks workspace-sensitive operations such as `MAX_POOL2D` as not region-bindable. `RuntimeMemoryBinder` binds dtype-specific slot arrays for `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, `INT64`, and `BOOL`; the binding remains a runtime storage optimization and does not change semantic tensor shapes, dtypes, or graph edges.

**Edge Cases**

- Alias/view operations are bound by aliasing their input runtime tensor rather than assigning a new slot.
- A region slot is bound only when slot use count is at least two and slot size equals runtime tensor flat size.
- Workspace-sensitive nodes do not disable binding for independent region values.

**Misconceptions**

- Memory planning does not change graph semantics.
- Runtime binding is not stored on the semantic tensor graph.

**Related Mechanisms**

- Prepared Execution
- Fused ASM Execution
- [Metal Backend: Buffer Residency And Materialization](metal-backend.md#buffer-residency-and-materialization)
- Autodiff / Backward Graph

## CPU Dispatch

**Problem**

Each prepared node must run the correct CPU kernel with dtype, layout, dispatch, reduction, matmul, conv2d, and workspace metadata already resolved.

**Mental Model**

CPU execution reads a prepared recipe, resolves runtime input tensors, applies the CPU execution plan, and calls the dtype-specific kernel method.

**Key Concepts**

- `ComputeEngine`
- `CpuBackend`
- `CpuKernelResolver`
- `CpuExecutionPlanner`
- `CpuNodeExecutionPlan`
- dispatch hints
- compute contract

**Where It Lives**

- [`ComputeEngine.java`](../src/main/java/backend/ComputeEngine.java)
- [`CpuBackend.java`](../src/main/java/backend/cpu/CpuBackend.java)
- [`CpuKernelResolver.java`](../src/main/java/backend/cpu/registry/CpuKernelResolver.java)
- [`CpuExecutionPlanner.java`](../src/main/java/backend/cpu/kernels/plan/CpuExecutionPlanner.java)
- [`CpuNodePreparer.java`](../src/main/java/backend/cpu/prepare/CpuNodePreparer.java)

**Step-By-Step**

1. `PreparedExecution` calls `ComputeEngine.compute(node, metadata, context)`.
2. `ComputeEngine` switches on `metadata.backend()`.
3. For CPU, `CpuBackend.execute(...)` resolves the operation from metadata override or compiled node.
4. It fetches the runtime output tensor by node id.
5. It requires a prepared CPU kernel and CPU plan.
6. It resolves runtime inputs using execution input node ids when present.
7. It applies the CPU node plan, which may materialize or remap inputs.
8. If the plan uses a strided elementwise path, it calls `CpuStridedElementWise`.
9. Otherwise it calls the dtype-specific kernel method.

**Worked Example**

For `FLOAT64` `ADD`, prepared metadata contains `CpuAddKernel` and a CPU plan. Runtime calls:

```text
CpuAddKernel.forwardF64(addDescriptor, runtimeInputs, runtimeOutput, kernelContext)
```

For a fused node, metadata contains `CpuFusedKernel`, a `FusedOperation`, and a prepared fused executable.

**Internals**

`CpuExecutionPlanner.from(runtimeConfig.cpuKernelConfig())` creates sub-planners for elementwise dispatch, fused dispatch, reductions, matmul, conv2d, attention, and compute-contract resolution. `CpuKernelResolver.resolve(opType)` maps every supported `Operation.OpType` to a concrete CPU kernel or throws for unsupported internal types.

**Edge Cases**

- `CONST_SCALAR` has no standalone CPU kernel.
- `UNKNOWN` cannot resolve a CPU kernel.
- Non-`FLOAT64` execution marks the tensor's double data view stale after kernel execution.
- Partition-interior nodes are ignored by `ComputeEngine`.

**Misconceptions**

- The CPU backend should not decide whether an operation should have been fused; it only executes prepared metadata.
- Kernel resolution is not the same as dispatch planning.

**Related Mechanisms**

- Prepared Execution
- Fused ASM Execution
- Tuning / Calibration / Persistence

## Fused ASM Execution

**Problem**

Elementwise chains can be cheaper as one fused loop than as multiple materialized intermediates, but the fused loop still needs dtype, layout, vector, and parallel policy resolved before execution.

**Mental Model**

Graph optimization identifies regions. CPU lowering turns a fused region into a `FusedOperation`. Prepare compiles or retrieves an ASM-backed `PreparedFusedExecutable`. Runtime invokes scalar, vector, parallel, or parallel-vector ranges based on prepared hints.

**Key Concepts**

- `FusedOperation`
- `FusedExpressionPlan`
- `FusedExecutionPlan`
- `FusedExecutionBackendResolver`
- `AsmPreparedFusedExecutableFactory`
- `PreparedFusedExecutable`
- `CpuFusedKernel`
- `FusedExecutor`

**Where It Lives**

- [`FusedOperation.java`](../src/main/java/backend/cpu/fused/plan/FusedOperation.java)
- [`FusedExecutionBackendResolver.java`](../src/main/java/backend/cpu/fused/exec/FusedExecutionBackendResolver.java)
- [`AsmFusedExecutionBackend.java`](../src/main/java/backend/cpu/fused/asm/AsmFusedExecutionBackend.java)
- [`AsmPreparedFusedExecutableFactory.java`](../src/main/java/backend/cpu/fused/asm/AsmPreparedFusedExecutableFactory.java)
- [`CpuFusedKernel.java`](../src/main/java/backend/cpu/kernels/fused/CpuFusedKernel.java)
- [`FusedExecutor.java`](../src/main/java/backend/cpu/kernels/fused/FusedExecutor.java)
- [`FusedExecutionModesTest.java`](../src/test/java/FusedExecutionModesTest.java)

**Step-By-Step**

1. Region optimization creates optimized regions from backend planning information.
2. CPU lowering produces a lowered fused anchor and `FusedOperationPreparation`.
3. `CpuNodePreparer` resolves `CpuFusedKernel`, compute contract, fused dispatch hints, and CPU plan.
4. It creates a `FusedExecutionPlan`.
5. `FusedExecutionBackendResolver` selects the ASM backend.
6. `AsmPreparedFusedExecutableFactory` builds a cache key from scheduler signature, precision mode, vector width, and specialization.
7. On a cache miss, bytecode is generated and loaded with `CustomClassLoader`.
8. Runtime `CpuFusedKernel` calls `FusedExecutor.execute(...)`.
9. `FusedExecutor` dispatches scalar, vector, parallel, or parallel-vector ranges.

**Worked Example**

```java
Tensor out = a.add(b).mul(c).add(a.mul(0.25)).max(b).min(c).sigmoid();
```

With fusion enabled, this elementwise chain can become one fused operation. Tests compare fused output against a no-optimization baseline across scalar, vector, parallel, and parallel-vector modes.

**Internals**

The generated class name includes specialization and vector width, for example a suffix like `W1` or `W2`. Tests verify different fused ASM widths create different executable classes. `PreparedFusedExecutable.applyRangeVector(...)` defaults to scalar execution unless the generated executable overrides it.

**Edge Cases**

- The resolver currently throws if the ASM backend does not support a plan.
- BFLOAT16 fused execution uses BFLOAT16 storage with F32 compute contract.
- Broadcast inputs, non-contiguous inputs, `where`, float32, float64, and BFLOAT16 paths are covered by tests.

**Misconceptions**

- Fused ASM execution is not a runtime rewrite; it is prepared before execution.
- `FusedOperation.isCheap()` is a planning hint, not a correctness rule.

**Related Mechanisms**

- Optimizer Stages
- Prepared Execution
- CPU Dispatch

## Tuning / Calibration / Persistence

**Problem**

The fastest runtime policy depends on dtype, workload, hardware, JDK, and backend settings. Synaptik needs measured profiles rather than hardcoded assumptions.

**Mental Model**

Tuning evaluates executable `ExecutionProfile` candidates. Platform calibration searches reusable runtime defaults and persists a `PlatformRuntimeProfile`; graph autotune uses graph policy plus runtime profile to assemble executable profiles.

**Key Concepts**

- `ExecutionProfile`
- `GraphExecutionPolicy`
- `PlatformRuntimeProfile`
- `ExecutionProfileAssembler`
- `AutotuneRequest`
- `AutotuneSession`
- `PlatformCalibrationRequest`
- persistence policy

**Where It Lives**

- [`ExecutionProfile.java`](../src/main/java/config/profile/ExecutionProfile.java)
- [`ExecutionProfileAssembler.java`](../src/main/java/config/profile/ExecutionProfileAssembler.java)
- [`PlatformRuntimeProfile.java`](../src/main/java/config/profile/PlatformRuntimeProfile.java)
- [`AutotuneRequest.java`](../src/main/java/tuning/autotune/AutotuneRequest.java)
- [`DefaultAutotuneSession.java`](../src/main/java/tuning/autotune/DefaultAutotuneSession.java)
- [`DefaultPlatformCalibrationSession.java`](../src/main/java/tuning/calibration/DefaultPlatformCalibrationSession.java)
- [`PlatformCalibrationPaths.java`](../src/main/java/tuning/calibration/store/PlatformCalibrationPaths.java)

**Step-By-Step**

1. Start from a graph policy and runtime profile.
2. Generate candidate executable profiles.
3. Validate a candidate on a workload instance.
4. Measure the candidate with `MeasurementEngine`.
5. Rank candidates by measurement score.
6. Persist history entries when enabled.
7. Persist the best profile when enabled.
8. For platform calibration, carry the winning runtime profile from one calibration family into the next.

**Worked Example**

Platform calibration for `FLOAT64` forward/backward may tune matmul microkernel, tiles, BLAS threshold, fused thresholds, elementwise thresholds, reduction thresholds, scheduler chunks, materialization thresholds, and numerics policy. The CLI layout normalizes the dtype to `f64` and the execution mode to `forward-backward`.

Default calibration layout:

```text
profiles/platform/<platform-id>/calibration/schema-v2/
  latest/f64/forward-backward/profile.json
  latest/f64/forward-backward/manifest.json
  history/f64/forward-backward/<family-id>.jsonl
  runs/<run-id>/f64/forward-backward/<family-id>/result.json
  runs/<run-id>/f64/forward-backward/<family-id>/result.txt
  runs/<run-id>/f64/forward-backward/<family-id>/selected-profile.json
  runs/<run-id>/f64/forward-backward/<family-id>/candidates.jsonl
```

**Internals**

`DefaultAutotuneSession` validates and measures candidates, tracks fingerprints to avoid duplicate evaluation, selects finalists by median steady-state milliseconds, and persists best-profile/history records through store interfaces. `DefaultPlatformCalibrationSession` generates runtime-profile candidates for each family, benchmarks candidate entries across workloads, selects the winner by the family score policy, and optionally saves the final platform runtime profile.

**Edge Cases**

- Persistence is disabled by default unless paths and flags are provided.
- Legacy seed-profile adapters exist but are marked deprecated in `AutotuneRequest`.
- New platform ids use only canonical OS and architecture, for example `macos-arm64`.
- Older local platform directories that include JVM vendor and core count are read only as compatibility fallback.

**Misconceptions**

- Calibration is not graph autotune.
- Tuning does not define execution semantics; it selects measured executable profiles.
- `PlatformRuntimeProfile` does not replace `ExecutionProfile`; it is assembled with graph policy into one.

**Related Mechanisms**

- CPU Dispatch
- Prepared Execution
- Optimizer Stages
