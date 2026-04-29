# Architecture

**Analysis Date:** 2026-04-29

## Pattern Overview

**Overall:** Layered compiled tensor graph runtime.

**Key Characteristics:**
- User code builds a semantic tensor DAG through `src/main/java/tensor/Tensor.java` and `src/main/java/tensor/TensorOps.java`.
- Primitive meaning is separated from public builders in immutable descriptors under `src/main/java/operations`.
- Graph compilation snapshots, canonicalizes, optimizes, partitions, and memory-plans graphs through `src/main/java/graph/CompiledGraph.java`, `src/main/java/graph/compile/GraphCompiler.java`, and `src/main/java/graph/optimizer`.
- Runtime preparation converts compile artifacts plus `src/main/java/config/runtime/RuntimeConfig.java` into ordered executable steps through `src/main/java/backend/prepare/PreparedExecutionBuilder.java`.
- Execution uses fresh per-run state in `src/main/java/graph/execution/ExecutionState.java` and backend dispatch through `src/main/java/backend/ComputeEngine.java`.
- CPU implementation is complete under `src/main/java/backend/cpu`; Metal, CUDA, and OpenCL live under `src/main/java/backend/metal`, `src/main/java/backend/cuda`, and `src/main/java/backend/opencl`.
- Tuning, calibration, benchmark, validation, search, reporting, and profile persistence live under `src/main/java/tuning` and feed runnable `src/main/java/config/profile/ExecutionProfile.java` values back into compile/prepare/execute.

## Layers

**Public Tensor API:**
- Purpose: Build semantic graph nodes, hold typed tensor storage/layout metadata, expose fluent and static user APIs, and bridge convenience compute calls to compile/prepare/execute.
- Location: `src/main/java/tensor`
- Contains: `src/main/java/tensor/Tensor.java`, `src/main/java/tensor/TensorOps.java`, `src/main/java/tensor/TensorExecutionSupport.java`, dtype storage classes such as `src/main/java/tensor/Float64Storage.java`, and family builders under `src/main/java/tensor/ops`.
- Depends on: `src/main/java/operations`, `src/main/java/graph`, `src/main/java/backend/runtime`, `src/main/java/config/profile`, and `src/main/java/config/runtime`.
- Used by: Tests under `src/test/java`, tuning workloads under `src/main/java/tuning/workload`, numerics harnesses under `src/main/java/numerics`, and CLI/application entry points under `src/main/java/synaptik/app`.
- Pattern: Add public operation behavior through `src/main/java/tensor/ops/<family>` first, then expose through `src/main/java/tensor/TensorOps.java` and `src/main/java/tensor/Tensor.java`.

**Primitive Operation Descriptors:**
- Purpose: Represent what a graph node means without embedding builder logic, backward lambdas, runtime state, kernels, or backend policy.
- Location: `src/main/java/operations`
- Contains: `src/main/java/operations/Operation.java` and descriptor families such as `src/main/java/operations/elementwise/binary/add.java`, `src/main/java/operations/reduction/sum.java`, `src/main/java/operations/layout/reshape.java`, `src/main/java/operations/linalg/matmul.java`, and `src/main/java/operations/nn/conv/conv2d.java`.
- Depends on: Mostly Java value semantics and public option records from `src/main/java/tensor/options` where semantic parameters require them.
- Used by: Tensor builders in `src/main/java/tensor/ops`, optimizer signatures in `src/main/java/graph/optimizer`, CPU kernel resolution in `src/main/java/backend/cpu/registry/CpuKernelResolver.java`, and backend dispatch metadata in `src/main/java/graph/execution/CompiledNodeExecutionMetadata.java`.
- Pattern: Each descriptor implements `Operation.opType()` and `Operation.getExpression()`; semantic parameters belong in descriptor fields.

**Graph Compile Layer:**
- Purpose: Convert a semantic tensor DAG into immutable compile artifacts with compiled-node snapshots, forward/backward boundaries, gradient bindings, partition candidates, optimizer state, and memory plan.
- Location: `src/main/java/graph` and `src/main/java/graph/compile`
- Contains: `src/main/java/graph/CompiledGraph.java`, `src/main/java/graph/CompiledNode.java`, `src/main/java/graph/compile/GraphCompiler.java`, `src/main/java/graph/compile/CompileArtifacts.java`, `src/main/java/graph/compile/BackwardGraphBuilder.java`, `src/main/java/graph/compile/GradientBindingCollector.java`, `src/main/java/graph/compile/OptimizerGraphSnapshot.java`, and `src/main/java/graph/compile/PartitionPlanningSnapshotBuilder.java`.
- Depends on: Semantic tensors in `src/main/java/tensor`, operation descriptors in `src/main/java/operations`, optimizer config in `src/main/java/config/optimizer`, optimizer stages in `src/main/java/graph/optimizer`, and backend partition descriptors in `src/main/java/backend/partition`.
- Used by: `src/main/java/tensor/TensorExecutionSupport.java`, `src/main/java/synaptik/app/Main.java`, `src/main/java/tuning`, and tests under `src/test/java`.
- Pattern: Compile captures an `OptimizerGraphSnapshot` before optimization so rewrites do not accumulate directly on user-owned `Tensor` objects.

**Graph Optimizer Layer:**
- Purpose: Run ordered graph-level transformations and planning stages over a compile-time snapshot.
- Location: `src/main/java/graph/optimizer`
- Contains: `src/main/java/graph/optimizer/OptimizerFactory.java`, `src/main/java/graph/optimizer/rewrite/RewriteRule.java`, `src/main/java/graph/optimizer/cse/CommonSubexpressionEliminationRule.java`, `src/main/java/graph/optimizer/partition/PartitionIntentRule.java`, `src/main/java/graph/optimizer/region/RegionOptimizationRule.java`, `src/main/java/graph/optimizer/memory/MemoryOptimizerRule.java`, and `src/main/java/graph/optimizer/state/OptimizerState.java`.
- Depends on: Stage order and policy records in `src/main/java/config/optimizer`, graph snapshots from `src/main/java/graph/compile`, operation descriptors from `src/main/java/operations`, and backend legality adapters through `src/main/java/backend/partition`.
- Used by: `src/main/java/graph/compile/GraphCompiler.java`.
- Pattern: `src/main/java/config/optimizer/OptimizerConfig.java` validates `AR -> CSE -> PART -> FUSE -> MEM` dependencies; `FUSE` requires `PART`, and `MEM` requires `FUSE`.

**Backend-Neutral Prepare, Selection, Lowering, And Runtime Contracts:**
- Purpose: Translate compile artifacts and runtime policy into backend metadata without rebuilding optimizer artifacts.
- Location: `src/main/java/backend/prepare`, `src/main/java/backend/select`, `src/main/java/backend/lowering`, `src/main/java/backend/partition`, `src/main/java/backend/runtime`, and `src/main/java/backend/memory`
- Contains: `src/main/java/backend/prepare/PreparedExecutionBuilder.java`, `src/main/java/backend/prepare/BackendPrepareDispatcher.java`, `src/main/java/backend/select/DefaultBackendSelectionPolicy.java`, `src/main/java/backend/lowering/LoweringPipeline.java`, `src/main/java/backend/partition/BackendPartitionDescriptorRegistry.java`, `src/main/java/backend/runtime/ExecutionContext.java`, and `src/main/java/backend/memory/TensorResidencyState.java`.
- Depends on: Compile artifacts from `src/main/java/graph/compile/CompileArtifacts.java`, optimized regions from `src/main/java/graph/optimizer/state/OptimizerState.java`, and runtime config from `src/main/java/config/runtime/RuntimeConfig.java`.
- Used by: `src/main/java/graph/CompiledGraph.java` and `src/main/java/graph/execution/PreparedExecution.java`.
- Pattern: Generic orchestration remains backend-neutral; concrete preparers live under backend roots such as `src/main/java/backend/cpu/prepare/CpuNodePreparer.java`, `src/main/java/backend/metal/prepare/MetalNodePreparer.java`, and `src/main/java/backend/cuda/prepare/CudaGpuNodePreparer.java`.

**Prepared Execution And Per-Run State:**
- Purpose: Execute prepared forward/backward steps, bind memory plans, publish forward outputs, and publish detached gradients.
- Location: `src/main/java/graph/execution`
- Contains: `src/main/java/graph/execution/PreparedExecution.java`, `src/main/java/graph/execution/PreparedNodeExecution.java`, `src/main/java/graph/execution/CompiledNodeExecutionMetadata.java`, `src/main/java/graph/execution/ExecutionState.java`, `src/main/java/graph/execution/RuntimeMemoryBinder.java`, and trace records under `src/main/java/graph/execution/trace`.
- Depends on: Backend dispatch in `src/main/java/backend/ComputeEngine.java`, runtime context in `src/main/java/backend/runtime/ExecutionContext.java`, compiled nodes in `src/main/java/graph/CompiledNode.java`, and memory plans in `src/main/java/graph/optimizer/memory`.
- Used by: Public convenience compute in `src/main/java/tensor/TensorExecutionSupport.java`, explicit prepared execution users, and tuning/benchmark sessions under `src/main/java/tuning`.
- Pattern: `PreparedExecution` is reusable, while every `execute(...)` call creates a new `ExecutionState`.

**CPU Backend Implementation:**
- Purpose: Prepare and execute complete CPU runtime kernels across elementwise, layout, reduction, linalg, neural-network, index, gradient, and fused families.
- Location: `src/main/java/backend/cpu`
- Contains: `src/main/java/backend/cpu/CpuBackend.java`, `src/main/java/backend/cpu/prepare/CpuNodePreparer.java`, `src/main/java/backend/cpu/registry/CpuKernelResolver.java`, planners under `src/main/java/backend/cpu/kernels/plan`, kernels under `src/main/java/backend/cpu/kernels`, and fused planning/codegen/executable support under `src/main/java/backend/cpu/fused`.
- Depends on: Prepared metadata in `src/main/java/graph/execution/CompiledNodeExecutionMetadata.java`, runtime context in `src/main/java/backend/runtime/ExecutionContext.java`, runtime config in `src/main/java/config/runtime/RuntimeConfig.java`, and operation descriptors in `src/main/java/operations`.
- Used by: `src/main/java/backend/ComputeEngine.java` and backend-specific prepare dispatch through `src/main/java/backend/prepare/BackendPrepareDispatcher.java`.
- Pattern: Preparation resolves `CpuKernel`, `CpuNodeExecutionPlan`, workspaces, fused executables, dispatch hints, matmul hints, reduction hints, conv2d hints, and dtype contracts before runtime loops execute.

**Accelerator Backend Scaffolding:**
- Purpose: Represent accelerator partitions, DAG/lowering artifacts, bridge executables, prepared accelerator executables, device buffers, and runtime fallbacks.
- Location: `src/main/java/backend/accelerator`, `src/main/java/backend/metal`, `src/main/java/backend/cuda`, and `src/main/java/backend/opencl`
- Contains: Shared accelerator records under `src/main/java/backend/accelerator/dag`, Metal bridge/buffer/lowering/prepare/exec classes under `src/main/java/backend/metal`, CUDA bridge/lowering/prepare/exec classes under `src/main/java/backend/cuda`, and minimal OpenCL classes under `src/main/java/backend/opencl`.
- Depends on: Backend-neutral lowering contracts in `src/main/java/backend/lowering`, partition descriptors in `src/main/java/backend/partition`, memory residency records in `src/main/java/backend/memory`, and runtime accelerator config in `src/main/java/config/runtime/AcceleratorConfig.java`.
- Used by: `src/main/java/backend/select/DefaultBackendSelectionPolicy.java`, `src/main/java/backend/lowering/LoweringPipeline.java`, `src/main/java/backend/prepare/BackendPrepareDispatcher.java`, and `src/main/java/backend/ComputeEngine.java`.
- Pattern: Accelerator execution is region/partition anchored; partition interiors are skipped by `src/main/java/backend/ComputeEngine.java` when `PartitionExecutionRole.INTERIOR` is set.

**Configuration And Profiles:**
- Purpose: Keep optimizer policy, runtime policy, backend knobs, and persisted executable profile records separate.
- Location: `src/main/java/config`
- Contains: Optimizer config under `src/main/java/config/optimizer`, runtime config under `src/main/java/config/runtime`, backend tuning records under `src/main/java/config/backend`, and profile records under `src/main/java/config/profile`.
- Depends on: Backend enums such as `src/main/java/backend/ApproxMode.java` and operation/runtime needs from compile and prepare layers.
- Used by: `src/main/java/tensor/TensorExecutionSupport.java`, `src/main/java/graph/CompiledGraph.java`, `src/main/java/backend/prepare/PreparedExecutionBuilder.java`, and `src/main/java/tuning`.
- Pattern: `src/main/java/config/profile/ExecutionProfile.java` combines optimizer policy and runtime policy into a runnable profile.

**Tuning, Calibration, Benchmark, And Validation:**
- Purpose: Measure candidate profiles, search graph/runtime policy, validate results, report timings, and persist best profiles or platform runtime defaults.
- Location: `src/main/java/tuning`
- Contains: Fluent APIs under `src/main/java/tuning/api`, autotune under `src/main/java/tuning/autotune`, benchmark under `src/main/java/tuning/benchmark`, calibration under `src/main/java/tuning/calibration`, candidate spaces under `src/main/java/tuning/candidate`, measurement under `src/main/java/tuning/measure`, persistence under `src/main/java/tuning/store`, validation under `src/main/java/tuning/validate`, and workloads under `src/main/java/tuning/workload`.
- Depends on: Tensor graphs in `src/main/java/tensor`, compile/execute artifacts in `src/main/java/graph`, and config/profile records in `src/main/java/config`.
- Used by: `src/main/java/synaptik/app/TuningCli.java`, `src/main/java/synaptik/app/Main.java`, and public compute autotune in `src/main/java/tensor/TensorExecutionSupport.java`.
- Pattern: Tuning chooses or persists `ExecutionProfile` values; it does not define execution semantics.

**Numerics Harness:**
- Purpose: Compare numerical drift between execution profiles and optimizer stage sets.
- Location: `src/main/java/numerics`
- Contains: `src/main/java/numerics/NumericsCli.java`, `src/main/java/numerics/NumericsHarness.java`, `src/main/java/numerics/NumericsGraphFactory.java`, `src/main/java/numerics/NumericsPolicy.java`, and `src/main/java/numerics/NumericsReport.java`.
- Depends on: Tensor graph construction in `src/main/java/tensor` and execution profiles in `src/main/java/config/profile`.
- Used by: CLI-style diagnostics via `src/main/java/numerics/NumericsCli.java`.

## Data Flow

**Forward-Only Tensor Compute:**

1. User code creates leaf tensors and calls methods on `src/main/java/tensor/Tensor.java`; the methods delegate through `src/main/java/tensor/TensorOps.java` into family builders under `src/main/java/tensor/ops`.
2. Builders validate shape/dtype contracts, instantiate descriptors from `src/main/java/operations`, and create derived semantic graph nodes with `src/main/java/tensor/TensorPrimitiveBuilder.java`.
3. `src/main/java/tensor/TensorExecutionSupport.java` resolves a default or autotuned `src/main/java/config/profile/ExecutionProfile.java`.
4. `src/main/java/graph/CompiledGraph.java` invokes `src/main/java/graph/compile/GraphCompiler.java`.
5. `GraphCompiler` wraps the root with `Tensor.forwardOutput()`, optionally canonicalizes via `src/main/java/graph/SemanticForwardCanonicalizer.java`, captures an optimizer snapshot, runs `src/main/java/graph/optimizer/GraphOptimizer.java`, snapshots `src/main/java/graph/CompiledNode.java` records, builds partition artifacts, and finalizes a memory plan.
6. `src/main/java/backend/prepare/PreparedExecutionBuilder.java` selects backend plans with `src/main/java/backend/select/DefaultBackendSelectionPolicy.java`, lowers selected regions with `src/main/java/backend/lowering/LoweringPipeline.java`, and creates `src/main/java/graph/execution/PreparedNodeExecution.java` steps.
7. `src/main/java/graph/execution/PreparedExecution.java` creates a fresh `ExecutionState`, binds memory through `src/main/java/graph/execution/RuntimeMemoryBinder.java`, creates `ExecutionContext`, and calls `src/main/java/backend/ComputeEngine.java` for each forward step.
8. `ComputeEngine` dispatches to `src/main/java/backend/cpu/CpuBackend.java`, `src/main/java/backend/metal/MetalBackend.java`, `src/main/java/backend/cuda/CudaGpuBackend.java`, `src/main/java/backend/cuda/CudaBackend.java`, or `src/main/java/backend/opencl/OpenClBackend.java` based on prepared metadata.
9. `PreparedExecution` publishes the computed forward result back to the source root tensor in `src/main/java/tensor/Tensor.java`.

**Training / Forward-Backward Compute:**

1. `src/main/java/tensor/TensorExecutionSupport.java` chooses training defaults when `src/main/java/tensor/CompileMode.java` is `TRAINING` or `AUTO` with trainable leaf inputs.
2. `src/main/java/graph/compile/GraphCompiler.java` detects trainable leaf tensors, seeds the forward root, and builds backward nodes with `src/main/java/graph/compile/BackwardGraphBuilder.java`.
3. `GraphCompiler` creates a system super-root using `operations.layout.noop` so forward and backward nodes are optimized as one closure.
4. `src/main/java/graph/compile/GradientBindingCollector.java` records mappings from semantic tensors to compiled gradient nodes.
5. `src/main/java/backend/prepare/PreparedExecutionBuilder.java` splits prepared nodes into forward and backward lists using the compile-time forward boundary.
6. `src/main/java/graph/execution/PreparedExecution.java` seeds the root gradient, executes the full step sequence, synchronizes the forward root, and publishes detached gradients to source tensors.

**Backend Selection And Region Lowering:**

1. `src/main/java/graph/compile/PartitionPlanningSnapshotBuilder.java` derives partition and backend candidates from compiled nodes and `src/main/java/backend/partition/BackendPartitionDescriptorRegistry.java`.
2. `src/main/java/backend/select/DefaultBackendSelectionPolicy.java` filters candidates by compatibility, runtime enablement, runtime availability, and accelerator cost model.
3. `src/main/java/backend/lowering/LoweringPipeline.java` receives optimized regions plus selected partition plans and invokes registered lowerers from `BackendPartitionDescriptorRegistry`.
4. Backend-specific lowerers such as `src/main/java/backend/cpu/lowering/CpuRegionLowerer.java`, `src/main/java/backend/metal/lowering/MetalRegionLowerer.java`, and `src/main/java/backend/cuda/lowering/CudaRegionLowerer.java` produce lowered artifacts consumed by backend-specific preparers.
5. `src/main/java/backend/prepare/BackendPrepareContext.java` indexes selected plans, lowered regions, partition roles, and prepared metadata so concrete preparers can consume decisions without recomputing optimizer state.

**State Management:**
- Semantic graph state lives on mutable `Tensor` objects in `src/main/java/tensor/Tensor.java`.
- Compile artifacts live in memory on `src/main/java/graph/CompiledGraph.java` through `src/main/java/graph/compile/CompileArtifacts.java`.
- Prepared executable metadata lives on `src/main/java/graph/execution/PreparedExecution.java`.
- Per-run tensors, workspaces, residency, runtime outputs, and CPU materialization traces live in `src/main/java/graph/execution/ExecutionState.java`.
- Runtime policy, metadata lookup, and family-specific caches are exposed to kernels through `src/main/java/backend/runtime/ExecutionContext.java`.
- Persisted tuning profiles live under `profiles/platform` for platform calibration and under `build/tuning/tensor` for generic tensor autotune paths in `src/main/java/tensor/TensorExecutionSupport.java`.

## Key Abstractions

**Tensor:**
- Purpose: Mutable tensor value, graph node, data publication target, and public API gateway.
- Examples: `src/main/java/tensor/Tensor.java`, `src/main/java/tensor/TensorMetadata.java`, `src/main/java/tensor/TensorStorage.java`, `src/main/java/tensor/TensorInternalAccess.java`.
- Pattern: Use family builders under `src/main/java/tensor/ops` for new semantics; keep `Tensor.java` as facade and storage/metadata owner.

**Operation:**
- Purpose: Immutable primitive descriptor with stable `Operation.OpType`, expression text, category, and fusable marker.
- Examples: `src/main/java/operations/Operation.java`, `src/main/java/operations/elementwise/unary/relu.java`, `src/main/java/operations/linalg/scaledDotProductAttention.java`, `src/main/java/operations/loss/crossEntropyLossIndices.java`.
- Pattern: Descriptors carry semantic parameters only; runtime policy and kernels belong outside `src/main/java/operations`.

**CompiledGraph / CompileArtifacts:**
- Purpose: Public compile facade and immutable compile artifact bundle.
- Examples: `src/main/java/graph/CompiledGraph.java`, `src/main/java/graph/compile/CompileArtifacts.java`, `src/main/java/graph/compile/GraphCompiler.java`.
- Pattern: Compile creates fresh artifacts from current graph state; prepare consumes artifacts instead of mutating graph semantics.

**CompiledNode:**
- Purpose: Snapshot of a tensor node at compile time with ids, labels, dtype/shape/layout metadata, operation, inputs, backend intent, and semantic source mapping.
- Examples: `src/main/java/graph/CompiledNode.java`, `src/main/java/graph/compile/GraphCompiler.java`.
- Pattern: Backend metadata is attached later through `CompiledNodeExecutionMetadata`, not directly onto `CompiledNode`.

**OptimizerState:**
- Purpose: Container for graph optimizer output, execution metadata, partitions, optimized regions, memory plans, and traces.
- Examples: `src/main/java/graph/optimizer/state/OptimizerState.java`, `src/main/java/graph/optimizer/state/OptimizerTrace.java`.
- Pattern: Optimizer rules transform `OptimizerState`; stage-specific code lives in `src/main/java/graph/optimizer/rewrite`, `cse`, `partition`, `region`, or `memory`.

**PreparedExecution:**
- Purpose: Reusable runtime-bound artifact with forward/backward steps, prepared metadata, memory plan, gradient bindings, root tensor, and prepare trace.
- Examples: `src/main/java/graph/execution/PreparedExecution.java`, `src/main/java/graph/execution/PreparedNodeExecution.java`.
- Pattern: Reuse `PreparedExecution` for repeated runs with the same graph and runtime config; each run receives a new `ExecutionState`.

**CompiledNodeExecutionMetadata:**
- Purpose: Prepare-time metadata for one compiled node or partition anchor.
- Examples: `src/main/java/graph/execution/CompiledNodeExecutionMetadata.java`, `src/main/java/backend/cpu/prepare/CpuNodePreparer.java`, `src/main/java/backend/metal/prepare/MetalNodePreparer.java`.
- Pattern: Store backend, CPU kernel, CPU plan, fused executable, CPU workspace template, accelerator executable, execution operation override, execution input ids, and partition role in one immutable record.

**ComputeEngine:**
- Purpose: Stateless dispatcher from prepared metadata to concrete backend implementations.
- Examples: `src/main/java/backend/ComputeEngine.java`, `src/main/java/backend/ComputeBackend.java`.
- Pattern: Do not add concrete backend helpers to root `src/main/java/backend`; `src/test/java/SourceTreeHygieneTest.java` restricts the root package to facade files.

**CpuNodeExecutionPlan:**
- Purpose: CPU execution recipe resolved at prepare time.
- Examples: `src/main/java/backend/cpu/kernels/CpuNodeExecutionPlan.java`, `src/main/java/backend/cpu/kernels/plan/CpuExecutionPlanner.java`, `src/main/java/backend/cpu/kernels/plan/CpuPlanAssembler.java`.
- Pattern: Runtime kernels consume this plan through `src/main/java/backend/cpu/kernels/CpuKernelContext.java`.

**ExecutionProfile:**
- Purpose: Runnable profile combining dtype, execution mode, optimizer config, runtime config, and workload metadata.
- Examples: `src/main/java/config/profile/ExecutionProfile.java`, `src/main/java/config/profile/GraphExecutionPolicy.java`, `src/main/java/config/profile/PlatformRuntimeProfile.java`.
- Pattern: Benchmark, autotune, calibration, and public tensor autotune all resolve to `ExecutionProfile` values before execution.

## Entry Points

**Gradle Application:**
- Location: `build.gradle`
- Triggers: `./gradlew run`
- Responsibilities: Runs `src/main/java/synaptik/app/TuningCli.java` with Vector API and native access JVM args.

**Tuning CLI:**
- Location: `src/main/java/synaptik/app/TuningCli.java`
- Triggers: `./gradlew run --args="..."` or direct Java application launch.
- Responsibilities: Parse commands such as `calibrate`, `autotune`, `benchmark-winner`, `benchmark-graph-space`, and `full`; orchestrate calibration, graph autotune, and benchmark sessions through `src/main/java/tuning`.

**Programmatic Tuning Example:**
- Location: `src/main/java/synaptik/app/Main.java`
- Triggers: Direct Java main invocation.
- Responsibilities: Demonstrate fluent `src/main/java/tuning/api/Synaptik.java` calibration and benchmark APIs.

**Tensor Compile/Compute API:**
- Location: `src/main/java/tensor/Tensor.java` and `src/main/java/tensor/TensorExecutionSupport.java`
- Triggers: User calls such as `Tensor.compute()`, `Tensor.compile(...)`, `Tensor.prepare(...)`, or `Tensor.compute(ComputeOptions)`.
- Responsibilities: Resolve defaults/profile, compile graph, prepare runtime metadata, execute, and publish results.

**Compiled Graph Facade:**
- Location: `src/main/java/graph/CompiledGraph.java`
- Triggers: `CompiledGraph.compile(root, optimizerConfig, compileMode)` or `Tensor.compile(...)`.
- Responsibilities: Build compile artifacts, prepare runtime execution, and expose convenience execute methods.

**Prepared Execution Runtime:**
- Location: `src/main/java/graph/execution/PreparedExecution.java`
- Triggers: `PreparedExecution.execute(ExecutionMode)` or `PreparedExecution.executeTraced(ExecutionMode)`.
- Responsibilities: Create per-run state, bind memory, run forward/backward steps, collect traces, close resources, and publish outputs/gradients.

**Backend Dispatcher:**
- Location: `src/main/java/backend/ComputeEngine.java`
- Triggers: Each prepared execution step.
- Responsibilities: Dispatch to CPU, Metal, CUDA, or OpenCL based on `CompiledNodeExecutionMetadata.backend()`.

**Numerics CLI:**
- Location: `src/main/java/numerics/NumericsCli.java`
- Triggers: Direct Java main invocation with `numerics.*` system properties.
- Responsibilities: Build comparison profiles and run the numerical drift harness.

**Native Metal Shim Build:**
- Location: `scripts/build-metal-mps-shim.sh` and `src/main/native/apple/synaptik_apple_mps_stub.m`
- Triggers: `./gradlew buildMetalMpsShim`, `./gradlew nativeBuild`, or `./gradlew metalTest`.
- Responsibilities: Build optional macOS Metal MPS dynamic library into `build/native/apple/libsynaptik_apple_mps.dylib`.

## Error Handling

**Strategy:** Fail fast with Java exceptions at the layer that owns the violated contract, and keep traces as diagnostic records rather than hidden control flow.

**Patterns:**
- Public and facade entry points validate required arguments with `IllegalArgumentException` or `NullPointerException`, as in `src/main/java/graph/CompiledGraph.java`, `src/main/java/tensor/TensorExecutionSupport.java`, and `src/main/java/backend/ComputeEngine.java`.
- Unsupported semantic/runtime combinations throw `UnsupportedOperationException`, such as BOOL/INT32 backward roots in `src/main/java/graph/compile/GraphCompiler.java` and unavailable backend paths in `src/main/java/backend/ComputeEngine.java`.
- Missing prepare-time metadata throws `IllegalStateException`, as in `src/main/java/backend/cpu/CpuBackend.java`, `src/main/java/backend/cpu/prepare/CpuNodePreparer.java`, and `src/main/java/backend/cpu/registry/CpuKernelResolver.java`.
- Runtime execution closes resources in `src/main/java/graph/execution/PreparedExecution.java` and suppresses close failures onto the original failure when needed.
- CLI parsing in `src/main/java/synaptik/app/TuningCli.java` prints usage and rethrows invalid command errors.
- Trace objects under `src/main/java/graph/execution/trace` expose compile, prepare, backend selection, and run diagnostics without changing execution semantics.

## Cross-Cutting Concerns

**Logging:** CLI and report output uses `System.out` in `src/main/java/synaptik/app/TuningCli.java`, `src/main/java/synaptik/app/Main.java`, `src/main/java/numerics/NumericsCli.java`, and report renderers under `src/main/java/tuning`. There is no centralized logging framework detected in `build.gradle`.

**Validation:** Shape, dtype, broadcast, and semantic validation belongs in `src/main/java/tensor/ops` and helpers such as `src/main/java/tensor/BroadcastPlanner.java`; optimizer stage-order validation belongs in `src/main/java/config/optimizer/OptimizerConfig.java`; runtime config normalization belongs in `src/main/java/config/runtime/RuntimeConfig.java`; source/package boundary validation is enforced by `src/test/java/SourceTreeHygieneTest.java`.

**Authentication:** Not applicable. No application authentication layer is present in `src/main/java`.

**Configuration:** Build configuration is in `build.gradle`, `settings.gradle`, and `gradle.properties`; runtime/optimizer/profile configuration records are in `src/main/java/config`; persisted optimizer/profile examples include `config/optimizer-profile.json` and `profiles/platform`.

**Native And External Runtime Binding:** BLAS and accelerator bridge lookup is handled under `src/main/java/backend/blas`, `src/main/java/backend/metal/bridge`, and `src/main/java/backend/cuda/bridge`; Metal native source is in `src/main/native/apple`.

**Testing Anchors:** Tests mirror architecture packages under `src/test/java/backend`, `src/test/java/graph`, `src/test/java/config`, `src/test/java/tuning`, and root-level operation/execution tests such as `src/test/java/TensorComputeConvenienceApiTest.java`, `src/test/java/PreparedExecutionBuildTest.java`, and `src/test/java/SourceTreeHygieneTest.java`.

---

*Architecture analysis: 2026-04-29*
