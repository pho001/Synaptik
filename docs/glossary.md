<!-- generated-by: gsd-doc-writer -->
# Glossary

Navigation: [Index](index.md#recommended-reading-paths) | [Framework Concepts](framework-concepts.md#tensors-as-graph-nodes) | [Architecture](architecture.md#core-artifact-boundaries) | [Compute Flow](compute-flow.md#primary-artifacts) | [Graph Optimizer](graph-optimizer.md#graph-optimizer) | [Native Bridges & BLAS](native-bridges-and-blas.md#term-map-at-a-glance) | [Metal Backend](metal-backend.md#native-buffer-abi) | [Calibration & Autotune](calibration-autotune.md#core-distinction)

Chapters: [A](#a) | [B](#b) | [C](#c) | [D](#d) | [E](#e) | [F](#f) | [G](#g) | [J](#j) | [L](#l) | [M](#m) | [N](#n) | [O](#o) | [P](#p) | [R](#r) | [S](#s) | [T](#t) | [W](#w)

Project-specific terms used in Synaptik, with source references.

## Table Of Contents

- [A](#a)
- [B](#b)
- [C](#c)
- [D](#d)
- [E](#e)
- [F](#f)
- [G](#g)
- [J](#j)
- [L](#l)
- [M](#m)
- [N](#n)
- [O](#o)
- [P](#p)
- [R](#r)
- [S](#s)
- [T](#t)
- [W](#w)

## A

**Accelerator config**: Runtime policy for accelerator backends, held by `RuntimeConfig.accelerator()`. Source: [`RuntimeConfig.java`](../src/main/java/config/runtime/RuntimeConfig.java), [`AcceleratorConfig.java`](../src/main/java/config/runtime/AcceleratorConfig.java).

**Application Binary Interface (ABI)**: Runtime/binary contract between Java FFM code and the native Metal shim. It defines native symbol names, primitive argument layout, pointer meaning, buffer ownership, lifetime, and synchronization expectations. It is different from a Java API, which is a source-level method/class contract. In Synaptik, ABI-sensitive code lives around [`MetalMpsFfmBridge.java`](../src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java) and [`synaptik_apple_mps_stub.m`](../src/main/native/apple/synaptik_apple_mps_stub.m).

**Arena**: Java FFM lifetime scope for native allocations and library lookup resources. The OpenBLAS bridge keeps a shared arena in its cached state so symbol lookup resources stay valid for later downcalls. Source: [`OpenBlasFfmBridge.java`](../src/main/java/backend/blas/OpenBlasFfmBridge.java), [Native Bridges & BLAS: Java FFM Step-By-Step](native-bridges-and-blas.md#java-ffm-step-by-step).

**Autodiff**: Reverse-mode gradient graph construction over tensor DAGs. Source: [`BackwardGraphBuilder.java`](../src/main/java/graph/compile/BackwardGraphBuilder.java), [`TensorBinaryOps.java`](../src/main/java/tensor/ops/binary/TensorBinaryOps.java).

**Autograd compilation scope**: Scope opened while compile builds backward graph state. Source: [`AutogradCompilationScope.java`](../src/main/java/tensor/AutogradCompilationScope.java), [`GraphCompiler.java`](../src/main/java/graph/compile/GraphCompiler.java).

**Autotune**: Evaluation of candidate `ExecutionProfile` objects for a workload. Source: [`AutotuneSession.java`](../src/main/java/tuning/autotune/AutotuneSession.java), [`DefaultAutotuneSession.java`](../src/main/java/tuning/autotune/DefaultAutotuneSession.java).

## B

**BF16 / BFLOAT16**: 16-bit floating-point storage format with an 8-bit exponent and 7 explicit mantissa bits. In the CPU path it is commonly stored as `short[]`, unpacked to F32 for elementwise compute, and packed back to BF16 storage. Source: [`BFloat16Storage.java`](../src/main/java/tensor/BFloat16Storage.java), [CPU BF16 Runtime](cpu-bf16.md#cpu-bf16-runtime).

**Backward graph**: Gradient-producing graph nodes built from forward-node backward lambdas during training compile. Source: [`BackwardGraphBuilder.java`](../src/main/java/graph/compile/BackwardGraphBuilder.java).

**Backward node**: A tensor node marked as part of backward-stage execution and captured in `CompiledNode.backwardNode()`. Source: [`CompiledNode.java`](../src/main/java/graph/CompiledNode.java), [`BackwardGraphBuilder.java`](../src/main/java/graph/compile/BackwardGraphBuilder.java).

**Backend prepare context**: Prepare-time context containing runtime config, compiled nodes, consumers, selected plans, lowered regions, and prepared metadata. Source: [`BackendPrepareContext.java`](../src/main/java/backend/prepare/BackendPrepareContext.java).

**Backend selection**: Prepare-time selection of backend partition plans using runtime config. Source: [`DefaultBackendSelectionPolicy.java`](../src/main/java/backend/select/DefaultBackendSelectionPolicy.java), [`PreparedExecutionBuilder.java`](../src/main/java/backend/prepare/PreparedExecutionBuilder.java).

**BLAS**: Basic Linear Algebra Subprograms, a standard family of optimized vector/matrix routines. In Synaptik, BLAS currently matters mainly for GEMM-backed matmul and GEMM-lowered conv2d through the OpenBLAS FFM bridge. Source: [`BlasProvider.java`](../src/main/java/backend/blas/BlasProvider.java), [`OpenBlasFfmBridge.java`](../src/main/java/backend/blas/OpenBlasFfmBridge.java), [Native Bridges & BLAS: What BLAS Is](native-bridges-and-blas.md#what-blas-is).

**Broadcast plan**: Shape, stride, and gradient-reduction metadata for broadcasted binary operations. Source: [`BroadcastPlan.java`](../src/main/java/tensor/BroadcastPlan.java), [`BroadcastPlanner.java`](../src/main/java/tensor/BroadcastPlanner.java).

## C

**Calibration**: Search for platform runtime defaults across calibration families and workloads. Source: [`DefaultPlatformCalibrationSession.java`](../src/main/java/tuning/calibration/DefaultPlatformCalibrationSession.java), [`PlatformCalibrationDefaults.java`](../src/main/java/tuning/calibration/PlatformCalibrationDefaults.java).

**CBLAS**: C language interface to BLAS. Synaptik's OpenBLAS bridge looks up CBLAS symbols such as `cblas_sgemm`, `cblas_dgemm`, and optional `cblas_sbgemm`, then calls them through Java FFM. Source: [`OpenBlasFfmBridge.java`](../src/main/java/backend/blas/OpenBlasFfmBridge.java), [Native Bridges & BLAS: OpenBLAS In Synaptik](native-bridges-and-blas.md#openblas-in-synaptik).

**Compile artifact**: The immutable result set from graph compile, including compiled nodes, gradient bindings, optimizer state, partitions, memory plan, and traces. Source: [`CompileArtifacts.java`](../src/main/java/graph/compile/CompileArtifacts.java).

**Compile mode**: Public compile intent: `INFERENCE_ONLY`, `TRAINING`, or `AUTO`. Source: [`CompileMode.java`](../src/main/java/tensor/CompileMode.java), [`GraphCompiler.java`](../src/main/java/graph/compile/GraphCompiler.java).

**Compile trace**: Timing and graph-size metadata captured from compile. Source: [`CompileTrace.java`](../src/main/java/graph/execution/trace/CompileTrace.java), [`GraphCompiler.java`](../src/main/java/graph/compile/GraphCompiler.java).

**Compiled graph**: Facade around compiled artifacts with prepare and execute helpers. Source: [`CompiledGraph.java`](../src/main/java/graph/CompiledGraph.java).

**CPU materialization reason**: Explicit reason why execution needs CPU-readable tensor storage, such as graph output publication, gradient publication, CPU consumer, public data access, or CPU fallback. It is a residency/trace contract, not a copy implementation. Source: [`CpuMaterializationReason.java`](../src/main/java/backend/memory/CpuMaterializationReason.java), [`ExecutionState.java`](../src/main/java/graph/execution/ExecutionState.java), [`PreparedExecution.java`](../src/main/java/graph/execution/PreparedExecution.java).

**CPU materialization trace**: Run-trace entry that records a failed or completed request for CPU-readable tensor storage, including node id, reason, source backend/residency, logical bytes, duration, completion flag, and diagnostic detail. Source: [`CpuMaterializationTrace.java`](../src/main/java/graph/execution/trace/CpuMaterializationTrace.java), [`RunTrace.java`](../src/main/java/graph/execution/trace/RunTrace.java), [`ExecutionState.java`](../src/main/java/graph/execution/ExecutionState.java).

**Device-to-CPU materializer**: Per-run backend hook that synchronizes an active device buffer binding into a runtime tensor's CPU-visible storage when a CPU consumer, graph output publication, gradient publication, or public data access needs current CPU bytes. Source: [`DeviceToCpuMaterializer.java`](../src/main/java/backend/memory/DeviceToCpuMaterializer.java), [`CpuMaterializationResult.java`](../src/main/java/backend/memory/CpuMaterializationResult.java), [`ExecutionState.java`](../src/main/java/graph/execution/ExecutionState.java).

**Compiled node**: Immutable compile-time snapshot of a tensor node. Source: [`CompiledNode.java`](../src/main/java/graph/CompiledNode.java).

**Compute contract**: CPU prepared metadata describing storage dtype, compute dtype, accumulate dtype, and execution backend. Source: [`ResolvedCpuComputeContract.java`](../src/main/java/backend/cpu/kernels/ResolvedCpuComputeContract.java), [`CpuExecutionPlanner.java`](../src/main/java/backend/cpu/kernels/plan/CpuExecutionPlanner.java).

**Compute engine**: Backend dispatch facade called by `PreparedExecution`. Source: [`ComputeEngine.java`](../src/main/java/backend/ComputeEngine.java).

**CPU execution plan**: Prepared per-node CPU recipe containing layout, compute contract, dispatch hints, reduction/matmul/conv hints, and application behavior. Source: [`CpuNodeExecutionPlan.java`](../src/main/java/backend/cpu/kernels/CpuNodeExecutionPlan.java), [`CpuPlanAssembler.java`](../src/main/java/backend/cpu/kernels/plan/CpuPlanAssembler.java).

**CPU kernel resolver**: Mapping from `Operation.OpType` to concrete CPU kernel instances. Source: [`CpuKernelResolver.java`](../src/main/java/backend/cpu/registry/CpuKernelResolver.java).

**CSE**: Common subexpression elimination graph optimization stage. Source: [`CommonSubexpressionEliminationRule.java`](../src/main/java/graph/optimizer/cse/CommonSubexpressionEliminationRule.java), [`OptimizerFactory.java`](../src/main/java/graph/optimizer/OptimizerFactory.java).

## D

**Detached gradient publication**: Copying runtime gradient tensors back to semantic tensors without aliasing runtime buffers. Source: [`PreparedExecution.java`](../src/main/java/graph/execution/PreparedExecution.java).

**Device buffer binding**: Backend-neutral runtime descriptor that associates a compiled node with a usable device-visible buffer. Metal implements it through `MetalBufferBinding`, while execution state stores only the small common contract. A binding can be reserved for a future writable output without marking the value current, or attached after execution as the active current value. Source: [`DeviceBufferBinding.java`](../src/main/java/backend/memory/DeviceBufferBinding.java), [`MetalBufferBinding.java`](../src/main/java/backend/metal/buffer/MetalBufferBinding.java), [`ExecutionState.java`](../src/main/java/graph/execution/ExecutionState.java).

**Dispatch hints**: Prepared elementwise/fused execution mode metadata such as scalar/vector/parallel mode, vector width, workers, and chunk sizes. Source: [`ResolvedDispatchHints.java`](../src/main/java/backend/cpu/kernels/elementwise/plan/ResolvedDispatchHints.java), [`CpuExecutionPlanner.java`](../src/main/java/backend/cpu/kernels/plan/CpuExecutionPlanner.java).

**Downcall**: Java FFM call from Java into native code. Synaptik's OpenBLAS bridge creates downcall `MethodHandle`s for CBLAS functions and invokes them from Java CPU matmul wrappers. Source: [`OpenBlasFfmBridge.java`](../src/main/java/backend/blas/OpenBlasFfmBridge.java), [Native Bridges & BLAS: What Java FFM Is](native-bridges-and-blas.md#what-java-ffm-is).

## E

**Execution context**: Per-run context containing execution mode, runtime config-derived policies, metadata index, execution state, and family-specific runtime caches. Source: [`ExecutionContext.java`](../src/main/java/backend/runtime/ExecutionContext.java).

**Execution mode**: Runtime mode such as `FORWARD` or `FORWARD_BACKWARD`. Source: [`ExecutionMode.java`](../src/main/java/backend/runtime/ExecutionMode.java).

**Execution profile**: Runnable profile combining dtype, execution mode, compile config, runtime config, and workload metadata. Source: [`ExecutionProfile.java`](../src/main/java/config/profile/ExecutionProfile.java).

**Execution state**: Per-run runtime tensor storage and metadata state built for prepared execution. Source: [`ExecutionState.java`](../src/main/java/graph/execution/ExecutionState.java).

## F

**Forward boundary**: Compiled node id/index separating forward and backward execution steps. Source: [`CompileArtifacts.java`](../src/main/java/graph/compile/CompileArtifacts.java), [`PreparedExecutionBuilder.java`](../src/main/java/backend/prepare/PreparedExecutionBuilder.java).

**Forward output wrapper**: System `NOOP` tensor created by `Tensor.forwardOutput()` to normalize publication. Source: [`Tensor.java`](../src/main/java/tensor/Tensor.java), [`GraphCompiler.java`](../src/main/java/graph/compile/GraphCompiler.java).

**Fused region optimization**: Compile phase that creates fused and unit execution units inside already owned regions. It is no longer modeled as a graph optimizer stage. Source: [`DefaultRegionOptimizer.java`](../src/main/java/graph/optimizer/region/DefaultRegionOptimizer.java), [`RegionOptimizationConfig.java`](../src/main/java/config/compile/RegionOptimizationConfig.java).

**Fused operation**: Backend-owned `Operation` descriptor for a fused expression plan. Source: [`FusedOperation.java`](../src/main/java/backend/cpu/fused/plan/FusedOperation.java).

**Fused ASM executable**: Generated bytecode implementation of a fused execution plan, loaded and cached during prepare. Source: [`AsmPreparedFusedExecutableFactory.java`](../src/main/java/backend/cpu/fused/asm/AsmPreparedFusedExecutableFactory.java), [`PreparedFusedExecutable.java`](../src/main/java/backend/cpu/fused/exec/PreparedFusedExecutable.java).

## G

**Graph execution policy**: Profile component wrapping `CompileConfig` for graph/autotune candidate assembly. Source: [`GraphExecutionPolicy.java`](../src/main/java/config/profile/GraphExecutionPolicy.java).

**GEMM**: General Matrix Multiply, usually written as `C = alpha * A @ B + beta * C`. Synaptik uses GEMM for direct matmul, linear-style matrix products, attention matmuls, and conv2d after im2col lowering. Source: [`MatMulBlasBackend.java`](../src/main/java/backend/cpu/kernels/linalg/matmul/blas/MatMulBlasBackend.java), [`Conv2dGemmBackend.java`](../src/main/java/backend/cpu/kernels/nn/Conv2dGemmBackend.java), [Native Bridges & BLAS: GEMM Mental Model](native-bridges-and-blas.md#gemm-mental-model).

**Graph optimizer**: Backend-neutral graph cleanup and lowering pipeline: `AR`, `CF`, `CSE`, `DCE`, and optional `LOWER`. Source: [`GraphOptimizer.java`](../src/main/java/graph/optimizer/GraphOptimizer.java), [`OptimizerFactory.java`](../src/main/java/graph/optimizer/OptimizerFactory.java).

**Gradient binding**: Mapping from semantic/source tensors to compiled gradient nodes or constant gradient templates. Source: [`CompiledGradientBinding.java`](../src/main/java/graph/CompiledGradientBinding.java), [`GradientBindingCollector.java`](../src/main/java/graph/compile/GradientBindingCollector.java).

## J

**Java FFM**: Java Foreign Function and Memory API. Synaptik uses it to load native libraries, find exported symbols, describe native function signatures, and call native code through downcall `MethodHandle`s. The OpenBLAS bridge uses FFM to call CBLAS; Metal and CUDA bridges use FFM to call Synaptik native shims. Source: [`OpenBlasFfmBridge.java`](../src/main/java/backend/blas/OpenBlasFfmBridge.java), [`MetalMpsFfmBridge.java`](../src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java), [`CudaFfmBridge.java`](../src/main/java/backend/cuda/bridge/CudaFfmBridge.java), [Native Bridges & BLAS: What Java FFM Is](native-bridges-and-blas.md#what-java-ffm-is).

## L

**Leaf tensor**: Tensor with no operation descriptor, usually user input or parameter data. Source: [`CompiledNode.java`](../src/main/java/graph/CompiledNode.java), [`Tensor.java`](../src/main/java/tensor/Tensor.java).

**Leading dimension**: BLAS stride parameter describing the distance between adjacent logical rows for row-major CBLAS calls. For dense row-major `[M,K]` input `A`, Synaptik passes `lda = K`; for dense `[K,N]` input `B`, it passes `ldb = N`. Source: [`OpenBlasFfmBridge.java`](../src/main/java/backend/blas/OpenBlasFfmBridge.java), [Native Bridges & BLAS: Matrix Storage Terms](native-bridges-and-blas.md#matrix-storage-terms).

**Lowering**: Conversion from higher-level graph/region structure into backend-executable units or primitive descriptors. Source: [`LoweringPipeline.java`](../src/main/java/backend/lowering/LoweringPipeline.java), [`PreparedExecutionBuilder.java`](../src/main/java/backend/prepare/PreparedExecutionBuilder.java).

## M

**Memory planning**: Compile phase that builds `MemoryPlan` artifacts, including lifetimes, reusable intervals, slots, region-value bindings, and runtime binding policy. Source: [`MemoryPlanner.java`](../src/main/java/graph/optimizer/memory/MemoryPlanner.java), [`MemoryPlanningConfig.java`](../src/main/java/config/compile/MemoryPlanningConfig.java).

**Memory plan**: Compile-time lifetimes, reusable intervals, slots, region-value memory bindings, and runtime binding policies. Source: [`MemoryPlan.java`](../src/main/java/graph/optimizer/memory/MemoryPlan.java), [`MemoryPlanner.java`](../src/main/java/graph/optimizer/memory/MemoryPlanner.java).

**Memory role**: Classification used by memory planning for temporaries, saved forward values, gradient targets, and related storage owners. Source: [`MemoryRole.java`](../src/main/java/graph/optimizer/memory/MemoryRole.java), [`NodeLifetime.java`](../src/main/java/graph/optimizer/memory/NodeLifetime.java).

**Memory segment**: Java FFM view of a region of memory. In the OpenBLAS bridge, `MemorySegment.ofArray(...)` wraps Java tensor arrays and `asSlice(...)` applies element offsets for batched GEMM calls. Source: [`OpenBlasFfmBridge.java`](../src/main/java/backend/blas/OpenBlasFfmBridge.java), [Native Bridges & BLAS: Java FFM Step-By-Step](native-bridges-and-blas.md#java-ffm-step-by-step).

**Metal buffer binding**: Java-side descriptor for the current native Metal buffer execution path. It ties a compiled node id, dtype, shape, element count, access intent, and native buffer handle together without exposing semantic `Tensor` objects to native code. Source: [`MetalBufferBinding.java`](../src/main/java/backend/metal/buffer/MetalBufferBinding.java), [`MetalBufferHandle.java`](../src/main/java/backend/metal/buffer/MetalBufferHandle.java), [Metal Backend: Buffer Residency And Materialization](metal-backend.md#buffer-residency-and-materialization).

**Metal bridge execution stats**: Per-execution diagnostics from the Metal MPSGraph bridge, including input/output bytes, Java-to-native copy time, native execute time, native-to-Java copy time, CPU fallback flag, and fallback reason. Source: [`MetalMpsBridgeExecutionStats.java`](../src/main/java/backend/metal/bridge/MetalMpsBridgeExecutionStats.java), [`PreparedMetalExecutable.java`](../src/main/java/backend/metal/exec/PreparedMetalExecutable.java).

**Metal bridge execution path**: Trace label for the actual runtime path of a selected Metal executable: CPU fallback, tensor-array copy bridge, or explicit buffer-binding bridge. Source: [`MetalMpsBridgeExecutionPath.java`](../src/main/java/backend/metal/bridge/MetalMpsBridgeExecutionPath.java), [`PreparedExecution.java`](../src/main/java/graph/execution/PreparedExecution.java), [Metal Backend: Trace Reading](metal-backend.md#trace-reading).

**MPSGraph**: Apple's graph-level Metal Performance Shaders API used by the native shim to compile and run selected `FLOAT32` accelerator DAG regions. Synaptik calls it through the Objective-C shim, not directly from Java tensor code. Source: [`synaptik_apple_mps_stub.m`](../src/main/native/apple/synaptik_apple_mps_stub.m), [Metal Backend: Objective-C Native Shim](metal-backend.md#objective-c-native-shim).

**MTLBuffer**: Metal buffer object used by the native shim to pass tensor bytes to and from MPSGraph. Java sees it only as an opaque handle inside `MetalBufferHandle`; ownership and release are handled by the native shim and run-scoped execution resources. Source: [`MetalBufferHandle.java`](../src/main/java/backend/metal/buffer/MetalBufferHandle.java), [`synaptik_apple_mps_stub.m`](../src/main/native/apple/synaptik_apple_mps_stub.m), [Metal Backend: Native Buffer ABI](metal-backend.md#native-buffer-abi).

**Metal transfer model**: Compile planning cost preset used by scored Metal region planning to penalize input/output transfer bytes and credit avoided intermediate materialization. Source: [`MetalTransferModel.java`](../src/main/java/config/optimizer/MetalTransferModel.java), [`BackendPlanningCostConfig.java`](../src/main/java/config/compile/BackendPlanningCostConfig.java), [`ScoredCandidatePartitionPlanner.java`](../src/main/java/graph/optimizer/partition/ScoredCandidatePartitionPlanner.java).

## N

**Native bridge**: Java adapter that loads a native library, discovers symbols, converts Java-side metadata/memory into the native ABI, and calls native code. Source: [`OpenBlasFfmBridge.java`](../src/main/java/backend/blas/OpenBlasFfmBridge.java), [`MetalMpsFfmBridge.java`](../src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java), [`CudaFfmBridge.java`](../src/main/java/backend/cuda/bridge/CudaFfmBridge.java), [Native Bridges & BLAS: Term Map At A Glance](native-bridges-and-blas.md#term-map-at-a-glance).

**Native library**: Loadable compiled binary outside the JVM, such as OpenBLAS or the Synaptik Metal `.dylib`. Java FFM uses `SymbolLookup.libraryLookup(...)` to load and inspect these libraries. Source: [`OpenBlasFfmBridge.java`](../src/main/java/backend/blas/OpenBlasFfmBridge.java), [`MetalMpsFfmBridge.java`](../src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java).

## O

**OpenBLAS**: Native BLAS implementation optionally used by Synaptik's CPU matmul and GEMM-lowered conv2d paths. It is selected with `BlasProvider.OPENBLAS_FFM`, but each node still needs to pass planner legality and profitability gates before the bridge is called. Source: [`OpenBlasFfmBridge.java`](../src/main/java/backend/blas/OpenBlasFfmBridge.java), [`MatMulPlanner.java`](../src/main/java/backend/cpu/kernels/linalg/matmul/plan/MatMulPlanner.java), [Native Bridges & BLAS: OpenBLAS In Synaptik](native-bridges-and-blas.md#openblas-in-synaptik).

**Operation descriptor**: Immutable semantic descriptor implementing `Operation`. Source: [`Operation.java`](../src/main/java/operations/Operation.java), [`operations/README.md`](../src/main/java/operations/README.md#core-contract).

**Optimizer state**: Mutable optimizer pipeline carrier for graph, forward output, execution metadata, partitions, optimized regions, and memory plan. Source: [`OptimizerState.java`](../src/main/java/graph/optimizer/state/OptimizerState.java).

**Optimizer stage**: Current documentation uses this term only for backend-neutral graph optimization phases: `AR`, `CF`, `CSE`, `DCE`, and `LOWER`. Backend planning, region optimization, and memory planning are compile phases, not optimizer stages. Source: [`GraphOptimizationConfig.java`](../src/main/java/config/compile/GraphOptimizationConfig.java), [`OptimizerFactory.java`](../src/main/java/graph/optimizer/OptimizerFactory.java).

## P

**Backend planning**: Compile-time phase that propagates backend intent and creates CPU or accelerator ownership regions. Source: [`BackendPlanningConfig.java`](../src/main/java/config/compile/BackendPlanningConfig.java), [`BackendPlanningService.java`](../src/main/java/graph/compile/BackendPlanningService.java), [`BackendPlanningJobResolver.java`](../src/main/java/graph/compile/BackendPlanningJobResolver.java).

**Partition**: Internal implementation object for a backend-compatible ownership region discovered during backend planning. Architecture docs prefer "ownership region" to avoid confusing this with a public graph optimizer stage. Source: [`Partition.java`](../src/main/java/graph/optimizer/partition/Partition.java), [`GreedyMaxRegionPartitionPlanner.java`](../src/main/java/graph/optimizer/partition/GreedyMaxRegionPartitionPlanner.java).

**Partition anchor**: Executable node representing a lowered partition while interior nodes are skipped as standalone execution steps. Source: [`PartitionExecutionRole.java`](../src/main/java/backend/accelerator/exec/PartitionExecutionRole.java), [`CpuNodePreparer.java`](../src/main/java/backend/cpu/prepare/CpuNodePreparer.java).

**Platform runtime profile**: Machine-oriented runtime-default profile produced by calibration and convertible to `RuntimeConfig`. Source: [`PlatformRuntimeProfile.java`](../src/main/java/config/profile/PlatformRuntimeProfile.java).

**Prepare trace**: Timing and step-count metadata captured by prepare. Source: [`PrepareTrace.java`](../src/main/java/graph/execution/trace/PrepareTrace.java), [`PreparedExecutionBuilder.java`](../src/main/java/backend/prepare/PreparedExecutionBuilder.java).

**Prepared execution**: Reusable runtime-bound artifact containing forward/backward steps and metadata. Source: [`PreparedExecution.java`](../src/main/java/graph/execution/PreparedExecution.java).

**Prepared node execution**: One executable prepared node step: compiled node plus metadata. Source: [`PreparedNodeExecution.java`](../src/main/java/graph/execution/PreparedNodeExecution.java).

## R

**Region optimization**: FUSE-stage conversion of partitions into optimized regions. Source: [`RegionOptimizationRule.java`](../src/main/java/graph/optimizer/region/RegionOptimizationRule.java), [`DefaultRegionOptimizer.java`](../src/main/java/graph/optimizer/region/DefaultRegionOptimizer.java).

**Row-major**: Matrix storage order where adjacent elements in the same row are adjacent in memory. Synaptik's OpenBLAS bridge uses row-major no-transpose CBLAS calls for dense contiguous matmul buffers. Source: [`OpenBlasFfmBridge.java`](../src/main/java/backend/blas/OpenBlasFfmBridge.java), [Native Bridges & BLAS: Matrix Storage Terms](native-bridges-and-blas.md#matrix-storage-terms).

**Runtime config**: Runtime/backend policy used by prepare and execution. Source: [`RuntimeConfig.java`](../src/main/java/config/runtime/RuntimeConfig.java).

**Runtime memory binder**: Execution-start binder that aliases view nodes and assigns slot arrays from `MemoryPlan` to runtime tensors. Source: [`RuntimeMemoryBinder.java`](../src/main/java/graph/execution/RuntimeMemoryBinder.java).

**Run trace**: Execution-mode timing plus optional per-step trace metadata. Source: [`RunTrace.java`](../src/main/java/graph/execution/trace/RunTrace.java), [`PreparedExecution.java`](../src/main/java/graph/execution/PreparedExecution.java).

**Runtime storage residency**: Per-run state describing where the newest value for a compiled node is materialized: normal CPU array, host-shared device buffer, or device-owned buffer. Source: [`StorageResidency.java`](../src/main/java/backend/memory/StorageResidency.java), [`TensorResidencyState.java`](../src/main/java/backend/memory/TensorResidencyState.java), [`ExecutionState.java`](../src/main/java/graph/execution/ExecutionState.java).

## S

**Semantic canonicalization**: Pre-autograd forward graph rebuilding into canonical semantic forms. Source: [`SemanticForwardCanonicalizer.java`](../src/main/java/graph/SemanticForwardCanonicalizer.java).

**Source tensor mapping**: Mapping from canonicalized/optimized tensors back to original semantic tensors for publication and gradient binding. Source: [`GraphCompiler.java`](../src/main/java/graph/compile/GraphCompiler.java), [`CompiledNode.java`](../src/main/java/graph/CompiledNode.java).

**Storage offset**: Offset into backing storage for view tensors. Source: [`TensorMetadata.java`](../src/main/java/tensor/TensorMetadata.java).

**Storage residency**: Physical residency class for a runtime tensor value, distinct from semantic tensor dtype/layout. `CPU_ARRAY` means Java typed storage is current; `HOST_SHARED_DEVICE_BUFFER` and `DEVICE_OWNED` are explicit states for shared-buffer or GPU-owned execution paths. Source: [`StorageResidency.java`](../src/main/java/backend/memory/StorageResidency.java).

**Stride**: Per-axis step used to translate logical indices to storage positions. Source: [`TensorMetadata.java`](../src/main/java/tensor/TensorMetadata.java).

**Symbol**: Exported native function or object name found in a native library. Java FFM looks up symbols such as `cblas_sgemm`, `cblas_dgemm`, and `synaptik_apple_mps_execute_partition_f32_buffers` before it can create downcall handles. Source: [`OpenBlasFfmBridge.java`](../src/main/java/backend/blas/OpenBlasFfmBridge.java), [`MetalMpsFfmBridge.java`](../src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java).

## T

**Tensor**: Public value type and semantic graph node. Source: [`Tensor.java`](../src/main/java/tensor/Tensor.java), [`tensor/README.md`](../src/main/java/tensor/README.md#what-tensor-represents).

**Tensor storage**: Dtype-specific backing storage with mutation versioning. Source: [`TensorStorage.java`](../src/main/java/tensor/TensorStorage.java).

**Tuning history**: Persisted per-candidate measurement history for autotune. Source: [`TuningHistoryEntry.java`](../src/main/java/tuning/store/TuningHistoryEntry.java), [`JsonFileTuningHistoryStore.java`](../src/main/java/tuning/store/JsonFileTuningHistoryStore.java).

## W

**Workload fingerprint**: Key representing workload and profile context for persisted tuning records. Source: [`WorkloadFingerprint.java`](../src/main/java/tuning/store/WorkloadFingerprint.java).

**Workspace**: Per-node CPU scratch state such as float continuation buffers, packed weights, or max-pool index storage. Source: [`CpuNodeWorkspace.java`](../src/main/java/backend/cpu/kernels/CpuNodeWorkspace.java), [`CpuNodePreparer.java`](../src/main/java/backend/cpu/prepare/CpuNodePreparer.java).
