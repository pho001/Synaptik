<!-- generated-by: gsd-doc-writer -->
# Glossary

Navigation: [Index](index.md) | [Framework Concepts](framework-concepts.md) | [Architecture](architecture.md) | [Compute Flow](compute-flow.md) | [Graph Optimizer](graph-optimizer.md) | [Calibration & Autotune](calibration-autotune.md)

Chapters: [A](#a) | [B](#b) | [C](#c) | [D](#d) | [E](#e) | [F](#f) | [G](#g) | [L](#l) | [M](#m) | [O](#o) | [P](#p) | [R](#r) | [S](#s) | [T](#t) | [W](#w)

Project-specific terms used in Synaptik, with source references.

## Table Of Contents

- [A](#a)
- [B](#b)
- [C](#c)
- [D](#d)
- [E](#e)
- [F](#f)
- [G](#g)
- [L](#l)
- [M](#m)
- [O](#o)
- [P](#p)
- [R](#r)
- [S](#s)
- [T](#t)
- [W](#w)

## A

**Accelerator config**: Runtime policy for accelerator backends, held by `RuntimeConfig.accelerator()`. Source: [`RuntimeConfig.java`](../src/main/java/config/runtime/RuntimeConfig.java), [`AcceleratorConfig.java`](../src/main/java/config/runtime/AcceleratorConfig.java).

**Autodiff**: Reverse-mode gradient graph construction over tensor DAGs. Source: [`BackwardGraphBuilder.java`](../src/main/java/graph/compile/BackwardGraphBuilder.java), [`TensorBinaryOps.java`](../src/main/java/tensor/ops/binary/TensorBinaryOps.java).

**Autograd compilation scope**: Scope opened while compile builds backward graph state. Source: [`AutogradCompilationScope.java`](../src/main/java/tensor/AutogradCompilationScope.java), [`GraphCompiler.java`](../src/main/java/graph/compile/GraphCompiler.java).

**Autotune**: Evaluation of candidate `ExecutionProfile` objects for a workload. Source: [`AutotuneSession.java`](../src/main/java/tuning/autotune/AutotuneSession.java), [`DefaultAutotuneSession.java`](../src/main/java/tuning/autotune/DefaultAutotuneSession.java).

## B

**Backward graph**: Gradient-producing graph nodes built from forward-node backward lambdas during training compile. Source: [`BackwardGraphBuilder.java`](../src/main/java/graph/compile/BackwardGraphBuilder.java).

**Backward node**: A tensor node marked as part of backward-stage execution and captured in `CompiledNode.backwardNode()`. Source: [`CompiledNode.java`](../src/main/java/graph/CompiledNode.java), [`BackwardGraphBuilder.java`](../src/main/java/graph/compile/BackwardGraphBuilder.java).

**Backend prepare context**: Prepare-time context containing runtime config, compiled nodes, consumers, selected plans, lowered regions, and prepared metadata. Source: [`BackendPrepareContext.java`](../src/main/java/backend/prepare/BackendPrepareContext.java).

**Backend selection**: Prepare-time selection of backend partition plans using runtime config. Source: [`DefaultBackendSelectionPolicy.java`](../src/main/java/backend/select/DefaultBackendSelectionPolicy.java), [`PreparedExecutionBuilder.java`](../src/main/java/backend/prepare/PreparedExecutionBuilder.java).

**Broadcast plan**: Shape, stride, and gradient-reduction metadata for broadcasted binary operations. Source: [`BroadcastPlan.java`](../src/main/java/tensor/BroadcastPlan.java), [`BroadcastPlanner.java`](../src/main/java/tensor/BroadcastPlanner.java).

## C

**Calibration**: Search for platform runtime defaults across calibration families and workloads. Source: [`DefaultPlatformCalibrationSession.java`](../src/main/java/tuning/calibration/DefaultPlatformCalibrationSession.java), [`PlatformCalibrationDefaults.java`](../src/main/java/tuning/calibration/PlatformCalibrationDefaults.java).

**Compile artifact**: The immutable result set from graph compile, including compiled nodes, gradient bindings, optimizer state, partitions, memory plan, and traces. Source: [`CompileArtifacts.java`](../src/main/java/graph/compile/CompileArtifacts.java).

**Compile mode**: Public compile intent: `INFERENCE_ONLY`, `TRAINING`, or `AUTO`. Source: [`CompileMode.java`](../src/main/java/tensor/CompileMode.java), [`GraphCompiler.java`](../src/main/java/graph/compile/GraphCompiler.java).

**Compile trace**: Timing and graph-size metadata captured from compile. Source: [`CompileTrace.java`](../src/main/java/graph/execution/trace/CompileTrace.java), [`GraphCompiler.java`](../src/main/java/graph/compile/GraphCompiler.java).

**Compiled graph**: Facade around compiled artifacts with prepare and execute helpers. Source: [`CompiledGraph.java`](../src/main/java/graph/CompiledGraph.java).

**CPU materialization reason**: Explicit reason why execution needs CPU-readable tensor storage, such as graph output publication, gradient publication, CPU consumer, public data access, or CPU fallback. It is a residency/trace contract, not a copy implementation. Source: [`CpuMaterializationReason.java`](../src/main/java/backend/memory/CpuMaterializationReason.java), [`ExecutionState.java`](../src/main/java/graph/execution/ExecutionState.java), [`PreparedExecution.java`](../src/main/java/graph/execution/PreparedExecution.java).

**Compiled node**: Immutable compile-time snapshot of a tensor node. Source: [`CompiledNode.java`](../src/main/java/graph/CompiledNode.java).

**Compute contract**: CPU prepared metadata describing storage dtype, compute dtype, accumulate dtype, and execution backend. Source: [`ResolvedCpuComputeContract.java`](../src/main/java/backend/cpu/kernels/ResolvedCpuComputeContract.java), [`CpuExecutionPlanner.java`](../src/main/java/backend/cpu/kernels/plan/CpuExecutionPlanner.java).

**Compute engine**: Backend dispatch facade called by `PreparedExecution`. Source: [`ComputeEngine.java`](../src/main/java/backend/ComputeEngine.java).

**CPU execution plan**: Prepared per-node CPU recipe containing layout, compute contract, dispatch hints, reduction/matmul/conv hints, and application behavior. Source: [`CpuNodeExecutionPlan.java`](../src/main/java/backend/cpu/kernels/CpuNodeExecutionPlan.java), [`CpuPlanAssembler.java`](../src/main/java/backend/cpu/kernels/plan/CpuPlanAssembler.java).

**CPU kernel resolver**: Mapping from `Operation.OpType` to concrete CPU kernel instances. Source: [`CpuKernelResolver.java`](../src/main/java/backend/cpu/registry/CpuKernelResolver.java).

**CSE**: Common subexpression elimination optimizer stage. Source: [`CommonSubexpressionEliminationRule.java`](../src/main/java/graph/optimizer/cse/CommonSubexpressionEliminationRule.java), [`OptimizerFactory.java`](../src/main/java/graph/optimizer/OptimizerFactory.java).

## D

**Detached gradient publication**: Copying runtime gradient tensors back to semantic tensors without aliasing runtime buffers. Source: [`PreparedExecution.java`](../src/main/java/graph/execution/PreparedExecution.java).

**Device buffer binding**: Backend-neutral runtime descriptor that associates a compiled node with a usable device-visible buffer. Metal implements it through `MetalBufferBinding`, while execution state stores only the small common contract. Source: [`DeviceBufferBinding.java`](../src/main/java/backend/memory/DeviceBufferBinding.java), [`MetalBufferBinding.java`](../src/main/java/backend/metal/buffer/MetalBufferBinding.java), [`ExecutionState.java`](../src/main/java/graph/execution/ExecutionState.java).

**Dispatch hints**: Prepared elementwise/fused execution mode metadata such as scalar/vector/parallel mode, vector width, workers, and chunk sizes. Source: [`ResolvedDispatchHints.java`](../src/main/java/backend/cpu/kernels/elementwise/plan/ResolvedDispatchHints.java), [`CpuExecutionPlanner.java`](../src/main/java/backend/cpu/kernels/plan/CpuExecutionPlanner.java).

## E

**Execution context**: Per-run context containing execution mode, runtime config-derived policies, metadata index, execution state, and family-specific runtime caches. Source: [`ExecutionContext.java`](../src/main/java/backend/runtime/ExecutionContext.java).

**Execution mode**: Runtime mode such as `FORWARD` or `FORWARD_BACKWARD`. Source: [`ExecutionMode.java`](../src/main/java/backend/runtime/ExecutionMode.java).

**Execution profile**: Runnable profile combining dtype, execution mode, optimizer config, runtime config, and workload metadata. Source: [`ExecutionProfile.java`](../src/main/java/config/profile/ExecutionProfile.java).

**Execution state**: Per-run runtime tensor storage and metadata state built for prepared execution. Source: [`ExecutionState.java`](../src/main/java/graph/execution/ExecutionState.java).

## F

**Forward boundary**: Compiled node id/index separating forward and backward execution steps. Source: [`CompileArtifacts.java`](../src/main/java/graph/compile/CompileArtifacts.java), [`PreparedExecutionBuilder.java`](../src/main/java/backend/prepare/PreparedExecutionBuilder.java).

**Forward output wrapper**: System `NOOP` tensor created by `Tensor.forwardOutput()` to normalize publication. Source: [`Tensor.java`](../src/main/java/tensor/Tensor.java), [`GraphCompiler.java`](../src/main/java/graph/compile/GraphCompiler.java).

**FUSE**: Optimizer stage that creates optimized regions from partitions. Source: [`RegionOptimizationRule.java`](../src/main/java/graph/optimizer/region/RegionOptimizationRule.java), [`OptimizerFactory.java`](../src/main/java/graph/optimizer/OptimizerFactory.java).

**Fused operation**: Backend-owned `Operation` descriptor for a fused expression plan. Source: [`FusedOperation.java`](../src/main/java/backend/cpu/fused/plan/FusedOperation.java).

**Fused ASM executable**: Generated bytecode implementation of a fused execution plan, loaded and cached during prepare. Source: [`AsmPreparedFusedExecutableFactory.java`](../src/main/java/backend/cpu/fused/asm/AsmPreparedFusedExecutableFactory.java), [`PreparedFusedExecutable.java`](../src/main/java/backend/cpu/fused/exec/PreparedFusedExecutable.java).

## G

**Graph execution policy**: Profile component holding optimizer/graph policy. Source: [`GraphExecutionPolicy.java`](../src/main/java/config/profile/GraphExecutionPolicy.java).

**Graph optimizer**: Ordered pipeline of optimization rules. Source: [`GraphOptimizer.java`](../src/main/java/graph/optimizer/GraphOptimizer.java), [`OptimizerFactory.java`](../src/main/java/graph/optimizer/OptimizerFactory.java).

**Gradient binding**: Mapping from semantic/source tensors to compiled gradient nodes or constant gradient templates. Source: [`CompiledGradientBinding.java`](../src/main/java/graph/CompiledGradientBinding.java), [`GradientBindingCollector.java`](../src/main/java/graph/compile/GradientBindingCollector.java).

## L

**Leaf tensor**: Tensor with no operation descriptor, usually user input or parameter data. Source: [`CompiledNode.java`](../src/main/java/graph/CompiledNode.java), [`Tensor.java`](../src/main/java/tensor/Tensor.java).

**Lowering**: Conversion from higher-level graph/region structure into backend-executable units or primitive descriptors. Source: [`LoweringPipeline.java`](../src/main/java/backend/lowering/LoweringPipeline.java), [`PreparedExecutionBuilder.java`](../src/main/java/backend/prepare/PreparedExecutionBuilder.java).

## M

**MEM**: Optimizer stage that builds `MemoryPlan` artifacts. Source: [`MemoryOptimizerRule.java`](../src/main/java/graph/optimizer/memory/MemoryOptimizerRule.java), [`MemoryPlanner.java`](../src/main/java/graph/optimizer/memory/MemoryPlanner.java).

**Memory plan**: Compile-time lifetimes, reusable intervals, slots, region-value memory bindings, and runtime binding policies. Source: [`MemoryPlan.java`](../src/main/java/graph/optimizer/memory/MemoryPlan.java), [`MemoryPlanner.java`](../src/main/java/graph/optimizer/memory/MemoryPlanner.java).

**Memory role**: Classification used by memory planning for temporaries, saved forward values, gradient targets, and related storage owners. Source: [`MemoryRole.java`](../src/main/java/graph/optimizer/memory/MemoryRole.java), [`NodeLifetime.java`](../src/main/java/graph/optimizer/memory/NodeLifetime.java).

**Metal buffer binding**: Java-side descriptor for a future native Metal shared-buffer execution path. It ties a compiled node id, dtype, shape, element count, access intent, and native buffer handle together without exposing semantic `Tensor` objects to native code. Source: [`MetalBufferBinding.java`](../src/main/java/backend/metal/buffer/MetalBufferBinding.java), [`MetalBufferHandle.java`](../src/main/java/backend/metal/buffer/MetalBufferHandle.java).

**Metal bridge execution stats**: Per-execution diagnostics from the Metal MPSGraph bridge, including input/output bytes, Java-to-native copy time, native execute time, native-to-Java copy time, CPU fallback flag, and fallback reason. Source: [`MetalMpsBridgeExecutionStats.java`](../src/main/java/backend/metal/bridge/MetalMpsBridgeExecutionStats.java), [`PreparedMetalExecutable.java`](../src/main/java/backend/metal/exec/PreparedMetalExecutable.java).

**Metal bridge execution path**: Trace label for the actual runtime path of a selected Metal executable: CPU fallback, current tensor-array copy bridge, or future explicit buffer-binding bridge. Source: [`MetalMpsBridgeExecutionPath.java`](../src/main/java/backend/metal/bridge/MetalMpsBridgeExecutionPath.java), [`PreparedExecution.java`](../src/main/java/graph/execution/PreparedExecution.java).

**Metal transfer model**: Graph-level scoring preset used by scored Metal partition planning to penalize input/output transfer bytes and credit avoided intermediate materialization. Source: [`MetalTransferModel.java`](../src/main/java/config/optimizer/MetalTransferModel.java), [`PartitionConfig.java`](../src/main/java/config/optimizer/PartitionConfig.java), [`ScoredCandidatePartitionPlanner.java`](../src/main/java/graph/optimizer/partition/ScoredCandidatePartitionPlanner.java).

## O

**Operation descriptor**: Immutable semantic descriptor implementing `Operation`. Source: [`Operation.java`](../src/main/java/operations/Operation.java), [`operations/README.md`](../src/main/java/operations/README.md).

**Optimizer state**: Mutable optimizer pipeline carrier for graph, forward output, execution metadata, partitions, optimized regions, and memory plan. Source: [`OptimizerState.java`](../src/main/java/graph/optimizer/state/OptimizerState.java).

**Optimizer stage**: Public enum stage: `AR`, `CSE`, `PART`, `FUSE`, `MEM`. Source: [`OptimizerStage.java`](../src/main/java/config/optimizer/OptimizerStage.java).

## P

**PART**: Optimizer stage that propagates backend intent and creates partition plans. Source: [`PartitionIntentRule.java`](../src/main/java/graph/optimizer/partition/PartitionIntentRule.java).

**Partition**: Backend-compatible node group discovered during partition planning. Source: [`Partition.java`](../src/main/java/graph/optimizer/partition/Partition.java), [`GreedyMaxRegionPartitionPlanner.java`](../src/main/java/graph/optimizer/partition/GreedyMaxRegionPartitionPlanner.java).

**Partition anchor**: Executable node representing a lowered partition while interior nodes are skipped as standalone execution steps. Source: [`PartitionExecutionRole.java`](../src/main/java/backend/accelerator/exec/PartitionExecutionRole.java), [`CpuNodePreparer.java`](../src/main/java/backend/cpu/prepare/CpuNodePreparer.java).

**Platform runtime profile**: Machine-oriented runtime-default profile produced by calibration and convertible to `RuntimeConfig`. Source: [`PlatformRuntimeProfile.java`](../src/main/java/config/profile/PlatformRuntimeProfile.java).

**Prepare trace**: Timing and step-count metadata captured by prepare. Source: [`PrepareTrace.java`](../src/main/java/graph/execution/trace/PrepareTrace.java), [`PreparedExecutionBuilder.java`](../src/main/java/backend/prepare/PreparedExecutionBuilder.java).

**Prepared execution**: Reusable runtime-bound artifact containing forward/backward steps and metadata. Source: [`PreparedExecution.java`](../src/main/java/graph/execution/PreparedExecution.java).

**Prepared node execution**: One executable prepared node step: compiled node plus metadata. Source: [`PreparedNodeExecution.java`](../src/main/java/graph/execution/PreparedNodeExecution.java).

## R

**Region optimization**: FUSE-stage conversion of partitions into optimized regions. Source: [`RegionOptimizationRule.java`](../src/main/java/graph/optimizer/region/RegionOptimizationRule.java), [`DefaultRegionOptimizer.java`](../src/main/java/graph/optimizer/region/DefaultRegionOptimizer.java).

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

## T

**Tensor**: Public value type and semantic graph node. Source: [`Tensor.java`](../src/main/java/tensor/Tensor.java), [`tensor/README.md`](../src/main/java/tensor/README.md).

**Tensor storage**: Dtype-specific backing storage with mutation versioning. Source: [`TensorStorage.java`](../src/main/java/tensor/TensorStorage.java).

**Tuning history**: Persisted per-candidate measurement history for autotune. Source: [`TuningHistoryEntry.java`](../src/main/java/tuning/store/TuningHistoryEntry.java), [`JsonFileTuningHistoryStore.java`](../src/main/java/tuning/store/JsonFileTuningHistoryStore.java).

## W

**Workload fingerprint**: Key representing workload and profile context for persisted tuning records. Source: [`WorkloadFingerprint.java`](../src/main/java/tuning/store/WorkloadFingerprint.java).

**Workspace**: Per-node CPU scratch state such as float continuation buffers, packed weights, or max-pool index storage. Source: [`CpuNodeWorkspace.java`](../src/main/java/backend/cpu/kernels/CpuNodeWorkspace.java), [`CpuNodePreparer.java`](../src/main/java/backend/cpu/prepare/CpuNodePreparer.java).
