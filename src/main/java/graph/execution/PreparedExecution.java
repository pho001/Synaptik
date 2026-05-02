package graph.execution;

import backend.ComputeBackend;
import backend.ComputeEngine;
import backend.accelerator.lowering.GpuCompoundPatternType;
import backend.cpu.fused.plan.FusedOperation;
import backend.cpu.kernels.CpuDTypeOps;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.CompiledGradientBinding;
import graph.execution.trace.ComputeTraceMetadata;
import graph.execution.trace.DispatchTraceMetadata;
import graph.execution.trace.ExecutionStepTrace;
import graph.execution.trace.FusedTraceMetadata;
import graph.execution.trace.LayoutTraceMetadata;
import graph.execution.trace.MatMulTraceMetadata;
import graph.execution.trace.ConvTraceMetadata;
import graph.execution.trace.PrepareTrace;
import graph.execution.trace.ReductionTraceMetadata;
import graph.execution.trace.RunTrace;
import graph.execution.trace.StepExecutionMetadata;
import graph.optimizer.memory.MemoryPlan;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Runtime plan produced from {@link graph.compile.CompileArtifacts}.
 *
 * <p>Preparation resolves compile-time nodes to concrete execution operations, CPU kernels, backend metadata, memory
 * binding policy, and ordered forward/backward step lists for one {@link RuntimeConfig}. The prepared object is an
 * immutable description of how to run; each execution creates a fresh {@link ExecutionState}.
 *
 * <p>Running a prepared execution has side effects on the graph's tensors: output storage is synchronized back to the
 * source root tensor, backward mode seeds the root gradient, and compiled gradient bindings publish computed gradients.
 * Concurrent calls against shared source tensors or shared backend workspaces are not supported.
 */
public final class PreparedExecution {
    private final RuntimeConfig runtimeConfig;
    private final boolean supportsBackward;
    private final List<PreparedNodeExecution> executionSteps;
    private final List<PreparedNodeExecution> forwardSteps;
    private final List<PreparedNodeExecution> backwardSteps;
    private final List<CompiledNode> allNodes;
    private final Map<Tensor, CompiledGradientBinding> compiledGradients;
    private final Tensor rootTensor;
    private final CompiledNode forwardOutputNode;
    private final CompiledGradientBinding forwardSeedGradient;
    private final MemoryPlan memoryPlan;
    private final PrepareTrace prepareTrace;
    private final Map<Integer, CompiledNodeExecutionMetadata> metadataIndex;

    /**
     * Creates a prepared execution from already lowered step metadata.
     *
     * @param runtimeConfig runtime configuration used to select kernels and execution metadata
     * @param supportsBackward whether backward steps are available
     * @param executionSteps full ordered step list used by forward-backward execution
     * @param forwardSteps forward-only step list
     * @param backwardSteps backward-only step list
     * @param allNodes all compiled nodes in graph order
     * @param compiledGradients gradient publication bindings
     * @param rootTensor source root tensor to synchronize after execution
     * @param forwardOutputNode compiled node that holds the forward result
     * @param forwardSeedGradient binding used to seed backward execution
     * @param memoryPlan memory reuse and region binding plan, possibly {@code null}
     * @param prepareTrace preparation diagnostics and timing metadata
     */
    public PreparedExecution(
            RuntimeConfig runtimeConfig,
            boolean supportsBackward,
            List<PreparedNodeExecution> executionSteps,
            List<PreparedNodeExecution> forwardSteps,
            List<PreparedNodeExecution> backwardSteps,
            List<CompiledNode> allNodes,
            Map<Tensor, CompiledGradientBinding> compiledGradients,
            Tensor rootTensor,
            CompiledNode forwardOutputNode,
            CompiledGradientBinding forwardSeedGradient,
            MemoryPlan memoryPlan,
            PrepareTrace prepareTrace
    ) {
        this.runtimeConfig = Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        this.supportsBackward = supportsBackward;
        this.executionSteps = List.copyOf(executionSteps == null ? List.of() : executionSteps);
        this.forwardSteps = List.copyOf(forwardSteps == null ? List.of() : forwardSteps);
        this.backwardSteps = List.copyOf(backwardSteps == null ? List.of() : backwardSteps);
        this.allNodes = List.copyOf(allNodes == null ? List.of() : allNodes);
        this.compiledGradients = Map.copyOf(compiledGradients == null ? Map.of() : compiledGradients);
        this.rootTensor = Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        this.forwardOutputNode = Objects.requireNonNull(forwardOutputNode, "forwardOutputNode cannot be null");
        this.forwardSeedGradient = forwardSeedGradient;
        this.memoryPlan = memoryPlan;
        this.prepareTrace = prepareTrace == null ? PrepareTrace.skipped() : prepareTrace;
        this.metadataIndex = buildMetadataIndex(this.executionSteps);
    }

    /**
     * Returns the runtime configuration used to prepare this plan.
     *
     * @return runtime configuration
     */
    public RuntimeConfig runtimeConfig() {
        return runtimeConfig;
    }

    /**
     * Returns whether this plan contains backward work.
     *
     * @return {@code true} when {@link ExecutionMode#FORWARD_BACKWARD} is valid
     */
    public boolean supportsBackward() {
        return supportsBackward;
    }

    /**
     * Returns immutable forward execution steps.
     *
     * @return forward step list
     */
    public List<PreparedNodeExecution> forwardSteps() {
        return forwardSteps;
    }

    /**
     * Returns immutable backward execution steps.
     *
     * @return backward step list, empty for inference-only plans
     */
    public List<PreparedNodeExecution> backwardSteps() {
        return backwardSteps;
    }

    /**
     * Returns the full execution sequence used for forward-backward mode.
     *
     * @return immutable full step list
     */
    public List<PreparedNodeExecution> executionSteps() {
        return executionSteps;
    }

    /**
     * Returns preparation timing and backend-selection metadata.
     *
     * @return prepare trace
     */
    public PrepareTrace prepareTrace() {
        return prepareTrace;
    }

    /**
     * Executes this prepared plan without collecting per-step trace metadata.
     *
     * @param mode execution mode to run
     * @throws NullPointerException if {@code mode} is {@code null}
     * @throws IllegalStateException if backward execution is requested but this plan has no backward steps
     */
    public void execute(ExecutionMode mode) {
        executeInternal(mode, false);
    }

    /**
     * Executes this prepared plan and returns run-level diagnostics.
     *
     * @param mode execution mode to run
     * @return run trace containing duration and per-step metadata
     * @throws NullPointerException if {@code mode} is {@code null}
     * @throws IllegalStateException if backward execution is requested but this plan has no backward steps
     */
    public RunTrace executeTraced(ExecutionMode mode) {
        return executeInternal(mode, true);
    }

    private RunTrace executeInternal(ExecutionMode mode, boolean captureTrace) {
        Objects.requireNonNull(mode, "mode cannot be null");
        if (mode == ExecutionMode.FORWARD_BACKWARD && !supportsBackward) {
            throw new IllegalStateException("Prepared execution does not support backward execution.");
        }

        long runStart = System.nanoTime();
        java.util.ArrayList<ExecutionStepTrace> steps = captureTrace ? new java.util.ArrayList<>() : null;
        ExecutionState executionState = ExecutionState.create(allNodes, metadataIndex, forwardOutputNode.id());
        RuntimeException executionFailure = null;
        Error executionError = null;
        try {
            RuntimeMemoryBinder.bind(memoryPlan, allNodes, executionState);
            ExecutionContext context = ExecutionContext.fromRuntimeConfig(runtimeConfig, mode, metadataIndex, executionState);

            if (mode == ExecutionMode.FORWARD_BACKWARD) {
                seedRootGradient(executionState);
                executeSteps(executionSteps, context, captureTrace, steps, 0);
                syncRootData(mode, executionState);
                publishCompiledGradients(executionState);
            } else {
                executeSteps(forwardSteps, context, captureTrace, steps, 0);
                syncRootData(mode, executionState);
            }
            return new RunTrace(
                    mode,
                    System.nanoTime() - runStart,
                    steps == null ? List.of() : steps,
                    executionState.cpuMaterializationTraces()
            );
        } catch (RuntimeException ex) {
            executionFailure = ex;
            throw ex;
        } catch (Error err) {
            executionError = err;
            throw err;
        } finally {
            try {
                executionState.closeResources();
            } catch (RuntimeException closeFailure) {
                if (executionFailure != null) {
                    executionFailure.addSuppressed(closeFailure);
                } else if (executionError != null) {
                    executionError.addSuppressed(closeFailure);
                } else {
                    throw closeFailure;
                }
            }
        }
    }

    /**
     * Convenience wrapper for {@link #execute(ExecutionMode)} in forward-backward mode.
     *
     * <p>If the plan has no backward steps, this method prints an informational message and returns without mutation.
     */
    public void backward() {
        if (!supportsBackward) {
            System.out.println("Info: No gradients to compute.");
            return;
        }
        execute(ExecutionMode.FORWARD_BACKWARD);
    }

    private static Map<Integer, CompiledNodeExecutionMetadata> buildMetadataIndex(List<PreparedNodeExecution> executionSteps) {
        Map<Integer, CompiledNodeExecutionMetadata> out = new HashMap<>();
        for (PreparedNodeExecution step : executionSteps) {
            out.put(step.compiledNode().id(), step.metadata());
        }
        return Map.copyOf(out);
    }

    private static void executeSteps(
            List<PreparedNodeExecution> steps,
            ExecutionContext context,
            boolean captureTrace,
            List<ExecutionStepTrace> traces,
            int startIndex
    ) {
        for (int i = 0; i < steps.size(); i++) {
            PreparedNodeExecution step = steps.get(i);
            long t0 = captureTrace ? System.nanoTime() : 0L;
            if (DeviceLayoutViewPropagator.tryPropagate(step, context)) {
                if (captureTrace) {
                    traces.add(toStepTrace(startIndex + i, step, System.nanoTime() - t0, context));
                }
                continue;
            }
            requireCpuReadableInputs(step, context);
            ComputeEngine.compute(step.compiledNode(), step.metadata(), context);
            markResidencyAfterStep(step, context);
            if (captureTrace) {
                traces.add(toStepTrace(startIndex + i, step, System.nanoTime() - t0, context));
            }
        }
    }

    private static void markResidencyAfterStep(PreparedNodeExecution step, ExecutionContext context) {
        int nodeId = step.compiledNode().id();
        if (step.metadata().backend() == ComputeBackend.CPU) {
            context.markCpuCurrent(nodeId, residencyReason(step));
            return;
        }
        var residency = context.residencyForNodeId(nodeId);
        if (residency == null || (!residency.cpuCurrent() && !residency.deviceCurrent())) {
            context.markCpuCurrent(nodeId, residencyReason(step));
        }
    }

    private static void requireCpuReadableInputs(PreparedNodeExecution step, ExecutionContext context) {
        if (step.metadata().backend() != ComputeBackend.CPU) {
            return;
        }
        List<Integer> inputIds = step.metadata().executionInputNodeIds().isEmpty()
                ? step.compiledNode().inputIds()
                : step.metadata().executionInputNodeIds();
        for (int inputId : inputIds) {
            context.requireCpuReadable(inputId, CpuMaterializationReason.CPU_CONSUMER);
        }
    }

    private static String residencyReason(PreparedNodeExecution step) {
        if (step != null && step.metadata().acceleratorExecutable() instanceof backend.metal.exec.PreparedMetalExecutable metal) {
            if (metal.lastExecutionStats().usedCpuFallback()) {
                return "metal cpu fallback wrote CPU array";
            }
            return metal.lastExecutionStats().executionPath() == backend.metal.bridge.MetalMpsBridgeExecutionPath.BUFFER_BINDING
                    ? "metal buffer binding execution wrote device buffer"
                    : "metal bridge copied output to CPU array";
        }
        return "backend wrote CPU array";
    }

    private static ExecutionStepTrace toStepTrace(int index, PreparedNodeExecution step, long durationNs, ExecutionContext context) {
        CompiledNode node = step.compiledNode();
        Tensor semanticNode = step.node();
        var metadata = step.metadata();
        operations.Operation executionOperation = metadata.executionOperation() == null
                ? node.operation()
                : metadata.executionOperation();
        String opType = executionOperation == null ? "LEAF" : executionOperation.opType().name();
        String kernel = metadata.cpuKernel() == null ? "" : metadata.cpuKernel().getClass().getSimpleName();
        return new ExecutionStepTrace(
                index,
                node.label(),
                opType,
                java.util.Arrays.stream(node.shape()).boxed().toList(),
                node.dataType(),
                metadata.backend().name(),
                kernel,
                durationNs,
                buildStepMetadata(node, semanticNode, step, context)
        );
    }

    private static StepExecutionMetadata buildStepMetadata(CompiledNode node, Tensor semanticNode, PreparedNodeExecution step, ExecutionContext context) {
        var metadata = step.metadata();
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        ComputeTraceMetadata compute = null;
        LayoutTraceMetadata layout = new LayoutTraceMetadata(
                node.storageOffset(),
                node.contiguous(),
                metadata.cpuPlan() != null && metadata.cpuPlan().stridedPath(),
                metadata.cpuPlan() == null ? "" : metadata.cpuPlan().targetType().name()
        );
        DispatchTraceMetadata dispatch = null;
        ReductionTraceMetadata reduction = null;
        MatMulTraceMetadata matMul = null;
        ConvTraceMetadata conv = null;
        FusedTraceMetadata fusedMeta = null;

        if (metadata.cpuPlan() != null) {
            var plan = metadata.cpuPlan();
            compute = new ComputeTraceMetadata(
                    plan.computeContract().computeType().name(),
                    plan.computeContract().storageType().name(),
                    plan.computeContract().computeType().name(),
                    plan.computeContract().backend().name(),
                    plan.computeContract().accumulateType().name()
            );
            if (plan.dispatchHints() != null) {
                dispatch = new DispatchTraceMetadata(
                        plan.dispatchHints().mode().name(),
                        plan.dispatchHints().vectorWidth(),
                        plan.dispatchHints().plannedWorkers(),
                        plan.dispatchHints().scalarChunkSize(),
                        plan.dispatchHints().vectorChunkSize()
                );
            }
            if (plan.reductionHints() != null) {
                reduction = new ReductionTraceMetadata(
                        plan.reductionHints().mode().name(),
                        plan.reductionHints().plannedWorkers(),
                        plan.reductionHints().chunkSize(),
                        plan.reductionHints().vectorWidth(),
                        plan.reductionHints().accuracyMode().name()
                );
            }
            if (plan.matMulHints() != null) {
                matMul = new MatMulTraceMetadata(
                        plan.matMulHints().useBlas(),
                        plan.matMulHints().useBatchedBlas(),
                        plan.matMulHints().parallel(),
                        plan.matMulHints().tileM(),
                        plan.matMulHints().tileN(),
                        plan.matMulHints().tileK(),
                        plan.matMulHints().plannedWorkers(),
                        plan.matMulHints().work(),
                        plan.matMulHints().microKernel().name()
                );
            }
        }

        ConvTraceMetadata trace = context.convTraceForNodeId(node.id());
        if (trace != null) {
            conv = trace;
        }

        operations.Operation executionOperation = metadata.executionOperation() == null
                ? node.operation()
                : metadata.executionOperation();
        if (executionOperation instanceof FusedOperation fused) {
            String executionBackend = step.metadata().fusedExecutable() == null
                    ? ""
                    : step.metadata().fusedExecutable().getClass().getSimpleName();
            fusedMeta = new FusedTraceMetadata(
                    fused.getPrecisionMode(),
                    fused.isLowCostHint(),
                    fused.getDispatchFamily().id(),
                    fused.getSchedulerSignature(),
                    executionBackend,
                    fused.getDispatchScale(),
                    fused.getPlan().nodeCount(),
                    fused.getPlan().inputCount()
            );
        }

        if (metadata.acceleratorExecutable() != null) {
            var decision = metadata.acceleratorExecutable().lastAcceleratorBufferDecision();
            attrs.put("acceleratorBufferMode", decision.mode().name());
            attrs.put("acceleratorBufferBackend", decision.backend().name());
            attrs.put("acceleratorBufferDecision", decision.path().name());
            attrs.put("acceleratorBufferExecutionPath", decision.path().name());
            attrs.put("acceleratorBufferReasonCode", decision.reasonCode().name());
            attrs.put("acceleratorBufferReason", decision.reason());
            attrs.put("acceleratorBufferPreparedInputUsed", decision.preparedInputUsed());
            attrs.put("acceleratorBufferInputCount", decision.inputs().size());
            attrs.put("acceleratorBufferOutputCount", decision.outputs().size());
            attrs.put("cpuMaterializationCount", context.cpuMaterializationTraceCount());
            attrs.put("deviceHandoffCount", deviceHandoffCount(decision));
            var manifest = metadata.acceleratorExecutable().gpuLoweredRegionManifest();
            if (manifest != null && !manifest.regionId().isBlank()) {
                attrs.put("gpuRegionId", manifest.regionId());
                attrs.put("gpuLoweredRegionId", manifest.regionId());
            }
            if (manifest != null) {
                attrs.put("selectedRegionLength", manifest.selectedRegionLength());
                attrs.put("loweredPrimitiveCount", manifest.loweredPrimitives().size());
                attrs.put("gpuFusedSubpatternCount", manifest.fusedSubpatterns().size());
                attrs.put("gpuFusedSubpatternTypes", manifest.fusedSubpatterns().stream()
                        .map(subpattern -> subpattern.patternType().name())
                        .toList());
            }
            if (manifest != null && !manifest.fusedSubpatterns().isEmpty()) {
                attrs.put("gpuFusedSubpatternOriginalNodeIds", manifest.fusedSubpatterns().stream()
                        .map(subpattern -> subpattern.originalOperationNodeIds())
                        .toList());
                attrs.put("gpuFusedSubpatternLoweredPrimitiveCount", manifest.fusedSubpatterns().stream()
                        .map(subpattern -> subpattern.loweredPrimitiveCount())
                        .toList());
                attrs.put("gpuFusedSubpatternReasons", manifest.fusedSubpatterns().stream()
                        .map(subpattern -> subpattern.reason().name())
                        .toList());
            }
            var compoundSummary = metadata.acceleratorExecutable().compoundSummary();
            if (compoundSummary != null && compoundSummary.patternType() != GpuCompoundPatternType.NONE) {
                attrs.put("gpuCompoundPattern", compoundSummary.patternType().name());
                attrs.put("gpuCompoundSupported", compoundSummary.supported());
                attrs.put("gpuCompoundReason", compoundSummary.reason().name());
                attrs.put("gpuCompoundNodeCount", compoundSummary.orderedNodeIds().size());
                attrs.put("gpuCompoundOrderedNodeIds", compoundSummary.orderedNodeIds());
                attrs.put("gpuCompoundDagNodeTypes", compoundSummary.dagNodeTypes());
                attrs.put("gpuCompoundPostOps", compoundSummary.postOps());
            }
        }
        var layoutTransformDecision = context.layoutTransformDecisionForNodeId(node.id());
        if (layoutTransformDecision != null) {
            attrs.put("gpuLayoutTransformKind", layoutTransformDecision.kind().name());
            attrs.put("gpuLayoutTransformOp", layoutTransformDecision.opType().name());
            attrs.put("gpuLayoutTransformSourceNodeId", layoutTransformDecision.sourceNodeId());
            attrs.put("gpuLayoutTransformTargetNodeId", layoutTransformDecision.targetNodeId());
            attrs.put("gpuLayoutTransformAccepted", layoutTransformDecision.accepted());
            attrs.put("gpuLayoutTransformSourceLayoutClass", layoutTransformDecision.sourceLayout().layoutClass().name());
            attrs.put("gpuLayoutTransformTargetLayoutClass", layoutTransformDecision.targetLayout().layoutClass().name());
            attrs.put("gpuLayoutTransformBytes", layoutTransformDecision.targetLayout().logicalByteLength());
            attrs.put("gpuLayoutMaterializationCount",
                    layoutTransformDecision.kind() == backend.accelerator.buffer.AcceleratorLayoutTransformKind.DENSE_GPU_MATERIALIZATION
                            && layoutTransformDecision.accepted()
                            ? 1
                            : 0);
            attrs.put("gpuLayoutMaterializationBytes",
                    layoutTransformDecision.kind() == backend.accelerator.buffer.AcceleratorLayoutTransformKind.DENSE_GPU_MATERIALIZATION
                            && layoutTransformDecision.accepted()
                            ? layoutTransformDecision.targetLayout().logicalByteLength()
                            : 0L);
            attrs.putIfAbsent("acceleratorBufferBackend", layoutTransformDecision.backendId());
            attrs.putIfAbsent("acceleratorBufferDecision", layoutTransformDecision.kind().name());
            attrs.putIfAbsent("acceleratorBufferExecutionPath", layoutTransformDecision.accepted()
                    ? layoutTransformDecision.kind().name()
                    : "UNAVAILABLE");
            attrs.putIfAbsent("acceleratorBufferReasonCode", layoutTransformDecision.reasonCode().name());
            attrs.putIfAbsent("acceleratorBufferReason", layoutTransformDecision.reason());
        }

        if (metadata.acceleratorExecutable() instanceof backend.metal.exec.PreparedMetalExecutable metal) {
            var metalStats = metal.lastExecutionStats();
            attrs.put("metalBridgeAvailable", metal.bridge().isAvailable());
            attrs.put("metalBridgeContextAvailable", metal.bridgeContext().available());
            attrs.put("metalBridgeExecutableAvailable", metal.bridgeExecutable().available());
            attrs.put("metalBridgeCacheHit", metal.bridgeExecutable().cacheHit());
            attrs.put("metalSupportsBufferBindings", metal.bridge().supportsBufferBindings());
            attrs.put("metalBufferBindingDecision", metal.lastBufferBindingDecision());
            attrs.put("metalSubgraphNodeCount", metal.plan().nodeIds().size());
            attrs.put("metalSubgraphOps", metal.plan().subgraph().ops().stream().map(op -> op.opType().name()).toList());
            attrs.put("metalEstimatedWork", metal.plan().estimatedWork());
            attrs.put("metalUsedCpuFallback", metalStats.usedCpuFallback());
            attrs.put("metalFallbackReason", metalStats.fallbackReason());
            attrs.put("metalExecutionPath", metalStats.executionPath().name());
            attrs.put("metalExternalInputCount", metalStats.externalInputCount());
            attrs.put("metalOutputCount", metalStats.outputCount());
            attrs.put("metalInputBytes", metalStats.inputBytes());
            attrs.put("metalOutputBytes", metalStats.outputBytes());
            attrs.put("metalJavaToNativeCopyNs", metalStats.javaToNativeCopyNs());
            attrs.put("metalOutputAllocationNs", metalStats.outputAllocationNs());
            attrs.put("metalNativeExecuteNs", metalStats.nativeExecuteNs());
            attrs.put("metalNativeDeviceCopyNs", metalStats.nativeDeviceCopyNs());
            attrs.put("metalNativeToJavaCopyNs", metalStats.nativeToJavaCopyNs());
            attrs.put("metalBridgeTotalNs", metalStats.totalNs());
        }
        if (metadata.acceleratorExecutable() instanceof backend.cuda.exec.PreparedCudaExecutable cuda) {
            var cudaStats = cuda.lastExecutionStats();
            attrs.put("cudaBridgeAvailable", cuda.bridge().isAvailable());
            attrs.put("cudaBridgeContextAvailable", cuda.bridgeContext().available());
            attrs.put("cudaBridgeExecutableAvailable", cuda.bridgeExecutable().available());
            attrs.put("cudaSupportsBufferBindings", cuda.bridge().supportsBufferBindings());
            attrs.put("cudaUsedCpuFallback", cudaStats.usedCpuFallback());
            attrs.put("cudaFallbackReason", cudaStats.fallbackReason());
            attrs.put("cudaExecutionPath", cudaStats.executionPath().name());
            attrs.put("cudaExternalInputCount", cudaStats.externalInputCount());
            attrs.put("cudaOutputCount", cudaStats.outputCount());
            attrs.put("cudaInputBytes", cudaStats.inputBytes());
            attrs.put("cudaOutputBytes", cudaStats.outputBytes());
            attrs.put("cudaJavaToNativeCopyNs", cudaStats.javaToNativeCopyNs());
            attrs.put("cudaNativeExecuteNs", cudaStats.nativeExecuteNs());
            attrs.put("cudaNativeDeviceCopyNs", cudaStats.nativeDeviceCopyNs());
            attrs.put("cudaNativeToJavaCopyNs", cudaStats.nativeToJavaCopyNs());
            attrs.put("cudaBridgeTotalNs", cudaStats.totalNs());
            attrs.put("acceleratorInputBytes", cudaStats.inputBytes());
            attrs.put("acceleratorOutputBytes", cudaStats.outputBytes());
            attrs.put("acceleratorJavaToNativeCopyNs", cudaStats.javaToNativeCopyNs());
            attrs.put("acceleratorNativeToJavaCopyNs", cudaStats.nativeToJavaCopyNs());
            attrs.put("acceleratorNativeDeviceCopyNs", cudaStats.nativeDeviceCopyNs());
        }
        var residency = context.residencyForNodeId(node.id());
        if (residency != null) {
            attrs.put("storageResidency", residency.residency().name());
            attrs.put("storageCpuCurrent", residency.cpuCurrent());
            attrs.put("storageDeviceCurrent", residency.deviceCurrent());
            attrs.put("storageDeviceBackend", residency.deviceBackend());
            attrs.put("storageTransitionReason", residency.lastTransitionReason());
        }
        var deviceBinding = context.deviceBufferBindingForNodeId(node.id());
        if (deviceBinding != null) {
            attrs.put("deviceBufferBackend", deviceBinding.backendId());
            attrs.put("deviceBufferBytes", deviceBinding.logicalByteLength());
            attrs.put("deviceBufferAvailable", deviceBinding.available());
            attrs.put("deviceBuffer", deviceBinding.describe());
        }

        return new StepExecutionMetadata("node", attrs, compute, layout, dispatch, reduction, matMul, conv, fusedMeta);
    }

    private static int deviceHandoffCount(backend.accelerator.buffer.AcceleratorBufferDecision decision) {
        if (decision == null
                || decision.path() != backend.accelerator.buffer.AcceleratorBufferExecutionPath.BUFFER_BINDING) {
            return 0;
        }
        return decision.inputs().size() + decision.outputs().size();
    }

    private void syncRootData(ExecutionMode mode, ExecutionState executionState) {
        Integer semanticRootNodeId = nodeIdForSemanticTensor(rootTensor);
        int actualRootNodeId = semanticRootNodeId == null ? resolveForwardRuntimeRootNodeId() : semanticRootNodeId;
        Tensor publishTarget = resolveSemanticPublishTarget(rootTensor);
        Integer publishNodeId = nodeIdForSemanticTensor(publishTarget);
        if (publishNodeId != null) {
            if (publishNodeId != actualRootNodeId
                    && shouldPublishActualRootForAlias(executionState, publishNodeId, actualRootNodeId)) {
                publishRuntimeTensor(mode, executionState, rootTensor, actualRootNodeId);
                repairSemanticAliasChain(rootTensor);
                return;
            }
            executionState.requireCpuReadable(publishNodeId, CpuMaterializationReason.GRAPH_OUTPUT);
            Tensor runtimePublished = executionState.runtimeTensorForNodeId(publishNodeId);
            if (publishTarget.getStorage() == runtimePublished.getStorage()) {
                repairSemanticAliasChain(rootTensor);
                return;
            }
            if (mode == ExecutionMode.FORWARD_BACKWARD || runtimePublished != publishTarget) {
                publishTarget.copyDataFrom(runtimePublished);
            }
            repairSemanticAliasChain(rootTensor);
            return;
        }

        publishRuntimeTensor(mode, executionState, rootTensor, actualRootNodeId);
        repairSemanticAliasChain(rootTensor);
    }

    private static boolean shouldPublishActualRootForAlias(
            ExecutionState executionState,
            int publishNodeId,
            int actualRootNodeId
    ) {
        var publishState = executionState.residencyForNodeId(publishNodeId);
        var actualRootState = executionState.residencyForNodeId(actualRootNodeId);
        return !publishState.cpuCurrent()
                && (actualRootState.cpuCurrent() || actualRootState.requiresCpuMaterialization());
    }

    private static void publishRuntimeTensor(
            ExecutionMode mode,
            ExecutionState executionState,
            Tensor publishTarget,
            int nodeId
    ) {
        executionState.requireCpuReadable(nodeId, CpuMaterializationReason.GRAPH_OUTPUT);
        Tensor runtimeTensor = executionState.runtimeTensorForNodeId(nodeId);
        if (mode == ExecutionMode.FORWARD_BACKWARD || runtimeTensor != publishTarget) {
            publishTarget.copyDataFrom(runtimeTensor);
        }
    }

    private Tensor resolveSemanticPublishTarget(Tensor tensor) {
        Tensor current = tensor;
        while (isAliasViewOp(current) && current.getPrevTensors() != null && !current.getPrevTensors().isEmpty()) {
            current = current.getPrevTensors().getFirst();
        }
        return current;
    }

    private Integer nodeIdForSemanticTensor(Tensor tensor) {
        if (tensor == null) {
            return null;
        }
        for (CompiledNode node : allNodes) {
            if (node.semanticTensor() == tensor || node.sourceTensor() == tensor) {
                return node.id();
            }
        }
        return null;
    }

    private int resolveForwardRuntimeRootNodeId() {
        if (forwardOutputNode.operation() != null
                && forwardOutputNode.operation().opType() == operations.Operation.OpType.NOOP
                && Tensor.SYSTEM_FORWARD_OUTPUT_LABEL.equals(forwardOutputNode.label())
                && !forwardOutputNode.inputIds().isEmpty()) {
            return forwardOutputNode.inputIds().getFirst();
        }
        return forwardOutputNode.id();
    }

    private boolean isAliasViewOp(Tensor tensor) {
        if (tensor == null || tensor.getOperation() == null) {
            return false;
        }
        if (tensor.getPrevTensors() == null || tensor.getPrevTensors().isEmpty()) {
            return false;
        }
        return switch (tensor.getOperation().opType()) {
            case NOOP, EXPAND, SELECT, PERMUTE, EXPAND_DIMS, SQUEEZE -> true;
            case RESHAPE -> tensor.getPrevTensors().getFirst().isContiguous();
            default -> false;
        };
    }

    private void repairSemanticAliasChain(Tensor tensor) {
        if (!isAliasViewOp(tensor) || tensor.getPrevTensors() == null || tensor.getPrevTensors().isEmpty()) {
            return;
        }
        Tensor source = tensor.getPrevTensors().getFirst();
        repairSemanticAliasChain(source);
        TensorInternalAccess.aliasRuntimeFrom(tensor, source);
    }

    private void seedRootGradient(ExecutionState executionState) {
        if (!(forwardSeedGradient instanceof CompiledGradientBinding.NodeBinding nodeBinding)) {
            return;
        }
        fillGradientOnes(executionState.runtimeTensorForNodeId(nodeBinding.nodeId()));
    }

    private void publishCompiledGradients(ExecutionState executionState) {
        for (CompiledNode node : allNodes) {
            if (node.backwardNode()) {
                continue;
            }
            Tensor tensor = node.sourceTensor();
            CompiledGradientBinding binding = compiledGradients.get(tensor);
            if (binding == null) {
                TensorInternalAccess.setGradient(tensor, null);
                continue;
            }
            Tensor published;
            if (binding instanceof CompiledGradientBinding.NodeBinding nodeBinding) {
                executionState.requireCpuReadable(nodeBinding.nodeId(), CpuMaterializationReason.GRADIENT_PUBLICATION);
                published = detachedCopy(executionState.runtimeTensorForNodeId(nodeBinding.nodeId()));
            } else if (binding instanceof CompiledGradientBinding.ConstantBinding constantBinding) {
                published = detachedCopy(constantBinding.template());
            } else {
                throw new IllegalStateException("Unsupported gradient binding type: " + binding.getClass().getName());
            }
            TensorInternalAccess.setGradient(tensor, published);
        }
    }

    private static Tensor detachedCopy(Tensor source) {
        Tensor copy = new Tensor(source.getShape(), null, source.getLabel(), source.getDataType());
        copy.copyDataFrom(source);
        return copy;
    }

    private static void fillGradientOnes(Tensor gradient) {
        switch (gradient.getDataType()) {
            case FLOAT64 -> Arrays.fill(gradient.getFloat64Data(), 1.0);
            case FLOAT32 -> Arrays.fill(gradient.getFloat32Data(), 1.0f);
            case BFLOAT16 -> Arrays.fill(gradient.getBFloat16Data(), CpuDTypeOps.toBFloat16Bits(1.0f));
            case INT32, BOOL -> throw new UnsupportedOperationException("INT32/BOOL tensors do not support gradient seeding.");
        }
    }

}
