package graph.execution;

import backend.ComputeEngine;
import backend.cpu.fused.plan.FusedOperation;
import backend.kernels.cpu.CpuDTypeOps;
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

    public RuntimeConfig runtimeConfig() {
        return runtimeConfig;
    }

    public boolean supportsBackward() {
        return supportsBackward;
    }

    public List<PreparedNodeExecution> forwardSteps() {
        return forwardSteps;
    }

    public List<PreparedNodeExecution> backwardSteps() {
        return backwardSteps;
    }

    public List<PreparedNodeExecution> executionSteps() {
        return executionSteps;
    }

    public PrepareTrace prepareTrace() {
        return prepareTrace;
    }

    public void execute(ExecutionMode mode) {
        executeInternal(mode, false);
    }

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
        return new RunTrace(mode, System.nanoTime() - runStart, steps == null ? List.of() : steps);
    }

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
            ComputeEngine.compute(step.compiledNode(), step.metadata(), context);
            if (captureTrace) {
                traces.add(toStepTrace(startIndex + i, step, System.nanoTime() - t0, context));
            }
        }
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

        if (metadata.acceleratorExecutable() instanceof backend.metal.exec.PreparedMetalExecutable metal) {
            attrs.put("metalBridgeAvailable", metal.bridge().isAvailable());
            attrs.put("metalBridgeContextAvailable", metal.bridgeContext().available());
            attrs.put("metalBridgeExecutableAvailable", metal.bridgeExecutable().available());
            attrs.put("metalBridgeCacheHit", metal.bridgeExecutable().cacheHit());
            attrs.put("metalSubgraphNodeCount", metal.plan().nodeIds().size());
            attrs.put("metalSubgraphOps", metal.plan().subgraph().ops().stream().map(op -> op.opType().name()).toList());
            attrs.put("metalEstimatedWork", metal.plan().estimatedWork());
        }

        return new StepExecutionMetadata("node", attrs, compute, layout, dispatch, reduction, matMul, conv, fusedMeta);
    }

    private void syncRootData(ExecutionMode mode, ExecutionState executionState) {
        Tensor publishTarget = resolveSemanticPublishTarget(rootTensor);
        Integer publishNodeId = nodeIdForSemanticTensor(publishTarget);
        if (publishNodeId != null) {
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

        int actualRootNodeId = resolveForwardRuntimeRootNodeId();
        Tensor actualRoot = executionState.runtimeTensorForNodeId(actualRootNodeId);
        if (mode == ExecutionMode.FORWARD_BACKWARD || actualRoot != rootTensor) {
            rootTensor.copyDataFrom(actualRoot);
        }
        repairSemanticAliasChain(rootTensor);
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
