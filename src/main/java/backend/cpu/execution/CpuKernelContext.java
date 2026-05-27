package backend.cpu.execution;

import backend.cpu.plan.CpuLayoutPlan;
import backend.cpu.CpuFusedExecutionArtifact;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.plan.ResolvedCpuComputeContract;
import backend.cpu.plan.elementwise.ResolvedDispatchHints;
import backend.cpu.plan.layout.ResolvedBroadcastPlan;
import backend.cpu.plan.layout.ResolvedWhereBroadcastPlan;
import backend.cpu.plan.linalg.attention.ResolvedScaledDotProductAttentionPlan;
import backend.cpu.provider.linalg.matmul.PreparedMatMulExecutable;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;
import backend.cpu.plan.reduction.ResolvedReductionHints;
import backend.runtime.ExecutionContext;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.trace.ConvTraceMetadata;
import backend.cpu.fused.exec.PreparedFusedExecutable;
import operations.Operation;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

public final class CpuKernelContext {
    private final CpuNodeExecutionPlan nodePlan;
    private final int nodeId;
    private final List<Integer> inputNodeIds;
    private final ExecutionContext executionContext;
    private final CompiledNodeExecutionMetadata executionMetadata;
    private final List<CompiledNodeExecutionMetadata> inputMetadatas;
    private final Operation executionOperation;

    public CpuKernelContext(
            int nodeId,
            List<Integer> inputNodeIds,
            CpuNodeExecutionPlan nodePlan,
            ExecutionContext executionContext,
            CompiledNodeExecutionMetadata executionMetadata,
            List<CompiledNodeExecutionMetadata> inputMetadatas
    ) {
        this(nodeId, inputNodeIds, nodePlan, executionContext, executionMetadata, inputMetadatas, null);
    }

    public CpuKernelContext(
            int nodeId,
            List<Integer> inputNodeIds,
            CpuNodeExecutionPlan nodePlan,
            ExecutionContext executionContext,
            CompiledNodeExecutionMetadata executionMetadata,
            List<CompiledNodeExecutionMetadata> inputMetadatas,
            Operation executionOperation
    ) {
        this.nodeId = nodeId;
        this.inputNodeIds = Collections.unmodifiableList(new ArrayList<>(inputNodeIds == null ? List.of() : inputNodeIds));
        this.nodePlan = Objects.requireNonNull(nodePlan, "nodePlan cannot be null");
        this.executionContext = Objects.requireNonNull(executionContext, "executionContext cannot be null");
        this.executionMetadata = Objects.requireNonNull(executionMetadata, "executionMetadata cannot be null");
        this.inputMetadatas = Collections.unmodifiableList(new ArrayList<>(inputMetadatas == null ? List.of() : inputMetadatas));
        this.executionOperation = executionOperation;
    }

    public CpuNodeExecutionPlan nodePlan() {
        return nodePlan;
    }

    public int nodeId() {
        return nodeId;
    }

    public List<Integer> inputNodeIds() {
        return inputNodeIds;
    }

    public CpuLayoutPlan layoutPlan() {
        return nodePlan.layoutPlan();
    }

    public ResolvedCpuComputeContract computeContract() {
        return nodePlan.computeContract();
    }

    public int fusedAsmVectorWidth() {
        ResolvedDispatchHints hints = nodePlan.dispatchHints();
        return hints == null ? 1 : hints.vectorWidth();
    }

    public ResolvedDispatchHints dispatchHints() {
        return nodePlan.dispatchHints();
    }

    public ResolvedReductionHints reductionHints() {
        return nodePlan.reductionHints();
    }

    public ResolvedMatMulHints matMulHints() {
        return nodePlan.matMulHints();
    }

    public PreparedMatMulExecutable matMulExecutable() {
        return nodePlan.matMulExecutable();
    }

    public ResolvedScaledDotProductAttentionPlan attentionPlan() {
        return nodePlan.attentionPlan();
    }

    public ResolvedBroadcastPlan broadcastPlan() {
        return nodePlan.broadcastPlan();
    }

    public ResolvedWhereBroadcastPlan whereBroadcastPlan() {
        return nodePlan.whereBroadcastPlan();
    }

    public boolean useFastExpApprox() {
        return executionContext.useFastExpApprox();
    }

    public boolean useFastTanhApprox() {
        return executionContext.useFastTanhApprox();
    }

    public ExecutionContext executionContext() {
        return executionContext;
    }

    public <T> T runtimeStateFor(Tensor tensor, Class<T> type) {
        return executionContext.runtimeStateFor(tensor, type);
    }

    public void putRuntimeState(Tensor tensor, Object runtimeState) {
        executionContext.putRuntimeState(tensor, runtimeState);
    }

    public void clearRuntimeState(Tensor tensor) {
        executionContext.clearRuntimeState(tensor);
    }

    public void publishConvTrace(Tensor tensor, ConvTraceMetadata trace) {
        executionContext.publishConvTrace(nodeId, trace);
    }

    public CompiledNodeExecutionMetadata executionMetadata() {
        return executionMetadata;
    }

    public Operation executionOperation() {
        return executionOperation == null ? executionMetadata.executionOperation() : executionOperation;
    }

    public PreparedFusedExecutable fusedExecutable() {
        return executionMetadata.artifact() instanceof CpuFusedExecutionArtifact artifact
                ? artifact.fusedExecutable()
                : null;
    }

    public CpuNodeWorkspace cpuWorkspace() {
        return executionContext.cpuWorkspaceForNodeId(nodeId);
    }

    public boolean publishFloatContinuation() {
        return nodePlan.publishFloatContinuation();
    }

    public int plannedWorkers() {
        return nodePlan.plannedWorkers();
    }

    public int contiguousMaterializeThreshold() {
        return nodePlan.contiguousMaterializeThreshold();
    }

    public int computeChunkSize(int totalLength, int alignment, int targetChunksPerWorker, int minChunkSize) {
        int length = Math.max(1, totalLength);
        int workers = plannedWorkers();
        int targets = Math.max(workers, workers * Math.max(1, targetChunksPerWorker));
        int candidate = (length + targets - 1) / targets;
        int chunk = Math.max(Math.max(1, minChunkSize), candidate);
        int align = Math.max(1, alignment);
        if (align > 1) {
            int rem = chunk % align;
            if (rem != 0) {
                chunk += (align - rem);
            }
        }
        return chunk;
    }

    public float[] inputFloatContinuation(int inputIndex, int requiredLength) {
        if (inputIndex < 0 || inputIndex >= inputMetadatas.size()) {
            return null;
        }
        CompiledNodeExecutionMetadata metadata = inputMetadatas.get(inputIndex);
        Integer inputNodeId = inputIndex < inputNodeIds.size() ? inputNodeIds.get(inputIndex) : null;
        return resolveInputFloatContinuation(inputNodeId, metadata, requiredLength, 0);
    }

    private float[] resolveInputFloatContinuation(
            Integer nodeId,
            CompiledNodeExecutionMetadata metadata,
            int requiredLength,
            int depth
    ) {
        if (nodeId == null || metadata == null || depth > 8) {
            return null;
        }
        CpuNodeWorkspace workspace = executionContext.cpuWorkspaceForNodeId(nodeId);
        if (workspace != null && workspace.hasFloatContinuation(requiredLength)) {
            return workspace.requireFloatWorkspace();
        }
        if (!isContinuationPassthroughNode(nodeId)) {
            return null;
        }
        Tensor runtimeTensor = executionContext.runtimeTensorForNodeId(nodeId);
        List<Tensor> prev = runtimeTensor.getPrevTensors();
        if (prev == null || prev.size() != 1) {
            return null;
        }
        Integer upstreamNodeId = executionContext.nodeIdForRuntimeTensor(prev.getFirst());
        if (upstreamNodeId == null) {
            return null;
        }
        return resolveInputFloatContinuation(
                upstreamNodeId,
                executionContext.metadataForNodeId(upstreamNodeId),
                requiredLength,
                depth + 1
        );
    }

    private boolean isContinuationPassthroughNode(int nodeId) {
        Tensor runtimeTensor = executionContext.runtimeTensorForNodeId(nodeId);
        Operation operation = runtimeTensor.getOperation();
        if (operation == null || operation.opType() == null) {
            return false;
        }
        return switch (operation.opType()) {
            case RESHAPE, CONTIGUOUS -> true;
            default -> false;
        };
    }
}
