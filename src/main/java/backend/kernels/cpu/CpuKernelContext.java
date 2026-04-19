package backend.kernels.cpu;

import backend.CpuLayoutPlan;
import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;
import backend.kernels.cpu.layout.plan.ResolvedBroadcastPlan;
import backend.kernels.cpu.layout.plan.ResolvedWhereBroadcastPlan;
import backend.kernels.cpu.linalg.attention.plan.ResolvedScaledDotProductAttentionPlan;
import backend.kernels.cpu.linalg.matmul.plan.ResolvedMatMulHints;
import backend.kernels.cpu.nn.conv2d.plan.ResolvedConv2dHints;
import backend.kernels.cpu.reduction.plan.ResolvedReductionHints;
import backend.runtime.ExecutionContext;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.execution.trace.ConvTraceMetadata;
import graph.fused.PreparedFusedExecutable;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

public final class CpuKernelContext {
    private final CpuNodeExecutionPlan nodePlan;
    private final ExecutionContext executionContext;
    private final CompiledNodeExecutionMetadata executionMetadata;
    private final List<CompiledNodeExecutionMetadata> inputMetadatas;

    public CpuKernelContext(
            CpuNodeExecutionPlan nodePlan,
            ExecutionContext executionContext,
            CompiledNodeExecutionMetadata executionMetadata,
            List<CompiledNodeExecutionMetadata> inputMetadatas
    ) {
        this.nodePlan = Objects.requireNonNull(nodePlan, "nodePlan cannot be null");
        this.executionContext = Objects.requireNonNull(executionContext, "executionContext cannot be null");
        this.executionMetadata = Objects.requireNonNull(executionMetadata, "executionMetadata cannot be null");
        this.inputMetadatas = Collections.unmodifiableList(new ArrayList<>(inputMetadatas == null ? List.of() : inputMetadatas));
    }

    public CpuNodeExecutionPlan nodePlan() {
        return nodePlan;
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

    public ResolvedConv2dHints conv2dHints() {
        return nodePlan.conv2dHints();
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
        executionContext.publishConvTrace(tensor, trace);
    }

    public CompiledNodeExecutionMetadata executionMetadata() {
        return executionMetadata;
    }

    public PreparedFusedExecutable fusedExecutable() {
        return executionMetadata.fusedExecutable();
    }

    public CpuNodeWorkspace cpuWorkspace() {
        return executionMetadata.cpuWorkspace();
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
        if (metadata == null || metadata.cpuWorkspace() == null || !metadata.cpuWorkspace().hasFloatContinuation(requiredLength)) {
            return null;
        }
        return metadata.cpuWorkspace().requireFloatWorkspace();
    }
}
