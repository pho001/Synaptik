package backend.kernels.cpu;

import backend.CpuLayoutPlan;
import backend.runtime.BlasConfig;
import backend.runtime.ExecutionContext;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.fused.PreparedFusedExecutable;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

public final class CpuKernelContext {
    private final CpuExecutionPlanner planner;
    private final CpuNodeExecutionPlan nodePlan;
    private final ExecutionContext executionContext;
    private final CompiledNodeExecutionMetadata executionMetadata;
    private final List<CompiledNodeExecutionMetadata> inputMetadatas;

    public CpuKernelContext(
            CpuExecutionPlanner planner,
            CpuNodeExecutionPlan nodePlan,
            ExecutionContext executionContext,
            CompiledNodeExecutionMetadata executionMetadata,
            List<CompiledNodeExecutionMetadata> inputMetadatas
    ) {
        this.planner = Objects.requireNonNull(planner, "planner cannot be null");
        this.nodePlan = Objects.requireNonNull(nodePlan, "nodePlan cannot be null");
        this.executionContext = Objects.requireNonNull(executionContext, "executionContext cannot be null");
        this.executionMetadata = Objects.requireNonNull(executionMetadata, "executionMetadata cannot be null");
        this.inputMetadatas = Collections.unmodifiableList(new ArrayList<>(inputMetadatas == null ? List.of() : inputMetadatas));
    }

    public CpuExecutionPlanner planner() {
        return planner;
    }

    public CpuNodeExecutionPlan nodePlan() {
        return nodePlan;
    }

    public CpuLayoutPlan layoutPlan() {
        return nodePlan.layoutPlan();
    }

    public CpuComputeMode computeMode() {
        return nodePlan.computeMode();
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

    public BlasConfig blasConfig() {
        return executionContext.runtimeConfig().blasConfig();
    }

    public ExecutionContext executionContext() {
        return executionContext;
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
