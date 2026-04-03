package backend.kernels.cpu;

import backend.CpuLayoutPlan;
import backend.runtime.BlasConfig;
import backend.runtime.ExecutionContext;
import backend.kernels.cpu.fused.CompiledFusedKernel;
import graph.execution.CompiledNodeExecutionMetadata;

import java.util.Objects;

public final class CpuKernelContext {
    private final CpuExecutionPlanner planner;
    private final CpuNodeExecutionPlan nodePlan;
    private final ExecutionContext executionContext;
    private final CompiledNodeExecutionMetadata executionMetadata;

    public CpuKernelContext(
            CpuExecutionPlanner planner,
            CpuNodeExecutionPlan nodePlan,
            ExecutionContext executionContext,
            CompiledNodeExecutionMetadata executionMetadata
    ) {
        this.planner = Objects.requireNonNull(planner, "planner cannot be null");
        this.nodePlan = Objects.requireNonNull(nodePlan, "nodePlan cannot be null");
        this.executionContext = Objects.requireNonNull(executionContext, "executionContext cannot be null");
        this.executionMetadata = Objects.requireNonNull(executionMetadata, "executionMetadata cannot be null");
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

    public CompiledFusedKernel fusedKernel() {
        return executionMetadata.fusedKernel();
    }
}
