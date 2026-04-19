package backend.kernels.cpu.reduction.plan;

import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.linalg.attention.plan.ResolvedAttentionHints;
import backend.kernels.cpu.ResolvedCpuComputeContract;
import backend.kernels.cpu.reduction.plan.ResolvedReductionHints;
import backend.kernels.cpu.plan.CpuPlanningPolicy;

public final class ReductionPlanner {
    private final CpuPlanningPolicy policy;

    public ReductionPlanner(CpuPlanningPolicy policy) {
        this.policy = policy;
    }

    public ResolvedReductionHints resolve(int logicalSize, ResolvedCpuComputeContract contract) {
        int size = Math.max(0, logicalSize);
        int vectorWidth = policy.preferredVectorWidth(contract);
        boolean vectorAllowed = vectorWidth > 1 && size >= policy.reductionVectorMinSize();

        CpuExecutionMode mode;
        if (size >= policy.reductionParallelMinSize()) {
            mode = vectorAllowed ? CpuExecutionMode.PARALLEL_VECTOR : CpuExecutionMode.PARALLEL;
        } else {
            mode = vectorAllowed ? CpuExecutionMode.VECTOR : CpuExecutionMode.SCALAR;
        }

        int chunkSize = policy.computeChunkSize(
                size,
                mode == CpuExecutionMode.VECTOR || mode == CpuExecutionMode.PARALLEL_VECTOR ? vectorWidth : 1,
                1,
                policy.minReductionChunkSize()
        );

        return new ResolvedReductionHints(
                size,
                mode,
                chunkSize,
                vectorWidth,
                policy.plannedWorkers(),
                policy.sumAccuracyMode()
        );
    }

    public ResolvedAttentionHints resolveAttentionHints(
            int independentTasks,
            int workPerTask,
            int vectorSpan,
            ResolvedCpuComputeContract contract
    ) {
        int tasks = Math.max(1, independentTasks);
        int scalarWorkPerTask = Math.max(1, workPerTask);
        long totalWorkLong = (long) tasks * scalarWorkPerTask;
        int totalWork = totalWorkLong > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) totalWorkLong;
        long vectorWorkLong = (long) tasks * Math.max(1, vectorSpan);
        int vectorWork = vectorWorkLong > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) vectorWorkLong;
        int vectorWidth = policy.preferredVectorWidth(contract);
        boolean vectorAllowed = vectorWidth > 1
                && vectorSpan >= vectorWidth
                && vectorWork >= policy.attentionVectorMinSize();
        boolean parallelAllowed = tasks > 1 && totalWork >= policy.attentionParallelMinSize();
        CpuExecutionMode mode;
        if (parallelAllowed) {
            mode = vectorAllowed ? CpuExecutionMode.PARALLEL_VECTOR : CpuExecutionMode.PARALLEL;
        } else {
            mode = vectorAllowed ? CpuExecutionMode.VECTOR : CpuExecutionMode.SCALAR;
        }
        int workChunk = policy.computeChunkSize(
                totalWork,
                vectorAllowed ? vectorWidth : 1,
                1,
                policy.minReductionChunkSize()
        );
        int taskChunkSize = Math.max(1, workChunk / scalarWorkPerTask);
        return new ResolvedAttentionHints(
                mode,
                Math.min(tasks, taskChunkSize),
                vectorWidth,
                policy.plannedWorkers()
        );
    }
}
