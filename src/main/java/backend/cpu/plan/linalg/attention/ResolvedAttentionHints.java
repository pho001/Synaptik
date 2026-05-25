package backend.cpu.plan.linalg.attention;

import backend.cpu.plan.CpuExecutionMode;

public record ResolvedAttentionHints(
        CpuExecutionMode mode,
        int taskChunkSize,
        int vectorWidth,
        int plannedWorkers
) {
    public ResolvedAttentionHints {
        mode = mode == null ? CpuExecutionMode.SCALAR : mode;
        taskChunkSize = Math.max(1, taskChunkSize);
        vectorWidth = Math.max(1, vectorWidth);
        plannedWorkers = Math.max(1, plannedWorkers);
    }

    public boolean vectorized() {
        return mode == CpuExecutionMode.VECTOR || mode == CpuExecutionMode.PARALLEL_VECTOR;
    }

    public boolean parallel() {
        return mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR;
    }
}
