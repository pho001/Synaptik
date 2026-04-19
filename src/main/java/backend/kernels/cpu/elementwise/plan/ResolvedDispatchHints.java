package backend.kernels.cpu.elementwise.plan;

import backend.kernels.cpu.CpuExecutionMode;

public record ResolvedDispatchHints(
        int totalLength,
        CpuExecutionMode mode,
        int scalarChunkSize,
        int vectorChunkSize,
        int vectorWidth,
        int plannedWorkers,
        boolean useCommonPool
) {
    public ResolvedDispatchHints {
        totalLength = Math.max(0, totalLength);
        mode = mode == null ? CpuExecutionMode.SCALAR : mode;
        scalarChunkSize = Math.max(1, scalarChunkSize);
        vectorChunkSize = Math.max(1, vectorChunkSize);
        vectorWidth = Math.max(1, vectorWidth);
        plannedWorkers = Math.max(1, plannedWorkers);
    }

    public boolean vectorized() {
        return mode == CpuExecutionMode.VECTOR || mode == CpuExecutionMode.PARALLEL_VECTOR;
    }

    public boolean parallel() {
        return mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR;
    }

    public int chunkSizeForCurrentMode() {
        return vectorized() ? vectorChunkSize : scalarChunkSize;
    }
}
