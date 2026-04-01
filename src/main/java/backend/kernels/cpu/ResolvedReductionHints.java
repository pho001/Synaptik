package backend.kernels.cpu;

import backend.kernels.cpu.CpuExecutionMode;
import config.backend.SumAccuracyMode;

public record ResolvedReductionHints(
        int logicalSize,
        CpuExecutionMode mode,
        int chunkSize,
        int vectorWidth,
        int plannedWorkers,
        SumAccuracyMode accuracyMode
) {
    public ResolvedReductionHints {
        logicalSize = Math.max(0, logicalSize);
        mode = mode == null ? CpuExecutionMode.SCALAR : mode;
        chunkSize = Math.max(1, chunkSize);
        vectorWidth = Math.max(1, vectorWidth);
        plannedWorkers = Math.max(1, plannedWorkers);
        accuracyMode = accuracyMode == null ? SumAccuracyMode.FAST : accuracyMode;
    }

    public boolean vectorized() {
        return mode == CpuExecutionMode.VECTOR || mode == CpuExecutionMode.PARALLEL_VECTOR;
    }

    public boolean parallel() {
        return mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR;
    }
}