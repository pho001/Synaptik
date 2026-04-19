package backend.kernels.cpu.linalg.matmul.plan;

import config.backend.CpuMatMulMicroKernel;

public record ResolvedMatMulHints(
        boolean useBlas,
        boolean useBatchedBlas,
        boolean parallel,
        int tileM,
        int tileN,
        int tileK,
        int plannedWorkers,
        long work,
        CpuMatMulMicroKernel microKernel
) {
    public ResolvedMatMulHints {
        tileM = Math.max(1, tileM);
        tileN = Math.max(1, tileN);
        tileK = Math.max(1, tileK);
        plannedWorkers = Math.max(1, plannedWorkers);
        work = Math.max(0L, work);
        microKernel = microKernel == null ? CpuMatMulMicroKernel.AUTO : microKernel;
    }
}
