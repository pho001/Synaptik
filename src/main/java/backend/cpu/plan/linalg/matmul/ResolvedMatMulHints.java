package backend.cpu.plan.linalg.matmul;

import config.backend.CpuMatMulMicroKernel;

public record ResolvedMatMulHints(
        boolean useBlas,
        boolean useBatchedBlas,
        MatMulExecutionRoute route,
        boolean parallel,
        int tileM,
        int tileN,
        int tileK,
        int plannedWorkers,
        long work,
        CpuMatMulMicroKernel microKernel,
        String blasProvider,
        boolean blasDebug,
        int openBlasThreads,
        boolean openblasSgemmAvailable,
        boolean openblasDgemmAvailable,
        boolean openblasSbgemmAvailable,
        boolean openblasBgemmAvailable,
        String openblasLookupSource,
        String threadPolicy,
        String cpuStorageProfile,
        String requestedCpuStorage,
        String nativeCpuFailurePolicy
) {
    public ResolvedMatMulHints(
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
        this(
                useBlas,
                useBatchedBlas,
                useBlas || useBatchedBlas ? MatMulExecutionRoute.OPENBLAS_ARRAY_COPYING : MatMulExecutionRoute.JAVA_DIRECT,
                parallel,
                tileM,
                tileN,
                tileK,
                plannedWorkers,
                work,
                microKernel,
                "",
                false,
                0,
                false,
                false,
                false,
                false,
                "UNAVAILABLE",
                "",
                "AUTO",
                "CPU_ARRAY",
                ""
        );
    }

    public ResolvedMatMulHints {
        route = route == null
                ? (useBlas || useBatchedBlas ? MatMulExecutionRoute.OPENBLAS_ARRAY_COPYING : MatMulExecutionRoute.JAVA_DIRECT)
                : route;
        tileM = Math.max(1, tileM);
        tileN = Math.max(1, tileN);
        tileK = Math.max(1, tileK);
        plannedWorkers = Math.max(1, plannedWorkers);
        work = Math.max(0L, work);
        microKernel = microKernel == null ? CpuMatMulMicroKernel.AUTO : microKernel;
        blasProvider = blasProvider == null ? "" : blasProvider;
        openBlasThreads = Math.max(0, openBlasThreads);
        openblasLookupSource = openblasLookupSource == null ? "UNAVAILABLE" : openblasLookupSource;
        threadPolicy = threadPolicy == null ? "" : threadPolicy;
        cpuStorageProfile = cpuStorageProfile == null ? "AUTO" : cpuStorageProfile;
        requestedCpuStorage = requestedCpuStorage == null ? "CPU_ARRAY" : requestedCpuStorage;
        nativeCpuFailurePolicy = nativeCpuFailurePolicy == null ? "" : nativeCpuFailurePolicy;
    }

    public boolean usesOpenBlasMemorySegment() {
        return route == MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT;
    }

    public boolean usesOpenBlasArrayCopying() {
        return route == MatMulExecutionRoute.OPENBLAS_ARRAY_COPYING;
    }

    public boolean usesJavaArrays() {
        return route == MatMulExecutionRoute.JAVA_DIRECT;
    }
}
