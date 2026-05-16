package graph.execution.trace;

/**
 * Matrix multiplication kernel metadata for a step.
 *
 * @param useBlas whether BLAS was used
 * @param useBatchedBlas whether batched BLAS was used
 * @param blasProvider BLAS provider selected for this matmul
 * @param blasSymbol native BLAS symbol used or targeted by the route
 * @param blasRoute stable BLAS route label
 * @param route concrete runtime route selected for the step
 * @param cpuStorageProfile runtime-level CPU storage policy
 * @param nativeCpuFailurePolicy native CPU fallback policy
 * @param requestedCpuStorage effective requested CPU storage after runtime/BLAS policy resolution
 * @param actualCpuStorage storage route actually used for this matmul step
 * @param nativeCpuFallbackReason native CPU route fallback reason, when present
 * @param openblasSgemmAvailable whether {@code cblas_sgemm} is available from the selected OpenBLAS FFM bridge
 * @param openblasDgemmAvailable whether {@code cblas_dgemm} is available from the selected OpenBLAS FFM bridge
 * @param openblasSbgemmAvailable whether {@code cblas_sbgemm} is available from the selected OpenBLAS FFM bridge
 * @param openblasBgemmAvailable whether {@code cblas_bgemm} is available from the selected OpenBLAS FFM bridge
 * @param bf16ContinuationRoute BF16 continuation route, for example {@code SBGEMM} or {@code JAVA}
 * @param bf16OutputRoute BF16 public-output route, for example {@code BGEMM}, {@code PROMOTED_F32}, or {@code JAVA}
 * @param bf16ComputePrecision BF16 compute precision contract for the selected route
 * @param bf16OutputPrecision BF16 output precision contract for the selected route
 * @param copyInBytes estimated bytes copied into a native library boundary, or {@code -1} when unknown
 * @param copyOutBytes estimated bytes copied out of a native library boundary, or {@code -1} when unknown
 * @param nativeTempBytes native temporary bytes allocated by the route, or {@code -1} when unknown
 * @param threadPolicy effective OpenBLAS thread policy
 * @param fallbackReason dynamic fallback reason, when a preferred route could not run
 * @param parallel whether parallel execution was planned
 * @param tileM M tile size
 * @param tileN N tile size
 * @param tileK K tile size
 * @param plannedWorkers planned worker count
 * @param work estimated operation work
 * @param microKernel selected micro-kernel label
 */
public record MatMulTraceMetadata(
        boolean useBlas,
        boolean useBatchedBlas,
        String blasProvider,
        String blasSymbol,
        String blasRoute,
        String route,
        String cpuStorageProfile,
        String nativeCpuFailurePolicy,
        String requestedCpuStorage,
        String actualCpuStorage,
        String nativeCpuFallbackReason,
        boolean openblasSgemmAvailable,
        boolean openblasDgemmAvailable,
        boolean openblasSbgemmAvailable,
        boolean openblasBgemmAvailable,
        String bf16ContinuationRoute,
        String bf16OutputRoute,
        String bf16ComputePrecision,
        String bf16OutputPrecision,
        long copyInBytes,
        long copyOutBytes,
        long nativeTempBytes,
        String threadPolicy,
        String fallbackReason,
        boolean parallel,
        int tileM,
        int tileN,
        int tileK,
        int plannedWorkers,
        long work,
        String microKernel
) {
    public MatMulTraceMetadata {
        blasProvider = blasProvider == null ? "" : blasProvider;
        blasSymbol = blasSymbol == null ? "" : blasSymbol;
        blasRoute = blasRoute == null ? "" : blasRoute;
        route = route == null ? "" : route;
        cpuStorageProfile = cpuStorageProfile == null ? "" : cpuStorageProfile;
        nativeCpuFailurePolicy = nativeCpuFailurePolicy == null ? "" : nativeCpuFailurePolicy;
        requestedCpuStorage = requestedCpuStorage == null ? "" : requestedCpuStorage;
        actualCpuStorage = actualCpuStorage == null ? "" : actualCpuStorage;
        nativeCpuFallbackReason = nativeCpuFallbackReason == null ? "" : nativeCpuFallbackReason;
        bf16ContinuationRoute = bf16ContinuationRoute == null ? "" : bf16ContinuationRoute;
        bf16OutputRoute = bf16OutputRoute == null ? "" : bf16OutputRoute;
        bf16ComputePrecision = bf16ComputePrecision == null ? "" : bf16ComputePrecision;
        bf16OutputPrecision = bf16OutputPrecision == null ? "" : bf16OutputPrecision;
        copyInBytes = Math.max(-1L, copyInBytes);
        copyOutBytes = Math.max(-1L, copyOutBytes);
        nativeTempBytes = Math.max(-1L, nativeTempBytes);
        threadPolicy = threadPolicy == null ? "" : threadPolicy;
        fallbackReason = fallbackReason == null ? "" : fallbackReason;
        microKernel = microKernel == null ? "" : microKernel;
    }
}
