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
        copyInBytes = Math.max(-1L, copyInBytes);
        copyOutBytes = Math.max(-1L, copyOutBytes);
        nativeTempBytes = Math.max(-1L, nativeTempBytes);
        threadPolicy = threadPolicy == null ? "" : threadPolicy;
        fallbackReason = fallbackReason == null ? "" : fallbackReason;
        microKernel = microKernel == null ? "" : microKernel;
    }
}
