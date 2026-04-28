package graph.execution.trace;

/**
 * Matrix multiplication kernel metadata for a step.
 *
 * @param useBlas whether BLAS was used
 * @param useBatchedBlas whether batched BLAS was used
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
        boolean parallel,
        int tileM,
        int tileN,
        int tileK,
        int plannedWorkers,
        long work,
        String microKernel
) {
}
