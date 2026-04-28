package graph.execution.trace;

/**
 * Dispatch scheduler metadata for a step.
 *
 * @param mode dispatch mode label
 * @param vectorWidth vector width selected by the kernel
 * @param plannedWorkers planned worker count
 * @param scalarChunkSize scalar chunk size
 * @param vectorChunkSize vectorized chunk size
 */
public record DispatchTraceMetadata(
        String mode,
        int vectorWidth,
        int plannedWorkers,
        int scalarChunkSize,
        int vectorChunkSize
) {
}
