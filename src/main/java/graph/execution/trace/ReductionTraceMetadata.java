package graph.execution.trace;

/**
 * Reduction kernel metadata for a step.
 *
 * @param mode reduction mode label
 * @param plannedWorkers planned worker count
 * @param chunkSize chunk size selected for reduction work
 * @param vectorWidth vector width selected by the kernel
 * @param accuracyMode numerical accuracy mode label
 */
public record ReductionTraceMetadata(
        String mode,
        int plannedWorkers,
        int chunkSize,
        int vectorWidth,
        String accuracyMode
) {
}
