package graph.execution.trace;

public record ReductionTraceMetadata(
        String mode,
        int plannedWorkers,
        int chunkSize,
        int vectorWidth,
        String accuracyMode
) {
}
