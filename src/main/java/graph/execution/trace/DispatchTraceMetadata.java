package graph.execution.trace;

public record DispatchTraceMetadata(
        String mode,
        int vectorWidth,
        int plannedWorkers,
        int scalarChunkSize,
        int vectorChunkSize
) {
}
