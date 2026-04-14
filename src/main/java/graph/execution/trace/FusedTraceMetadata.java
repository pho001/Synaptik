package graph.execution.trace;

public record FusedTraceMetadata(
        int precisionMode,
        boolean lowCostHint,
        String dispatchFamily,
        String schedulerSignature,
        String executionBackend,
        int dispatchScale,
        int fusedNodeCount,
        int fusedInputCount
) {
}
