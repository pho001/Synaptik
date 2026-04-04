package graph.execution.trace;

public record MatMulTraceMetadata(
        boolean useBlas,
        boolean useBatchedBlas,
        boolean parallel,
        int tileM,
        int tileN,
        int tileK,
        int plannedWorkers,
        long work
) {
}
