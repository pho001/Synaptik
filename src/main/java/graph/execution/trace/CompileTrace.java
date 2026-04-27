package graph.execution.trace;

public record CompileTrace(
        boolean measured,
        long durationNs,
        int totalNodeCount,
        int forwardNodeCount,
        boolean supportsBackward,
        PartitionCompileTrace partitionPlanning
) {
    public static CompileTrace skipped() {
        return new CompileTrace(false, 0L, 0, 0, false, PartitionCompileTrace.empty());
    }
}
