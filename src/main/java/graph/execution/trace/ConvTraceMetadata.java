package graph.execution.trace;

public record ConvTraceMetadata(
        String executionKind,
        boolean lowered,
        boolean blasUsed,
        String blasProvider,
        int m,
        int n,
        int k,
        int blasCalls,
        int javaCalls
) {
    public ConvTraceMetadata {
        executionKind = (executionKind == null || executionKind.isBlank()) ? "UNKNOWN" : executionKind;
        blasProvider = (blasProvider == null || blasProvider.isBlank()) ? "NONE" : blasProvider;
        m = Math.max(0, m);
        n = Math.max(0, n);
        k = Math.max(0, k);
        blasCalls = Math.max(0, blasCalls);
        javaCalls = Math.max(0, javaCalls);
    }
}
