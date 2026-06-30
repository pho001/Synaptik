package trace.backend;

/**
 * Convolution kernel metadata for a step.
 *
 * @param executionKind convolution execution strategy
 * @param lowered whether the operation was lowered before execution
 * @param blasUsed whether BLAS was used
 * @param blasProvider BLAS provider label
 * @param m GEMM M dimension, when applicable
 * @param n GEMM N dimension, when applicable
 * @param k GEMM K dimension, when applicable
 * @param blasCalls number of BLAS calls
 * @param javaCalls number of Java fallback calls
 */
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
