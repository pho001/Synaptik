package backend.metal.bridge;

/**
 * Per-execution diagnostics reported by the Metal MPSGraph bridge.
 *
 * <p>The values are intentionally measured at the Java bridge boundary. Native
 * GPU command timing may be refined later, but these counters already expose
 * the costs that currently dominate the bridge path: Java-to-native input
 * copies, native invocation, output allocation, and native-to-Java copy-back.</p>
 *
 * @param usedCpuFallback whether this execution was served by CPU fallback instead of Metal
 * @param fallbackReason stable diagnostic reason when {@code usedCpuFallback} is true
 * @param executionPath runtime path used for this execution attempt
 * @param externalInputCount number of external tensors passed to the bridge
 * @param outputCount number of output tensors requested from the bridge
 * @param inputBytes total logical input payload bytes
 * @param outputBytes total logical output payload bytes
 * @param javaToNativeCopyNs time spent copying Java tensor arrays into native bridge memory
 * @param outputAllocationNs time spent allocating temporary native output buffers
 * @param nativeExecuteNs time spent inside the native execute call as observed by Java
 * @param nativeToJavaCopyNs time spent copying native outputs back into Java tensor arrays
 * @param totalNs total measured bridge execution time
 */
public record MetalMpsBridgeExecutionStats(
        boolean usedCpuFallback,
        String fallbackReason,
        MetalMpsBridgeExecutionPath executionPath,
        int externalInputCount,
        int outputCount,
        long inputBytes,
        long outputBytes,
        long javaToNativeCopyNs,
        long outputAllocationNs,
        long nativeExecuteNs,
        long nativeToJavaCopyNs,
        long totalNs
) {
    public MetalMpsBridgeExecutionStats {
        fallbackReason = fallbackReason == null ? "" : fallbackReason;
        executionPath = executionPath == null
                ? (usedCpuFallback ? MetalMpsBridgeExecutionPath.CPU_FALLBACK : MetalMpsBridgeExecutionPath.TENSOR_ARRAY_COPY)
                : executionPath;
    }

    /**
     * Creates a stats value for an execution that fell back before entering the bridge.
     *
     * @param reason human-readable fallback reason
     * @param externalInputCount number of resolved external inputs, if known
     * @param outputCount number of resolved outputs, if known
     * @param inputBytes logical input payload bytes, if known
     * @param outputBytes logical output payload bytes, if known
     * @return fallback diagnostics with timing counters set to zero
     */
    public static MetalMpsBridgeExecutionStats fallback(
            String reason,
            int externalInputCount,
            int outputCount,
            long inputBytes,
            long outputBytes
    ) {
        return new MetalMpsBridgeExecutionStats(
                true,
                reason,
                MetalMpsBridgeExecutionPath.CPU_FALLBACK,
                externalInputCount,
                outputCount,
                inputBytes,
                outputBytes,
                0L,
                0L,
                0L,
                0L,
                0L
        );
    }
}
