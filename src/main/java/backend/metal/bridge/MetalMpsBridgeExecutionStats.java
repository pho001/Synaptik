package backend.metal.bridge;

/**
 * Per-execution diagnostics reported by the Metal MPSGraph bridge.
 *
 * <p>The values are intentionally measured at the Java bridge boundary. Native
 * GPU command timing may be refined later, but these counters already expose
 * the costs that currently dominate the bridge path: Java-to-native input
 * copies, native invocation, output allocation, native shim result copies into
 * caller-provided buffers, and native-to-Java copy-back.</p>
 *
 * @param usedCpuFallback whether this execution was served by CPU fallback instead of Metal
 * @param fallbackReason stable diagnostic reason when {@code usedCpuFallback} is true
 * @param executionPath runtime path used for this execution attempt
 * @param nativeCopyStrategy native-side output copy/write classification
 * @param externalInputCount number of external tensors passed to the bridge
 * @param outputCount number of output tensors requested from the bridge
 * @param inputBytes total logical input payload bytes
 * @param outputBytes total logical output payload bytes
 * @param javaToNativeCopyNs time spent copying Java tensor arrays into native bridge memory
 * @param outputAllocationNs time spent allocating temporary native output buffers
 * @param nativeExecuteNs time spent inside the native execute call as observed by Java
 * @param nativeDeviceCopyNs time spent by the native shim copying MPSGraph result storage into caller-provided buffers
 * @param nativeToJavaCopyNs time spent copying native outputs back into Java tensor arrays
 * @param totalNs total measured bridge execution time
 */
public record MetalMpsBridgeExecutionStats(
        boolean usedCpuFallback,
        String fallbackReason,
        MetalMpsBridgeExecutionPath executionPath,
        MetalNativeCopyStrategy nativeCopyStrategy,
        int externalInputCount,
        int outputCount,
        long inputBytes,
        long outputBytes,
        long javaToNativeCopyNs,
        long outputAllocationNs,
        long nativeExecuteNs,
        long nativeDeviceCopyNs,
        long nativeToJavaCopyNs,
        long totalNs
) {
    public MetalMpsBridgeExecutionStats {
        fallbackReason = fallbackReason == null ? "" : fallbackReason;
        executionPath = executionPath == null
                ? (usedCpuFallback ? MetalMpsBridgeExecutionPath.CPU_FALLBACK : MetalMpsBridgeExecutionPath.TENSOR_ARRAY_COPY)
                : executionPath;
        nativeCopyStrategy = nativeCopyStrategy == null
                ? defaultCopyStrategy(usedCpuFallback, executionPath)
                : nativeCopyStrategy;
    }

    public MetalMpsBridgeExecutionStats(
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
            long nativeDeviceCopyNs,
            long nativeToJavaCopyNs,
            long totalNs
    ) {
        this(
                usedCpuFallback,
                fallbackReason,
                executionPath,
                null,
                externalInputCount,
                outputCount,
                inputBytes,
                outputBytes,
                javaToNativeCopyNs,
                outputAllocationNs,
                nativeExecuteNs,
                nativeDeviceCopyNs,
                nativeToJavaCopyNs,
                totalNs
        );
    }

    /**
     * Returns whether the native bridge has proven direct writes into caller-provided output buffers.
     */
    public boolean outputBufferWriteProven() {
        return nativeCopyStrategy == MetalNativeCopyStrategy.TRUE_OUTPUT_BUFFER_WRITE;
    }

    /**
     * Returns a stable report label for the output-buffer write/copy contract.
     */
    public String outputBufferWriteStatus() {
        return switch (nativeCopyStrategy) {
            case TRUE_OUTPUT_BUFFER_WRITE -> "PROVEN_TRUE_WRITE";
            case MPSGRAPH_RESULT_COPY -> "COPY_REQUIRED";
            case UNKNOWN_OR_UNPROVEN -> "UNKNOWN_OR_UNPROVEN";
        };
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
                MetalNativeCopyStrategy.UNKNOWN_OR_UNPROVEN,
                externalInputCount,
                outputCount,
                inputBytes,
                outputBytes,
                0L,
                0L,
                0L,
                0L,
                0L,
                0L
        );
    }

    private static MetalNativeCopyStrategy defaultCopyStrategy(
            boolean usedCpuFallback,
            MetalMpsBridgeExecutionPath executionPath
    ) {
        if (usedCpuFallback || executionPath == MetalMpsBridgeExecutionPath.CPU_FALLBACK) {
            return MetalNativeCopyStrategy.UNKNOWN_OR_UNPROVEN;
        }
        return MetalNativeCopyStrategy.MPSGRAPH_RESULT_COPY;
    }
}
