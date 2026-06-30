package backend.cuda.bridge;

import runtime.device.buffer.AcceleratorBufferExecutionPath;

/**
 * Per-execution diagnostics reported by the CUDA bridge.
 *
 * <p>The counters are observed at the Java/native boundary. The current CUDA shim
 * does not expose native sub-timers for every path, so device-copy counters can
 * legitimately be zero while still preserving a stable trace/report contract.</p>
 *
 * @param usedCpuFallback whether this execution was served by CPU fallback instead of CUDA
 * @param fallbackReason stable diagnostic reason when {@code usedCpuFallback} is true
 * @param executionPath runtime path used for this execution attempt
 * @param externalInputCount number of external tensors or buffers passed to the bridge
 * @param outputCount number of output tensors or buffers requested from the bridge
 * @param inputBytes total logical input payload bytes
 * @param outputBytes total logical output payload bytes
 * @param javaToNativeCopyNs time spent copying Java tensors into native bridge memory
 * @param nativeExecuteNs time spent inside the native execute call as observed by Java
 * @param nativeDeviceCopyNs time spent by the native shim on device-side copies
 * @param nativeToJavaCopyNs time spent copying native outputs back into Java tensors
 * @param totalNs total measured bridge execution time
 */
public record CudaBridgeExecutionStats(
        boolean usedCpuFallback,
        String fallbackReason,
        AcceleratorBufferExecutionPath executionPath,
        int externalInputCount,
        int outputCount,
        long inputBytes,
        long outputBytes,
        long javaToNativeCopyNs,
        long nativeExecuteNs,
        long nativeDeviceCopyNs,
        long nativeToJavaCopyNs,
        long totalNs
) {
    public CudaBridgeExecutionStats {
        fallbackReason = fallbackReason == null ? "" : fallbackReason;
        executionPath = executionPath == null
                ? (usedCpuFallback ? AcceleratorBufferExecutionPath.CPU_FALLBACK : AcceleratorBufferExecutionPath.TENSOR_ARRAY)
                : executionPath;
    }

    /**
     * Creates diagnostics for an execution that fell back before entering native CUDA.
     *
     * @param reason human-readable fallback reason
     * @param externalInputCount number of resolved external inputs, if known
     * @param outputCount number of resolved outputs, if known
     * @param inputBytes logical input payload bytes, if known
     * @param outputBytes logical output payload bytes, if known
     * @return fallback diagnostics with timing counters set to zero
     */
    public static CudaBridgeExecutionStats fallback(
            String reason,
            int externalInputCount,
            int outputCount,
            long inputBytes,
            long outputBytes
    ) {
        return new CudaBridgeExecutionStats(
                true,
                reason,
                AcceleratorBufferExecutionPath.CPU_FALLBACK,
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
