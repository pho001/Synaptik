package graph.execution.trace;

/**
 * Native CPU allocation counters captured for one run.
 *
 * @param allocationCount successful native CPU allocations
 * @param releaseCount released native CPU allocations
 * @param retainCount allocations marked retained beyond normal run ownership
 * @param allocationFailureCount failed native CPU allocation attempts
 * @param requestedBytes caller-requested bytes
 * @param allocatedBytes bytes requested from FFM after allocator rounding
 * @param currentLiveBytes bytes still live when the trace snapshot was captured
 * @param peakLiveBytes maximum simultaneously live allocated bytes
 * @param retainedBytes bytes marked retained when the trace snapshot was captured
 */
public record NativeCpuMemoryTrace(
        long allocationCount,
        long releaseCount,
        long retainCount,
        long allocationFailureCount,
        long requestedBytes,
        long allocatedBytes,
        long currentLiveBytes,
        long peakLiveBytes,
        long retainedBytes
) {
    public NativeCpuMemoryTrace {
        allocationCount = Math.max(0L, allocationCount);
        releaseCount = Math.max(0L, releaseCount);
        retainCount = Math.max(0L, retainCount);
        allocationFailureCount = Math.max(0L, allocationFailureCount);
        requestedBytes = Math.max(0L, requestedBytes);
        allocatedBytes = Math.max(0L, allocatedBytes);
        currentLiveBytes = Math.max(0L, currentLiveBytes);
        peakLiveBytes = Math.max(0L, peakLiveBytes);
        retainedBytes = Math.max(0L, retainedBytes);
    }

    public boolean present() {
        return allocationCount > 0L
                || releaseCount > 0L
                || retainCount > 0L
                || allocationFailureCount > 0L
                || requestedBytes > 0L
                || allocatedBytes > 0L
                || peakLiveBytes > 0L
                || retainedBytes > 0L;
    }

    public static NativeCpuMemoryTrace empty() {
        return new NativeCpuMemoryTrace(0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L);
    }
}
