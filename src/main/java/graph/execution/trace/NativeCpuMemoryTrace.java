package graph.execution.trace;

/**
 * Native CPU allocation counters captured for one run.
 *
 * @param allocationCount successful native CPU allocations
 * @param releaseCount released native CPU allocations
 * @param retainCount allocations marked retained beyond normal run ownership
 * @param allocationFailureCount failed native CPU allocation attempts
 * @param requestedPoolPolicy requested native memory pool policy
 * @param effectivePoolPolicy effective native memory pool policy
 * @param requestedBytes caller-requested bytes
 * @param allocatedBytes bytes requested from FFM after allocator rounding
 * @param currentLiveBytes bytes still live when the trace snapshot was captured
 * @param peakLiveBytes maximum simultaneously live allocated bytes
 * @param retainedBytes bytes marked retained when the trace snapshot was captured
 * @param poolHitCount pool-backed allocations served from an idle block
 * @param poolMissCount pool-backed allocations that needed a fresh block
 * @param pooledBytes bytes idle in a native memory pool
 * @param reusedBytes bytes served from idle pool blocks
 * @param discardedBytes bytes closed instead of kept in a pool
 * @param wastedBytes allocated bytes beyond requested bytes
 */
public record NativeCpuMemoryTrace(
        long allocationCount,
        long releaseCount,
        long retainCount,
        long allocationFailureCount,
        String requestedPoolPolicy,
        String effectivePoolPolicy,
        long requestedBytes,
        long allocatedBytes,
        long currentLiveBytes,
        long peakLiveBytes,
        long retainedBytes,
        long poolHitCount,
        long poolMissCount,
        long pooledBytes,
        long reusedBytes,
        long discardedBytes,
        long wastedBytes
) {
    public NativeCpuMemoryTrace {
        allocationCount = Math.max(0L, allocationCount);
        releaseCount = Math.max(0L, releaseCount);
        retainCount = Math.max(0L, retainCount);
        allocationFailureCount = Math.max(0L, allocationFailureCount);
        requestedPoolPolicy = requestedPoolPolicy == null ? "" : requestedPoolPolicy;
        effectivePoolPolicy = effectivePoolPolicy == null ? "" : effectivePoolPolicy;
        requestedBytes = Math.max(0L, requestedBytes);
        allocatedBytes = Math.max(0L, allocatedBytes);
        currentLiveBytes = Math.max(0L, currentLiveBytes);
        peakLiveBytes = Math.max(0L, peakLiveBytes);
        retainedBytes = Math.max(0L, retainedBytes);
        poolHitCount = Math.max(0L, poolHitCount);
        poolMissCount = Math.max(0L, poolMissCount);
        pooledBytes = Math.max(0L, pooledBytes);
        reusedBytes = Math.max(0L, reusedBytes);
        discardedBytes = Math.max(0L, discardedBytes);
        wastedBytes = Math.max(0L, wastedBytes);
    }

    public NativeCpuMemoryTrace(
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
        this(
                allocationCount,
                releaseCount,
                retainCount,
                allocationFailureCount,
                "",
                "",
                requestedBytes,
                allocatedBytes,
                currentLiveBytes,
                peakLiveBytes,
                retainedBytes,
                0L,
                0L,
                0L,
                0L,
                0L,
                Math.max(0L, allocatedBytes - requestedBytes)
        );
    }

    public boolean present() {
        return allocationCount > 0L
                || releaseCount > 0L
                || retainCount > 0L
                || allocationFailureCount > 0L
                || requestedBytes > 0L
                || allocatedBytes > 0L
                || peakLiveBytes > 0L
                || retainedBytes > 0L
                || poolHitCount > 0L
                || poolMissCount > 0L
                || pooledBytes > 0L
                || reusedBytes > 0L
                || discardedBytes > 0L
                || wastedBytes > 0L;
    }

    public static NativeCpuMemoryTrace empty() {
        return new NativeCpuMemoryTrace(0L, 0L, 0L, 0L, "", "", 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L);
    }
}
