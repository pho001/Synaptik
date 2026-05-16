package backend.cpu.nativecpu;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Allocation counters for native CPU tensor memory.
 */
public final class NativeCpuMemoryStats {
    private final AtomicLong allocationCount = new AtomicLong();
    private final AtomicLong releaseCount = new AtomicLong();
    private final AtomicLong retainCount = new AtomicLong();
    private final AtomicLong allocationFailureCount = new AtomicLong();
    private final AtomicLong requestedBytes = new AtomicLong();
    private final AtomicLong allocatedBytes = new AtomicLong();
    private final AtomicLong currentLiveBytes = new AtomicLong();
    private final AtomicLong peakLiveBytes = new AtomicLong();
    private final AtomicLong retainedBytes = new AtomicLong();

    void recordAllocation(long requested, long allocated) {
        long safeRequested = Math.max(0L, requested);
        long safeAllocated = Math.max(0L, allocated);
        allocationCount.incrementAndGet();
        requestedBytes.addAndGet(safeRequested);
        allocatedBytes.addAndGet(safeAllocated);
        long live = currentLiveBytes.addAndGet(safeAllocated);
        updatePeak(live);
    }

    void recordRelease(long allocated, boolean retained) {
        long safeAllocated = Math.max(0L, allocated);
        releaseCount.incrementAndGet();
        currentLiveBytes.updateAndGet(current -> Math.max(0L, current - safeAllocated));
        if (retained) {
            retainedBytes.updateAndGet(current -> Math.max(0L, current - safeAllocated));
        }
    }

    void recordRetain(long allocated) {
        long safeAllocated = Math.max(0L, allocated);
        retainCount.incrementAndGet();
        retainedBytes.addAndGet(safeAllocated);
    }

    void recordAllocationFailure() {
        allocationFailureCount.incrementAndGet();
    }

    public Snapshot snapshot() {
        return new Snapshot(
                allocationCount.get(),
                releaseCount.get(),
                retainCount.get(),
                allocationFailureCount.get(),
                requestedBytes.get(),
                allocatedBytes.get(),
                currentLiveBytes.get(),
                peakLiveBytes.get(),
                retainedBytes.get()
        );
    }

    private void updatePeak(long live) {
        long currentPeak;
        do {
            currentPeak = peakLiveBytes.get();
            if (live <= currentPeak) {
                return;
            }
        } while (!peakLiveBytes.compareAndSet(currentPeak, live));
    }

    public record Snapshot(
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
    }
}
