package backend.cpu.nativecpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Allocates native CPU memory for {@code MemorySegment}-backed tensor storage.
 */
public final class NativeCpuAllocator {
    public static final long DEFAULT_ALIGNMENT_BYTES = 64L;

    private final NativeCpuMemoryStats stats;

    public NativeCpuAllocator() {
        this(new NativeCpuMemoryStats());
    }

    public NativeCpuAllocator(NativeCpuMemoryStats stats) {
        this.stats = stats == null ? new NativeCpuMemoryStats() : stats;
    }

    public NativeCpuAllocation allocate(long byteSize, String label) {
        return allocate(byteSize, DEFAULT_ALIGNMENT_BYTES, label);
    }

    public NativeCpuAllocation allocate(long byteSize, long alignment, String label) {
        if (byteSize < 0L) {
            throw new IllegalArgumentException("byteSize cannot be negative: " + byteSize);
        }
        if (alignment <= 0L) {
            throw new IllegalArgumentException("alignment must be positive: " + alignment);
        }
        Arena arena = Arena.ofShared();
        try {
            long allocatedBytes = Math.max(byteSize, 1L);
            MemorySegment segment = arena.allocate(allocatedBytes, alignment);
            stats.recordAllocation(byteSize, allocatedBytes);
            return new NativeCpuAllocation(arena, segment, byteSize, allocatedBytes, alignment, label, stats);
        } catch (RuntimeException ex) {
            stats.recordAllocationFailure();
            arena.close();
            throw ex;
        }
    }

    public NativeCpuMemoryStats.Snapshot statsSnapshot() {
        return stats.snapshot();
    }
}
