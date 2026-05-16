package backend.cpu.nativecpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Allocates native CPU memory for {@code MemorySegment}-backed tensor storage.
 */
public final class NativeCpuAllocator {
    public static final long DEFAULT_ALIGNMENT_BYTES = 64L;

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
            MemorySegment segment = arena.allocate(Math.max(byteSize, 1L), alignment);
            return new NativeCpuAllocation(arena, segment, byteSize, alignment, label);
        } catch (RuntimeException ex) {
            arena.close();
            throw ex;
        }
    }
}
