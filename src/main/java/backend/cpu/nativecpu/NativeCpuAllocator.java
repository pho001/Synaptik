package backend.cpu.nativecpu;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

import config.runtime.NativeCpuMemoryConfig;
import config.runtime.NativeMemoryPoolPolicy;

/**
 * Allocates native CPU memory for {@code MemorySegment}-backed tensor storage.
 */
public final class NativeCpuAllocator {
    public static final long DEFAULT_ALIGNMENT_BYTES = 64L;

    private final NativeCpuMemoryStats stats;
    private final NativeCpuMemoryConfig config;
    private final NativeMemoryPoolPolicy effectivePoolPolicy;
    private final NativeCpuMemoryPool pool;

    public NativeCpuAllocator() {
        this(NativeCpuMemoryConfig.disabled(), new NativeCpuMemoryStats());
    }

    public NativeCpuAllocator(NativeCpuMemoryStats stats) {
        this(NativeCpuMemoryConfig.disabled(), stats);
    }

    public NativeCpuAllocator(NativeCpuMemoryConfig config) {
        this(config, new NativeCpuMemoryStats());
    }

    public NativeCpuAllocator(NativeCpuMemoryConfig config, NativeCpuMemoryStats stats) {
        this.stats = stats == null ? new NativeCpuMemoryStats() : stats;
        this.config = config == null ? NativeCpuMemoryConfig.disabled() : config;
        this.effectivePoolPolicy = this.config.poolPolicy() == NativeMemoryPoolPolicy.PER_EXECUTION
                ? NativeMemoryPoolPolicy.PER_EXECUTION
                : NativeMemoryPoolPolicy.DISABLED;
        this.pool = effectivePoolPolicy == NativeMemoryPoolPolicy.PER_EXECUTION
                ? new NativeCpuMemoryPool(this.config.maxPoolBytes())
                : null;
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
        long effectiveAlignment = config.alignmentBytes() > 0
                ? Math.max(alignment, config.alignmentBytes())
                : alignment;
        if (effectivePoolPolicy == NativeMemoryPoolPolicy.PER_EXECUTION) {
            return allocatePooled(byteSize, effectiveAlignment, label);
        }
        return allocateDirect(byteSize, effectiveAlignment, label);
    }

    private NativeCpuAllocation allocateDirect(long byteSize, long alignment, String label) {
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

    private NativeCpuAllocation allocatePooled(long byteSize, long alignment, String label) {
        long allocatedBytes = sizeClass(byteSize);
        NativeCpuMemoryPool.Block block = pool.acquire(allocatedBytes, alignment);
        if (block != null) {
            stats.recordAllocation(byteSize, allocatedBytes);
            stats.recordPoolHit(allocatedBytes);
            return new NativeCpuAllocation(block, pool, byteSize, label, stats);
        }
        Arena arena = Arena.ofShared();
        try {
            MemorySegment segment = arena.allocate(allocatedBytes, alignment);
            stats.recordAllocation(byteSize, allocatedBytes);
            stats.recordPoolMiss();
            return new NativeCpuAllocation(
                    new NativeCpuMemoryPool.Block(arena, segment, allocatedBytes, alignment),
                    pool,
                    byteSize,
                    label,
                    stats
            );
        } catch (RuntimeException ex) {
            stats.recordAllocationFailure();
            arena.close();
            throw ex;
        }
    }

    public long drainPool() {
        if (pool == null) {
            return 0L;
        }
        long drained = pool.drain();
        stats.recordDrain(drained);
        return drained;
    }

    public NativeCpuMemoryStats.Snapshot statsSnapshot() {
        return stats.snapshot();
    }

    public NativeMemoryPoolPolicy requestedPoolPolicy() {
        return config.poolPolicy();
    }

    public NativeMemoryPoolPolicy effectivePoolPolicy() {
        return effectivePoolPolicy;
    }

    private static long sizeClass(long byteSize) {
        long size = Math.max(1L, byteSize);
        long rounded = 64L;
        while (rounded < size && rounded < (1L << 62)) {
            rounded <<= 1;
        }
        return Math.max(rounded, size);
    }
}
