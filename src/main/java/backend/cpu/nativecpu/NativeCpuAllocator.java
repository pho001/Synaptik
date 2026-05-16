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
        this(config, stats, null);
    }

    public NativeCpuAllocator(
            NativeCpuMemoryConfig config,
            NativeCpuMemoryStats stats,
            NativeCpuMemoryPool preparedPool
    ) {
        this.stats = stats == null ? new NativeCpuMemoryStats() : stats;
        this.config = config == null ? NativeCpuMemoryConfig.disabled() : config;
        if (this.config.poolPolicy() == NativeMemoryPoolPolicy.PER_EXECUTION) {
            this.effectivePoolPolicy = NativeMemoryPoolPolicy.PER_EXECUTION;
            this.pool = new NativeCpuMemoryPool(this.config.maxPoolBytes());
        } else if (this.config.poolPolicy() == NativeMemoryPoolPolicy.PER_PREPARED_EXECUTION && preparedPool != null) {
            this.effectivePoolPolicy = NativeMemoryPoolPolicy.PER_PREPARED_EXECUTION;
            this.pool = preparedPool;
        } else {
            this.effectivePoolPolicy = NativeMemoryPoolPolicy.DISABLED;
            this.pool = null;
        }
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
        if (effectivePoolPolicy == NativeMemoryPoolPolicy.PER_EXECUTION
                || effectivePoolPolicy == NativeMemoryPoolPolicy.PER_PREPARED_EXECUTION) {
            return allocatePooled(byteSize, effectiveAlignment, label);
        }
        return allocateDirect(byteSize, effectiveAlignment, label);
    }

    private NativeCpuAllocation allocateDirect(long byteSize, long alignment, String label) {
        Arena arena = Arena.ofShared();
        try {
            long allocatedBytes = Math.max(byteSize, 1L);
            MemorySegment segment = arena.allocate(allocatedBytes, alignment);
            poisonAllocated(segment);
            stats.recordAllocation(byteSize, allocatedBytes);
            return new NativeCpuAllocation(
                    arena,
                    segment,
                    byteSize,
                    allocatedBytes,
                    alignment,
                    label,
                    stats,
                    config.debugPoisonReleasedBuffers()
            );
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
            poisonAllocated(block.segment());
            stats.recordAllocation(byteSize, allocatedBytes);
            stats.recordPoolHit(allocatedBytes);
            return new NativeCpuAllocation(block, pool, byteSize, label, stats, config.debugPoisonReleasedBuffers());
        }
        Arena arena = Arena.ofShared();
        try {
            MemorySegment segment = arena.allocate(allocatedBytes, alignment);
            poisonAllocated(segment);
            stats.recordAllocation(byteSize, allocatedBytes);
            stats.recordPoolMiss();
            return new NativeCpuAllocation(
                    new NativeCpuMemoryPool.Block(arena, segment, allocatedBytes, alignment),
                    pool,
                    byteSize,
                    label,
                    stats,
                    config.debugPoisonReleasedBuffers()
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

    public long drainRunLocalPool() {
        if (effectivePoolPolicy != NativeMemoryPoolPolicy.PER_EXECUTION) {
            return 0L;
        }
        return drainPool();
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

    private void poisonAllocated(MemorySegment segment) {
        if (config.debugPoisonReleasedBuffers() && segment != null) {
            segment.fill((byte) 0xAB);
        }
    }
}
