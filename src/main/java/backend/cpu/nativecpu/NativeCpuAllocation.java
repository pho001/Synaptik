package backend.cpu.nativecpu;

import tensor.storage.NativeMemoryAllocation;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Native CPU memory allocation owned by a prepared execution run or a test.
 *
 * <p>The allocation owns its {@link Arena}; closing the allocation releases every segment allocated from
 * that arena. Instances are idempotent {@link ExecutionResource}s so they can be registered with
 * {@link graph.execution.state.ExecutionState} and closed from a {@code finally} block.</p>
 */
public final class NativeCpuAllocation implements NativeMemoryAllocation {
    private final Arena arena;
    private final MemorySegment segment;
    private final NativeCpuMemoryPool.Block poolBlock;
    private final NativeCpuMemoryPool pool;
    private final long byteSize;
    private final long allocatedBytes;
    private final long alignment;
    private final String label;
    private final NativeCpuMemoryStats stats;
    private final boolean poisonReleasedBuffers;
    private final AtomicBoolean released = new AtomicBoolean();
    private final AtomicBoolean retained = new AtomicBoolean();
    private volatile String retainedReason;

    NativeCpuAllocation(Arena arena, MemorySegment segment, long byteSize, long alignment, String label) {
        this(arena, segment, byteSize, Math.max(byteSize, 1L), alignment, label, new NativeCpuMemoryStats(), false);
    }

    NativeCpuAllocation(
            Arena arena,
            MemorySegment segment,
            long byteSize,
            long allocatedBytes,
            long alignment,
            String label,
            NativeCpuMemoryStats stats,
            boolean poisonReleasedBuffers
    ) {
        this.arena = Objects.requireNonNull(arena, "arena cannot be null");
        this.segment = Objects.requireNonNull(segment, "segment cannot be null");
        this.poolBlock = null;
        this.pool = null;
        this.byteSize = byteSize;
        this.allocatedBytes = Math.max(0L, allocatedBytes);
        this.alignment = alignment;
        this.label = label == null ? "" : label;
        this.stats = stats == null ? new NativeCpuMemoryStats() : stats;
        this.poisonReleasedBuffers = poisonReleasedBuffers;
    }

    NativeCpuAllocation(
            NativeCpuMemoryPool.Block poolBlock,
            NativeCpuMemoryPool pool,
            long byteSize,
            String label,
            NativeCpuMemoryStats stats,
            boolean poisonReleasedBuffers
    ) {
        this.arena = null;
        this.segment = Objects.requireNonNull(poolBlock, "poolBlock cannot be null").segment();
        this.poolBlock = poolBlock;
        this.pool = Objects.requireNonNull(pool, "pool cannot be null");
        this.byteSize = byteSize;
        this.allocatedBytes = poolBlock.allocatedBytes();
        this.alignment = poolBlock.alignment();
        this.label = label == null ? "" : label;
        this.stats = stats == null ? new NativeCpuMemoryStats() : stats;
        this.poisonReleasedBuffers = poisonReleasedBuffers;
    }

    public MemorySegment segment() {
        ensureOpen();
        return segment;
    }

    public long byteSize() {
        return byteSize;
    }

    public long allocatedBytes() {
        return allocatedBytes;
    }

    public long alignment() {
        return alignment;
    }

    public String label() {
        return label;
    }

    public boolean closed() {
        return released();
    }

    public boolean released() {
        return released.get();
    }

    public boolean retainedAfterExecute() {
        return retained.get();
    }

    public String retainedReason() {
        return retainedReason == null ? "" : retainedReason;
    }

    public void retain(String reason) {
        ensureOpen();
        if (retained.compareAndSet(false, true)) {
            retainedReason = reason == null ? "" : reason;
            stats.recordRetain(allocatedBytes);
        }
    }

    public void ensureOpen() {
        if (released.get()) {
            throw new IllegalStateException("Native CPU allocation is closed: " + label);
        }
    }

    public void release() {
        if (released.compareAndSet(false, true)) {
            boolean wasRetained = retained.get();
            try {
                poisonReleased();
                if (poolBlock != null && !wasRetained && pool.release(poolBlock)) {
                    stats.recordPooledRelease(allocatedBytes);
                } else {
                    if (poolBlock != null) {
                        poolBlock.close();
                    } else {
                        arena.close();
                    }
                    stats.recordDiscarded(allocatedBytes);
                }
            } finally {
                stats.recordRelease(allocatedBytes, wasRetained);
            }
        }
    }

    @Override
    public void close() {
        release();
    }

    private void poisonReleased() {
        if (poisonReleasedBuffers) {
            segment.fill((byte) 0xCD);
        }
    }
}
