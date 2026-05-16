package backend.cpu.nativecpu;

import backend.memory.ExecutionResource;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Native CPU memory allocation owned by a prepared execution run or a test.
 *
 * <p>The allocation owns its {@link Arena}; closing the allocation releases every segment allocated from
 * that arena. Instances are idempotent {@link ExecutionResource}s so they can be registered with
 * {@link graph.execution.ExecutionState} and closed from a {@code finally} block.</p>
 */
public final class NativeCpuAllocation implements ExecutionResource {
    private final Arena arena;
    private final MemorySegment segment;
    private final long byteSize;
    private final long allocatedBytes;
    private final long alignment;
    private final String label;
    private final NativeCpuMemoryStats stats;
    private final AtomicBoolean released = new AtomicBoolean();
    private final AtomicBoolean retained = new AtomicBoolean();
    private volatile String retainedReason;

    NativeCpuAllocation(Arena arena, MemorySegment segment, long byteSize, long alignment, String label) {
        this(arena, segment, byteSize, Math.max(byteSize, 1L), alignment, label, new NativeCpuMemoryStats());
    }

    NativeCpuAllocation(
            Arena arena,
            MemorySegment segment,
            long byteSize,
            long allocatedBytes,
            long alignment,
            String label,
            NativeCpuMemoryStats stats
    ) {
        this.arena = Objects.requireNonNull(arena, "arena cannot be null");
        this.segment = Objects.requireNonNull(segment, "segment cannot be null");
        this.byteSize = byteSize;
        this.allocatedBytes = Math.max(0L, allocatedBytes);
        this.alignment = alignment;
        this.label = label == null ? "" : label;
        this.stats = stats == null ? new NativeCpuMemoryStats() : stats;
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
            try {
                arena.close();
            } finally {
                stats.recordRelease(allocatedBytes, retained.get());
            }
        }
    }

    @Override
    public void close() {
        release();
    }
}
