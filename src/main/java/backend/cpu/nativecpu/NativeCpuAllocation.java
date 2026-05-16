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
    private final long alignment;
    private final String label;
    private final AtomicBoolean closed = new AtomicBoolean();

    NativeCpuAllocation(Arena arena, MemorySegment segment, long byteSize, long alignment, String label) {
        this.arena = Objects.requireNonNull(arena, "arena cannot be null");
        this.segment = Objects.requireNonNull(segment, "segment cannot be null");
        this.byteSize = byteSize;
        this.alignment = alignment;
        this.label = label == null ? "" : label;
    }

    public MemorySegment segment() {
        ensureOpen();
        return segment;
    }

    public long byteSize() {
        return byteSize;
    }

    public long alignment() {
        return alignment;
    }

    public String label() {
        return label;
    }

    public boolean closed() {
        return closed.get();
    }

    public void ensureOpen() {
        if (closed.get()) {
            throw new IllegalStateException("Native CPU allocation is closed: " + label);
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            arena.close();
        }
    }
}
