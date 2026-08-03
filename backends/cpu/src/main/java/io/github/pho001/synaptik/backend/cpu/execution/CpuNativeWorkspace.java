package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Run-owned aligned native CPU scratch workspace backed by one distinct shared arena.
 *
 * <p>The workspace has physical byte geometry but no logical buffer-validity state. Runtime owns
 * cleanup orchestration for a successful run-state construction; this representation owns only
 * its arena and exact native segment.</p>
 */
final class CpuNativeWorkspace implements WorkspaceRepresentation {
    private final long byteSize;
    private final long byteAlignment;
    private final Arena arena;
    private final MemorySegment segment;
    private final AtomicBoolean closed = new AtomicBoolean();

    private CpuNativeWorkspace(long size, long alignment, Arena arena, MemorySegment segment) {
        this.byteSize = size; this.byteAlignment = alignment; this.arena = arena; this.segment = segment;
    }

    /**
     * Allocates one exact-size aligned segment in a distinct shared arena.
     *
     * @param byteSize non-negative allocation size in bytes; zero remains a live workspace
     * @param byteAlignment positive power-of-two alignment in bytes
     * @return a new open run-owned workspace; never {@code null}
     * @throws IllegalArgumentException if size or alignment is invalid
     */
    static CpuNativeWorkspace allocate(long byteSize, long byteAlignment) {
        if (byteSize < 0) throw new IllegalArgumentException("byteSize must be non-negative");
        if (byteAlignment <= 0 || (byteAlignment & (byteAlignment - 1)) != 0) {
            throw new IllegalArgumentException("byteAlignment must be a positive power of two");
        }
        Arena arena = Arena.ofShared();
        try {
            return new CpuNativeWorkspace(byteSize, byteAlignment, arena,
                    arena.allocate(byteSize, byteAlignment));
        } catch (RuntimeException | Error failure) {
            try { arena.close(); }
            catch (RuntimeException | Error cleanup) { if (cleanup != failure) failure.addSuppressed(cleanup); }
            throw failure;
        }
    }

    /** @return exact non-negative workspace size in bytes */
    long byteSize() { return byteSize; }
    /** @return requested positive power-of-two alignment in bytes */
    long byteAlignment() { return byteAlignment; }
    /** @return exact retained segment; never {@code null}
     * @throws IllegalStateException if this workspace is closed */
    MemorySegment segment() { ensureOpen(); return segment; }
    /** @return whether close has begun; safe to query concurrently */
    boolean isClosed() { return closed.get(); }
    /** @return whether this workspace is open and accessible to the calling thread */
    boolean isAccessible() { return !isClosed() && segment.scope().isAlive()
            && segment.isAccessibleBy(Thread.currentThread()); }
    /** @throws IllegalStateException if this workspace or its arena scope is closed */
    private void ensureOpen() { if (isClosed() || !segment.scope().isAlive())
        throw new IllegalStateException("CPU representation is closed"); }

    /**
     * Closes the owning arena at most once.
     * Repeated and concurrent calls are inert after the first attempt; an unchecked arena-close
     * failure propagates unchanged and is never retried.
     */
    @Override public void close() { if (closed.compareAndSet(false, true)) arena.close(); }
}
