package io.github.pho001.synaptik.backend.cpu.internal.memory;

import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Run-owned aligned native CPU scratch workspace.
 *
 * <p>Each instance owns one shared arena and exactly one writable segment so an invoking thread
 * may complete a selected materialization before workers consume it, or disjoint selected
 * scatter ranges may reuse separate exact-product slices. Closing is idempotent and ends
 * accessibility; the workspace never owns input buffers, workers, or generated artifacts.
 */
public final class CpuContiguousWorkspace implements WorkspaceRepresentation {
    private final Arena arena;
    private final MemorySegment segment;
    private final long byteAlignment;
    private final AtomicBoolean closed = new AtomicBoolean();
    private CpuContiguousWorkspace(Arena arena, MemorySegment segment, long byteAlignment) {
        this.arena = arena; this.segment = segment; this.byteAlignment = byteAlignment;
    }
    /**
     * Allocates one shared-arena workspace, including a valid zero-byte workspace.
     *
     * @param byteSize non-negative requested size in bytes
     * @param byteAlignment positive power-of-two byte alignment
     * @return a new open run-owned workspace; never {@code null}
     * @throws IllegalArgumentException if size is negative or alignment is not a positive power
     *     of two
     * @throws RuntimeException if native allocation fails
     */
    public static CpuContiguousWorkspace allocate(long byteSize, long byteAlignment) {
        if (byteSize < 0 || byteAlignment <= 0 || (byteAlignment & (byteAlignment - 1)) != 0) {
            throw new IllegalArgumentException("invalid workspace geometry");
        }
        Arena arena = Arena.ofShared();
        try { return new CpuContiguousWorkspace(arena,
                arena.allocate(byteSize, byteAlignment), byteAlignment); }
        catch (RuntimeException | Error failure) { try { arena.close(); } catch (Throwable cleanup) {
            if (cleanup != failure) failure.addSuppressed(cleanup); } throw failure; }
    }
    /**
     * Returns the allocated size.
     *
     * @return the exact non-negative allocated size in bytes
     */
    public long byteSize() { return segment.byteSize(); }
    /**
     * Returns the allocation alignment.
     *
     * @return the exact positive power-of-two alignment in bytes
     */
    public long byteAlignment() { return byteAlignment; }
    /**
     * Reports current-thread access.
     *
     * @return whether the segment is live and accessible
     */
    public boolean isAccessible() { return !closed.get() && segment.scope().isAlive()
            && segment.isAccessibleBy(Thread.currentThread()); }
    /**
     * Returns the exact writable workspace segment for CPU-private cold binding.
     *
     * @return the live writable segment; never {@code null}
     * @throws IllegalStateException if this workspace is closed or inaccessible to the current
     *     thread
     */
    public MemorySegment writableSegment() {
        if (!isAccessible()) throw new IllegalStateException("workspace is not accessible");
        return segment;
    }
    /** Closes the owned arena at most once. */
    @Override public void close() { if (closed.compareAndSet(false, true)) arena.close(); }
}
