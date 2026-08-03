package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Run-owned aligned native CPU buffer backed by one distinct shared arena.
 *
 * <p>The shared arena permits access from the orchestrating thread, CPU workers, and compatible
 * Foreign Function and Memory calls during the run. The representation owns the arena and closes
 * it at most once; no cleaner, pooling, or ownership transfer exists here.</p>
 */
final class CpuNativeBuffer extends CpuBufferRepresentation {
    private final Arena arena;
    private final long byteAlignment;
    private final AtomicBoolean closed = new AtomicBoolean();

    private CpuNativeBuffer(DataType type, long size, long alignment, Arena arena, MemorySegment segment) {
        super(type, size, segment);
        this.byteAlignment = alignment;
        this.arena = arena;
    }

    /**
     * Allocates one exact-size aligned segment in a distinct shared arena.
     *
     * @param dataType non-null logical data type retained by the representation
     * @param byteSize non-negative allocation size in bytes; zero is a real allocation geometry
     * @param byteAlignment positive power-of-two alignment in bytes
     * @return a new open run-owned representation; never {@code null}
     * @throws NullPointerException if {@code dataType} is {@code null}, after geometry validation
     * @throws IllegalArgumentException if size or alignment is invalid
     */
    static CpuNativeBuffer allocate(DataType dataType, long byteSize, long byteAlignment) {
        NativeAllocation.checkGeometry(byteSize, byteAlignment);
        Objects.requireNonNull(dataType, "dataType");
        Arena arena = Arena.ofShared();
        try {
            MemorySegment segment = arena.allocate(byteSize, byteAlignment);
            return new CpuNativeBuffer(dataType, byteSize, byteAlignment, arena, segment);
        } catch (RuntimeException | Error failure) {
            NativeAllocation.closeAfterFailure(arena, failure);
            throw failure;
        }
    }

    /** @return the positive power-of-two requested allocation alignment in bytes */
    long byteAlignment() { return byteAlignment; }

    /** @return whether close has begun; safe to query concurrently */
    @Override boolean isClosed() { return closed.get(); }

    /**
     * Closes the owning arena at most once, with closed state visible before physical close.
     * Repeated and concurrent calls are inert after the first attempt; an unchecked arena-close
     * failure propagates unchanged and is never retried.
     */
    @Override public void close() { if (closed.compareAndSet(false, true)) arena.close(); }

    /** Shared validation and partial-failure cleanup mechanics for native buffer allocation. */
    private static final class NativeAllocation {
        /** Validates exact byte geometry before any arena is created. */
        static void checkGeometry(long size, long alignment) {
            if (size < 0) throw new IllegalArgumentException("byteSize must be non-negative");
            if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
                throw new IllegalArgumentException("byteAlignment must be a positive power of two");
            }
        }

        /** Closes a created arena while preserving the exact primary and distinct cleanup failure. */
        static void closeAfterFailure(Arena arena, Throwable primary) {
            try { arena.close(); }
            catch (RuntimeException | Error cleanup) { if (cleanup != primary) primary.addSuppressed(cleanup); }
        }
    }
}
