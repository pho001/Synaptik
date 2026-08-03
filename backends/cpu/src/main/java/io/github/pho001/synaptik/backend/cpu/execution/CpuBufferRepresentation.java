package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Common non-public CPU representation metadata and its once-classified direct argument.
 *
 * <p>The representation retains exact data type, byte size, and segment identity. Construction
 * performs the cold storage classification once. Accessors remain valid only while the concrete
 * owning or borrowed representation is open and accessible to the calling thread.</p>
 */
abstract class CpuBufferRepresentation implements BufferRepresentation {
    private final DataType dataType;
    private final long byteSize;
    private final MemorySegment segment;
    private final CpuBufferArgument argument;

    /**
     * Retains and classifies one exact segment without copying it.
     *
     * @param dataType non-null logical data type
     * @param byteSize exact selected size in bytes
     * @param segment non-null exact selected segment or slice
     * @throws NullPointerException if {@code dataType} or {@code segment} is {@code null}
     * @throws IllegalArgumentException if segment geometry cannot form a valid direct argument
     * @throws IllegalStateException if the segment is not alive or accessible to this thread
     */
    CpuBufferRepresentation(DataType dataType, long byteSize, MemorySegment segment) {
        this.dataType = Objects.requireNonNull(dataType, "dataType");
        this.segment = Objects.requireNonNull(segment, "segment");
        this.byteSize = byteSize;
        CpuBufferArgument classified;
        try {
            classified = classify(dataType, segment, byteSize);
        } catch (IllegalArgumentException incompatibleCarrier) {
            classified = null;
        }
        this.argument = classified;
    }

    /** @return retained non-null logical data type */
    final DataType dataType() { return dataType; }

    /** @return exact non-negative byte size */
    final long byteSize() { return byteSize; }

    /** @return exact retained segment, while this representation is open */
    final MemorySegment segment() { ensureOpen(); return segment; }

    /**
     * Returns the direct typed argument classified from the exact retained segment.
     *
     * @return non-null array or exact-segment argument; owns no memory
     * @throws IllegalStateException if the representation is closed
     * @throws IllegalArgumentException if an observable heap carrier conflicts with the data type
     */
    final CpuBufferArgument argument() {
        ensureOpen();
        if (argument == null) {
            throw new IllegalArgumentException("heap carrier is incompatible with data type");
        }
        return argument;
    }

    /** @return whether this representation can no longer be accessed */
    abstract boolean isClosed();

    /** @return whether the exact segment is open, size-consistent, and accessible to this thread */
    final boolean isAccessible() {
        return !isClosed() && segment.scope().isAlive()
                && segment.isAccessibleBy(Thread.currentThread()) && segment.byteSize() == byteSize;
    }

    /** @throws IllegalStateException if this representation or its segment scope is closed */
    final void ensureOpen() {
        if (isClosed() || !segment.scope().isAlive()) {
            throw new IllegalStateException("CPU representation is closed");
        }
    }

    /** Classifies one exact segment as an observable typed carrier or exact-segment argument. */
    private static CpuBufferArgument classify(DataType type, MemorySegment segment, long size) {
        Object base = segment.heapBase().orElse(null);
        boolean readOnly = segment.isReadOnly();
        if (base == null) return new CpuBufferArgument.Segment(type, segment, size, readOnly);
        long offset = segment.address();
        return switch (type) {
            case FLOAT64 -> base instanceof double[] values
                    ? new CpuBufferArgument.Doubles(values, offset, size, readOnly) : mismatch();
            case FLOAT32 -> base instanceof float[] values
                    ? new CpuBufferArgument.Floats(values, offset, size, readOnly) : mismatch();
            case BFLOAT16 -> base instanceof short[] values
                    ? new CpuBufferArgument.Shorts(values, offset, size, readOnly) : mismatch();
            case INT32 -> base instanceof int[] values
                    ? new CpuBufferArgument.Ints(values, offset, size, readOnly) : mismatch();
            case INT64 -> base instanceof long[] values
                    ? new CpuBufferArgument.Longs(values, offset, size, readOnly) : mismatch();
            case BOOL -> base instanceof byte[] values
                    ? new CpuBufferArgument.Bytes(values, offset, size, readOnly) : mismatch();
        };
    }

    /** Throws the stable cold-classification failure for an observable wrong carrier. */
    private static CpuBufferArgument mismatch() {
        throw new IllegalArgumentException("heap carrier is incompatible with data type");
    }
}
