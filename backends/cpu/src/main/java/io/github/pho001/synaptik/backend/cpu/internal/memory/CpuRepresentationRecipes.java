package io.github.pho001.synaptik.backend.cpu.internal.memory;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Constructs immutable Runtime representation recipes for the supported CPU lifecycle adapter.
 *
 * <p>The type owns no state. Its callbacks defer allocation and constant initialization until
 * Runtime creates a run, and every invocation returns a fresh run-owned representation. Recipe
 * construction itself performs no allocation or mutation.</p>
 */
public final class CpuRepresentationRecipes {
    /** Prevents construction of this field-free recipe namespace. */
    private CpuRepresentationRecipes() {}

    /**
     * Creates a deferred recipe for fresh aligned native CPU buffer storage.
     *
     * @param type the non-null represented element type
     * @param entry the non-null prepared buffer geometry; its byte size must contain a whole
     *     number of {@code type} elements
     * @return a non-null immutable creator whose invocations return distinct run-owned buffers
     * @throws NullPointerException if {@code type} or {@code entry} is {@code null}
     * @throws IllegalArgumentException if the prepared byte size is not a multiple of the type's
     *     byte width
     */
    public static PreparedRepresentationPlan.BufferCreator buffer(DataType type,
            PreparedMemoryPlan.BufferEntry entry) {
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(entry, "entry");
        requireElementGeometry(type, entry.byteSize());
        return () -> CpuNativeBuffer.allocate(type, entry.byteSize(), entry.byteAlignment());
    }

    /**
     * Creates a deferred recipe for a fresh aligned CPU buffer initialized as a scalar splat.
     *
     * <p>Each invocation writes the exact represented scalar value to every complete element of
     * its new buffer. If initialization fails, the new buffer is closed before the failure is
     * propagated.</p>
     *
     * @param type the non-null represented element type
     * @param entry the non-null prepared buffer geometry; its byte size must contain a whole
     *     number of {@code type} elements
     * @param value the non-null scalar whose data type must equal {@code type}
     * @return a non-null immutable creator whose invocations return distinct initialized
     *     run-owned buffers
     * @throws NullPointerException if {@code entry} or {@code value} is {@code null}
     * @throws IllegalArgumentException if {@code type} is {@code null}, the scalar type differs
     *     from {@code type}, or the prepared byte size is not a multiple of the type's byte width
     */
    public static PreparedRepresentationPlan.BufferCreator initializedBuffer(DataType type,
            PreparedMemoryPlan.BufferEntry entry, ScalarValue value) {
        Objects.requireNonNull(value, "value");
        if (value.dataType() != type) {
            throw new IllegalArgumentException("constant scalar type disagrees with CPU buffer");
        }
        requireElementGeometry(type, entry.byteSize());
        return () -> initialize(CpuNativeBuffer.allocate(type, entry.byteSize(),
                entry.byteAlignment()), value);
    }

    /**
     * Creates a deferred recipe for fresh aligned native CPU workspace storage.
     *
     * @param entry the non-null prepared workspace byte size and alignment
     * @return a non-null immutable creator whose invocations return distinct run-owned workspaces
     * @throws NullPointerException if {@code entry} is {@code null}
     */
    public static PreparedRepresentationPlan.WorkspaceCreator workspace(
            PreparedMemoryPlan.WorkspaceEntry entry) {
        Objects.requireNonNull(entry, "entry");
        return () -> CpuContiguousWorkspace.allocate(entry.byteSize(), entry.byteAlignment());
    }

    /**
     * Fills a newly allocated buffer and closes it if initialization fails.
     *
     * @param buffer the non-null newly allocated buffer owned by this operation
     * @param value the non-null matching scalar value to write into every element
     * @return the same initialized {@code buffer} reference, with ownership passed to the caller
     * @throws RuntimeException if represented-value access or initialization fails
     * @throws Error if initialization or failure cleanup reports an error
     */
    private static CpuNativeBuffer initialize(CpuNativeBuffer buffer, ScalarValue value) {
        try {
            long count = buffer.byteSize() / buffer.dataType().byteWidth();
            var segment = buffer.segment();
            for (long index = 0; index < count; index++) {
                switch (buffer.dataType()) {
                    case FLOAT64 -> segment.setAtIndex(ValueLayout.JAVA_DOUBLE, index,
                            value.float64Value());
                    case FLOAT32 -> segment.setAtIndex(ValueLayout.JAVA_FLOAT, index,
                            value.float32Value());
                    case BFLOAT16 -> segment.setAtIndex(ValueLayout.JAVA_SHORT, index,
                            value.bfloat16Bits());
                    case INT64 -> segment.setAtIndex(ValueLayout.JAVA_LONG, index,
                            value.int64Value());
                    case INT32 -> segment.setAtIndex(ValueLayout.JAVA_INT, index,
                            value.int32Value());
                    case BOOL -> segment.setAtIndex(ValueLayout.JAVA_BYTE, index,
                            (byte) (value.booleanValue() ? 1 : 0));
                }
            }
            return buffer;
        } catch (RuntimeException | Error failure) {
            try {
                buffer.close();
            } catch (RuntimeException | Error cleanup) {
                if (cleanup != failure) failure.addSuppressed(cleanup);
            }
            throw failure;
        }
    }

    /**
     * Validates that a byte extent represents complete elements of one type.
     *
     * @param type the non-null represented element type
     * @param byteSize the non-negative prepared byte extent
     * @throws IllegalArgumentException if {@code byteSize} is not a multiple of the type width
     */
    private static void requireElementGeometry(DataType type, long byteSize) {
        if (byteSize % type.byteWidth() != 0) {
            throw new IllegalArgumentException(
                    "CPU buffer byte size is not a complete element multiple");
        }
    }
}
