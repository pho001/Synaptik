package io.github.pho001.synaptik.backend.cpu.internal.memory;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import java.lang.foreign.ValueLayout;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * Constructs immutable Runtime representation recipes for the supported CPU lifecycle adapter.
 *
 * <p>The type owns no state. Its callbacks defer allocation and constant initialization until
 * Runtime creates a run, and every invocation returns a fresh run-owned representation. Recipe
 * construction itself performs no allocation or mutation.</p>
 */
public final class CpuRepresentationRecipes {
    /**
     * Numeric affine usable-byte declaration for the ordinary int-limited CPU batch domain.
     * The declaration remains run-bound even when bytesPerBatch is zero.
     * @param constantBytes non-negative constant usable bytes, divisible by elementByteWidth
     * @param bytesPerBatch non-negative usable bytes per row, divisible by elementByteWidth
     * @param elementByteWidth supported represented width: 1, 2, 4 or 8 bytes
     */
    public record BatchByteSize(long constantBytes, long bytesPerBatch, int elementByteWidth)
            implements PreparedMemoryPlan.ByteSize {
        /**
         * Validates immutable coefficients without evaluating a representative batch.
         * @param constantBytes non-negative whole-element constant bytes
         * @param bytesPerBatch non-negative whole-element bytes per batch row
         * @param elementByteWidth represented width in bytes
         * @throws IllegalArgumentException if a coefficient or width is invalid
         */
        public BatchByteSize {
            if (constantBytes < 0 || bytesPerBatch < 0
                    || (elementByteWidth != 1 && elementByteWidth != 2
                        && elementByteWidth != 4 && elementByteWidth != 8)
                    || constantBytes % elementByteWidth != 0
                    || bytesPerBatch % elementByteWidth != 0) {
                throw new IllegalArgumentException("invalid CPU batch byte coefficients");
            }
        }
        /** Retains run-bound declaration identity even when the affine slope is zero.
         * @return empty, including structurally run-bound constant-byte declarations */
        @Override public OptionalLong fixedByteSize() { return OptionalLong.empty(); }
        /**
         * Resolves the exact usable span with checked element and byte limits.
         * @param batchSize non-negative actual B, no greater than Integer.MAX_VALUE
         * @return non-negative exact usable bytes
         * @throws IllegalArgumentException if B is negative
         * @throws ArithmeticException if B, element count or byte arithmetic is unrepresentable
         */
        @Override public long byteSize(long batchSize) {
            if (batchSize < 0) throw new IllegalArgumentException("batchSize must be non-negative");
            Math.toIntExact(batchSize);
            long bytes = Math.addExact(constantBytes, Math.multiplyExact(bytesPerBatch, batchSize));
            Math.toIntExact(bytes / elementByteWidth);
            return bytes;
        }
    }

    /**
     * Source backing-size recipe retaining a proved canonical batch read layout.
     *
     * <p>The existing binding owns actual-B extent, stride, element-count and referenced-span
     * limits. This adapter only converts that span to bytes at the represented type's width.
     * It never samples B and remains structurally batch-bound even for permanent zero bytes.</p>
     *
     * @param binding non-null existing checked batch read binding
     * @param dataType non-null represented source type, with its actual byte width
     */
    public record BatchLayoutByteSize(CpuAccessPlan.BatchBinding binding, DataType dataType)
            implements PreparedMemoryPlan.ByteSize {
        /**
         * Retains immutable checked source geometry without resolving a representative batch.
         *
         * @param binding non-null existing read binding
         * @param dataType non-null represented source type
         * @throws NullPointerException if an argument is null
         * @throws IllegalArgumentException if {@code binding} is not a read binding
         */
        public BatchLayoutByteSize {
            Objects.requireNonNull(binding, "binding");
            Objects.requireNonNull(dataType, "dataType");
            if (binding.plan().accessKind() != CpuAccessPlan.AccessKind.READ) {
                throw new IllegalArgumentException("CPU source byte geometry requires a read binding");
            }
        }

        /**
         * Preserves structural invocation binding, including permanently empty layouts.
         *
         * @return empty for every source batch-layout recipe
         */
        @Override
        public OptionalLong fixedByteSize() {
            return OptionalLong.empty();
        }

        /**
         * Resolves the actual invocation's checked referenced span and converts it to bytes.
         *
         * @param batchSize non-negative actual B within the existing binding's int-limited domain
         * @return exact non-negative backing bytes, including zero for an empty canonical source
         * @throws IllegalArgumentException if {@code batchSize} is negative
         * @throws ArithmeticException if actual geometry exceeds existing CPU limits or byte
         *     multiplication overflows
         */
        @Override
        public long byteSize(long batchSize) {
            return Math.multiplyExact(
                    binding.bind(batchSize).referencedElementSpan(), dataType.byteWidth());
        }
    }

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
        entry.size().fixedByteSize().ifPresent(bytes -> requireElementGeometry(type, bytes));
        return (bytes, alignment) -> {
            requireElementGeometry(type, bytes);
            requireRequest(entry.size(), entry.byteAlignment(), bytes, alignment);
            return CpuNativeBuffer.allocate(type, bytes, alignment);
        };
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
        Objects.requireNonNull(entry, "entry");
        entry.size().fixedByteSize().ifPresent(bytes -> requireElementGeometry(type, bytes));
        return (bytes, alignment) -> {
            requireElementGeometry(type, bytes);
            requireRequest(entry.size(), entry.byteAlignment(), bytes, alignment);
            return initialize(CpuNativeBuffer.allocate(type, bytes, alignment), value);
        };
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
        return (bytes, alignment) -> {
            requireRequest(entry.size(), entry.byteAlignment(), bytes, alignment);
            return CpuContiguousWorkspace.allocate(bytes, alignment);
        };
    }

    private static void requireRequest(PreparedMemoryPlan.ByteSize size, long expectedAlignment,
            long bytes, long alignment) {
        if (bytes < 0 || alignment != expectedAlignment
                || size.fixedByteSize().isPresent() && size.fixedByteSize().getAsLong() != bytes) {
            throw new IllegalArgumentException("creator request differs from prepared declaration");
        }
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
                    case FLOAT16 -> segment.setAtIndex(ValueLayout.JAVA_SHORT, index,
                            value.float16Bits());
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
        if (byteSize < 0 || byteSize % type.byteWidth() != 0) {
            throw new IllegalArgumentException(
                    "CPU buffer byte size is not a complete element multiple");
        }
    }
}
