package io.github.pho001.synaptik.backend.cpu.internal.memory;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Copies supported CPU physical representations into detached canonical host snapshots.
 *
 * <p>This cold-path owner validates the complete provable representation and descriptor geometry
 * before allocation or source access. It has no mutable state, retains no source or result, and
 * performs no Runtime, route, provider, schedule, or ownership operation. Segment-backed source
 * elements are read in native byte order, while every result uses canonical row-major logical
 * traversal and fixed big-endian encoding.</p>
 */
public final class CpuHostSnapshotExporter {
    private static final ValueLayout.OfDouble NATIVE_DOUBLE =
            ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfFloat NATIVE_FLOAT =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfShort NATIVE_SHORT =
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfInt NATIVE_INT =
            ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfLong NATIVE_LONG =
            ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.nativeOrder());

    private CpuHostSnapshotExporter() {
    }

    /**
     * Validates and copies one exact current CPU representation.
     *
     * <p>The fully static resolved layout supplies a non-negative storage offset and positive or
     * zero element strides. Rank zero reads its offset element; a zero-element shape returns a fresh
     * empty array without accessing an element. All six current data types preserve represented
     * bits without conversion, including floating NaN payloads and signed zeros; BOOL additionally
     * requires each logical value to be exactly {@code 0} or {@code 1}.</p>
     *
     * <p>The call is synchronous and changes no source ownership, validity, content, or lifetime.
     * The caller must retain the source owner's lease, keep the representation open and accessible
     * to this thread, and prevent mutation or closure until return. Independent valid calls share no
     * exporter state, but concurrent mutation has no atomic-snapshot guarantee.</p>
     *
     * @param representation non-null nominal representation whose exact concrete class must be
     *     {@link CpuBorrowedBuffer} or {@link CpuNativeBuffer}; inspected but not retained, closed,
     *     or mutated, and ownership remains unchanged
     * @param descriptor non-null fully static descriptor with a present resolved layout and a data
     *     type equal to the representation type; inspected but not retained or mutated
     * @param maximumBytes non-negative caller payload limit in bytes
     * @return fresh non-null caller-owned mutable bytes in canonical row-major logical order and
     *     fixed big-endian element encoding
     * @throws NullPointerException if an object argument is {@code null}
     * @throws IllegalStateException if the representation is closed or inaccessible to this thread
     * @throws IllegalArgumentException if the byte limit is negative; the shape is not fully static
     *     or the layout is unresolved; the result exceeds the limit or JVM array ceiling; the
     *     representation class, data type, element geometry, carrier, or capacity is incompatible;
     *     or a BOOL element is not canonical
     * @throws ArithmeticException if checked element-count, byte-count, source-address, or
     *     destination-address arithmetic overflows
     * @throws OutOfMemoryError if the JVM cannot allocate the otherwise valid result array
     */
    public static byte[] copy(BufferRepresentation representation, TensorDescriptor descriptor,
            long maximumBytes) {
        Objects.requireNonNull(representation, "representation");
        Objects.requireNonNull(descriptor, "descriptor");
        if (maximumBytes < 0) {
            throw new IllegalArgumentException(
                    "maximumBytes must be non-negative: " + maximumBytes);
        }
        if (!descriptor.shape().isFullyStatic()) {
            throw new IllegalArgumentException(
                    "host snapshot requires a fully static shape: " + descriptor.shape());
        }
        LayoutDescriptor layout = descriptor.layout().orElseThrow(() ->
                new IllegalArgumentException("host snapshot requires a resolved layout"));
        long elementCount = descriptor.shape().knownElementCount().orElseThrow();
        int width = descriptor.dataType().byteWidth();
        long byteCount = Math.multiplyExact(elementCount, width);
        if (byteCount > maximumBytes) {
            throw new IllegalArgumentException("canonical byte count exceeds maximumBytes: required="
                    + byteCount + ", maximum=" + maximumBytes);
        }
        if (byteCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "canonical byte count exceeds JVM byte[] limit: " + byteCount);
        }
        if (representation.getClass() != CpuBorrowedBuffer.class
                && representation.getClass() != CpuNativeBuffer.class) {
            throw new IllegalArgumentException(
                    "representation is not a supported CPU buffer representation");
        }

        CpuBufferRepresentation cpu = (CpuBufferRepresentation) representation;
        cpu.ensureOpen();
        if (!cpu.isAccessible()) {
            throw new IllegalStateException(
                    "CPU representation is not accessible to current thread");
        }
        if (cpu.dataType() != descriptor.dataType()) {
            throw new IllegalArgumentException(
                    "CPU representation data type differs from descriptor: representation="
                            + cpu.dataType() + ", descriptor=" + descriptor.dataType());
        }
        if (cpu.byteSize() % width != 0) {
            throw new IllegalArgumentException(
                    "CPU representation byte size is not a complete element multiple");
        }
        long capacity = cpu.byteSize() / width;
        long requiredSpan = layout.referencedElementSpan();
        if (capacity < requiredSpan) {
            throw new IllegalArgumentException(
                    "CPU representation capacity is smaller than descriptor span: capacity="
                            + capacity + ", required=" + requiredSpan);
        }
        CpuBufferArgument argument = cpu.argument();

        byte[] result = new byte[Math.toIntExact(byteCount)];
        if (elementCount == 0) {
            return result;
        }
        if (layout.kind() == LayoutKind.DENSE_CONTIGUOUS && layout.storageOffset() == 0) {
            for (long logicalIndex = 0; logicalIndex < elementCount; logicalIndex++) {
                encode(argument, descriptor.dataType(), logicalIndex, logicalIndex, result);
            }
            return result;
        }

        long[] extents = descriptor.shape().toLongArray();
        long[] strides = layout.strides();
        long[] coordinates = new long[extents.length];
        for (long logicalIndex = 0; logicalIndex < elementCount; logicalIndex++) {
            long sourceIndex = layout.storageOffset();
            for (int axis = 0; axis < coordinates.length; axis++) {
                sourceIndex = Math.addExact(sourceIndex,
                        Math.multiplyExact(coordinates[axis], strides[axis]));
            }
            encode(argument, descriptor.dataType(), sourceIndex, logicalIndex, result);
            increment(coordinates, extents);
        }
        return result;
    }

    /** Advances one row-major coordinate, with the final axis changing fastest. */
    private static void increment(long[] coordinates, long[] extents) {
        for (int axis = coordinates.length - 1; axis >= 0; axis--) {
            long next = Math.addExact(coordinates[axis], 1);
            if (next < extents[axis]) {
                coordinates[axis] = next;
                return;
            }
            coordinates[axis] = 0;
        }
    }

    /** Reads represented bits and writes one canonical element at its checked destination. */
    private static void encode(CpuBufferArgument argument, DataType type, long sourceIndex,
            long logicalIndex, byte[] destination) {
        long bits = readBits(argument, type, sourceIndex);
        if (type == DataType.BOOL && bits > 1) {
            throw new IllegalArgumentException(
                    "BOOL representation contains non-canonical byte at logical index "
                            + logicalIndex + ": " + bits);
        }
        int width = type.byteWidth();
        int destinationOffset = Math.toIntExact(Math.multiplyExact(logicalIndex, width));
        for (int byteIndex = width - 1; byteIndex >= 0; byteIndex--) {
            destination[Math.addExact(destinationOffset, byteIndex)] = (byte) bits;
            bits >>>= Byte.SIZE;
        }
    }

    /** Reads one element through the once-classified direct carrier using checked byte addressing. */
    private static long readBits(CpuBufferArgument argument, DataType type, long elementIndex) {
        int width = type.byteWidth();
        long relativeBytes = Math.multiplyExact(elementIndex, width);
        if (argument instanceof CpuBufferArgument.Segment segment) {
            MemorySegment memory = segment.segment();
            return switch (type) {
                case FLOAT64 -> Double.doubleToRawLongBits(memory.get(NATIVE_DOUBLE, relativeBytes));
                case FLOAT32 -> Integer.toUnsignedLong(
                        Float.floatToRawIntBits(memory.get(NATIVE_FLOAT, relativeBytes)));
                case BFLOAT16 -> Short.toUnsignedLong(memory.get(NATIVE_SHORT, relativeBytes));
                case INT64 -> memory.get(NATIVE_LONG, relativeBytes);
                case INT32 -> Integer.toUnsignedLong(memory.get(NATIVE_INT, relativeBytes));
                case BOOL -> Byte.toUnsignedLong(memory.get(ValueLayout.JAVA_BYTE, relativeBytes));
            };
        }

        long carrierBytes = Math.addExact(argument.byteOffset(), relativeBytes);
        return switch (type) {
            case FLOAT64 -> Double.doubleToRawLongBits(((CpuBufferArgument.Doubles) argument)
                    .carrier()[arrayIndex(carrierBytes, width)]);
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(
                    ((CpuBufferArgument.Floats) argument)
                            .carrier()[arrayIndex(carrierBytes, width)]));
            case BFLOAT16 -> Short.toUnsignedLong(((CpuBufferArgument.Shorts) argument)
                    .carrier()[arrayIndex(carrierBytes, width)]);
            case INT64 -> ((CpuBufferArgument.Longs) argument)
                    .carrier()[arrayIndex(carrierBytes, width)];
            case INT32 -> Integer.toUnsignedLong(((CpuBufferArgument.Ints) argument)
                    .carrier()[arrayIndex(carrierBytes, width)]);
            case BOOL -> Byte.toUnsignedLong(((CpuBufferArgument.Bytes) argument)
                    .carrier()[arrayIndex(carrierBytes, width)]);
        };
    }

    /** Converts an aligned checked carrier-relative byte address to a Java array index. */
    private static int arrayIndex(long byteAddress, int width) {
        return Math.toIntExact(byteAddress / width);
    }
}
