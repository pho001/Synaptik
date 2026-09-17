package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * Immutable detached host snapshot of one materialized publication occurrence.
 *
 * <p>The value retains the exact logical data type and immutable Shape from the final publication
 * descriptor plus canonical dense row-major bytes. {@code FLOAT64} and {@code FLOAT32} preserve
 * raw represented bits, {@code BFLOAT16} preserves its stored 16 bits, {@code INT64} and
 * {@code INT32} preserve their two's-complement patterns, and {@code BOOL} is one byte
 * {@code 0} or {@code 1}; every multi-byte element is big-endian. The source layout is not part
 * of this detached value.</p>
 *
 * <p>The private payload is defensively copied and never exposed mutably. The value retains no
 * result, Engine, Runtime representation, storage, arena, or other closeable resource, so all
 * accessors may be called concurrently and remain usable after the originating result or Engine
 * closes. Instances deliberately retain {@link Object} reference identity rather than defining
 * value equality.</p>
 */
public final class HostTensorValue {
    private final DataType dataType;
    private final Shape shape;
    private final long elementCount;
    private final long byteSize;
    private final byte[] canonicalBytes;

    /**
     * Creates a detached value by copying exact canonical bytes.
     *
     * @param dataType non-null exact logical element type retained by reference
     * @param shape non-null fully static immutable logical Shape retained by reference
     * @param canonicalBytes non-null canonical dense row-major payload copied defensively and
     *     never exposed mutably
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if {@code shape} is not fully static
     * @throws IllegalStateException if the payload length differs from its checked logical byte
     *     count
     * @throws ArithmeticException if the logical element or byte count overflows
     */
    HostTensorValue(DataType dataType, Shape shape, byte[] canonicalBytes) {
        this.dataType = Objects.requireNonNull(dataType, "dataType");
        this.shape = Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(canonicalBytes, "canonicalBytes");
        if (!shape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    "host snapshot requires a fully static shape: " + shape);
        }
        elementCount = shape.knownElementCount().orElseThrow();
        byteSize = Math.multiplyExact(elementCount, dataType.byteWidth());
        if (canonicalBytes.length != byteSize) {
            throw new IllegalStateException(
                    "backend canonical byte count does not match descriptor: expected="
                            + byteSize + ", actual=" + canonicalBytes.length);
        }
        this.canonicalBytes = canonicalBytes.clone();
    }

    /**
     * Returns the logical element type from the final publication descriptor.
     *
     * @return the exact non-null Model data type reference
     */
    public DataType dataType() {
        return dataType;
    }

    /**
     * Returns the logical Shape from the final publication descriptor.
     *
     * @return the exact non-null fully static immutable Shape reference
     */
    public Shape shape() {
        return shape;
    }

    /**
     * Returns the checked logical Shape count; rank zero is one and a zero extent makes it zero.
     *
     * @return the checked non-negative logical element count
     */
    public long elementCount() {
        return elementCount;
    }

    /**
     * Returns the canonical payload size, equal to element count times data-type byte width.
     *
     * @return the checked non-negative payload size in bytes
     */
    public long byteSize() {
        return byteSize;
    }

    /**
     * Returns a new view whose position, limit, mark, and byte order are independent of every
     * other view. The buffer exposes no mutable backing array.
     *
     * @return a fresh non-null read-only big-endian buffer with position zero and limit and
     *     capacity equal to {@link #byteSize()}
     */
    public ByteBuffer bytes() {
        return ByteBuffer.wrap(canonicalBytes).asReadOnlyBuffer().order(ByteOrder.BIG_ENDIAN);
    }
}
