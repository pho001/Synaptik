package io.github.pho001.synaptik.model.storage;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * Borrowed, non-owning host tensor storage backed by one exact-size {@link MemorySegment}.
 *
 * <p>The wrapper accepts live heap, native, mapped, global, confined, shared, read-only,
 * writable, and sliced segments. It retains the exact supplied segment but does not allocate,
 * copy, slice, reinterpret, retain, or close memory. The caller owns any arena and must keep the
 * segment's scope alive and obey its thread-access policy whenever performing memory access.</p>
 *
 * <p>Construction requires exact equality between the segment byte size and the checked product
 * of physical element capacity and logical data-type width. No alignment or byte-order policy is
 * implied, and no typed access or tensor, descriptor, or layout compatibility is validated.
 * After caller-controlled scope closure, metadata remains available, {@link #isAlive()} reports
 * false, and {@link #segment()} still returns the exact dead segment so JDK access rules enforce
 * failure.</p>
 *
 * <p>This class deliberately uses object identity for equality, hashing, synchronization, and
 * diagnostic text. Two wrappers do not become equal because they refer to the same segment or
 * describe the same metadata.</p>
 */
public final class MemorySegmentStorage implements HostTensorStorage {
    private final DataType dataType;
    private final long elementCapacity;
    private final long byteSize;
    private final MemorySegment segment;

    /**
     * Creates a borrowed wrapper around an exact-size live memory segment.
     *
     * <p>Validation occurs in this order: {@code dataType} nullity, {@code segment} nullity,
     * non-negative capacity, checked capacity-to-byte multiplication, exact segment byte size,
     * and initial segment-scope liveness. Null failures use the messages {@code dataType} and
     * {@code segment}. Negative capacity reports
     * {@code elementCapacity must be non-negative: <value>}. Overflow reports
     * {@code element byte size overflows long: elementCapacity=<value>, byteWidth=<width>}.
     * A size mismatch reports
     * {@code segment byte size must equal required byte size: required=<required>, actual=<actual>},
     * and a dead initial scope reports {@code segment scope is not alive}. An earlier invalid
     * condition wins when more than one input is invalid.</p>

     * <p>Zero capacity is valid only with a live zero-byte segment. The largest arithmetic
     * capacity is {@code Long.MAX_VALUE / dataType.byteWidth()}; exceeding it fails before
     * segment-size comparison.</p>
     *
     * <p>The exact data-type and segment references are retained. The caller retains ownership of
     * the segment's arena or other lifetime mechanism and is responsible for keeping it alive and
     * using it only from threads allowed by the JDK scope. Exact-size slices are accepted without
     * an alignment promise. This constructor performs no allocation, copy, memory access,
     * address lookup, alignment validation, byte-order selection, or arena operation.</p>
     *
     * @param dataType the non-null logical type whose byte width sizes each physical element; the
     *     exact reference is retained
     * @param elementCapacity the non-negative physical number of complete elements; zero is valid
     *     with an exact zero-byte segment
     * @param segment the non-null, initially live, exact-size memory segment to borrow; the exact
     *     reference is retained and the caller continues to own its lifetime
     * @throws NullPointerException if {@code dataType} is null, or if {@code segment} is null
     * @throws IllegalArgumentException if {@code elementCapacity} is negative, or if the segment
     *     byte size differs from the required checked byte size
     * @throws ArithmeticException if multiplying {@code elementCapacity} by the data-type byte
     *     width exceeds {@link Long#MAX_VALUE}
     * @throws IllegalStateException if the segment scope is not alive after the preceding
     *     validation succeeds
     */
    public MemorySegmentStorage(
            DataType dataType,
            long elementCapacity,
            MemorySegment segment) {
        this.dataType = Objects.requireNonNull(dataType, "dataType");
        this.segment = Objects.requireNonNull(segment, "segment");
        if (elementCapacity < 0) {
            throw new IllegalArgumentException(
                    "elementCapacity must be non-negative: " + elementCapacity);
        }
        this.elementCapacity = elementCapacity;

        int byteWidth = dataType.byteWidth();
        try {
            this.byteSize = Math.multiplyExact(elementCapacity, byteWidth);
        } catch (ArithmeticException exception) {
            throw new ArithmeticException(
                    "element byte size overflows long: elementCapacity="
                            + elementCapacity
                            + ", byteWidth="
                            + byteWidth);
        }
        if (segment.byteSize() != byteSize) {
            throw new IllegalArgumentException(
                    "segment byte size must equal required byte size: required="
                            + byteSize
                            + ", actual="
                            + segment.byteSize());
        }
        if (!segment.scope().isAlive()) {
            throw new IllegalStateException("segment scope is not alive");
        }
    }

    /**
     * Returns the exact logical data type supplied at construction.
     *
     * <p>The value describes complete physical elements in the raw region; it does not select a
     * typed Java carrier, alignment, byte order, or backend representation.</p>
     *
     * @return the retained non-null data-type reference
     */
    @Override
    public DataType dataType() {
        return dataType;
    }

    /**
     * Returns the validated physical element capacity.
     *
     * <p>This capacity is independent of tensor logical element count and layout span.</p>
     *
     * @return the retained non-negative capacity measured in complete elements
     */
    @Override
    public long elementCapacity() {
        return elementCapacity;
    }

    /**
     * Returns the checked exact byte size established at construction.
     *
     * <p>The stored value is not recalculated and remains available after the segment scope
     * closes. Zero element capacity produces zero bytes.</p>
     *
     * @return the non-negative capacity-to-width product in bytes, exactly equal to the supplied
     *     segment's byte size
     */
    @Override
    public long byteSize() {
        return byteSize;
    }

    /**
     * Returns the exact borrowed segment supplied at construction, even after its scope closes.
     *
     * <p>The result is never copied, sliced, reinterpreted, or replaced. It may be a dead segment;
     * JDK memory access through that segment then enforces the applicable closed-scope failure.
     * The caller continues to own any arena and must obey the segment's scope and thread-access
     * rules.</p>
     *
     * @return the retained non-null memory-segment reference
     */
    @Override
    public MemorySegment segment() {
        return segment;
    }

    /**
     * Reports the retained segment's JDK read-only state.
     *
     * <p>A writable result permits raw mutation only under the segment's JDK scope and
     * thread-access rules and adds no synchronization or version behavior. This query does not
     * grant write access to a read-only segment.</p>
     *
     * @return {@code true} when the supplied segment is read-only; otherwise {@code false}
     */
    @Override
    public boolean isReadOnly() {
        return segment.isReadOnly();
    }

    /**
     * Reports the retained segment scope's liveness at the instant of the query.
     *
     * <p>The result may become stale immediately and says nothing about access from the current
     * thread. A false result does not prevent {@link #segment()} from returning the exact retained
     * segment; JDK access operations remain responsible for rejecting the dead scope.</p>
     *
     * @return the current result of {@code segment.scope().isAlive()}
     */
    @Override
    public boolean isAlive() {
        return segment.scope().isAlive();
    }
}
