package io.github.pho001.synaptik.model;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable, backend-independent descriptor of resolved logical tensor element geometry.
 *
 * <p>A descriptor records rank, element strides, an element-based storage offset, geometric layout
 * kind, explicit view or alias metadata, and the minimum referenced element span. It deliberately
 * does not retain the source {@link Shape} and contains no storage object, byte address, ownership
 * policy, device state, materialization decision, or backend information.</p>
 *
 * <p>Numeric layout descriptors can be created only for fully static shapes. Arrays supplied to or
 * returned by this class are defensively copied.</p>
 */
public final class LayoutDescriptor {
    private final int rank;
    private final long[] strides;
    private final long storageOffset;
    private final LayoutKind kind;
    private final boolean view;
    private final long referencedElementSpan;

    private LayoutDescriptor(
            int rank,
            long[] strides,
            long storageOffset,
            LayoutKind kind,
            boolean view,
            long referencedElementSpan) {
        this.rank = rank;
        this.strides = strides;
        this.storageOffset = storageOffset;
        this.kind = kind;
        this.view = view;
        this.referencedElementSpan = referencedElementSpan;
    }

    /**
     * Creates canonical row-major layout geometry with element offset zero.
     *
     * <p>The returned descriptor is not marked as a view. A scalar receives no stride entries and
     * references one element; a shape containing a zero-sized dimension references no elements.</p>
     *
     * @param shape non-null fully static shape used to resolve rank and canonical element strides;
     *     the shape is not retained
     * @return non-null immutable dense contiguous layout descriptor
     * @throws NullPointerException if {@code shape} is {@code null}
     * @throws IllegalArgumentException if {@code shape} contains a dynamic dimension
     * @throws ArithmeticException if a required stride product exceeds {@link Long#MAX_VALUE}
     */
    public static LayoutDescriptor contiguous(Shape shape) {
        Objects.requireNonNull(shape, "shape");
        long[] canonicalStrides = LayoutGeometry.canonicalStrides(shape);
        return create(shape, canonicalStrides, 0, false);
    }

    /**
     * Creates and classifies explicit resolved logical element geometry.
     *
     * <p>The supplied stride array is copied and never retained. Strides and offset are measured in
     * elements rather than bytes. A layout that repeats a dimension larger than one through stride
     * zero must be explicitly identified as a view.</p>
     *
     * @param shape non-null fully static shape used for validation and geometric calculations; the
     *     shape is not retained
     * @param strides non-null element strides with one non-negative entry per shape axis; the array
     *     remains owned and mutable by the caller
     * @param storageOffset non-negative offset of the logical origin, measured in storage elements
     * @param view {@code true} if the layout aliases or views another tensor's storage; required for
     *     broadcast zero-stride repetition
     * @return non-null immutable descriptor classified from the supplied geometry
     * @throws NullPointerException if {@code shape} or {@code strides} is {@code null}
     * @throws IllegalArgumentException if the shape is dynamic, stride count differs from rank, a
     *     stride or offset is negative, or broadcast repetition is not marked as a view
     * @throws ArithmeticException if canonical strides or referenced element span overflow
     */
    public static LayoutDescriptor of(
            Shape shape, long[] strides, long storageOffset, boolean view) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(strides, "strides");
        long[] copiedStrides = strides.clone();
        if (copiedStrides.length != shape.rank()) {
            throw new IllegalArgumentException(
                    "Stride count " + copiedStrides.length + " does not match rank " + shape.rank());
        }
        for (int axis = 0; axis < copiedStrides.length; axis++) {
            if (copiedStrides[axis] < 0) {
                throw new IllegalArgumentException(
                        "Stride at axis " + axis + " must be non-negative: " + copiedStrides[axis]);
            }
        }
        if (storageOffset < 0) {
            throw new IllegalArgumentException(
                    "Storage offset must be non-negative: " + storageOffset);
        }
        return create(shape, copiedStrides, storageOffset, view);
    }

    private static LayoutDescriptor create(
            Shape shape, long[] strides, long storageOffset, boolean view) {
        LayoutKind kind = LayoutGeometry.classify(shape, strides, storageOffset);
        if (kind == LayoutKind.BROADCAST_ZERO_STRIDE && !view) {
            throw new IllegalArgumentException("Broadcast zero-stride layout must be a view");
        }
        long span = LayoutGeometry.referencedElementSpan(shape, strides, storageOffset);
        return new LayoutDescriptor(shape.rank(), strides, storageOffset, kind, view, span);
    }

    /**
     * Returns the number of resolved layout axes.
     *
     * @return non-negative rank; zero identifies scalar layout geometry
     */
    public int rank() {
        return rank;
    }

    /**
     * Returns the derived geometric classification.
     *
     * @return non-null layout kind determined at construction time
     */
    public LayoutKind kind() {
        return kind;
    }

    /**
     * Copies the ordered element strides.
     *
     * @return newly allocated array with one non-negative element stride per axis; mutating it
     *     cannot affect this descriptor
     */
    public long[] strides() {
        return strides.clone();
    }

    /**
     * Returns the element stride at a positive or negative axis.
     *
     * @param axis axis in the inclusive range {@code [-rank, rank - 1]}; negative values count from
     *     the final axis
     * @return non-negative stride measured in storage elements
     * @throws IndexOutOfBoundsException if the axis is invalid, including every axis for rank zero
     */
    public long stride(int axis) {
        long normalized = axis;
        if (normalized < 0) {
            normalized += rank;
        }
        if (normalized < 0 || normalized >= rank) {
            throw new IndexOutOfBoundsException(
                    "Axis " + axis + " is outside layout rank " + rank);
        }
        return strides[(int) normalized];
    }

    /**
     * Returns the logical origin's offset in storage elements.
     *
     * @return non-negative element offset; the value is never a byte offset
     */
    public long storageOffset() {
        return storageOffset;
    }

    /**
     * Reports explicit view or alias metadata supplied during construction.
     *
     * @return {@code true} if the descriptor represents storage shared with another tensor;
     *     otherwise {@code false}
     */
    public boolean isView() {
        return view;
    }

    /**
     * Reports whether the element strides are canonical row-major strides.
     *
     * @return {@code true} for {@link LayoutKind#DENSE_CONTIGUOUS} and
     *     {@link LayoutKind#DENSE_WITH_OFFSET}; otherwise {@code false}
     */
    public boolean isContiguous() {
        return kind == LayoutKind.DENSE_CONTIGUOUS || kind == LayoutKind.DENSE_WITH_OFFSET;
    }

    /**
     * Reports whether the logical origin has a non-zero storage element offset.
     *
     * @return {@code true} when {@link #storageOffset()} is greater than zero; otherwise
     *     {@code false}
     */
    public boolean hasStorageOffset() {
        return storageOffset != 0;
    }

    /**
     * Reports whether any raw element stride is zero, regardless of the corresponding dimension.
     *
     * @return {@code true} if any stored stride is zero; otherwise {@code false}
     */
    public boolean hasZeroStride() {
        return LayoutGeometry.hasZeroStride(strides);
    }

    /**
     * Reports whether this layout repeats a non-singleton dimension through stride zero.
     *
     * @return {@code true} exactly when {@link #kind()} is
     *     {@link LayoutKind#BROADCAST_ZERO_STRIDE}
     */
    public boolean isBroadcast() {
        return kind == LayoutKind.BROADCAST_ZERO_STRIDE;
    }

    /**
     * Returns the minimum storage element count that contains every referenced element index.
     *
     * <p>A layout whose shape contains a zero-sized dimension has span zero regardless of offset.
     * For a layout that references elements, the value is the greatest referenced element index
     * plus one.</p>
     *
     * @return non-negative span measured in storage elements
     */
    public long referencedElementSpan() {
        return referencedElementSpan;
    }

    /**
     * Compares every stored and derived layout field structurally.
     *
     * @param other candidate object, which may be {@code null}
     * @return {@code true} when the other descriptor has equal rank, strides, offset, kind, view
     *     flag, and referenced span
     */
    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof LayoutDescriptor descriptor)) {
            return false;
        }
        return rank == descriptor.rank
                && storageOffset == descriptor.storageOffset
                && view == descriptor.view
                && referencedElementSpan == descriptor.referencedElementSpan
                && kind == descriptor.kind
                && Arrays.equals(strides, descriptor.strides);
    }

    /**
     * Returns a structural hash of every stored and derived layout field.
     *
     * @return hash code consistent with {@link #equals(Object)}
     */
    @Override
    public int hashCode() {
        int result = Objects.hash(rank, storageOffset, kind, view, referencedElementSpan);
        return 31 * result + Arrays.hashCode(strides);
    }

    /**
     * Returns a diagnostic summary of this resolved logical layout.
     *
     * @return non-null text containing kind, rank, strides, element offset, view flag, and element
     *     span; the format is not a serialization contract
     */
    @Override
    public String toString() {
        return "LayoutDescriptor["
                + "kind=" + kind
                + ", rank=" + rank
                + ", strides=" + Arrays.toString(strides)
                + ", storageOffset=" + storageOffset
                + ", view=" + view
                + ", referencedElementSpan=" + referencedElementSpan
                + ']';
    }
}
