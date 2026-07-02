package io.github.pho001.synaptik.model;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.StringJoiner;

/**
 * Immutable ordered dimensions describing the logical shape of a tensor value.
 *
 * <p>A shape may combine static and symbolic dynamic dimensions. Rank zero is the canonical scalar
 * shape and has one logical element. Static dimensions may be zero, in which case the fully static
 * shape describes an empty tensor. Shape does not contain strides, storage, or execution state.</p>
 */
public final class Shape {
    private static final Shape SCALAR = new Shape(List.of());

    private final List<Dimension> dimensions;

    /**
     * Creates a shape from an already validated dimension list.
     *
     * @param dimensions non-null immutable source whose elements are all non-null
     */
    private Shape(List<Dimension> dimensions) {
        this.dimensions = List.copyOf(dimensions);
    }

    /**
     * Returns the canonical rank-zero scalar shape.
     *
     * @return shared immutable scalar shape with rank zero and known element count one
     */
    public static Shape scalar() {
        return SCALAR;
    }

    /**
     * Creates a fully static shape from ordered numeric sizes.
     *
     * <p>The caller-owned array is read during construction and is never retained. An empty array
     * produces the canonical scalar shape.</p>
     *
     * @param sizes non-null ordered axis extents; each value must be non-negative
     * @return immutable shape containing corresponding {@link StaticDimension} values
     * @throws NullPointerException if {@code sizes} is {@code null}
     * @throws IllegalArgumentException if any size is negative
     */
    public static Shape of(long... sizes) {
        Objects.requireNonNull(sizes, "sizes");
        if (sizes.length == 0) {
            return scalar();
        }

        Dimension[] dimensions = new Dimension[sizes.length];
        for (int index = 0; index < sizes.length; index++) {
            dimensions[index] = new StaticDimension(sizes[index]);
        }
        return ofDimensions(dimensions);
    }

    /**
     * Creates a shape from ordered static or dynamic dimensions.
     *
     * <p>The caller-owned array is defensively copied. An empty array produces the canonical scalar
     * shape.</p>
     *
     * @param dimensions non-null ordered dimensions with no null elements
     * @return immutable shape structurally equal to the supplied dimensions
     * @throws NullPointerException if the array or any dimension is {@code null}
     */
    public static Shape ofDimensions(Dimension... dimensions) {
        Objects.requireNonNull(dimensions, "dimensions");
        if (dimensions.length == 0) {
            return scalar();
        }

        Dimension[] copied = dimensions.clone();
        for (int index = 0; index < copied.length; index++) {
            Objects.requireNonNull(copied[index], "dimensions[" + index + "]");
        }
        return new Shape(Arrays.asList(copied));
    }

    /**
     * Returns the number of axes in this shape.
     *
     * @return non-negative rank; zero identifies the scalar shape
     */
    public int rank() {
        return dimensions.size();
    }

    /**
     * Returns an immutable ordered view of this shape's dimensions.
     *
     * @return non-null unmodifiable list; attempts to mutate it fail and cannot affect this shape
     */
    public List<Dimension> dimensions() {
        return dimensions;
    }

    /**
     * Returns the dimension at a positive or negative axis.
     *
     * @param axis axis in the inclusive range {@code [-rank, rank - 1]}; negative values count from
     *     the final axis
     * @return non-null immutable dimension at the normalized axis
     * @throws IndexOutOfBoundsException if the axis is invalid, including every axis for rank zero
     */
    public Dimension dimension(int axis) {
        return dimensions.get(normalizeAxis(axis));
    }

    /**
     * Normalizes a positive or negative axis to its non-negative index.
     *
     * @param axis axis in the inclusive range {@code [-rank, rank - 1]}; negative values count from
     *     the final axis
     * @return normalized axis in the inclusive range {@code [0, rank - 1]}
     * @throws IndexOutOfBoundsException if the axis is invalid, including every axis for rank zero
     */
    public int normalizeAxis(int axis) {
        long normalized = axis;
        if (normalized < 0) {
            normalized += rank();
        }
        if (normalized < 0 || normalized >= rank()) {
            throw new IndexOutOfBoundsException(
                    "Axis " + axis + " is outside shape rank " + rank());
        }
        return (int) normalized;
    }

    /**
     * Reports whether every dimension has a known static size.
     *
     * @return {@code true} for scalar and fully static shapes; {@code false} when any dimension is
     *     dynamic
     */
    public boolean isFullyStatic() {
        for (Dimension dimension : dimensions) {
            if (dimension.isDynamic()) {
                return false;
            }
        }
        return true;
    }

    /**
     * Calculates the number of logical elements when all dimensions are static.
     *
     * <p>Rank zero has one element. Any fully static shape containing a zero dimension has zero
     * elements, even if multiplying its other dimensions would overflow. Dynamic shapes return an
     * empty optional because this model does not bind symbolic sizes.</p>
     *
     * @return present non-negative element count for a fully static shape, or an empty optional for
     *     a shape containing a dynamic dimension
     * @throws ArithmeticException if a non-zero fully static product exceeds {@link Long#MAX_VALUE}
     */
    public OptionalLong knownElementCount() {
        if (!isFullyStatic()) {
            return OptionalLong.empty();
        }
        for (Dimension dimension : dimensions) {
            if (((StaticDimension) dimension).size() == 0) {
                return OptionalLong.of(0);
            }
        }

        long count = 1;
        for (Dimension dimension : dimensions) {
            count = Math.multiplyExact(count, ((StaticDimension) dimension).size());
        }
        return OptionalLong.of(count);
    }

    /**
     * Copies this shape's static sizes into a primitive array.
     *
     * @return newly allocated ordered sizes; an empty array for the scalar shape
     * @throws IllegalStateException if any dimension is dynamic
     */
    public long[] toLongArray() {
        if (!isFullyStatic()) {
            throw new IllegalStateException("Cannot extract numeric sizes from dynamic shape " + this);
        }

        long[] sizes = new long[dimensions.size()];
        for (int index = 0; index < dimensions.size(); index++) {
            sizes[index] = ((StaticDimension) dimensions.get(index)).size();
        }
        return sizes;
    }

    /**
     * Compares shapes using ordered structural dimension equality.
     *
     * @param other candidate object, which may be {@code null}
     * @return {@code true} when {@code other} is a shape with equal ordered dimensions
     */
    @Override
    public boolean equals(Object other) {
        return this == other
                || (other instanceof Shape shape && dimensions.equals(shape.dimensions));
    }

    /**
     * Returns the structural hash of the ordered dimensions.
     *
     * @return hash code consistent with {@link #equals(Object)}
     */
    @Override
    public int hashCode() {
        return dimensions.hashCode();
    }

    /**
     * Returns a concise diagnostic representation of this shape.
     *
     * @return non-null text such as {@code Shape[]}, {@code Shape[2, 0, 3]}, or
     *     {@code Shape[N, 4]}; the format is not a serialization contract
     */
    @Override
    public String toString() {
        StringJoiner joiner = new StringJoiner(", ", "Shape[", "]");
        for (Dimension dimension : dimensions) {
            if (dimension instanceof StaticDimension staticDimension) {
                joiner.add(Long.toString(staticDimension.size()));
            } else if (dimension instanceof DynamicDimension dynamicDimension) {
                joiner.add(dynamicDimension.symbol());
            }
        }
        return joiner.toString();
    }
}
