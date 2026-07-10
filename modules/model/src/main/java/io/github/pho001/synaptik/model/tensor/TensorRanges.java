package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.math.BigInteger;
import java.util.Optional;

/**
 * Implements package-local integer-range construction without owning tensor identity.
 *
 * <p>This field-free helper validates and sizes ranges with overflow-safe arithmetic, creates one
 * canonical dense descriptor and exact primitive carrier, and delegates final copied construction
 * to {@link TensorFactory#fromFlatArray}. It retains no result or caller state.</p>
 */
final class TensorRanges {
    /** Prevents instances of this stateless package-local implementation helper. */
    private TensorRanges() {
    }

    /**
     * Creates an eager INT32 range with inclusive start and exclusive end.
     *
     * @param startInclusive first emitted signed 32-bit value
     * @param endExclusive exclusive signed 32-bit bound
     * @param step non-zero signed increment that must advance toward the end
     * @param label non-null optional label already checked by the public boundary
     * @return a fresh rank-one dense INT32 leaf tensor; never {@code null}
     */
    static Tensor range(
            int startInclusive, int endExclusive, int step, Optional<String> label) {
        int count = rangeCount(
                BigInteger.valueOf(startInclusive),
                BigInteger.valueOf(endExclusive),
                BigInteger.valueOf(step));
        TensorDescriptor descriptor = descriptor(Shape.of(count), DataType.INT32);
        int[] values = new int[count];
        int value = startInclusive;
        for (int index = 0; index < count; index++) {
            values[index] = value;
            if (index + 1 < count) {
                value += step;
            }
        }
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    /**
     * Creates an eager INT64 range with inclusive start and exclusive end.
     *
     * @param startInclusive first emitted signed 64-bit value
     * @param endExclusive exclusive signed 64-bit bound
     * @param step non-zero signed increment that must advance toward the end
     * @param label non-null optional label already checked by the public boundary
     * @return a fresh rank-one dense INT64 leaf tensor; never {@code null}
     */
    static Tensor range(
            long startInclusive, long endExclusive, long step, Optional<String> label) {
        int count = rangeCount(
                BigInteger.valueOf(startInclusive),
                BigInteger.valueOf(endExclusive),
                BigInteger.valueOf(step));
        TensorDescriptor descriptor = descriptor(Shape.of(count), DataType.INT64);
        long[] values = new long[count];
        long value = startInclusive;
        for (int index = 0; index < count; index++) {
            values[index] = value;
            if (index + 1 < count) {
                value += step;
            }
        }
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    /**
     * Validates range direction and calculates its positive ceiling-divided count exactly.
     *
     * @param start inclusive primitive value represented exactly
     * @param end exclusive primitive value represented exactly
     * @param step primitive step represented exactly
     * @return positive Java-array-sized range count
     * @throws IllegalArgumentException if the step is zero, range is empty, direction is invalid,
     *     or count exceeds {@link Integer#MAX_VALUE}, checked in that order
     */
    private static int rangeCount(BigInteger start, BigInteger end, BigInteger step) {
        int stepSign = step.signum();
        if (stepSign == 0) {
            throw new IllegalArgumentException("range step must not be zero");
        }
        int bounds = start.compareTo(end);
        if (bounds == 0) {
            throw new IllegalArgumentException("range must contain at least one element");
        }
        if ((stepSign > 0 && bounds > 0) || (stepSign < 0 && bounds < 0)) {
            throw new IllegalArgumentException(
                    "range step direction does not advance toward end");
        }

        BigInteger distance = stepSign > 0 ? end.subtract(start) : start.subtract(end);
        BigInteger magnitude = step.abs();
        BigInteger[] division = distance.divideAndRemainder(magnitude);
        BigInteger count = division[0];
        if (division[1].signum() != 0) {
            count = count.add(BigInteger.ONE);
        }
        if (count.compareTo(BigInteger.valueOf(Integer.MAX_VALUE)) > 0) {
            throw new IllegalArgumentException(
                    "range element count exceeds Java array limit: required="
                            + count
                            + ", maximum="
                            + Integer.MAX_VALUE);
        }
        return count.intValueExact();
    }

    /**
     * Constructs a canonical dense non-differentiable descriptor for a validated range.
     *
     * @param shape fully static rank-one result shape
     * @param dataType exact integral result type
     * @return a resolved dense-contiguous descriptor; never {@code null}
     */
    private static TensorDescriptor descriptor(Shape shape, DataType dataType) {
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        return new TensorDescriptor(dataType, shape, Optional.of(layout), false);
    }
}
