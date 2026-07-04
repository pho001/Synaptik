package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.Optional;

/**
 * Implements package-local deterministic range and flat-prefix population mechanics.
 *
 * <p>This stateless helper calculates integer range counts without primitive overflow and creates
 * canonical dense-contiguous descriptors for ranges and caller-shaped prefixes. Every successful
 * path allocates exactly one carrier containing the complete logical result and delegates exactly
 * once to the matching {@link TensorFactory#fromFlatArray} overload. It retains no source,
 * constructs no Tensor or storage directly, and owns no identifier allocation.</p>
 *
 * <p>Range validation completes before descriptor or carrier creation. Prefix validation requires
 * a fully static shape, a checked Java-array-sized logical count, sufficient strict input or a
 * repeatable cyclic input, and descriptor gradient eligibility before allocating the result
 * carrier. Numeric values and raw BFLOAT16 bits are copied unchanged; BOOL bytes remain raw until
 * flat import performs canonical zero-or-one normalization.</p>
 */
final class TensorPopulations {
    /** Prevents instances of this stateless package-local implementation helper. */
    private TensorPopulations() {
    }

    /**
     * Creates an eager INT32 range with inclusive start and exclusive end.
     *
     * <p>Exact sizing and direction validation precede descriptor and carrier creation. The fill
     * loop advances only when another value remains, which avoids an unused post-final primitive
     * overflow. The completed carrier is delegated once to matching flat import and is not
     * retained.</p>
     *
     * @param startInclusive first emitted signed 32-bit value
     * @param endExclusive exclusive signed 32-bit bound
     * @param step non-zero signed increment that must advance toward the end
     * @param label non-null optional label already checked by the public boundary and delegated
     *     unchanged to flat import
     * @return a non-null fresh rank-one dense INT32 tensor with gradients disabled and independent
     *     writable heap storage
     * @throws IllegalArgumentException if the step is zero, the range is empty, direction is
     *     invalid, the count exceeds the Java array limit, or delegated label validation fails
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if source or destination allocation fails
     */
    static Tensor range(
            int startInclusive, int endExclusive, int step, Optional<String> label) {
        int count = rangeCount(
                BigInteger.valueOf(startInclusive),
                BigInteger.valueOf(endExclusive),
                BigInteger.valueOf(step));
        TensorDescriptor descriptor = descriptor(Shape.of(count), DataType.INT32, false);
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
     * <p>Exact sizing and direction validation precede descriptor and carrier creation. The fill
     * loop advances only when another value remains, including at primitive boundaries. The
     * completed carrier is delegated once to matching flat import and is not retained.</p>
     *
     * @param startInclusive first emitted signed 64-bit value
     * @param endExclusive exclusive signed 64-bit bound
     * @param step non-zero signed increment that must advance toward the end
     * @param label non-null optional label already checked by the public boundary and delegated
     *     unchanged to flat import
     * @return a non-null fresh rank-one dense INT64 tensor with gradients disabled and independent
     *     writable heap storage
     * @throws IllegalArgumentException if the step is zero, the range is empty, direction is
     *     invalid, the count exceeds the Java array limit, or delegated label validation fails
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if source or destination allocation fails
     */
    static Tensor range(
            long startInclusive, long endExclusive, long step, Optional<String> label) {
        int count = rangeCount(
                BigInteger.valueOf(startInclusive),
                BigInteger.valueOf(endExclusive),
                BigInteger.valueOf(step));
        TensorDescriptor descriptor = descriptor(Shape.of(count), DataType.INT64, false);
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
     * Creates a FLOAT64 tensor from an exact copied strict prefix.
     *
     * @param shape non-null fully static shape for the canonical dense result
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit FLOAT64 gradient request
     * @param source non-null caller-owned source with at least the logical result count; its tail
     *     is ignored and the array is not retained
     * @return a non-null fresh dense FLOAT64 tensor containing the independent prefix copy
     * @throws IllegalArgumentException if shape, count, source sufficiency, gradient eligibility,
     *     or delegated label validation fails
     * @throws ArithmeticException if checked count or layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if prefix-carrier or destination allocation fails
     */
    static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, double[] source) {
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.FLOAT64, requiresGrad, source.length, false);
        double[] values = Arrays.copyOf(source, elementCount(descriptor));
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    /**
     * Creates a FLOAT32 tensor from an exact copied strict prefix without numeric conversion.
     *
     * @param shape non-null fully static shape for the canonical dense result
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit FLOAT32 gradient request
     * @param source non-null caller-owned source with at least the logical result count; its tail
     *     is ignored and the array is not retained
     * @return a non-null fresh dense FLOAT32 tensor containing the independent prefix copy
     * @throws IllegalArgumentException if shape, count, source sufficiency, gradient eligibility,
     *     or delegated label validation fails
     * @throws ArithmeticException if checked count or layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if prefix-carrier or destination allocation fails
     */
    static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, float[] source) {
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.FLOAT32, requiresGrad, source.length, false);
        float[] values = Arrays.copyOf(source, elementCount(descriptor));
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    /**
     * Creates a BFLOAT16 tensor from an exact copied strict prefix of raw short bits.
     *
     * @param shape non-null fully static shape for the canonical dense result
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit BFLOAT16 gradient request
     * @param source non-null caller-owned raw-bit source with at least the logical result count;
     *     its tail is ignored and the array is not retained
     * @return a non-null fresh dense BFLOAT16 tensor preserving the independent raw-bit prefix
     * @throws IllegalArgumentException if shape, count, source sufficiency, gradient eligibility,
     *     or delegated label validation fails
     * @throws ArithmeticException if checked count or layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if prefix-carrier or destination allocation fails
     */
    static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, short[] source) {
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.BFLOAT16, requiresGrad, source.length, false);
        short[] values = Arrays.copyOf(source, elementCount(descriptor));
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    /**
     * Creates an INT32 tensor from an exact copied strict prefix.
     *
     * @param shape non-null fully static shape for the canonical dense result
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad must be false because INT32 is not differentiable
     * @param source non-null caller-owned source with at least the logical result count; its tail
     *     is ignored and the array is not retained
     * @return a non-null fresh dense INT32 tensor containing the independent prefix copy
     * @throws IllegalArgumentException if shape, count, source sufficiency, gradient eligibility,
     *     or delegated label validation fails
     * @throws ArithmeticException if checked count or layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if prefix-carrier or destination allocation fails
     */
    static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, int[] source) {
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.INT32, requiresGrad, source.length, false);
        int[] values = Arrays.copyOf(source, elementCount(descriptor));
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    /**
     * Creates an INT64 tensor from an exact copied strict prefix.
     *
     * @param shape non-null fully static shape for the canonical dense result
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad must be false because INT64 is not differentiable
     * @param source non-null caller-owned source with at least the logical result count; its tail
     *     is ignored and the array is not retained
     * @return a non-null fresh dense INT64 tensor containing the independent prefix copy
     * @throws IllegalArgumentException if shape, count, source sufficiency, gradient eligibility,
     *     or delegated label validation fails
     * @throws ArithmeticException if checked count or layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if prefix-carrier or destination allocation fails
     */
    static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, long[] source) {
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.INT64, requiresGrad, source.length, false);
        long[] values = Arrays.copyOf(source, elementCount(descriptor));
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    /**
     * Creates a BOOL tensor from a strict prefix normalized later by flat import.
     *
     * @param shape non-null fully static shape for the canonical dense result
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad must be false because BOOL is not differentiable
     * @param source non-null caller-owned zero/non-zero source with at least the logical result
     *     count; its tail is ignored and the array is not retained
     * @return a non-null fresh dense BOOL tensor containing canonical zero-or-one prefix values
     * @throws IllegalArgumentException if shape, count, source sufficiency, gradient eligibility,
     *     or delegated label validation fails
     * @throws ArithmeticException if checked count or layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if prefix-carrier or destination allocation fails
     */
    static Tensor fromStrictFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, byte[] source) {
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.BOOL, requiresGrad, source.length, false);
        byte[] values = Arrays.copyOf(source, elementCount(descriptor));
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    /**
     * Creates a FLOAT64 tensor by cyclically repeating a caller-owned source.
     *
     * @param shape non-null fully static shape for the canonical dense result
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit FLOAT64 gradient request
     * @param source non-null caller-owned cycle, empty only for empty output and never retained
     * @return a non-null fresh dense FLOAT64 tensor containing the independent repetition
     * @throws IllegalArgumentException if shape, count, cyclic source, gradient eligibility, or
     *     delegated label validation fails
     * @throws ArithmeticException if checked count or layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if result-carrier or destination allocation fails
     */
    static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, double[] source) {
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.FLOAT64, requiresGrad, source.length, true);
        double[] values = new double[elementCount(descriptor)];
        for (int index = 0; index < values.length; index++) {
            values[index] = source[index % source.length];
        }
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    /**
     * Creates a FLOAT32 tensor by cyclically repeating a caller-owned source without conversion.
     *
     * @param shape non-null fully static shape for the canonical dense result
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit FLOAT32 gradient request
     * @param source non-null caller-owned cycle, empty only for empty output and never retained
     * @return a non-null fresh dense FLOAT32 tensor containing the independent repetition
     * @throws IllegalArgumentException if shape, count, cyclic source, gradient eligibility, or
     *     delegated label validation fails
     * @throws ArithmeticException if checked count or layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if result-carrier or destination allocation fails
     */
    static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, float[] source) {
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.FLOAT32, requiresGrad, source.length, true);
        float[] values = new float[elementCount(descriptor)];
        for (int index = 0; index < values.length; index++) {
            values[index] = source[index % source.length];
        }
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    /**
     * Creates a BFLOAT16 tensor by cyclically repeating raw short bit patterns unchanged.
     *
     * @param shape non-null fully static shape for the canonical dense result
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit BFLOAT16 gradient request
     * @param source non-null caller-owned raw-bit cycle, empty only for empty output and never
     *     retained
     * @return a non-null fresh dense BFLOAT16 tensor preserving the independent repeated raw bits
     * @throws IllegalArgumentException if shape, count, cyclic source, gradient eligibility, or
     *     delegated label validation fails
     * @throws ArithmeticException if checked count or layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if result-carrier or destination allocation fails
     */
    static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, short[] source) {
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.BFLOAT16, requiresGrad, source.length, true);
        short[] values = new short[elementCount(descriptor)];
        for (int index = 0; index < values.length; index++) {
            values[index] = source[index % source.length];
        }
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    /**
     * Creates an INT32 tensor by cyclically repeating a caller-owned source.
     *
     * @param shape non-null fully static shape for the canonical dense result
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad must be false because INT32 is not differentiable
     * @param source non-null caller-owned cycle, empty only for empty output and never retained
     * @return a non-null fresh dense INT32 tensor containing the independent repetition
     * @throws IllegalArgumentException if shape, count, cyclic source, gradient eligibility, or
     *     delegated label validation fails
     * @throws ArithmeticException if checked count or layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if result-carrier or destination allocation fails
     */
    static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, int[] source) {
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.INT32, requiresGrad, source.length, true);
        int[] values = new int[elementCount(descriptor)];
        for (int index = 0; index < values.length; index++) {
            values[index] = source[index % source.length];
        }
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    /**
     * Creates an INT64 tensor by cyclically repeating a caller-owned source.
     *
     * @param shape non-null fully static shape for the canonical dense result
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad must be false because INT64 is not differentiable
     * @param source non-null caller-owned cycle, empty only for empty output and never retained
     * @return a non-null fresh dense INT64 tensor containing the independent repetition
     * @throws IllegalArgumentException if shape, count, cyclic source, gradient eligibility, or
     *     delegated label validation fails
     * @throws ArithmeticException if checked count or layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if result-carrier or destination allocation fails
     */
    static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, long[] source) {
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.INT64, requiresGrad, source.length, true);
        long[] values = new long[elementCount(descriptor)];
        for (int index = 0; index < values.length; index++) {
            values[index] = source[index % source.length];
        }
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    /**
     * Creates a BOOL tensor by cyclic repetition and downstream flat-import normalization.
     *
     * @param shape non-null fully static shape for the canonical dense result
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad must be false because BOOL is not differentiable
     * @param source non-null caller-owned zero/non-zero cycle, empty only for empty output and
     *     never retained
     * @return a non-null fresh dense BOOL tensor containing canonical repeated zero-or-one values
     * @throws IllegalArgumentException if shape, count, cyclic source, gradient eligibility, or
     *     delegated label validation fails
     * @throws ArithmeticException if checked count or layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if result-carrier or destination allocation fails
     */
    static Tensor fromCyclicFlatPrefix(
            Shape shape, Optional<String> label, boolean requiresGrad, byte[] source) {
        TensorDescriptor descriptor = prefixDescriptor(
                shape, DataType.BOOL, requiresGrad, source.length, true);
        byte[] values = new byte[elementCount(descriptor)];
        for (int index = 0; index < values.length; index++) {
            values[index] = source[index % source.length];
        }
        return TensorFactory.fromFlatArray(descriptor, label, values);
    }

    /**
     * Validates range direction and calculates its positive ceiling-divided count exactly.
     *
     * <p>Arbitrary precision is confined to sizing so primitive subtraction, absolute value, and
     * addition cannot overflow. It does not create a public arbitrary-precision range or value
     * conversion contract.</p>
     *
     * @param start inclusive primitive value represented exactly
     * @param end exclusive primitive value represented exactly
     * @param step primitive step represented exactly
     * @return positive Java-array-sized range count
     * @throws IllegalArgumentException for zero step, empty range, wrong direction, or count above
     *     {@link Integer#MAX_VALUE}, in that order
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
     * Validates prefix shape, count, source availability, layout, and gradient eligibility.
     *
     * @param shape non-null proposed output shape
     * @param dataType exact data type inferred from the typed entry's source carrier
     * @param requiresGrad explicit model-level gradient request
     * @param sourceLength non-negative source length
     * @param cyclic true for cyclic availability rules; false for strict-prefix rules
     * @return a new resolved canonical dense descriptor
     * @throws IllegalArgumentException if static shape, Java-array limit, source availability, or
     *     gradient eligibility validation fails in the documented order
     * @throws ArithmeticException if checked count or layout arithmetic overflows
     */
    private static TensorDescriptor prefixDescriptor(
            Shape shape,
            DataType dataType,
            boolean requiresGrad,
            int sourceLength,
            boolean cyclic) {
        if (!shape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    "prefix tensor creation requires a fully static shape: " + shape);
        }
        long count = shape.knownElementCount().orElseThrow();
        if (count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "prefix tensor element count exceeds Java array limit: required="
                            + count
                            + ", maximum="
                            + Integer.MAX_VALUE);
        }
        if (!cyclic && sourceLength < count) {
            throw new IllegalArgumentException(
                    "strict flat prefix source is too short: required="
                            + count
                            + ", actual="
                            + sourceLength);
        }
        if (cyclic && count > 0 && sourceLength == 0) {
            throw new IllegalArgumentException(
                    "cyclic flat prefix source must not be empty for non-empty output");
        }
        return descriptor(shape, dataType, requiresGrad);
    }

    /**
     * Constructs one canonical dense descriptor from already validated logical facts.
     *
     * @param shape fully static shape
     * @param dataType exact inferred result type
     * @param requiresGrad explicit gradient request
     * @return a new resolved dense-contiguous descriptor
     * @throws IllegalArgumentException if dense layout rejects the shape or descriptor gradient
     *     eligibility rejects {@code requiresGrad}
     * @throws ArithmeticException if checked stride or referenced-span arithmetic overflows
     */
    private static TensorDescriptor descriptor(
            Shape shape, DataType dataType, boolean requiresGrad) {
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        return new TensorDescriptor(dataType, shape, Optional.of(layout), requiresGrad);
    }

    /**
     * Returns the already validated Java-array-sized logical count from a dense descriptor.
     *
     * @param descriptor resolved prefix descriptor
     * @return exact non-negative count narrowed safely to {@code int}
     * @throws ArithmeticException if the count unexpectedly does not fit {@code int}
     * @throws IllegalStateException if the descriptor unexpectedly lacks a known logical count
     */
    private static int elementCount(TensorDescriptor descriptor) {
        return Math.toIntExact(descriptor.shape().knownElementCount().orElseThrow());
    }
}
