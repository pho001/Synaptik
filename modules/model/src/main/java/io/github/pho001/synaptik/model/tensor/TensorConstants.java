package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Arrays;
import java.util.Optional;

/**
 * Implements the package-local descriptor construction and population mechanics for constants.
 *
 * <p>This stateless helper creates only rank-zero or caller-shaped dense-contiguous descriptors.
 * It validates static shape, Java-array count, layout geometry, and gradient eligibility before
 * allocating a scalar or one-filled source carrier. Zeros delegate directly to heap allocation;
 * scalars and ones delegate to one exact-carrier flat import. Every result therefore receives new
 * descriptor, layout, storage, backing-array, Tensor, and identifier objects, while no source
 * carrier or template is retained.</p>
 *
 * <p>Descriptor validation requires a fully static shape, obtains the checked logical element
 * count, enforces {@link Integer#MAX_VALUE}, constructs canonical contiguous geometry, and applies
 * {@link TensorDescriptor} gradient eligibility in that order. It completes before scalar or one
 * source-carrier allocation and before every destination or ID allocation. A blank label is
 * rejected only after destination and ID allocation and consumes that ID. Identifier exhaustion
 * is also observed after destination allocation. Scalar and one paths have already allocated
 * their source carrier at either point; the zero path has no source carrier. A JVM allocation
 * failure before ID allocation consumes no ID.</p>
 */
final class TensorConstants {
    /** Prevents instances of this stateless package-local implementation helper. */
    private TensorConstants() {
    }

    /**
     * Creates a rank-zero FLOAT64 constant through exact binary64 flat import.
     *
     * @param value exact binary64 value, including its raw signed-zero or NaN representation
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit model-level gradient request
     * @return a non-null fresh independent rank-zero dense tensor containing {@code value}
     * @throws NullPointerException if the internal {@code label} precondition is violated
     * @throws IllegalArgumentException if descriptor eligibility fails before source allocation or
     *     delegated label validation fails after destination and ID allocation
     * @throws ArithmeticException if checked scalar geometry overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if source or destination allocation fails
     */
    static Tensor scalar(double value, Optional<String> label, boolean requiresGrad) {
        TensorDescriptor descriptor = descriptor(Shape.scalar(), DataType.FLOAT64, requiresGrad);
        return TensorFactory.fromFlatArray(descriptor, label, new double[] {value});
    }

    /**
     * Creates a rank-zero FLOAT32 constant through exact binary32 flat import.
     *
     * @param value exact binary32 value, including its raw signed-zero or NaN representation
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit model-level gradient request
     * @return a non-null fresh independent rank-zero dense tensor containing {@code value}
     * @throws NullPointerException if the internal {@code label} precondition is violated
     * @throws IllegalArgumentException if descriptor eligibility fails before source allocation or
     *     delegated label validation fails after destination and ID allocation
     * @throws ArithmeticException if checked scalar geometry overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if source or destination allocation fails
     */
    static Tensor scalar(float value, Optional<String> label, boolean requiresGrad) {
        TensorDescriptor descriptor = descriptor(Shape.scalar(), DataType.FLOAT32, requiresGrad);
        return TensorFactory.fromFlatArray(descriptor, label, new float[] {value});
    }

    /**
     * Creates a rank-zero BFLOAT16 constant by explicitly rounding a binary32 semantic value.
     *
     * @param value binary32 value converted with {@link BFloat16Bits#fromFloat(float)}
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit model-level gradient request
     * @return a non-null fresh independent rank-zero dense tensor containing the converted BFLOAT16 bits
     * @throws NullPointerException if the internal {@code label} precondition is violated
     * @throws IllegalArgumentException if descriptor eligibility fails before source allocation or
     *     delegated label validation fails after destination and ID allocation
     * @throws ArithmeticException if checked scalar geometry overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if source or destination allocation fails
     */
    static Tensor scalarBFloat16(float value, Optional<String> label, boolean requiresGrad) {
        TensorDescriptor descriptor = descriptor(Shape.scalar(), DataType.BFLOAT16, requiresGrad);
        return TensorFactory.fromFlatArray(
                descriptor, label, new short[] {BFloat16Bits.fromFloat(value)});
    }

    /**
     * Creates a rank-zero INT32 constant through exact signed-integer flat import.
     *
     * @param value exact signed 32-bit value
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit request, which must be false for INT32
     * @return a non-null fresh independent rank-zero dense tensor containing {@code value}
     * @throws NullPointerException if the internal {@code label} precondition is violated
     * @throws IllegalArgumentException if descriptor eligibility fails before source allocation or
     *     delegated label validation fails after destination and ID allocation
     * @throws ArithmeticException if checked scalar geometry overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if source or destination allocation fails
     */
    static Tensor scalar(int value, Optional<String> label, boolean requiresGrad) {
        TensorDescriptor descriptor = descriptor(Shape.scalar(), DataType.INT32, requiresGrad);
        return TensorFactory.fromFlatArray(descriptor, label, new int[] {value});
    }

    /**
     * Creates a rank-zero INT64 constant through exact signed-integer flat import.
     *
     * @param value exact signed 64-bit value
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit request, which must be false for INT64
     * @return a non-null fresh independent rank-zero dense tensor containing {@code value}
     * @throws NullPointerException if the internal {@code label} precondition is violated
     * @throws IllegalArgumentException if descriptor eligibility fails before source allocation or
     *     delegated label validation fails after destination and ID allocation
     * @throws ArithmeticException if checked scalar geometry overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if source or destination allocation fails
     */
    static Tensor scalar(long value, Optional<String> label, boolean requiresGrad) {
        TensorDescriptor descriptor = descriptor(Shape.scalar(), DataType.INT64, requiresGrad);
        return TensorFactory.fromFlatArray(descriptor, label, new long[] {value});
    }

    /**
     * Creates a rank-zero BOOL constant with canonical byte-zero or byte-one representation.
     *
     * @param value semantic boolean value; no numeric truthiness conversion is performed
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit request, which must be false for BOOL
     * @return a non-null fresh independent rank-zero dense tensor containing {@code value}
     * @throws NullPointerException if the internal {@code label} precondition is violated
     * @throws IllegalArgumentException if descriptor eligibility fails before source allocation or
     *     delegated label validation fails after destination and ID allocation
     * @throws ArithmeticException if checked scalar geometry overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if source or destination allocation fails
     */
    static Tensor scalar(boolean value, Optional<String> label, boolean requiresGrad) {
        TensorDescriptor descriptor = descriptor(Shape.scalar(), DataType.BOOL, requiresGrad);
        return TensorFactory.fromFlatArray(descriptor, label, new byte[] {value ? (byte) 1 : 0});
    }

    /**
     * Creates a caller-shaped dense tensor using only JVM default-zero destination allocation.
     *
     * <p>No source carrier or fill loop is created. Scalar shapes allocate one element, and a
     * fully static shape containing a zero-sized dimension allocates an empty backing array.</p>
     *
     * @param shape non-null fully static logical shape; scalar and empty shapes are accepted
     * @param dataType non-null exact element type
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit model-level gradient request
     * @return a non-null fresh independent dense tensor containing raw zero or BOOL false in every element
     * @throws NullPointerException if an internal non-null argument precondition is violated
     * @throws IllegalArgumentException if shape is dynamic, count exceeds the Java array limit,
     *     gradient request is ineligible, or delegated label validation fails
     * @throws ArithmeticException if checked count, stride, or span arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if destination allocation fails
     */
    static Tensor zeros(
            Shape shape, DataType dataType, Optional<String> label, boolean requiresGrad) {
        TensorDescriptor descriptor = descriptor(shape, dataType, requiresGrad);
        return TensorFactory.allocate(descriptor, label);
    }

    /**
     * Creates a caller-shaped dense tensor by filling one exact typed carrier and importing it.
     *
     * <p>The fill values are {@code 1.0d}, {@code 1.0f}, converted BFLOAT16 {@code 0x3F80},
     * {@code 1}, {@code 1L}, and BOOL byte {@code 1}. Exactly one carrier is created, filled, and
     * passed to exactly one matching flat-import overload. Scalar shapes have one value; empty
     * shapes use an empty carrier. The carrier is not retained.</p>
     *
     * @param shape non-null fully static logical shape; scalar and empty shapes are accepted
     * @param dataType non-null exact element type selecting carrier and one representation
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit model-level gradient request
     * @return a non-null fresh independent dense tensor containing exact typed one in every element
     * @throws NullPointerException if an internal non-null argument precondition is violated
     * @throws IllegalArgumentException if shape is dynamic, count exceeds the Java array limit,
     *     gradient request is ineligible, or delegated label validation fails
     * @throws ArithmeticException if checked count, stride, or span arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if source or destination allocation fails
     */
    static Tensor ones(
            Shape shape, DataType dataType, Optional<String> label, boolean requiresGrad) {
        TensorDescriptor descriptor = descriptor(shape, dataType, requiresGrad);
        int length = (int) descriptor.layout().orElseThrow().referencedElementSpan();
        return importOnes(descriptor, label, length);
    }

    /**
     * Validates physical constant geometry and constructs its canonical dense descriptor.
     *
     * @param shape non-null proposed constant shape
     * @param dataType non-null exact constant data type
     * @param requiresGrad explicit gradient request validated by {@link TensorDescriptor}
     * @return a new resolved canonical dense descriptor
     * @throws IllegalArgumentException if shape is dynamic, count exceeds the Java array limit, or
     *     gradient request is ineligible
     * @throws ArithmeticException if checked count, stride, or span arithmetic overflows
     */
    private static TensorDescriptor descriptor(
            Shape shape, DataType dataType, boolean requiresGrad) {
        if (!shape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    "constant tensor creation requires a fully static shape: " + shape);
        }

        long elementCount = shape.knownElementCount().orElseThrow();
        if (elementCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "constant tensor element count exceeds Java array limit: required="
                            + elementCount
                            + ", maximum="
                            + Integer.MAX_VALUE);
        }

        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        return new TensorDescriptor(
                dataType, shape, Optional.of(layout), requiresGrad);
    }

    /**
     * Creates, fills, and imports exactly one carrier selected by the descriptor data type.
     *
     * @param descriptor validated canonical dense descriptor
     * @param label non-null optional label delegated to flat import
     * @param length validated Java-array element count
     * @return fresh independent tensor returned by exactly one matching flat import overload
     * @throws IllegalArgumentException if delegated import or label validation fails
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if source or destination allocation fails
     */
    private static Tensor importOnes(
            TensorDescriptor descriptor, Optional<String> label, int length) {
        return switch (descriptor.dataType()) {
            case FLOAT64 -> {
                double[] source = new double[length];
                Arrays.fill(source, 1.0d);
                yield TensorFactory.fromFlatArray(descriptor, label, source);
            }
            case FLOAT32 -> {
                float[] source = new float[length];
                Arrays.fill(source, 1.0f);
                yield TensorFactory.fromFlatArray(descriptor, label, source);
            }
            case BFLOAT16 -> {
                short[] source = new short[length];
                Arrays.fill(source, BFloat16Bits.fromFloat(1.0f));
                yield TensorFactory.fromFlatArray(descriptor, label, source);
            }
            case INT32 -> {
                int[] source = new int[length];
                Arrays.fill(source, 1);
                yield TensorFactory.fromFlatArray(descriptor, label, source);
            }
            case INT64 -> {
                long[] source = new long[length];
                Arrays.fill(source, 1L);
                yield TensorFactory.fromFlatArray(descriptor, label, source);
            }
            case BOOL -> {
                byte[] source = new byte[length];
                Arrays.fill(source, (byte) 1);
                yield TensorFactory.fromFlatArray(descriptor, label, source);
            }
        };
    }
}
