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
 * allocating a source carrier. Zeros delegate directly to heap allocation. Scalars and ones use
 * one exact carrier and one matching flat import. Each full-value entry allocates one exact
 * carrier, fills every position with {@link Arrays#fill}, and invokes one matching flat import.
 * Identity creation allocates one default-zero exact carrier, writes typed one only on the
 * row-major main diagonal, and invokes one matching flat import. Every result therefore receives
 * new descriptor, layout, source and destination carriers where applicable, storage, Tensor, and
 * identifier objects; no template or source carrier is retained.</p>
 *
 * <p>Descriptor validation requires a fully static shape, obtains the checked logical element
 * count, enforces {@link Integer#MAX_VALUE}, constructs canonical contiguous geometry, and applies
 * {@link TensorDescriptor} gradient eligibility in that order. It completes before source,
 * destination, or ID allocation. Identity validates a negative row count and then a negative
 * column count before creating its rank-two shape and entering that descriptor path. A blank label
 * is rejected only after source creation where applicable, destination allocation, and ID
 * allocation, and consumes that ID. Identifier exhaustion is observed after the source and
 * destination carriers exist but before publication or copying; no identifier is rolled back. An
 * unexpected copy failure also occurs after ID allocation. A JVM allocation failure before ID
 * allocation consumes no ID. All successful public initializers remain provenance-free leaves
 * and no operation, graph, runtime, or backend state is created here.</p>
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
     * Creates a dense FLOAT64 tensor filled from one exact binary64 scalar.
     *
     * <p>Descriptor validation completes before one {@code double[]} source is allocated and
     * filled with the exact value. One matching flat import then creates and copies into a new
     * destination. Scalar shapes contain one value; static shapes with a zero extent use empty
     * source and destination arrays. The source is neither exposed nor retained.</p>
     *
     * @param shape non-null fully static shape; scalar and empty shapes are accepted
     * @param value exact binary64 fill value, including signed-zero and NaN payload bits
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit model-level gradient request
     * @return a non-null fresh independent provenance-free dense tensor filled with {@code value}
     * @throws NullPointerException if an internal non-null shape or label precondition is violated
     * @throws IllegalArgumentException if the shape is dynamic, its count exceeds the Java array
     *     limit, the gradient request is ineligible, or delegated label validation rejects blank text
     * @throws ArithmeticException if checked element-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after destination allocation
     * @throws OutOfMemoryError if source or destination allocation fails before ID allocation
     */
    static Tensor full(
            Shape shape, double value, Optional<String> label, boolean requiresGrad) {
        TensorDescriptor descriptor = descriptor(shape, DataType.FLOAT64, requiresGrad);
        double[] source = new double[elementCount(descriptor)];
        Arrays.fill(source, value);
        return TensorFactory.fromFlatArray(descriptor, label, source);
    }

    /**
     * Creates a dense FLOAT32 tensor filled from one exact binary32 scalar.
     *
     * <p>Descriptor validation precedes allocation of one exactly filled {@code float[]} source.
     * One matching flat import copies it into a new destination. Scalar and empty static shapes
     * retain their requested shape semantics, and the source is not retained.</p>
     *
     * @param shape non-null fully static shape; scalar and empty shapes are accepted
     * @param value exact binary32 fill value, including signed-zero and NaN payload bits
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit model-level gradient request
     * @return a non-null fresh independent provenance-free dense tensor filled with {@code value}
     * @throws NullPointerException if an internal non-null shape or label precondition is violated
     * @throws IllegalArgumentException if the shape is dynamic, its count exceeds the Java array
     *     limit, the gradient request is ineligible, or delegated label validation rejects blank text
     * @throws ArithmeticException if checked element-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after destination allocation
     * @throws OutOfMemoryError if source or destination allocation fails before ID allocation
     */
    static Tensor full(
            Shape shape, float value, Optional<String> label, boolean requiresGrad) {
        TensorDescriptor descriptor = descriptor(shape, DataType.FLOAT32, requiresGrad);
        float[] source = new float[elementCount(descriptor)];
        Arrays.fill(source, value);
        return TensorFactory.fromFlatArray(descriptor, label, source);
    }

    /**
     * Creates a dense BFLOAT16 tensor by converting and repeating one binary32 semantic value.
     *
     * <p>After descriptor validation, the semantic value is converted once with
     * {@link BFloat16Bits#fromFloat(float)}, one {@code short[]} source is filled with those raw
     * bits, and one matching flat import copies it. Scalar and empty static shapes are supported;
     * the raw source carrier is neither exposed nor retained.</p>
     *
     * @param shape non-null fully static shape; scalar and empty shapes are accepted
     * @param value binary32 value converted once with {@link BFloat16Bits#fromFloat(float)}
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit model-level gradient request
     * @return a non-null fresh independent provenance-free dense tensor filled with converted bits
     * @throws NullPointerException if an internal non-null shape or label precondition is violated
     * @throws IllegalArgumentException if the shape is dynamic, its count exceeds the Java array
     *     limit, the gradient request is ineligible, or delegated label validation rejects blank text
     * @throws ArithmeticException if checked element-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after destination allocation
     * @throws OutOfMemoryError if source or destination allocation fails before ID allocation
     */
    static Tensor fullBFloat16(
            Shape shape, float value, Optional<String> label, boolean requiresGrad) {
        TensorDescriptor descriptor = descriptor(shape, DataType.BFLOAT16, requiresGrad);
        short[] source = new short[elementCount(descriptor)];
        Arrays.fill(source, BFloat16Bits.fromFloat(value));
        return TensorFactory.fromFlatArray(descriptor, label, source);
    }

    /**
     * Creates a dense INT32 tensor filled from one exact signed 32-bit scalar.
     *
     * <p>Descriptor validation, including the required false gradient request, precedes one
     * exactly filled {@code int[]} source and one matching flat import. Scalar and empty static
     * shapes are supported, and the source is not retained.</p>
     *
     * @param shape non-null fully static shape; scalar and empty shapes are accepted
     * @param value exact signed 32-bit fill value
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit request, which must be false for INT32
     * @return a non-null fresh independent provenance-free dense tensor filled with {@code value}
     * @throws NullPointerException if an internal non-null shape or label precondition is violated
     * @throws IllegalArgumentException if the shape is dynamic, its count exceeds the Java array
     *     limit, gradients are requested, or delegated label validation rejects blank text
     * @throws ArithmeticException if checked element-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after destination allocation
     * @throws OutOfMemoryError if source or destination allocation fails before ID allocation
     */
    static Tensor full(
            Shape shape, int value, Optional<String> label, boolean requiresGrad) {
        TensorDescriptor descriptor = descriptor(shape, DataType.INT32, requiresGrad);
        int[] source = new int[elementCount(descriptor)];
        Arrays.fill(source, value);
        return TensorFactory.fromFlatArray(descriptor, label, source);
    }

    /**
     * Creates a dense INT64 tensor filled from one exact signed 64-bit scalar.
     *
     * <p>Descriptor validation, including the required false gradient request, precedes one
     * exactly filled {@code long[]} source and one matching flat import. Scalar and empty static
     * shapes are supported, and the source is not retained.</p>
     *
     * @param shape non-null fully static shape; scalar and empty shapes are accepted
     * @param value exact signed 64-bit fill value
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit request, which must be false for INT64
     * @return a non-null fresh independent provenance-free dense tensor filled with {@code value}
     * @throws NullPointerException if an internal non-null shape or label precondition is violated
     * @throws IllegalArgumentException if the shape is dynamic, its count exceeds the Java array
     *     limit, gradients are requested, or delegated label validation rejects blank text
     * @throws ArithmeticException if checked element-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after destination allocation
     * @throws OutOfMemoryError if source or destination allocation fails before ID allocation
     */
    static Tensor full(
            Shape shape, long value, Optional<String> label, boolean requiresGrad) {
        TensorDescriptor descriptor = descriptor(shape, DataType.INT64, requiresGrad);
        long[] source = new long[elementCount(descriptor)];
        Arrays.fill(source, value);
        return TensorFactory.fromFlatArray(descriptor, label, source);
    }

    /**
     * Creates a dense BOOL tensor filled from one semantic boolean scalar.
     *
     * <p>Descriptor validation, including the required false gradient request, precedes one
     * {@code byte[]} source filled with canonical zero or one and one matching BOOL flat import.
     * Scalar and empty static shapes are supported, and the source is not retained.</p>
     *
     * @param shape non-null fully static shape; scalar and empty shapes are accepted
     * @param value semantic fill value mapped only to canonical byte zero or one
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit request, which must be false for BOOL
     * @return a non-null fresh independent provenance-free dense tensor filled with {@code value}
     * @throws NullPointerException if an internal non-null shape or label precondition is violated
     * @throws IllegalArgumentException if the shape is dynamic, its count exceeds the Java array
     *     limit, gradients are requested, or delegated label validation rejects blank text
     * @throws ArithmeticException if checked element-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after destination allocation
     * @throws OutOfMemoryError if source or destination allocation fails before ID allocation
     */
    static Tensor full(
            Shape shape, boolean value, Optional<String> label, boolean requiresGrad) {
        TensorDescriptor descriptor = descriptor(shape, DataType.BOOL, requiresGrad);
        byte[] source = new byte[elementCount(descriptor)];
        Arrays.fill(source, value ? (byte) 1 : (byte) 0);
        return TensorFactory.fromFlatArray(descriptor, label, source);
    }

    /**
     * Creates a rank-two dense rectangular identity matrix for any current data type.
     *
     * <p>Rows and columns are validated before shape and descriptor construction. One default-zero
     * exact carrier is allocated, typed one is written only at row-major main-diagonal positions,
     * and the carrier is delegated once to matching flat import. The diagonal positions are
     * {@code (i, i)} for {@code 0 <= i < min(rows, columns)}; all remaining positions keep their
     * exact JVM default-zero representation. Square, wide, tall, and zero-element matrices are
     * supported.</p>
     *
     * @param rows non-negative row count
     * @param columns non-negative column count
     * @param dataType non-null exact element type
     * @param label non-null optional label already checked by the public boundary
     * @param requiresGrad explicit model-level gradient request
     * @return a non-null fresh independent provenance-free dense matrix with typed ones on its
     *     main diagonal and typed zeros elsewhere
     * @throws NullPointerException if an internal non-null data-type or label precondition is violated
     * @throws IllegalArgumentException if a dimension is negative, count exceeds the Java array
     *     limit, gradient is ineligible, or delegated label validation fails
     * @throws ArithmeticException if checked element-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted after destination allocation
     * @throws OutOfMemoryError if source or destination allocation fails before ID allocation
     */
    static Tensor identityMatrix(
            long rows,
            long columns,
            DataType dataType,
            Optional<String> label,
            boolean requiresGrad) {
        if (rows < 0) {
            throw new IllegalArgumentException(
                    "identity matrix rows must be non-negative: " + rows);
        }
        if (columns < 0) {
            throw new IllegalArgumentException(
                    "identity matrix columns must be non-negative: " + columns);
        }

        TensorDescriptor descriptor = descriptor(Shape.of(rows, columns), dataType, requiresGrad);
        int length = elementCount(descriptor);
        int diagonalLength = (int) Math.min(rows, columns);
        return importIdentity(descriptor, label, length, diagonalLength, columns);
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
     * Returns the validated Java-array-sized logical count of a canonical dense descriptor.
     *
     * @param descriptor non-null fully static canonical dense descriptor created by
     *     {@link #descriptor(Shape, DataType, boolean)}; it is not retained
     * @return non-negative logical element count narrowed exactly to a Java array length
     * @throws java.util.NoSuchElementException if the internal fully static descriptor precondition
     *     is violated
     */
    private static int elementCount(TensorDescriptor descriptor) {
        return (int) descriptor.shape().knownElementCount().orElseThrow();
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

    /**
     * Allocates one default-zero exact carrier, writes typed diagonal ones, and imports it once.
     *
     * <p>The descriptor's data type selects exactly one primitive carrier and one flat-import
     * overload. Off-diagonal positions are never written. The source carrier is not exposed or
     * retained; flat import creates an independent destination and provenance-free Tensor.</p>
     *
     * @param descriptor non-null validated rank-two canonical dense descriptor selecting the carrier
     * @param label non-null optional label delegated unchanged to flat import
     * @param length non-negative validated Java-array-sized logical element count
     * @param diagonalLength non-negative number of main-diagonal positions to populate; no greater
     *     than either matrix extent
     * @param columns non-negative row-major column stride; validated index arithmetic fits
     *     {@code length}
     * @return non-null fresh matrix returned by exactly one matching flat-import overload
     * @throws IllegalArgumentException if delegated import or label validation fails
     * @throws IllegalStateException if tensor identifier space is exhausted
     * @throws OutOfMemoryError if source or destination allocation fails
     */
    private static Tensor importIdentity(
            TensorDescriptor descriptor,
            Optional<String> label,
            int length,
            int diagonalLength,
            long columns) {
        return switch (descriptor.dataType()) {
            case FLOAT64 -> {
                double[] source = new double[length];
                fillIdentityDiagonal(source, diagonalLength, columns);
                yield TensorFactory.fromFlatArray(descriptor, label, source);
            }
            case FLOAT32 -> {
                float[] source = new float[length];
                fillIdentityDiagonal(source, diagonalLength, columns);
                yield TensorFactory.fromFlatArray(descriptor, label, source);
            }
            case BFLOAT16 -> {
                short[] source = new short[length];
                fillIdentityDiagonal(source, diagonalLength, columns);
                yield TensorFactory.fromFlatArray(descriptor, label, source);
            }
            case INT32 -> {
                int[] source = new int[length];
                fillIdentityDiagonal(source, diagonalLength, columns);
                yield TensorFactory.fromFlatArray(descriptor, label, source);
            }
            case INT64 -> {
                long[] source = new long[length];
                fillIdentityDiagonal(source, diagonalLength, columns);
                yield TensorFactory.fromFlatArray(descriptor, label, source);
            }
            case BOOL -> {
                byte[] source = new byte[length];
                fillIdentityDiagonal(source, diagonalLength, columns);
                yield TensorFactory.fromFlatArray(descriptor, label, source);
            }
        };
    }

    /**
     * Writes exact binary64 one at every validated row-major main-diagonal position.
     *
     * @param values non-null default-zero destination carrier; off-diagonal values remain unchanged
     * @param diagonalLength non-negative number of diagonal positions to write
     * @param columns non-negative matrix column count used as the row-major stride
     */
    private static void fillIdentityDiagonal(double[] values, int diagonalLength, long columns) {
        for (int index = 0; index < diagonalLength; index++) {
            values[(int) (index * columns + index)] = 1.0d;
        }
    }

    /**
     * Writes exact binary32 one at every validated row-major main-diagonal position.
     *
     * @param values non-null default-zero destination carrier; off-diagonal values remain unchanged
     * @param diagonalLength non-negative number of diagonal positions to write
     * @param columns non-negative matrix column count used as the row-major stride
     */
    private static void fillIdentityDiagonal(float[] values, int diagonalLength, long columns) {
        for (int index = 0; index < diagonalLength; index++) {
            values[(int) (index * columns + index)] = 1.0f;
        }
    }

    /**
     * Writes BFLOAT16 one bits at every validated row-major main-diagonal position.
     *
     * @param values non-null default-zero raw-bit carrier; off-diagonal bits remain zero
     * @param diagonalLength non-negative number of diagonal positions to write
     * @param columns non-negative matrix column count used as the row-major stride
     */
    private static void fillIdentityDiagonal(short[] values, int diagonalLength, long columns) {
        short one = BFloat16Bits.fromFloat(1.0f);
        for (int index = 0; index < diagonalLength; index++) {
            values[(int) (index * columns + index)] = one;
        }
    }

    /**
     * Writes signed 32-bit one at every validated row-major main-diagonal position.
     *
     * @param values non-null default-zero destination carrier; off-diagonal values remain unchanged
     * @param diagonalLength non-negative number of diagonal positions to write
     * @param columns non-negative matrix column count used as the row-major stride
     */
    private static void fillIdentityDiagonal(int[] values, int diagonalLength, long columns) {
        for (int index = 0; index < diagonalLength; index++) {
            values[(int) (index * columns + index)] = 1;
        }
    }

    /**
     * Writes signed 64-bit one at every validated row-major main-diagonal position.
     *
     * @param values non-null default-zero destination carrier; off-diagonal values remain unchanged
     * @param diagonalLength non-negative number of diagonal positions to write
     * @param columns non-negative matrix column count used as the row-major stride
     */
    private static void fillIdentityDiagonal(long[] values, int diagonalLength, long columns) {
        for (int index = 0; index < diagonalLength; index++) {
            values[(int) (index * columns + index)] = 1L;
        }
    }

    /**
     * Writes canonical BOOL byte one at every validated row-major main-diagonal position.
     *
     * @param values non-null default-zero BOOL carrier; off-diagonal bytes remain canonical zero
     * @param diagonalLength non-negative number of diagonal positions to write
     * @param columns non-negative matrix column count used as the row-major stride
     */
    private static void fillIdentityDiagonal(byte[] values, int diagonalLength, long columns) {
        for (int index = 0; index < diagonalLength; index++) {
            values[(int) (index * columns + index)] = 1;
        }
    }
}
