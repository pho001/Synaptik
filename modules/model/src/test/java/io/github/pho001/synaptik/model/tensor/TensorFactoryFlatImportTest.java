package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.reflect.Field;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class TensorFactoryFlatImportTest {
    @Test
    void copiesEveryNumericCarrierAndRawBfloat16BitsIntoIndependentHeapArrays() {
        double[] float64 = {
            -0.0d, Double.longBitsToDouble(0x7ff8_0000_0000_0042L), Double.POSITIVE_INFINITY
        };
        float[] float32 = {
            -0.0f, Float.intBitsToFloat(0x7fc0_0042), Float.NEGATIVE_INFINITY
        };
        short[] bfloat16 = {(short) 0x8000, (short) 0x7fc1, (short) 0xffff};
        int[] int32 = {Integer.MIN_VALUE, -2, -1, 0, 1, Integer.MAX_VALUE};
        long[] int64 = {Long.MIN_VALUE, -1L, Long.MAX_VALUE};

        Tensor float64Tensor = TensorFactory.fromFlatArray(
                dense(DataType.FLOAT64, Shape.of(3)), Optional.empty(), float64);
        Tensor float32Tensor = TensorFactory.fromFlatArray(
                dense(DataType.FLOAT32, Shape.of(3)), Optional.empty(), float32);
        Tensor bfloat16Tensor = TensorFactory.fromFlatArray(
                dense(DataType.BFLOAT16, Shape.of(3)), Optional.empty(), bfloat16);
        Tensor int32Tensor = TensorFactory.fromFlatArray(
                dense(DataType.INT32, Shape.of(2, 3)), Optional.empty(), int32);
        Tensor int64Tensor = TensorFactory.fromFlatArray(
                dense(DataType.INT64, Shape.of(3)), Optional.empty(), int64);

        double[] float64Copy = heapArray(float64Tensor, double[].class);
        float[] float32Copy = heapArray(float32Tensor, float[].class);
        short[] bfloat16Copy = heapArray(bfloat16Tensor, short[].class);
        int[] int32Copy = heapArray(int32Tensor, int[].class);
        long[] int64Copy = heapArray(int64Tensor, long[].class);

        assertAll(
                () -> assertNotSame(float64, float64Copy),
                () -> assertEquals(
                        Double.doubleToRawLongBits(float64[0]),
                        Double.doubleToRawLongBits(float64Copy[0])),
                () -> assertEquals(
                        Double.doubleToRawLongBits(float64[1]),
                        Double.doubleToRawLongBits(float64Copy[1])),
                () -> assertEquals(
                        Double.doubleToRawLongBits(float64[2]),
                        Double.doubleToRawLongBits(float64Copy[2])),
                () -> assertNotSame(float32, float32Copy),
                () -> assertEquals(
                        Float.floatToRawIntBits(float32[0]),
                        Float.floatToRawIntBits(float32Copy[0])),
                () -> assertEquals(
                        Float.floatToRawIntBits(float32[1]),
                        Float.floatToRawIntBits(float32Copy[1])),
                () -> assertEquals(
                        Float.floatToRawIntBits(float32[2]),
                        Float.floatToRawIntBits(float32Copy[2])),
                () -> assertNotSame(bfloat16, bfloat16Copy),
                () -> assertArrayEquals(bfloat16, bfloat16Copy),
                () -> assertNotSame(int32, int32Copy),
                () -> assertArrayEquals(int32, int32Copy),
                () -> assertNotSame(int64, int64Copy),
                () -> assertArrayEquals(int64, int64Copy));

        float64[0] = 17.0d;
        float32[0] = 17.0f;
        bfloat16[0] = 17;
        int32[0] = 17;
        int64[0] = 17L;

        assertAll(
                () -> assertEquals(
                        Double.doubleToRawLongBits(-0.0d),
                        Double.doubleToRawLongBits(float64Copy[0])),
                () -> assertEquals(
                        Float.floatToRawIntBits(-0.0f),
                        Float.floatToRawIntBits(float32Copy[0])),
                () -> assertEquals((short) 0x8000, bfloat16Copy[0]),
                () -> assertEquals(Integer.MIN_VALUE, int32Copy[0]),
                () -> assertEquals(Long.MIN_VALUE, int64Copy[0]));
    }

    @Test
    void normalizesBoolBytesWithoutRetainingOrMutatingTheSource() {
        byte[] source = {0, -2, 3, Byte.MIN_VALUE, Byte.MAX_VALUE};
        byte[] original = source.clone();

        Tensor tensor = TensorFactory.fromFlatArray(
                dense(DataType.BOOL, Shape.of(5)), Optional.of("  mask\n"), source);
        byte[] destination = heapArray(tensor, byte[].class);

        assertAll(
                () -> assertEquals(Optional.of("mask"), tensor.label()),
                () -> assertNotSame(source, destination),
                () -> assertArrayEquals(new byte[] {0, 1, 1, 1, 1}, destination),
                () -> assertArrayEquals(original, source));

        source[0] = 9;
        source[1] = 0;
        assertArrayEquals(new byte[] {0, 1, 1, 1, 1}, destination);
    }

    @Test
    void importsScalarAndEmptyDenseTensors() {
        Tensor scalar = TensorFactory.fromFlatArray(
                dense(DataType.FLOAT64, Shape.scalar()),
                Optional.empty(),
                new double[] {4.25d});
        Tensor empty = TensorFactory.fromFlatArray(
                dense(DataType.INT32, Shape.of(2, 0, 4)),
                Optional.empty(),
                new int[0]);

        assertAll(
                () -> assertArrayEquals(new double[] {4.25d}, heapArray(scalar, double[].class)),
                () -> assertEquals(0, empty.hostStorage().orElseThrow().elementCapacity()),
                () -> assertEquals(0, heapArray(empty, int[].class).length));
    }

    @Test
    void rejectsNullArgumentsInOrderForEveryCarrierWithoutConsumingIdentifiers()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        TensorDescriptor float64 = dense(DataType.FLOAT64, Shape.scalar());
        TensorDescriptor float32 = dense(DataType.FLOAT32, Shape.scalar());
        TensorDescriptor bfloat16 = dense(DataType.BFLOAT16, Shape.scalar());
        TensorDescriptor int32 = dense(DataType.INT32, Shape.scalar());
        TensorDescriptor int64 = dense(DataType.INT64, Shape.scalar());
        TensorDescriptor bool = dense(DataType.BOOL, Shape.scalar());

        assertNullFailures(
                () -> TensorFactory.fromFlatArray(null, null, (double[]) null),
                () -> TensorFactory.fromFlatArray(float64, null, (double[]) null),
                () -> TensorFactory.fromFlatArray(float64, Optional.empty(), (double[]) null));
        assertNullFailures(
                () -> TensorFactory.fromFlatArray(null, null, (float[]) null),
                () -> TensorFactory.fromFlatArray(float32, null, (float[]) null),
                () -> TensorFactory.fromFlatArray(float32, Optional.empty(), (float[]) null));
        assertNullFailures(
                () -> TensorFactory.fromFlatArray(null, null, (short[]) null),
                () -> TensorFactory.fromFlatArray(bfloat16, null, (short[]) null),
                () -> TensorFactory.fromFlatArray(bfloat16, Optional.empty(), (short[]) null));
        assertNullFailures(
                () -> TensorFactory.fromFlatArray(null, null, (int[]) null),
                () -> TensorFactory.fromFlatArray(int32, null, (int[]) null),
                () -> TensorFactory.fromFlatArray(int32, Optional.empty(), (int[]) null));
        assertNullFailures(
                () -> TensorFactory.fromFlatArray(null, null, (long[]) null),
                () -> TensorFactory.fromFlatArray(int64, null, (long[]) null),
                () -> TensorFactory.fromFlatArray(int64, Optional.empty(), (long[]) null));
        assertNullFailures(
                () -> TensorFactory.fromFlatArray(null, null, (byte[]) null),
                () -> TensorFactory.fromFlatArray(bool, null, (byte[]) null),
                () -> TensorFactory.fromFlatArray(bool, Optional.empty(), (byte[]) null));

        assertEquals(before, next.get());
    }

    @Test
    void rejectsEveryWrongCarrierBeforeLayoutAndAllocationWithExactMessages()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        TensorDescriptor float32Unresolved = unresolved(DataType.FLOAT32, Shape.scalar());
        TensorDescriptor bfloat16Unresolved = unresolved(DataType.BFLOAT16, Shape.scalar());
        TensorDescriptor int32Unresolved = unresolved(DataType.INT32, Shape.scalar());
        TensorDescriptor int64Unresolved = unresolved(DataType.INT64, Shape.scalar());
        TensorDescriptor boolUnresolved = unresolved(DataType.BOOL, Shape.scalar());
        TensorDescriptor float64Unresolved = unresolved(DataType.FLOAT64, Shape.scalar());

        IllegalArgumentException float64 = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromFlatArray(
                        float32Unresolved, Optional.empty(), new double[0]));
        IllegalArgumentException float32 = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromFlatArray(
                        bfloat16Unresolved, Optional.empty(), new float[0]));
        IllegalArgumentException bfloat16 = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromFlatArray(
                        int32Unresolved, Optional.empty(), new short[0]));
        IllegalArgumentException int32 = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromFlatArray(
                        int64Unresolved, Optional.empty(), new int[0]));
        IllegalArgumentException int64 = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromFlatArray(
                        boolUnresolved, Optional.empty(), new long[0]));
        IllegalArgumentException bool = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromFlatArray(
                        float64Unresolved, Optional.empty(), new byte[0]));

        assertAll(
                () -> assertEquals(
                        "flat source data type must match descriptor: expected=FLOAT64, actual=FLOAT32",
                        float64.getMessage()),
                () -> assertEquals(
                        "flat source data type must match descriptor: expected=FLOAT32, actual=BFLOAT16",
                        float32.getMessage()),
                () -> assertEquals(
                        "flat source data type must match descriptor: expected=BFLOAT16, actual=INT32",
                        bfloat16.getMessage()),
                () -> assertEquals(
                        "flat source data type must match descriptor: expected=INT32, actual=INT64",
                        int32.getMessage()),
                () -> assertEquals(
                        "flat source data type must match descriptor: expected=INT64, actual=BOOL",
                        int64.getMessage()),
                () -> assertEquals(
                        "flat source data type must match descriptor: expected=BOOL, actual=FLOAT64",
                        bool.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void rejectsUnresolvedOffsetStridedAndBroadcastLayoutsBeforeLengthAndAllocation()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Shape shape = Shape.of(2, 3);
        TensorDescriptor unresolved = unresolved(DataType.FLOAT32, shape);
        TensorDescriptor offset = resolved(
                DataType.FLOAT32,
                shape,
                LayoutDescriptor.of(shape, new long[] {3, 1}, 2, true));
        TensorDescriptor strided = resolved(
                DataType.FLOAT32,
                shape,
                LayoutDescriptor.of(shape, new long[] {1, 2}, 0, true));
        TensorDescriptor broadcast = resolved(
                DataType.FLOAT32,
                shape,
                LayoutDescriptor.of(shape, new long[] {0, 1}, 0, true));

        IllegalArgumentException unresolvedFailure = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromFlatArray(unresolved, Optional.empty(), new float[6]));
        IllegalArgumentException offsetFailure = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromFlatArray(offset, Optional.empty(), new float[0]));
        IllegalArgumentException stridedFailure = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromFlatArray(strided, Optional.empty(), new float[0]));
        IllegalArgumentException broadcastFailure = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromFlatArray(broadcast, Optional.empty(), new float[0]));

        assertAll(
                () -> assertEquals(
                        "flat tensor import requires a resolved layout",
                        unresolvedFailure.getMessage()),
                () -> assertEquals(
                        "flat tensor import requires dense-contiguous layout: actual=DENSE_WITH_OFFSET",
                        offsetFailure.getMessage()),
                () -> assertEquals(
                        "flat tensor import requires dense-contiguous layout: actual=STRIDED",
                        stridedFailure.getMessage()),
                () -> assertEquals(
                        "flat tensor import requires dense-contiguous layout: actual=BROADCAST_ZERO_STRIDE",
                        broadcastFailure.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void rejectsTooShortAndTooLongSourcesBeforeAllocationWithExactMessages()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        TensorDescriptor descriptor = dense(DataType.INT64, Shape.of(2, 3));

        IllegalArgumentException tooShort = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromFlatArray(
                        descriptor, Optional.empty(), new long[5]));
        IllegalArgumentException tooLong = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromFlatArray(
                        descriptor, Optional.empty(), new long[7]));

        assertAll(
                () -> assertEquals(
                        "flat source length must equal logical element count: required=6, actual=5",
                        tooShort.getMessage()),
                () -> assertEquals(
                        "flat source length must equal logical element count: required=6, actual=7",
                        tooLong.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void delegatesBlankLabelAfterDestinationAndIdAllocation() throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        TensorDescriptor descriptor = dense(DataType.BOOL, Shape.of(3));

        IllegalArgumentException blank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromFlatArray(
                        descriptor, Optional.of(" \t\n "), new byte[] {0, 2, -1}));
        Tensor subsequent = TensorFactory.create(unresolved(DataType.FLOAT32, Shape.scalar()));

        assertAll(
                () -> assertEquals("label must not be blank", blank.getMessage()),
                () -> assertEquals(before + 1, subsequent.id().value()),
                () -> assertEquals(before + 2, next.get()));
    }

    @Test
    void propagatesIdentifierExhaustionFromAllocationBeforePopulation()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        AtomicBoolean maximumClaimed = maximumTensorIdClaimedState();
        long originalNext = next.get();
        boolean originalMaximumClaimed = maximumClaimed.get();
        byte[] source = {7};
        try {
            next.set(Long.MAX_VALUE);
            maximumClaimed.set(true);

            IllegalStateException exhausted = assertThrows(
                    IllegalStateException.class,
                    () -> TensorFactory.fromFlatArray(
                            dense(DataType.BOOL, Shape.scalar()), Optional.empty(), source));

            assertAll(
                    () -> assertEquals(
                            "tensor identifier space exhausted", exhausted.getMessage()),
                    () -> assertArrayEquals(new byte[] {7}, source),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(maximumClaimed.get()));
        } finally {
            next.set(originalNext);
            maximumClaimed.set(originalMaximumClaimed);
        }
    }

    @Test
    void acceptsAResolvedDenseDescriptorWithGradientEligibilityUnchanged() {
        Shape shape = Shape.of(2);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true);

        Tensor tensor = TensorFactory.fromFlatArray(
                descriptor, Optional.empty(), new float[] {1.0f, 2.0f});

        assertAll(
                () -> assertSame(descriptor, tensor.descriptor()),
                () -> assertTrue(tensor.descriptor().requiresGrad()),
                () -> assertArrayEquals(
                        new float[] {1.0f, 2.0f}, heapArray(tensor, float[].class)));
    }

    private static void assertNullFailures(
            ThrowingInvocation nullDescriptor,
            ThrowingInvocation nullLabel,
            ThrowingInvocation nullSource) {
        NullPointerException descriptorFailure = assertThrows(
                NullPointerException.class, nullDescriptor::invoke);
        NullPointerException labelFailure = assertThrows(
                NullPointerException.class, nullLabel::invoke);
        NullPointerException sourceFailure = assertThrows(
                NullPointerException.class, nullSource::invoke);

        assertAll(
                () -> assertEquals("descriptor", descriptorFailure.getMessage()),
                () -> assertEquals("label", labelFailure.getMessage()),
                () -> assertEquals("source", sourceFailure.getMessage()));
    }

    private static TensorDescriptor dense(DataType dataType, Shape shape) {
        return resolved(dataType, shape, LayoutDescriptor.contiguous(shape));
    }

    private static TensorDescriptor unresolved(DataType dataType, Shape shape) {
        return new TensorDescriptor(dataType, shape, Optional.empty(), false);
    }

    private static TensorDescriptor resolved(
            DataType dataType, Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(dataType, shape, Optional.of(layout), false);
    }

    private static <T> T heapArray(Tensor tensor, Class<T> carrier) {
        Object heapBase = tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow();
        return assertInstanceOf(carrier, heapBase);
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumTensorIdClaimedState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }

    @FunctionalInterface
    private interface ThrowingInvocation {
        void invoke();
    }
}
