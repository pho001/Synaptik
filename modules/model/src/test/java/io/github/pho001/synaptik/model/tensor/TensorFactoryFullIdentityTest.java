package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.Arrays;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class TensorFactoryFullIdentityTest {
    @Test
    void createsEveryExactFullCarrierWithDenseLeafMetadata() {
        Shape shape = Shape.of(2, 3);
        double doubleValue = Double.longBitsToDouble(0x7ff8_0000_0000_0042L);
        float floatValue = Float.intBitsToFloat(0x8000_0000);
        float bfloatValue = Float.intBitsToFloat(0x3f80_8000);

        Tensor float64 = TensorFactory.full(shape, doubleValue, Optional.of("  full  "), true);
        Tensor float32 = TensorFactory.full(shape, floatValue, Optional.empty(), true);
        Tensor bfloat16 = TensorFactory.fullBFloat16(shape, bfloatValue, Optional.empty(), true);
        Tensor int32 = TensorFactory.full(shape, Integer.MIN_VALUE, Optional.empty(), false);
        Tensor int64 = TensorFactory.full(shape, Long.MAX_VALUE, Optional.empty(), false);
        Tensor boolFalse = TensorFactory.full(shape, false, Optional.empty(), false);
        Tensor boolTrue = TensorFactory.full(shape, true, Optional.empty(), false);

        assertAll(
                () -> assertDenseLeaf(float64, shape, DataType.FLOAT64, true),
                () -> assertEquals(Optional.of("full"), float64.label()),
                () -> assertRawDoubleFill(heapArray(float64, double[].class), doubleValue),
                () -> assertDenseLeaf(float32, shape, DataType.FLOAT32, true),
                () -> assertRawFloatFill(heapArray(float32, float[].class), floatValue),
                () -> assertArrayEquals(
                        filledShort(6, BFloat16Bits.fromFloat(bfloatValue)),
                        heapArray(bfloat16, short[].class)),
                () -> assertDenseLeaf(bfloat16, shape, DataType.BFLOAT16, true),
                () -> assertArrayEquals(
                        filledInt(6, Integer.MIN_VALUE), heapArray(int32, int[].class)),
                () -> assertDenseLeaf(int32, shape, DataType.INT32, false),
                () -> assertArrayEquals(
                        filledLong(6, Long.MAX_VALUE), heapArray(int64, long[].class)),
                () -> assertDenseLeaf(int64, shape, DataType.INT64, false),
                () -> assertArrayEquals(new byte[6], heapArray(boolFalse, byte[].class)),
                () -> assertArrayEquals(filledByte(6, (byte) 1), heapArray(boolTrue, byte[].class)),
                () -> assertDenseLeaf(boolFalse, shape, DataType.BOOL, false),
                () -> assertDenseLeaf(boolTrue, shape, DataType.BOOL, false));
    }

    @Test
    void fullSupportsScalarEmptyAndIndependentResults() {
        Tensor scalar = TensorFactory.full(Shape.scalar(), -0.0d, Optional.empty(), true);
        Shape emptyShape = Shape.of(2, 0, 4);
        Tensor empty = TensorFactory.full(emptyShape, 17L, Optional.empty(), false);
        Tensor first = TensorFactory.full(Shape.of(2), 9, Optional.empty(), false);
        Tensor second = TensorFactory.full(Shape.of(2), 9, Optional.empty(), false);

        assertAll(
                () -> assertEquals(
                        Double.doubleToRawLongBits(-0.0d),
                        Double.doubleToRawLongBits(heapArray(scalar, double[].class)[0])),
                () -> assertEquals(Shape.scalar(), scalar.descriptor().shape()),
                () -> assertEquals(0, heapArray(empty, long[].class).length),
                () -> assertEquals(emptyShape, empty.descriptor().shape()),
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(first.descriptor(), second.descriptor()),
                () -> assertNotSame(
                        first.descriptor().layout().orElseThrow(),
                        second.descriptor().layout().orElseThrow()),
                () -> assertNotSame(
                        first.hostStorage().orElseThrow(), second.hostStorage().orElseThrow()),
                () -> assertNotSame(heapBase(first), heapBase(second)));
    }

    @Test
    void createsSquareIdentityForEveryDataType() {
        for (DataType dataType : DataType.values()) {
            Tensor tensor = TensorFactory.identityMatrix(
                    3, 3, dataType, Optional.of("  identity  "), dataType.isDifferentiable());
            assertDenseLeaf(tensor, Shape.of(3, 3), dataType, dataType.isDifferentiable());
            assertEquals(Optional.of("identity"), tensor.label());
            assertIdentityValues(tensor, dataType, 3, 3);
        }
    }

    @Test
    void identitySupportsWideTallAndZeroElementRectangles() {
        Tensor wide = TensorFactory.identityMatrix(
                2, 4, DataType.INT32, Optional.empty(), false);
        Tensor tall = TensorFactory.identityMatrix(
                4, 2, DataType.FLOAT32, Optional.empty(), false);
        Tensor zeroRows = TensorFactory.identityMatrix(
                0, Long.MAX_VALUE, DataType.INT64, Optional.empty(), false);
        Tensor zeroColumns = TensorFactory.identityMatrix(
                Long.MAX_VALUE, 0, DataType.BOOL, Optional.empty(), false);

        assertAll(
                () -> assertArrayEquals(
                        new int[] {1, 0, 0, 0, 0, 1, 0, 0},
                        heapArray(wide, int[].class)),
                () -> assertArrayEquals(
                        new float[] {1, 0, 0, 1, 0, 0, 0, 0},
                        heapArray(tall, float[].class)),
                () -> assertEquals(Shape.of(0, Long.MAX_VALUE), zeroRows.descriptor().shape()),
                () -> assertEquals(0, heapArray(zeroRows, long[].class).length),
                () -> assertEquals(Shape.of(Long.MAX_VALUE, 0), zeroColumns.descriptor().shape()),
                () -> assertEquals(0, heapArray(zeroColumns, byte[].class).length));
    }

    @Test
    void eyeMatchesCanonicalBehaviorButReturnsFreshIdentity() {
        Tensor canonical = TensorFactory.identityMatrix(
                2, 3, DataType.BFLOAT16, Optional.of("matrix"), true);
        Tensor alias = TensorFactory.eye(
                2, 3, DataType.BFLOAT16, Optional.of("matrix"), true);

        assertAll(
                () -> assertEquals(canonical.descriptor(), alias.descriptor()),
                () -> assertEquals(canonical.label(), alias.label()),
                () -> assertArrayEquals(
                        heapArray(canonical, short[].class), heapArray(alias, short[].class)),
                () -> assertNotSame(canonical, alias),
                () -> assertNotEquals(canonical.id(), alias.id()),
                () -> assertNotSame(canonical.descriptor(), alias.descriptor()),
                () -> assertNotSame(
                        canonical.descriptor().layout().orElseThrow(),
                        alias.descriptor().layout().orElseThrow()),
                () -> assertNotSame(
                        canonical.hostStorage().orElseThrow(), alias.hostStorage().orElseThrow()),
                () -> assertNotSame(heapBase(canonical), heapBase(alias)));
    }

    @Test
    void rejectsNullsAndNegativeDimensionsInRequiredOrderWithoutIdentifierUse()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        NullPointerException fullShape = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.full(null, 1.0d, null, false));
        NullPointerException fullLabel = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.full(Shape.scalar(), 1.0d, null, false));
        NullPointerException identityType = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.identityMatrix(-1, -1, null, null, false));
        NullPointerException identityLabel = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.identityMatrix(-1, -1, DataType.FLOAT32, null, false));
        IllegalArgumentException rows = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.identityMatrix(
                        -1, -2, DataType.FLOAT32, Optional.empty(), false));
        IllegalArgumentException columns = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.eye(
                        1, -2, DataType.FLOAT32, Optional.empty(), false));

        assertAll(
                () -> assertEquals("shape", fullShape.getMessage()),
                () -> assertEquals("label", fullLabel.getMessage()),
                () -> assertEquals("dataType", identityType.getMessage()),
                () -> assertEquals("label", identityLabel.getMessage()),
                () -> assertEquals(
                        "identity matrix rows must be non-negative: -1", rows.getMessage()),
                () -> assertEquals(
                        "identity matrix columns must be non-negative: -2", columns.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void rejectsDynamicLimitOverflowAndIneligibleGradientBeforeIdentifierUse()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Shape dynamic = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(2));

        IllegalArgumentException dynamicFailure = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.full(dynamic, 1.0f, Optional.empty(), false));
        IllegalArgumentException fullLimit = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.full(
                        Shape.of((long) Integer.MAX_VALUE + 1),
                        false,
                        Optional.empty(),
                        false));
        IllegalArgumentException identityLimit = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.identityMatrix(
                        Integer.MAX_VALUE,
                        2,
                        DataType.FLOAT32,
                        Optional.empty(),
                        false));
        ArithmeticException overflow = assertThrows(
                ArithmeticException.class,
                () -> TensorFactory.identityMatrix(
                        Long.MAX_VALUE,
                        2,
                        DataType.FLOAT32,
                        Optional.empty(),
                        false));
        IllegalArgumentException gradient = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.identityMatrix(
                        1, 1, DataType.INT64, Optional.empty(), true));

        assertAll(
                () -> assertEquals(
                        "constant tensor creation requires a fully static shape: " + dynamic,
                        dynamicFailure.getMessage()),
                () -> assertEquals(
                        "constant tensor element count exceeds Java array limit: required="
                                + ((long) Integer.MAX_VALUE + 1)
                                + ", maximum="
                                + Integer.MAX_VALUE,
                        fullLimit.getMessage()),
                () -> assertEquals(
                        "constant tensor element count exceeds Java array limit: required="
                                + (2L * Integer.MAX_VALUE)
                                + ", maximum="
                                + Integer.MAX_VALUE,
                        identityLimit.getMessage()),
                () -> assertEquals(
                        "Gradient eligibility requires a differentiable data type: INT64",
                        gradient.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void blankLabelsConsumeIdentifiersAndExhaustionRemainsPermanent()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long before = next.get();

        IllegalArgumentException fullBlank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.full(
                        Shape.scalar(), 1.0f, Optional.of(" \t "), false));
        IllegalArgumentException identityBlank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.eye(
                        1, 1, DataType.FLOAT32, Optional.of(" \n "), false));
        assertEquals(before + 2, next.get());

        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);
            IllegalStateException fullExhausted = assertThrows(
                    IllegalStateException.class,
                    () -> TensorFactory.full(
                            Shape.scalar(), 1L, Optional.empty(), false));
            IllegalStateException identityExhausted = assertThrows(
                    IllegalStateException.class,
                    () -> TensorFactory.identityMatrix(
                            1, 1, DataType.BOOL, Optional.empty(), false));
            assertAll(
                    () -> assertEquals("label must not be blank", fullBlank.getMessage()),
                    () -> assertEquals("label must not be blank", identityBlank.getMessage()),
                    () -> assertEquals(
                            "tensor identifier space exhausted", fullExhausted.getMessage()),
                    () -> assertEquals(
                            "tensor identifier space exhausted", identityExhausted.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    private static void assertDenseLeaf(
            Tensor tensor, Shape shape, DataType dataType, boolean requiresGrad) {
        assertAll(
                () -> assertEquals(shape, tensor.descriptor().shape()),
                () -> assertEquals(dataType, tensor.descriptor().dataType()),
                () -> assertEquals(requiresGrad, tensor.descriptor().requiresGrad()),
                () -> assertEquals(
                        LayoutKind.DENSE_CONTIGUOUS,
                        tensor.descriptor().layout().orElseThrow().kind()),
                () -> assertFalse(tensor.descriptor().layout().orElseThrow().isView()),
                () -> assertEquals(Optional.empty(), tensor.provenance()),
                () -> assertTrue(tensor.hostStorage().isPresent()));
    }

    private static void assertIdentityValues(
            Tensor tensor, DataType dataType, int rows, int columns) {
        int length = rows * columns;
        switch (dataType) {
            case FLOAT64 -> assertArrayEquals(
                    identityDouble(length, rows, columns), heapArray(tensor, double[].class));
            case FLOAT32 -> assertArrayEquals(
                    identityFloat(length, rows, columns), heapArray(tensor, float[].class));
            case BFLOAT16 -> assertArrayEquals(
                    identityShort(length, rows, columns), heapArray(tensor, short[].class));
            case INT32 -> assertArrayEquals(
                    identityInt(length, rows, columns), heapArray(tensor, int[].class));
            case INT64 -> assertArrayEquals(
                    identityLong(length, rows, columns), heapArray(tensor, long[].class));
            case BOOL -> assertArrayEquals(
                    identityByte(length, rows, columns), heapArray(tensor, byte[].class));
        }
    }

    private static void assertRawDoubleFill(double[] actual, double expected) {
        assertTrue(Arrays.stream(actual).allMatch(value ->
                Double.doubleToRawLongBits(value) == Double.doubleToRawLongBits(expected)));
    }

    private static void assertRawFloatFill(float[] actual, float expected) {
        for (float value : actual) {
            assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(value));
        }
    }

    private static double[] identityDouble(int length, int rows, int columns) {
        double[] result = new double[length];
        for (int index = 0; index < Math.min(rows, columns); index++) {
            result[index * columns + index] = 1.0d;
        }
        return result;
    }

    private static float[] identityFloat(int length, int rows, int columns) {
        float[] result = new float[length];
        for (int index = 0; index < Math.min(rows, columns); index++) {
            result[index * columns + index] = 1.0f;
        }
        return result;
    }

    private static short[] identityShort(int length, int rows, int columns) {
        short[] result = new short[length];
        for (int index = 0; index < Math.min(rows, columns); index++) {
            result[index * columns + index] = BFloat16Bits.fromFloat(1.0f);
        }
        return result;
    }

    private static int[] identityInt(int length, int rows, int columns) {
        int[] result = new int[length];
        for (int index = 0; index < Math.min(rows, columns); index++) {
            result[index * columns + index] = 1;
        }
        return result;
    }

    private static long[] identityLong(int length, int rows, int columns) {
        long[] result = new long[length];
        for (int index = 0; index < Math.min(rows, columns); index++) {
            result[index * columns + index] = 1L;
        }
        return result;
    }

    private static byte[] identityByte(int length, int rows, int columns) {
        byte[] result = new byte[length];
        for (int index = 0; index < Math.min(rows, columns); index++) {
            result[index * columns + index] = 1;
        }
        return result;
    }

    private static short[] filledShort(int length, short value) {
        short[] result = new short[length];
        Arrays.fill(result, value);
        return result;
    }

    private static int[] filledInt(int length, int value) {
        int[] result = new int[length];
        Arrays.fill(result, value);
        return result;
    }

    private static long[] filledLong(int length, long value) {
        long[] result = new long[length];
        Arrays.fill(result, value);
        return result;
    }

    private static byte[] filledByte(int length, byte value) {
        byte[] result = new byte[length];
        Arrays.fill(result, value);
        return result;
    }

    private static Object heapBase(Tensor tensor) {
        return tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow();
    }

    private static <T> T heapArray(Tensor tensor, Class<T> carrierType) {
        return carrierType.cast(heapBase(tensor));
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumClaimedState() throws ReflectiveOperationException {
        var field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }
}
