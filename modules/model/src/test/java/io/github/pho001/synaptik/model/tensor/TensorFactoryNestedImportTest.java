package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class TensorFactoryNestedImportTest {
    @Test
    void helperHasExactlyOnePackageEntryAndNoPublicProtectedOrMutableStaticState()
            throws ReflectiveOperationException {
        assertAll(
                () -> assertFalse(Modifier.isPublic(NestedTensorArray.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(NestedTensorArray.class.getModifiers())),
                () -> assertEquals(0, NestedTensorArray.class.getDeclaredFields().length),
                () -> assertEquals(0, NestedTensorArray.class.getDeclaredClasses().length));

        var constructors = NestedTensorArray.class.getDeclaredConstructors();
        assertEquals(1, constructors.length);
        assertAll(
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()));

        Method entry = NestedTensorArray.class.getDeclaredMethod(
                "importArray", Object.class, Optional.class, boolean.class);
        assertAll(
                () -> assertEquals(Tensor.class, entry.getReturnType()),
                () -> assertTrue(Modifier.isStatic(entry.getModifiers())),
                () -> assertFalse(Modifier.isPublic(entry.getModifiers())),
                () -> assertFalse(Modifier.isProtected(entry.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(entry.getModifiers())),
                () -> assertTrue(Arrays.stream(NestedTensorArray.class.getDeclaredMethods())
                        .filter(method -> !Modifier.isPrivate(method.getModifiers()))
                        .allMatch(entry::equals)),
                () -> assertTrue(Arrays.stream(NestedTensorArray.class.getDeclaredMethods())
                        .noneMatch(method -> Modifier.isPublic(method.getModifiers())
                                || Modifier.isProtected(method.getModifiers()))));
    }

    @Test
    void importsRankTwoAndRankThreeSourcesForEverySupportedCarrierInRowMajorOrder() {
        assertImported(
                new double[][] {{1.0d, 2.0d}, {3.0d, 4.0d}},
                DataType.FLOAT64,
                Shape.of(2, 2),
                new double[] {1.0d, 2.0d, 3.0d, 4.0d});
        assertImported(
                new double[][][] {{{1.0d, 2.0d}}, {{3.0d, 4.0d}}},
                DataType.FLOAT64,
                Shape.of(2, 1, 2),
                new double[] {1.0d, 2.0d, 3.0d, 4.0d});
        assertImported(
                new float[][] {{1.0f, 2.0f}, {3.0f, 4.0f}},
                DataType.FLOAT32,
                Shape.of(2, 2),
                new float[] {1.0f, 2.0f, 3.0f, 4.0f});
        assertImported(
                new float[][][] {{{1.0f}, {2.0f}}, {{3.0f}, {4.0f}}},
                DataType.FLOAT32,
                Shape.of(2, 2, 1),
                new float[] {1.0f, 2.0f, 3.0f, 4.0f});
        assertImported(
                new short[][] {{1, 2}, {3, 4}},
                DataType.BFLOAT16,
                Shape.of(2, 2),
                new short[] {1, 2, 3, 4});
        assertImported(
                new short[][][] {{{1, 2}, {3, 4}}},
                DataType.BFLOAT16,
                Shape.of(1, 2, 2),
                new short[] {1, 2, 3, 4});
        assertImported(
                new int[][] {{1, 2}, {3, 4}},
                DataType.INT32,
                Shape.of(2, 2),
                new int[] {1, 2, 3, 4});
        assertImported(
                new int[][][] {{{1}, {2}}, {{3}, {4}}},
                DataType.INT32,
                Shape.of(2, 2, 1),
                new int[] {1, 2, 3, 4});
        assertImported(
                new long[][] {{1L, 2L}, {3L, 4L}},
                DataType.INT64,
                Shape.of(2, 2),
                new long[] {1L, 2L, 3L, 4L});
        assertImported(
                new long[][][] {{{1L, 2L}}, {{3L, 4L}}},
                DataType.INT64,
                Shape.of(2, 1, 2),
                new long[] {1L, 2L, 3L, 4L});
        assertImported(
                new byte[][] {{0, 1}, {0, 1}},
                DataType.BOOL,
                Shape.of(2, 2),
                new byte[] {0, 1, 0, 1});
        assertImported(
                new byte[][][] {{{0}, {1}}, {{1}, {0}}},
                DataType.BOOL,
                Shape.of(2, 2, 1),
                new byte[] {0, 1, 1, 0});
    }

    @Test
    void preservesRawFloatingAndBfloat16Representations() {
        double negativeZero = -0.0d;
        double payloadNaN = Double.longBitsToDouble(0x7ff8_0000_0000_0042L);
        float negativeZero32 = -0.0f;
        float payloadNaN32 = Float.intBitsToFloat(0x7fc0_0042);
        short[] rawBfloat16 = {(short) 0x8000, (short) 0x7fc1, (short) 0xffff};

        double[] doubles = heapArray(
                TensorFactory.fromNestedArray(
                        new double[][] {{negativeZero, payloadNaN}}, Optional.empty(), false),
                double[].class);
        Tensor floatTensor = TensorFactory.fromNestedArray(
                new float[][] {{negativeZero32, payloadNaN32}}, Optional.empty(), true);
        float[] floats = heapArray(floatTensor, float[].class);
        short[] bfloat16 = heapArray(
                TensorFactory.fromNestedArray(
                        new short[][] {rawBfloat16.clone()}, Optional.empty(), false),
                short[].class);

        assertAll(
                () -> assertEquals(
                        Double.doubleToRawLongBits(negativeZero),
                        Double.doubleToRawLongBits(doubles[0])),
                () -> assertEquals(
                        Double.doubleToRawLongBits(payloadNaN),
                        Double.doubleToRawLongBits(doubles[1])),
                () -> assertEquals(
                        Float.floatToRawIntBits(negativeZero32),
                        Float.floatToRawIntBits(floats[0])),
                () -> assertEquals(
                        Float.floatToRawIntBits(payloadNaN32),
                        Float.floatToRawIntBits(floats[1])),
                () -> assertTrue(floatTensor.descriptor().requiresGrad()),
                () -> assertArrayEquals(rawBfloat16, bfloat16));
    }

    @Test
    void normalizesBoolAndDoesNotRetainOrMutateAnySourceLevel() {
        byte[] first = {0, -2};
        byte[] second = {3, 0};
        byte[][] source = {first, second};

        Tensor tensor = TensorFactory.fromNestedArray(
                source, Optional.of("  mask\n"), false);
        byte[] destination = heapArray(tensor, byte[].class);

        assertAll(
                () -> assertEquals(Optional.of("mask"), tensor.label()),
                () -> assertArrayEquals(new byte[] {0, 1, 1, 0}, destination),
                () -> assertArrayEquals(new byte[] {0, -2}, first),
                () -> assertArrayEquals(new byte[] {3, 0}, second),
                () -> assertNotSame(first, destination),
                () -> assertNotSame(second, destination));

        first[0] = 9;
        second[0] = 0;
        source[0] = new byte[] {1, 1};
        assertArrayEquals(new byte[] {0, 1, 1, 0}, destination);
    }

    @Test
    void acceptsEmptyFinalAxisAndRejectsAmbiguousEmptyAxesWithExactPaths()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        Tensor empty = TensorFactory.fromNestedArray(new int[2][0], Optional.empty(), false);
        long afterSuccess = next.get();

        IllegalArgumentException rootEmpty = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromNestedArray(new float[0][3], Optional.empty(), false));
        IllegalArgumentException innerEmpty = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromNestedArray(new long[2][0][4], Optional.empty(), false));

        assertAll(
                () -> assertEquals(Shape.of(2, 0), empty.descriptor().shape()),
                () -> assertEquals(0, empty.hostStorage().orElseThrow().elementCapacity()),
                () -> assertEquals(0, heapArray(empty, int[].class).length),
                () -> assertEquals(
                        "nested tensor source cannot infer dimensions after empty axis 0 at path []",
                        rootEmpty.getMessage()),
                () -> assertEquals(
                        "nested tensor source cannot infer dimensions after empty axis 1 at path [0]",
                        innerEmpty.getMessage()),
                () -> assertEquals(afterSuccess, next.get()));
    }

    @Test
    void rejectsNullNonArrayRankOneAndUnsupportedSourcesInOrderWithoutIds()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        NullPointerException nullSource = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.fromNestedArray(null, null, false));
        NullPointerException nullLabel = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.fromNestedArray(new int[][] {{1}}, null, false));
        IllegalArgumentException nonArray = failure("text");
        IllegalArgumentException rankOne = failure(new int[] {1});
        IllegalArgumentException bool = failure(new boolean[][] {{true}});
        IllegalArgumentException character = failure(new char[][] {{'x'}});
        IllegalArgumentException boxed = failure(new Integer[][] {{1}});
        IllegalArgumentException generic = failure(new Object[][] {{1}});

        assertAll(
                () -> assertEquals("source", nullSource.getMessage()),
                () -> assertEquals("label", nullLabel.getMessage()),
                () -> assertEquals(
                        "nested tensor source must be an array: actual=java.lang.String",
                        nonArray.getMessage()),
                () -> assertEquals(
                        "nested tensor source must have rank at least 2: actual=1",
                        rankOne.getMessage()),
                () -> assertEquals(
                        "nested tensor source leaf carrier is unsupported: boolean",
                        bool.getMessage()),
                () -> assertEquals(
                        "nested tensor source leaf carrier is unsupported: char",
                        character.getMessage()),
                () -> assertEquals(
                        "nested tensor source leaf carrier is unsupported: java.lang.Integer",
                        boxed.getMessage()),
                () -> assertEquals(
                        "nested tensor source leaf carrier is unsupported: java.lang.Object",
                        generic.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void rejectsNullAndRaggedSubarraysWithExactAxesAndPathsBeforeIds()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        int[][][] deepNull = {{{1}}, {null}};
        int[][][] deepRagged = {{{1}, {2}}, {{3}, {4, 5}}};

        IllegalArgumentException rootNull = failure(
                new double[][] {new double[] {1.0d}, null});
        IllegalArgumentException nestedNull = failure(deepNull);
        IllegalArgumentException ragged = failure(new double[][] {{1.0d}, {2.0d, 3.0d}});
        IllegalArgumentException nestedRagged = failure(deepRagged);

        assertAll(
                () -> assertEquals(
                        "nested tensor source contains null subarray at path [1]",
                        rootNull.getMessage()),
                () -> assertEquals(
                        "nested tensor source contains null subarray at path [1][0]",
                        nestedNull.getMessage()),
                () -> assertEquals(
                        "nested tensor source is ragged at axis 1, path [1]: expected=1, actual=2",
                        ragged.getMessage()),
                () -> assertEquals(
                        "nested tensor source is ragged at axis 2, path [1][1]: expected=1, actual=2",
                        nestedRagged.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void descriptorEligibilityFailureConsumesNoIdentifier() throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromNestedArray(
                        new int[][] {{1, 2}}, Optional.empty(), true));

        assertAll(
                () -> assertEquals(
                        "Gradient eligibility requires a differentiable data type: INT32",
                        failure.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void blankLabelDelegationAllocatesDestinationAndConsumesIdentifier()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        IllegalArgumentException blank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromNestedArray(
                        new byte[][] {{0, 2}}, Optional.of(" \t\n "), false));
        Tensor subsequent = TensorFactory.create(
                new TensorDescriptor(DataType.FLOAT32, Shape.scalar(), Optional.empty(), false));

        assertAll(
                () -> assertEquals("label must not be blank", blank.getMessage()),
                () -> assertEquals(before + 1, subsequent.id().value()),
                () -> assertEquals(before + 2, next.get()));
    }

    @Test
    void identifierExhaustionOccursAfterNestedAnalysisWithoutMutatingSource()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        AtomicBoolean maximumClaimed = maximumTensorIdClaimedState();
        long originalNext = next.get();
        boolean originalMaximumClaimed = maximumClaimed.get();
        byte[][] source = {{0, -7}};
        try {
            next.set(Long.MAX_VALUE);
            maximumClaimed.set(true);

            IllegalStateException exhausted = assertThrows(
                    IllegalStateException.class,
                    () -> TensorFactory.fromNestedArray(source, Optional.empty(), false));

            assertAll(
                    () -> assertEquals(
                            "tensor identifier space exhausted", exhausted.getMessage()),
                    () -> assertArrayEquals(new byte[] {0, -7}, source[0]),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(maximumClaimed.get()));
        } finally {
            next.set(originalNext);
            maximumClaimed.set(originalMaximumClaimed);
        }
    }

    private static IllegalArgumentException failure(Object source) {
        return assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromNestedArray(source, Optional.empty(), false));
    }

    private static void assertImported(
            Object source,
            DataType dataType,
            Shape shape,
            Object expectedFlat) {
        Tensor tensor = TensorFactory.fromNestedArray(source, Optional.empty(), false);
        TensorDescriptor descriptor = tensor.descriptor();

        assertAll(
                dataType.name() + " " + shape,
                () -> assertSame(dataType, descriptor.dataType()),
                () -> assertEquals(shape, descriptor.shape()),
                () -> assertEquals(
                        Optional.of(LayoutDescriptor.contiguous(shape)), descriptor.layout()),
                () -> assertEquals(
                        LayoutKind.DENSE_CONTIGUOUS,
                        descriptor.layout().orElseThrow().kind()),
                () -> assertFalse(descriptor.requiresGrad()),
                () -> assertPrimitiveArrayEquals(
                        expectedFlat,
                        tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow()));
    }

    private static void assertPrimitiveArrayEquals(Object expected, Object actual) {
        if (expected instanceof double[] values) {
            assertArrayEquals(values, assertInstanceOf(double[].class, actual));
        } else if (expected instanceof float[] values) {
            assertArrayEquals(values, assertInstanceOf(float[].class, actual));
        } else if (expected instanceof short[] values) {
            assertArrayEquals(values, assertInstanceOf(short[].class, actual));
        } else if (expected instanceof int[] values) {
            assertArrayEquals(values, assertInstanceOf(int[].class, actual));
        } else if (expected instanceof long[] values) {
            assertArrayEquals(values, assertInstanceOf(long[].class, actual));
        } else {
            assertArrayEquals(
                    assertInstanceOf(byte[].class, expected),
                    assertInstanceOf(byte[].class, actual));
        }
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
}
