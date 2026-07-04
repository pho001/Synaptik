package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class TensorFactoryPopulationTest {
    @Test
    void helperHasExactlyTheTypedPackageEntriesAndNoStateOrPublicSurface()
            throws ReflectiveOperationException {
        assertAll(
                () -> assertTrue(Modifier.isFinal(TensorPopulations.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorPopulations.class.getModifiers())),
                () -> assertFalse(Modifier.isProtected(TensorPopulations.class.getModifiers())),
                () -> assertEquals(0, TensorPopulations.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorPopulations.class.getDeclaredClasses().length),
                () -> assertEquals(1, TensorPopulations.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(
                        TensorPopulations.class.getDeclaredConstructors()[0].getModifiers())));

        Set<Method> entries = Set.of(
                TensorPopulations.class.getDeclaredMethod(
                        "range", int.class, int.class, int.class, Optional.class),
                TensorPopulations.class.getDeclaredMethod(
                        "range", long.class, long.class, long.class, Optional.class),
                entry("fromStrictFlatPrefix", double[].class),
                entry("fromStrictFlatPrefix", float[].class),
                entry("fromStrictFlatPrefix", short[].class),
                entry("fromStrictFlatPrefix", int[].class),
                entry("fromStrictFlatPrefix", long[].class),
                entry("fromStrictFlatPrefix", byte[].class),
                entry("fromCyclicFlatPrefix", double[].class),
                entry("fromCyclicFlatPrefix", float[].class),
                entry("fromCyclicFlatPrefix", short[].class),
                entry("fromCyclicFlatPrefix", int[].class),
                entry("fromCyclicFlatPrefix", long[].class),
                entry("fromCyclicFlatPrefix", byte[].class));

        assertAll(
                () -> assertTrue(entries.stream().allMatch(method ->
                        !Modifier.isPublic(method.getModifiers())
                                && !Modifier.isProtected(method.getModifiers())
                                && !Modifier.isPrivate(method.getModifiers())
                                && Modifier.isStatic(method.getModifiers())
                                && method.getReturnType() == Tensor.class)),
                () -> assertEquals(
                        entries,
                        Set.copyOf(Arrays.stream(TensorPopulations.class.getDeclaredMethods())
                                .filter(method -> !Modifier.isPrivate(method.getModifiers()))
                                .toList())),
                () -> assertTrue(Arrays.stream(TensorPopulations.class.getDeclaredMethods())
                        .noneMatch(method -> Modifier.isPublic(method.getModifiers())
                                || Modifier.isProtected(method.getModifiers()))));
    }

    @Test
    void createsTypedPositiveNegativeUnevenAndPrimitiveBoundaryRanges() {
        Tensor ascending = TensorFactory.range(1, 8, 3, Optional.of("  indices "));
        Tensor descending = TensorFactory.range(5L, -2L, -2L, Optional.empty());
        Tensor intMinimum = TensorFactory.range(
                Integer.MIN_VALUE, Integer.MIN_VALUE + 2, 1, Optional.empty());
        Tensor intMaximum = TensorFactory.range(
                Integer.MAX_VALUE, Integer.MAX_VALUE - 2, -1, Optional.empty());
        Tensor intExtremeStep = TensorFactory.range(
                Integer.MAX_VALUE, Integer.MIN_VALUE, Integer.MIN_VALUE, Optional.empty());
        Tensor longMinimum = TensorFactory.range(
                Long.MIN_VALUE, Long.MIN_VALUE + 2, 1L, Optional.empty());
        Tensor longMaximum = TensorFactory.range(
                Long.MAX_VALUE, Long.MAX_VALUE - 2, -1L, Optional.empty());
        Tensor longExtremeStep = TensorFactory.range(
                Long.MAX_VALUE, Long.MIN_VALUE, Long.MIN_VALUE, Optional.empty());

        assertAll(
                () -> assertRange(ascending, DataType.INT32, Shape.of(3), "indices"),
                () -> assertArrayEquals(new int[] {1, 4, 7}, heapArray(ascending, int[].class)),
                () -> assertRange(descending, DataType.INT64, Shape.of(4), null),
                () -> assertArrayEquals(
                        new long[] {5, 3, 1, -1}, heapArray(descending, long[].class)),
                () -> assertArrayEquals(
                        new int[] {Integer.MIN_VALUE, Integer.MIN_VALUE + 1},
                        heapArray(intMinimum, int[].class)),
                () -> assertArrayEquals(
                        new int[] {Integer.MAX_VALUE, Integer.MAX_VALUE - 1},
                        heapArray(intMaximum, int[].class)),
                () -> assertArrayEquals(
                        new int[] {Integer.MAX_VALUE, -1},
                        heapArray(intExtremeStep, int[].class)),
                () -> assertArrayEquals(
                        new long[] {Long.MIN_VALUE, Long.MIN_VALUE + 1},
                        heapArray(longMinimum, long[].class)),
                () -> assertArrayEquals(
                        new long[] {Long.MAX_VALUE, Long.MAX_VALUE - 1},
                        heapArray(longMaximum, long[].class)),
                () -> assertArrayEquals(
                        new long[] {Long.MAX_VALUE, -1},
                        heapArray(longExtremeStep, long[].class)));
    }

    @Test
    void rangeValidationUsesExactOrderMessagesAndNoIdentifiers() throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        NullPointerException nullLabel = assertThrows(
                NullPointerException.class, () -> TensorFactory.range(0, 0, 0, null));
        IllegalArgumentException zeroStep = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.range(0, 0, 0, Optional.empty()));
        IllegalArgumentException empty = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.range(2L, 2L, 1L, Optional.empty()));
        IllegalArgumentException ascendingWrong = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.range(0, 3, -1, Optional.empty()));
        IllegalArgumentException descendingWrong = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.range(3L, 0L, 1L, Optional.empty()));
        IllegalArgumentException intLimit = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.range(
                        Integer.MIN_VALUE, Integer.MAX_VALUE, 1, Optional.empty()));
        IllegalArgumentException longLimit = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.range(
                        0L, (long) Integer.MAX_VALUE + 1, 1L, Optional.empty()));

        assertAll(
                () -> assertEquals("label", nullLabel.getMessage()),
                () -> assertEquals("range step must not be zero", zeroStep.getMessage()),
                () -> assertEquals("range must contain at least one element", empty.getMessage()),
                () -> assertEquals(
                        "range step direction does not advance toward end",
                        ascendingWrong.getMessage()),
                () -> assertEquals(
                        "range step direction does not advance toward end",
                        descendingWrong.getMessage()),
                () -> assertEquals(
                        "range element count exceeds Java array limit: required=4294967295, maximum=2147483647",
                        intLimit.getMessage()),
                () -> assertEquals(
                        "range element count exceeds Java array limit: required=2147483648, maximum=2147483647",
                        longLimit.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void strictPrefixesCopyEveryCarrierIgnoreTailsAndPreserveRawValues() {
        double[] doubles = {-0.0d, 2.0d, 99.0d};
        float[] floats = {-0.0f, 2.0f, 99.0f};
        short[] bfloat16 = {(short) 0x8000, (short) 0x7fc1, (short) 0xffff};
        int[] ints = {Integer.MIN_VALUE, 2, 99};
        long[] longs = {Long.MIN_VALUE, 2L, 99L};
        byte[] bools = {0, -4, 3};

        Tensor float64 = TensorFactory.fromStrictFlatPrefix(
                Shape.of(2), Optional.empty(), true, doubles);
        Tensor float32 = TensorFactory.fromStrictFlatPrefix(
                Shape.of(2), Optional.empty(), true, floats);
        Tensor bfloat = TensorFactory.fromStrictFlatPrefix(
                Shape.of(2), Optional.empty(), true, bfloat16);
        Tensor int32 = TensorFactory.fromStrictFlatPrefix(
                Shape.of(2), Optional.empty(), false, ints);
        Tensor int64 = TensorFactory.fromStrictFlatPrefix(
                Shape.of(2), Optional.empty(), false, longs);
        Tensor bool = TensorFactory.fromStrictFlatPrefix(
                Shape.of(2), Optional.of(" mask "), false, bools);

        assertAll(
                () -> assertEquals(
                        Double.doubleToRawLongBits(-0.0d),
                        Double.doubleToRawLongBits(heapArray(float64, double[].class)[0])),
                () -> assertArrayEquals(new double[] {-0.0d, 2.0d}, heapArray(float64, double[].class)),
                () -> assertArrayEquals(new float[] {-0.0f, 2.0f}, heapArray(float32, float[].class)),
                () -> assertArrayEquals(
                        new short[] {(short) 0x8000, (short) 0x7fc1},
                        heapArray(bfloat, short[].class)),
                () -> assertArrayEquals(
                        new int[] {Integer.MIN_VALUE, 2}, heapArray(int32, int[].class)),
                () -> assertArrayEquals(
                        new long[] {Long.MIN_VALUE, 2L}, heapArray(int64, long[].class)),
                () -> assertArrayEquals(new byte[] {0, 1}, heapArray(bool, byte[].class)),
                () -> assertEquals(Optional.of("mask"), bool.label()),
                () -> assertTrue(float64.descriptor().requiresGrad()),
                () -> assertEquals(DataType.BFLOAT16, bfloat.descriptor().dataType()),
                () -> assertFalse(bool.descriptor().requiresGrad()));

        doubles[0] = 7.0d;
        floats[0] = 7.0f;
        bfloat16[0] = 7;
        ints[0] = 7;
        longs[0] = 7;
        bools[0] = 7;
        assertAll(
                () -> assertEquals(
                        Double.doubleToRawLongBits(-0.0d),
                        Double.doubleToRawLongBits(heapArray(float64, double[].class)[0])),
                () -> assertEquals(
                        Float.floatToRawIntBits(-0.0f),
                        Float.floatToRawIntBits(heapArray(float32, float[].class)[0])),
                () -> assertEquals((short) 0x8000, heapArray(bfloat, short[].class)[0]),
                () -> assertEquals(Integer.MIN_VALUE, heapArray(int32, int[].class)[0]),
                () -> assertEquals(Long.MIN_VALUE, heapArray(int64, long[].class)[0]),
                () -> assertEquals((byte) 0, heapArray(bool, byte[].class)[0]));
    }

    @Test
    void cyclicPrefixesRepeatEveryCarrierAndCopyLongEnoughInputs() {
        double[] doubles = {1.0d, 2.0d};
        float[] floats = {1.0f, 2.0f};
        short[] shorts = {1, 2};
        int[] ints = {1, 2, 3, 4, 5, 6, 7};
        long[] longs = {1L, 2L};
        byte[] bools = {0, -3};

        Tensor float64 = TensorFactory.fromCyclicFlatPrefix(
                Shape.of(2, 3), Optional.empty(), true, doubles);
        Tensor float32 = TensorFactory.fromCyclicFlatPrefix(
                Shape.of(6), Optional.empty(), true, floats);
        Tensor bfloat16 = TensorFactory.fromCyclicFlatPrefix(
                Shape.of(6), Optional.empty(), true, shorts);
        Tensor int32 = TensorFactory.fromCyclicFlatPrefix(
                Shape.of(6), Optional.empty(), false, ints);
        Tensor int64 = TensorFactory.fromCyclicFlatPrefix(
                Shape.of(6), Optional.empty(), false, longs);
        Tensor bool = TensorFactory.fromCyclicFlatPrefix(
                Shape.of(6), Optional.empty(), false, bools);

        assertAll(
                () -> assertArrayEquals(
                        new double[] {1, 2, 1, 2, 1, 2}, heapArray(float64, double[].class)),
                () -> assertArrayEquals(
                        new float[] {1, 2, 1, 2, 1, 2}, heapArray(float32, float[].class)),
                () -> assertArrayEquals(
                        new short[] {1, 2, 1, 2, 1, 2}, heapArray(bfloat16, short[].class)),
                () -> assertArrayEquals(
                        new int[] {1, 2, 3, 4, 5, 6}, heapArray(int32, int[].class)),
                () -> assertArrayEquals(
                        new long[] {1, 2, 1, 2, 1, 2}, heapArray(int64, long[].class)),
                () -> assertArrayEquals(
                        new byte[] {0, 1, 0, 1, 0, 1}, heapArray(bool, byte[].class)),
                () -> assertNotSame(doubles, heapArray(float64, double[].class)),
                () -> assertNotSame(ints, heapArray(int32, int[].class)));

        doubles[0] = 9.0d;
        ints[0] = 9;
        assertAll(
                () -> assertEquals(1.0d, heapArray(float64, double[].class)[0]),
                () -> assertEquals(1, heapArray(int32, int[].class)[0]));
    }

    @Test
    void emptyShapesAcceptEmptyStrictAndCyclicSourcesForEveryCarrier() {
        Shape empty = Shape.of(0, 3);
        List<Tensor> tensors = List.of(
                TensorFactory.fromStrictFlatPrefix(empty, Optional.empty(), false, new double[0]),
                TensorFactory.fromStrictFlatPrefix(empty, Optional.empty(), false, new float[1]),
                TensorFactory.fromStrictFlatPrefix(empty, Optional.empty(), false, new short[0]),
                TensorFactory.fromStrictFlatPrefix(empty, Optional.empty(), false, new int[1]),
                TensorFactory.fromStrictFlatPrefix(empty, Optional.empty(), false, new long[0]),
                TensorFactory.fromStrictFlatPrefix(empty, Optional.empty(), false, new byte[1]),
                TensorFactory.fromCyclicFlatPrefix(empty, Optional.empty(), false, new double[0]),
                TensorFactory.fromCyclicFlatPrefix(empty, Optional.empty(), false, new float[0]),
                TensorFactory.fromCyclicFlatPrefix(empty, Optional.empty(), false, new short[0]),
                TensorFactory.fromCyclicFlatPrefix(empty, Optional.empty(), false, new int[0]),
                TensorFactory.fromCyclicFlatPrefix(empty, Optional.empty(), false, new long[0]),
                TensorFactory.fromCyclicFlatPrefix(empty, Optional.empty(), false, new byte[0]));

        assertTrue(tensors.stream().allMatch(tensor ->
                tensor.descriptor().shape().equals(empty)
                        && tensor.descriptor().layout().orElseThrow().kind()
                                == LayoutKind.DENSE_CONTIGUOUS
                        && tensor.hostStorage().orElseThrow().elementCapacity() == 0));
    }

    @Test
    void prefixNullValidationRunsInPublicOrderForStrictAndCyclicCarriers()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Shape shape = Shape.scalar();

        assertNullOrder(
                () -> TensorFactory.fromStrictFlatPrefix(null, null, false, (double[]) null),
                () -> TensorFactory.fromStrictFlatPrefix(shape, null, false, (double[]) null),
                () -> TensorFactory.fromStrictFlatPrefix(shape, Optional.empty(), false, (double[]) null));
        assertNullOrder(
                () -> TensorFactory.fromStrictFlatPrefix(null, null, false, (float[]) null),
                () -> TensorFactory.fromStrictFlatPrefix(shape, null, false, (float[]) null),
                () -> TensorFactory.fromStrictFlatPrefix(shape, Optional.empty(), false, (float[]) null));
        assertNullOrder(
                () -> TensorFactory.fromStrictFlatPrefix(null, null, false, (short[]) null),
                () -> TensorFactory.fromStrictFlatPrefix(shape, null, false, (short[]) null),
                () -> TensorFactory.fromStrictFlatPrefix(shape, Optional.empty(), false, (short[]) null));
        assertNullOrder(
                () -> TensorFactory.fromCyclicFlatPrefix(null, null, false, (int[]) null),
                () -> TensorFactory.fromCyclicFlatPrefix(shape, null, false, (int[]) null),
                () -> TensorFactory.fromCyclicFlatPrefix(shape, Optional.empty(), false, (int[]) null));
        assertNullOrder(
                () -> TensorFactory.fromCyclicFlatPrefix(null, null, false, (long[]) null),
                () -> TensorFactory.fromCyclicFlatPrefix(shape, null, false, (long[]) null),
                () -> TensorFactory.fromCyclicFlatPrefix(shape, Optional.empty(), false, (long[]) null));
        assertNullOrder(
                () -> TensorFactory.fromCyclicFlatPrefix(null, null, false, (byte[]) null),
                () -> TensorFactory.fromCyclicFlatPrefix(shape, null, false, (byte[]) null),
                () -> TensorFactory.fromCyclicFlatPrefix(shape, Optional.empty(), false, (byte[]) null));

        assertEquals(before, next.get());
    }

    @Test
    void prefixValidationMessagesAndPrecedenceConsumeNoIdentifiers()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Shape dynamic = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(2));
        Shape overflow = Shape.of(Long.MAX_VALUE, 2);
        Shape overLimit = Shape.of((long) Integer.MAX_VALUE + 1);

        IllegalArgumentException dynamicFailure = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromStrictFlatPrefix(
                        dynamic, Optional.empty(), false, new int[0]));
        ArithmeticException overflowFailure = assertThrows(
                ArithmeticException.class,
                () -> TensorFactory.fromStrictFlatPrefix(
                        overflow, Optional.empty(), false, new int[0]));
        IllegalArgumentException limitFailure = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromStrictFlatPrefix(
                        overLimit, Optional.empty(), false, new int[0]));
        IllegalArgumentException shortFailure = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromStrictFlatPrefix(
                        Shape.of(4), Optional.empty(), false, new int[3]));
        IllegalArgumentException emptyCyclic = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromCyclicFlatPrefix(
                        Shape.scalar(), Optional.empty(), false, new long[0]));
        IllegalArgumentException gradient = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromStrictFlatPrefix(
                        Shape.scalar(), Optional.empty(), true, new byte[] {1}));

        assertAll(
                () -> assertEquals(
                        "prefix tensor creation requires a fully static shape: " + dynamic,
                        dynamicFailure.getMessage()),
                () -> assertEquals("long overflow", overflowFailure.getMessage()),
                () -> assertEquals(
                        "prefix tensor element count exceeds Java array limit: required=2147483648, maximum=2147483647",
                        limitFailure.getMessage()),
                () -> assertEquals(
                        "strict flat prefix source is too short: required=4, actual=3",
                        shortFailure.getMessage()),
                () -> assertEquals(
                        "cyclic flat prefix source must not be empty for non-empty output",
                        emptyCyclic.getMessage()),
                () -> assertEquals(
                        "Gradient eligibility requires a differentiable data type: BOOL",
                        gradient.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void blankLabelsConsumeIdentifiersAndExhaustionRemainsPermanent()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long before = next.get();

        IllegalArgumentException rangeBlank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.range(0, 2, 1, Optional.of("  ")));
        IllegalArgumentException prefixBlank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.fromCyclicFlatPrefix(
                        Shape.of(2), Optional.of("\t"), false, new int[] {1}));
        assertAll(
                () -> assertEquals("label must not be blank", rangeBlank.getMessage()),
                () -> assertEquals("label must not be blank", prefixBlank.getMessage()),
                () -> assertEquals(before + 2, next.get()));

        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);
            int[] source = {1, 2};

            IllegalStateException rangeExhausted = assertThrows(
                    IllegalStateException.class,
                    () -> TensorFactory.range(0L, 2L, 1L, Optional.empty()));
            IllegalStateException prefixExhausted = assertThrows(
                    IllegalStateException.class,
                    () -> TensorFactory.fromStrictFlatPrefix(
                            Shape.of(2), Optional.empty(), false, source));

            assertAll(
                    () -> assertEquals(
                            "tensor identifier space exhausted", rangeExhausted.getMessage()),
                    () -> assertEquals(
                            "tensor identifier space exhausted", prefixExhausted.getMessage()),
                    () -> assertArrayEquals(new int[] {1, 2}, source),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    private static Method entry(String name, Class<?> carrier) throws ReflectiveOperationException {
        return TensorPopulations.class.getDeclaredMethod(
                name, Shape.class, Optional.class, boolean.class, carrier);
    }

    private static void assertRange(
            Tensor tensor, DataType dataType, Shape shape, String expectedLabel) {
        TensorDescriptor descriptor = tensor.descriptor();
        assertAll(
                () -> assertEquals(dataType, descriptor.dataType()),
                () -> assertEquals(shape, descriptor.shape()),
                () -> assertEquals(LayoutKind.DENSE_CONTIGUOUS,
                        descriptor.layout().orElseThrow().kind()),
                () -> assertFalse(descriptor.requiresGrad()),
                () -> assertEquals(Optional.ofNullable(expectedLabel), tensor.label()),
                () -> assertEquals(
                        shape.knownElementCount().orElseThrow(),
                        tensor.hostStorage().orElseThrow().elementCapacity()));
    }

    private static void assertNullOrder(
            Invocation nullShape, Invocation nullLabel, Invocation nullSource) {
        NullPointerException shape = assertThrows(NullPointerException.class, nullShape::invoke);
        NullPointerException label = assertThrows(NullPointerException.class, nullLabel::invoke);
        NullPointerException source = assertThrows(NullPointerException.class, nullSource::invoke);
        assertAll(
                () -> assertEquals("shape", shape.getMessage()),
                () -> assertEquals("label", label.getMessage()),
                () -> assertEquals("source", source.getMessage()));
    }

    private static <T> T heapArray(Tensor tensor, Class<T> carrier) {
        return carrier.cast(
                tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow());
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumClaimedState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }

    @FunctionalInterface
    private interface Invocation {
        void invoke();
    }
}
