package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.Arena;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class TensorFactoryConstantTest {
    @Test
    void helperHasExactlyTheRequiredPackagePrivateStatelessShape()
            throws ReflectiveOperationException {
        assertAll(
                () -> assertTrue(Modifier.isFinal(TensorConstants.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorConstants.class.getModifiers())),
                () -> assertFalse(Modifier.isProtected(TensorConstants.class.getModifiers())),
                () -> assertEquals(0, TensorConstants.class.getDeclaredFields().length),
                () -> assertEquals(1, TensorConstants.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(
                        TensorConstants.class.getDeclaredConstructors()[0].getModifiers())),
                () -> assertEquals(
                        0, TensorConstants.class.getDeclaredConstructors()[0].getParameterCount()));

        Set<Method> entries = Set.of(
                TensorConstants.class.getDeclaredMethod(
                        "scalar", double.class, Optional.class, boolean.class),
                TensorConstants.class.getDeclaredMethod(
                        "scalar", float.class, Optional.class, boolean.class),
                TensorConstants.class.getDeclaredMethod(
                        "scalarBFloat16", float.class, Optional.class, boolean.class),
                TensorConstants.class.getDeclaredMethod(
                        "scalar", int.class, Optional.class, boolean.class),
                TensorConstants.class.getDeclaredMethod(
                        "scalar", long.class, Optional.class, boolean.class),
                TensorConstants.class.getDeclaredMethod(
                        "scalar", boolean.class, Optional.class, boolean.class),
                TensorConstants.class.getDeclaredMethod(
                        "zeros", Shape.class, DataType.class, Optional.class, boolean.class),
                TensorConstants.class.getDeclaredMethod(
                        "ones", Shape.class, DataType.class, Optional.class, boolean.class));

        assertAll(
                () -> assertTrue(entries.stream().allMatch(method ->
                        !Modifier.isPublic(method.getModifiers())
                                && !Modifier.isProtected(method.getModifiers())
                                && !Modifier.isPrivate(method.getModifiers())
                                && Modifier.isStatic(method.getModifiers())
                                && method.getReturnType() == Tensor.class)),
                () -> assertTrue(Arrays.stream(TensorConstants.class.getDeclaredMethods())
                        .filter(method -> !Modifier.isPrivate(method.getModifiers()))
                        .allMatch(entries::contains)),
                () -> assertEquals(
                        entries.size(),
                        Arrays.stream(TensorConstants.class.getDeclaredMethods())
                                .filter(method -> !Modifier.isPrivate(method.getModifiers()))
                                .count()),
                () -> assertTrue(Arrays.stream(TensorConstants.class.getDeclaredMethods())
                        .noneMatch(method -> Modifier.isPublic(method.getModifiers())
                                || Modifier.isProtected(method.getModifiers()))));
    }

    @Test
    void createsAllExactRankZeroScalarCarriers() {
        double doubleValue = Double.longBitsToDouble(0x7ff8_0000_0000_0042L);
        float floatValue = Float.intBitsToFloat(0x8000_0000);
        float bfloatValue = Float.intBitsToFloat(0x3f80_8000);

        Tensor float64 = TensorFactory.scalar(doubleValue, Optional.of("  d  "), true);
        Tensor float32 = TensorFactory.scalar(floatValue, Optional.empty(), true);
        Tensor bfloat16 = TensorFactory.scalarBFloat16(bfloatValue, Optional.empty(), true);
        Tensor int32 = TensorFactory.scalar(Integer.MIN_VALUE, Optional.empty(), false);
        Tensor int64 = TensorFactory.scalar(Long.MAX_VALUE, Optional.empty(), false);
        Tensor boolFalse = TensorFactory.scalar(false, Optional.empty(), false);
        Tensor boolTrue = TensorFactory.scalar(true, Optional.empty(), false);

        assertAll(
                () -> assertScalar(float64, DataType.FLOAT64, true),
                () -> assertEquals(Optional.of("d"), float64.label()),
                () -> assertEquals(
                        Double.doubleToRawLongBits(doubleValue),
                        Double.doubleToRawLongBits(heapArray(float64, double[].class)[0])),
                () -> assertScalar(float32, DataType.FLOAT32, true),
                () -> assertEquals(
                        Float.floatToRawIntBits(floatValue),
                        Float.floatToRawIntBits(heapArray(float32, float[].class)[0])),
                () -> assertScalar(bfloat16, DataType.BFLOAT16, true),
                () -> assertArrayEquals(
                        new short[] {BFloat16Bits.fromFloat(bfloatValue)},
                        heapArray(bfloat16, short[].class)),
                () -> assertScalar(int32, DataType.INT32, false),
                () -> assertArrayEquals(
                        new int[] {Integer.MIN_VALUE}, heapArray(int32, int[].class)),
                () -> assertScalar(int64, DataType.INT64, false),
                () -> assertArrayEquals(
                        new long[] {Long.MAX_VALUE}, heapArray(int64, long[].class)),
                () -> assertScalar(boolFalse, DataType.BOOL, false),
                () -> assertArrayEquals(new byte[] {0}, heapArray(boolFalse, byte[].class)),
                () -> assertScalar(boolTrue, DataType.BOOL, false),
                () -> assertArrayEquals(new byte[] {1}, heapArray(boolTrue, byte[].class)));
    }

    @Test
    void createsZerosAndOnesForEveryDataType() {
        Shape shape = Shape.of(2, 3);
        for (DataType dataType : DataType.values()) {
            boolean requiresGrad = dataType.isDifferentiable();
            Tensor zeros = TensorFactory.zeros(
                    shape, dataType, Optional.of("  zero  "), requiresGrad);
            Tensor ones = TensorFactory.ones(
                    shape, dataType, Optional.of("  one  "), requiresGrad);

            assertAll(
                    () -> assertDenseConstant(zeros, shape, dataType, requiresGrad, "zero"),
                    () -> assertDenseConstant(ones, shape, dataType, requiresGrad, "one"),
                    () -> assertNotSame(zeros.descriptor(), ones.descriptor()),
                    () -> assertNotSame(
                            zeros.descriptor().layout().orElseThrow(),
                            ones.descriptor().layout().orElseThrow()),
                    () -> assertNotSame(
                            zeros.hostStorage().orElseThrow(), ones.hostStorage().orElseThrow()),
                    () -> assertNotSame(heapBase(zeros), heapBase(ones)),
                    () -> assertNotEquals(zeros.id(), ones.id()));

            assertFilledValues(zeros, dataType, false, 6);
            assertFilledValues(ones, dataType, true, 6);
        }
    }

    @Test
    void preservesScalarAndZeroSizedShapeBehavior() {
        Tensor scalarZero = TensorFactory.zeros(
                Shape.scalar(), DataType.FLOAT32, Optional.empty(), false);
        Tensor scalarOne = TensorFactory.ones(
                Shape.scalar(), DataType.BFLOAT16, Optional.empty(), false);
        Shape emptyShape = Shape.of(2, 0, 4);
        Tensor emptyZero = TensorFactory.zeros(
                emptyShape, DataType.INT64, Optional.empty(), false);
        Tensor emptyOne = TensorFactory.ones(
                emptyShape, DataType.BOOL, Optional.empty(), false);

        assertAll(
                () -> assertEquals(Shape.scalar(), scalarZero.descriptor().shape()),
                () -> assertArrayEquals(new float[] {0.0f}, heapArray(scalarZero, float[].class)),
                () -> assertArrayEquals(
                        new short[] {BFloat16Bits.fromFloat(1.0f)},
                        heapArray(scalarOne, short[].class)),
                () -> assertEquals(emptyShape, emptyZero.descriptor().shape()),
                () -> assertEquals(0, heapArray(emptyZero, long[].class).length),
                () -> assertEquals(0, heapArray(emptyOne, byte[].class).length));
    }

    @Test
    void likeMethodsCopyOnlyShapeAndDataTypeAcrossTemplateLayoutsAndDeadStorage() {
        Shape shape = Shape.of(2, 2);
        Tensor unresolved;
        try (Arena arena = Arena.ofConfined()) {
            var storage = new MemorySegmentStorage(
                    DataType.FLOAT32, 0, arena.allocate(0, 1));
            unresolved = TensorFactory.create(
                    new TensorDescriptor(
                            DataType.FLOAT32, shape, Optional.empty(), true),
                    Optional.of("template"),
                    Optional.of(storage));
        }

        LayoutDescriptor strided = LayoutDescriptor.of(shape, new long[] {3, 1}, 0, true);
        Tensor viewTemplate = TensorFactory.create(
                new TensorDescriptor(
                        DataType.FLOAT32, shape, Optional.of(strided), true),
                Optional.of("view"),
                Optional.empty());
        Tensor denseTemplate = TensorFactory.zeros(
                shape, DataType.FLOAT32, Optional.of("dense"), true);

        Tensor fromDeadUnresolved = TensorFactory.onesLike(
                unresolved, Optional.of("fresh"), false);
        Tensor fromView = TensorFactory.zerosLike(
                viewTemplate, Optional.empty(), false);
        Tensor fromDense = TensorFactory.onesLike(
                denseTemplate, Optional.empty(), false);

        assertAll(
                () -> assertDenseConstant(
                        fromDeadUnresolved, shape, DataType.FLOAT32, false, "fresh"),
                () -> assertEquals(LayoutKind.DENSE_CONTIGUOUS,
                        fromView.descriptor().layout().orElseThrow().kind()),
                () -> assertFalse(fromView.descriptor().layout().orElseThrow().isView()),
                () -> assertEquals(LayoutKind.DENSE_CONTIGUOUS,
                        fromDense.descriptor().layout().orElseThrow().kind()),
                () -> assertEquals(Optional.empty(), fromView.label()),
                () -> assertFalse(fromView.descriptor().requiresGrad()),
                () -> assertNotEquals(unresolved.id(), fromDeadUnresolved.id()),
                () -> assertNotEquals(viewTemplate.id(), fromView.id()),
                () -> assertNotEquals(denseTemplate.id(), fromDense.id()),
                () -> assertNotSame(
                        denseTemplate.hostStorage().orElseThrow(),
                        fromDense.hostStorage().orElseThrow()),
                () -> assertNotSame(heapBase(denseTemplate), heapBase(fromDense)));
    }

    @Test
    void rejectsNullsInRequiredPublicOrderWithoutIdentifierUse()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Shape shape = Shape.scalar();
        Tensor template = TensorFactory.zeros(
                shape, DataType.FLOAT32, Optional.empty(), false);
        long afterTemplate = next.get();

        NullPointerException scalarLabel = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.scalar(1.0d, null, false));
        NullPointerException shapeFailure = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.zeros(null, null, null, false));
        NullPointerException typeFailure = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.ones(shape, null, null, false));
        NullPointerException labelFailure = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.zeros(shape, DataType.FLOAT32, null, false));
        NullPointerException templateFailure = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.zerosLike(null, null, false));
        NullPointerException likeLabelFailure = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.onesLike(template, null, false));

        assertAll(
                () -> assertEquals(before + 1, afterTemplate),
                () -> assertEquals("label", scalarLabel.getMessage()),
                () -> assertEquals("shape", shapeFailure.getMessage()),
                () -> assertEquals("dataType", typeFailure.getMessage()),
                () -> assertEquals("label", labelFailure.getMessage()),
                () -> assertEquals("template", templateFailure.getMessage()),
                () -> assertEquals("label", likeLabelFailure.getMessage()),
                () -> assertEquals(afterTemplate, next.get()));
    }

    @Test
    void rejectsDynamicOverLimitAndIneligibleGradientsBeforeIdentifierUse()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Shape dynamic = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(2));
        Shape overLimit = Shape.of((long) Integer.MAX_VALUE + 1);

        IllegalArgumentException dynamicFailure = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.ones(
                        dynamic, DataType.FLOAT32, Optional.empty(), false));
        IllegalArgumentException limitFailure = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.zeros(
                        overLimit, DataType.BOOL, Optional.empty(), false));
        IllegalArgumentException gradientFailure = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.scalar(1, Optional.empty(), true));

        assertAll(
                () -> assertEquals(
                        "constant tensor creation requires a fully static shape: " + dynamic,
                        dynamicFailure.getMessage()),
                () -> assertEquals(
                        "constant tensor element count exceeds Java array limit: required="
                                + ((long) Integer.MAX_VALUE + 1)
                                + ", maximum="
                                + Integer.MAX_VALUE,
                        limitFailure.getMessage()),
                () -> assertEquals(
                        "Gradient eligibility requires a differentiable data type: INT32",
                        gradientFailure.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void blankLabelsConsumeOneIdentifierForZerosOnesAndScalars()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        IllegalArgumentException zero = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.zeros(
                        Shape.scalar(), DataType.FLOAT32, Optional.of(" \t "), false));
        IllegalArgumentException one = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.ones(
                        Shape.scalar(), DataType.FLOAT32, Optional.of(" \n "), false));
        IllegalArgumentException scalar = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.scalar(1.0f, Optional.of("  "), false));

        assertAll(
                () -> assertEquals("label must not be blank", zero.getMessage()),
                () -> assertEquals("label must not be blank", one.getMessage()),
                () -> assertEquals("label must not be blank", scalar.getMessage()),
                () -> assertEquals(before + 3, next.get()));
    }

    @Test
    void permanentIdentifierExhaustionPropagatesAfterPopulationPreparation()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            IllegalStateException zero = assertThrows(
                    IllegalStateException.class,
                    () -> TensorFactory.zeros(
                            Shape.scalar(), DataType.FLOAT32, Optional.empty(), false));
            IllegalStateException one = assertThrows(
                    IllegalStateException.class,
                    () -> TensorFactory.ones(
                            Shape.scalar(), DataType.FLOAT32, Optional.empty(), false));
            IllegalStateException scalar = assertThrows(
                    IllegalStateException.class,
                    () -> TensorFactory.scalar(1.0f, Optional.empty(), false));

            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", zero.getMessage()),
                    () -> assertEquals("tensor identifier space exhausted", one.getMessage()),
                    () -> assertEquals("tensor identifier space exhausted", scalar.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    private static void assertScalar(Tensor tensor, DataType dataType, boolean requiresGrad) {
        assertAll(
                () -> assertEquals(dataType, tensor.descriptor().dataType()),
                () -> assertEquals(Shape.scalar(), tensor.descriptor().shape()),
                () -> assertEquals(LayoutKind.DENSE_CONTIGUOUS,
                        tensor.descriptor().layout().orElseThrow().kind()),
                () -> assertEquals(1,
                        tensor.descriptor().layout().orElseThrow().referencedElementSpan()),
                () -> assertEquals(requiresGrad, tensor.descriptor().requiresGrad()));
    }

    private static void assertDenseConstant(
            Tensor tensor,
            Shape shape,
            DataType dataType,
            boolean requiresGrad,
            String expectedLabel) {
        assertAll(
                () -> assertEquals(shape, tensor.descriptor().shape()),
                () -> assertEquals(dataType, tensor.descriptor().dataType()),
                () -> assertEquals(requiresGrad, tensor.descriptor().requiresGrad()),
                () -> assertEquals(LayoutKind.DENSE_CONTIGUOUS,
                        tensor.descriptor().layout().orElseThrow().kind()),
                () -> assertFalse(tensor.descriptor().layout().orElseThrow().isView()),
                () -> assertEquals(Optional.of(expectedLabel), tensor.label()));
    }

    private static void assertFilledValues(
            Tensor tensor, DataType dataType, boolean one, int length) {
        switch (dataType) {
            case FLOAT64 -> assertArrayEquals(
                    filledDouble(length, one ? 1.0d : 0.0d), heapArray(tensor, double[].class));
            case FLOAT32 -> assertArrayEquals(
                    filledFloat(length, one ? 1.0f : 0.0f), heapArray(tensor, float[].class));
            case BFLOAT16 -> assertArrayEquals(
                    filledShort(length, one ? BFloat16Bits.fromFloat(1.0f) : (short) 0),
                    heapArray(tensor, short[].class));
            case INT32 -> assertArrayEquals(
                    filledInt(length, one ? 1 : 0), heapArray(tensor, int[].class));
            case INT64 -> assertArrayEquals(
                    filledLong(length, one ? 1L : 0L), heapArray(tensor, long[].class));
            case BOOL -> assertArrayEquals(
                    filledByte(length, one ? (byte) 1 : (byte) 0),
                    heapArray(tensor, byte[].class));
        }
    }

    private static double[] filledDouble(int length, double value) {
        double[] result = new double[length];
        Arrays.fill(result, value);
        return result;
    }

    private static float[] filledFloat(int length, float value) {
        float[] result = new float[length];
        Arrays.fill(result, value);
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
