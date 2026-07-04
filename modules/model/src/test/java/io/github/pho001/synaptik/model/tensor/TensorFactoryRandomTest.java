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
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.random.RandomGenerator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class TensorFactoryRandomTest {
    @Test
    void helperHasExactlyFourPackageEntriesAndNoStateOrPublicSurface()
            throws ReflectiveOperationException {
        Method entry = TensorRandoms.class.getDeclaredMethod(
                "randomNormal",
                Shape.class,
                DataType.class,
                double.class,
                double.class,
                RandomGenerator.class,
                Optional.class,
                boolean.class);
        Method uniformEntry = TensorRandoms.class.getDeclaredMethod(
                "randomUniform",
                Shape.class,
                DataType.class,
                double.class,
                double.class,
                RandomGenerator.class,
                Optional.class,
                boolean.class);
        Method int32Entry = TensorRandoms.class.getDeclaredMethod(
                "randomInt",
                Shape.class,
                int.class,
                int.class,
                RandomGenerator.class,
                Optional.class);
        Method int64Entry = TensorRandoms.class.getDeclaredMethod(
                "randomInt",
                Shape.class,
                long.class,
                long.class,
                RandomGenerator.class,
                Optional.class);

        assertAll(
                () -> assertTrue(Modifier.isFinal(TensorRandoms.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorRandoms.class.getModifiers())),
                () -> assertFalse(Modifier.isProtected(TensorRandoms.class.getModifiers())),
                () -> assertEquals(0, TensorRandoms.class.getDeclaredFields().length),
                () -> assertEquals(1, TensorRandoms.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(
                        TensorRandoms.class.getDeclaredConstructors()[0].getModifiers())),
                () -> assertEquals(
                        0, TensorRandoms.class.getDeclaredConstructors()[0].getParameterCount()),
                () -> assertFalse(Modifier.isPublic(entry.getModifiers())),
                () -> assertFalse(Modifier.isProtected(entry.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(entry.getModifiers())),
                () -> assertTrue(Modifier.isStatic(entry.getModifiers())),
                () -> assertEquals(Tensor.class, entry.getReturnType()),
                () -> assertFalse(Modifier.isPublic(uniformEntry.getModifiers())),
                () -> assertFalse(Modifier.isProtected(uniformEntry.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(uniformEntry.getModifiers())),
                () -> assertTrue(Modifier.isStatic(uniformEntry.getModifiers())),
                () -> assertEquals(Tensor.class, uniformEntry.getReturnType()),
                () -> assertFalse(Modifier.isPublic(int32Entry.getModifiers())),
                () -> assertFalse(Modifier.isProtected(int32Entry.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(int32Entry.getModifiers())),
                () -> assertTrue(Modifier.isStatic(int32Entry.getModifiers())),
                () -> assertEquals(Tensor.class, int32Entry.getReturnType()),
                () -> assertFalse(Modifier.isPublic(int64Entry.getModifiers())),
                () -> assertFalse(Modifier.isProtected(int64Entry.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(int64Entry.getModifiers())),
                () -> assertTrue(Modifier.isStatic(int64Entry.getModifiers())),
                () -> assertEquals(Tensor.class, int64Entry.getReturnType()),
                () -> assertEquals(
                        4,
                        Arrays.stream(TensorRandoms.class.getDeclaredMethods())
                                .filter(method -> !Modifier.isPrivate(method.getModifiers()))
                                .count()),
                () -> assertTrue(Arrays.stream(TensorRandoms.class.getDeclaredMethods())
                        .noneMatch(method -> Modifier.isPublic(method.getModifiers())
                                || Modifier.isProtected(method.getModifiers()))),
                () -> assertTrue(Arrays.stream(TensorFactory.class.getDeclaredFields())
                        .noneMatch(field -> RandomGenerator.class.isAssignableFrom(field.getType()))));
    }

    @Test
    void convertsScriptedSamplesExactlyForAllThreeFloatingCarriersInOrder() {
        double[] gaussians = {-1.25d, 0.5d, 3.0d, -0.0d};
        double mean = 0x1.0000000000001p20;
        double deviation = 0x1.0000000000001p-20;
        ScriptedGenerator float64Source = new ScriptedGenerator(gaussians);
        ScriptedGenerator float32Source = new ScriptedGenerator(gaussians);
        ScriptedGenerator bfloat16Source = new ScriptedGenerator(gaussians);
        Shape float64Shape = Shape.of(2, 2);
        Shape vectorShape = Shape.of(4);

        Tensor float64 = TensorFactory.randomNormal(
                float64Shape,
                DataType.FLOAT64,
                mean,
                deviation,
                float64Source,
                Optional.of("  normal  "),
                true);
        Tensor float32 = TensorFactory.randomNormal(
                vectorShape,
                DataType.FLOAT32,
                mean,
                deviation,
                float32Source,
                Optional.empty(),
                true);
        Tensor bfloat16 = TensorFactory.randomNormal(
                vectorShape,
                DataType.BFLOAT16,
                mean,
                deviation,
                bfloat16Source,
                Optional.empty(),
                true);

        double[] expected64 = new double[gaussians.length];
        float[] expected32 = new float[gaussians.length];
        short[] expectedBfloat16 = new short[gaussians.length];
        for (int index = 0; index < gaussians.length; index++) {
            double sample = mean + gaussians[index] * deviation;
            expected64[index] = sample;
            expected32[index] = (float) sample;
            expectedBfloat16[index] = BFloat16Bits.fromFloat((float) sample);
        }

        assertAll(
                () -> assertArrayEquals(expected64, heapArray(float64, double[].class)),
                () -> assertArrayEquals(expected32, heapArray(float32, float[].class)),
                () -> assertArrayEquals(expectedBfloat16, heapArray(bfloat16, short[].class)),
                () -> assertEquals(gaussians.length, float64Source.calls()),
                () -> assertEquals(gaussians.length, float32Source.calls()),
                () -> assertEquals(gaussians.length, bfloat16Source.calls()),
                () -> assertEquals(Optional.of("normal"), float64.label()),
                () -> assertDescriptor(float64, float64Shape, DataType.FLOAT64, true),
                () -> assertDescriptor(float32, vectorShape, DataType.FLOAT32, true),
                () -> assertDescriptor(bfloat16, vectorShape, DataType.BFLOAT16, true),
                () -> assertNotSame(float64.hostStorage().orElseThrow(),
                        float32.hostStorage().orElseThrow()),
                () -> assertNotEquals(float64.id(), float32.id()));
    }

    @Test
    void scalarConsumesOneSampleAndEmptyShapeConsumesNone() {
        ScriptedGenerator scalarSource = new ScriptedGenerator(2.5d);
        ScriptedGenerator emptySource = new ScriptedGenerator();

        Tensor scalar = TensorFactory.randomNormal(
                Shape.scalar(),
                DataType.FLOAT64,
                -1.0d,
                2.0d,
                scalarSource,
                Optional.empty(),
                false);
        Shape emptyShape = Shape.of(2, 0, 3);
        Tensor empty = TensorFactory.randomNormal(
                emptyShape,
                DataType.FLOAT32,
                1.0d,
                3.0d,
                emptySource,
                Optional.empty(),
                false);

        assertAll(
                () -> assertArrayEquals(new double[] {4.0d}, heapArray(scalar, double[].class)),
                () -> assertEquals(1, scalarSource.calls()),
                () -> assertEquals(0, emptySource.calls()),
                () -> assertEquals(0, heapArray(empty, float[].class).length),
                () -> assertDescriptor(scalar, Shape.scalar(), DataType.FLOAT64, false),
                () -> assertDescriptor(empty, emptyShape, DataType.FLOAT32, false));
    }

    @Test
    void nullValidationUsesPublicOrderWithoutDrawingOrAllocatingAnIdentifier()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        ScriptedGenerator source = new ScriptedGenerator(0.0d);
        Shape shape = Shape.scalar();

        NullPointerException shapeFailure = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.randomNormal(
                        null, null, 0.0d, 1.0d, null, null, false));
        NullPointerException typeFailure = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.randomNormal(
                        shape, null, 0.0d, 1.0d, null, null, false));
        NullPointerException sourceFailure = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.randomNormal(
                        shape, DataType.FLOAT32, 0.0d, 1.0d, null, null, false));
        NullPointerException labelFailure = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.randomNormal(
                        shape, DataType.FLOAT32, 0.0d, 1.0d, source, null, false));

        assertAll(
                () -> assertEquals("shape", shapeFailure.getMessage()),
                () -> assertEquals("dataType", typeFailure.getMessage()),
                () -> assertEquals("randomGenerator", sourceFailure.getMessage()),
                () -> assertEquals("label", labelFailure.getMessage()),
                () -> assertEquals(0, source.calls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void metadataAndDistributionValidationPrecedesSamplingAndIdentifierAllocation()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        ScriptedGenerator source = new ScriptedGenerator(0.0d);
        Shape dynamic = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(2));
        Shape overflow = Shape.of(Long.MAX_VALUE, 2);
        Shape overLimit = Shape.of((long) Integer.MAX_VALUE + 1);

        IllegalArgumentException dynamicFailure = assertThrows(
                IllegalArgumentException.class,
                () -> random(dynamic, DataType.INT32, 0.0d, -1.0d, source));
        ArithmeticException overflowFailure = assertThrows(
                ArithmeticException.class,
                () -> random(overflow, DataType.FLOAT32, 0.0d, 1.0d, source));
        IllegalArgumentException limitFailure = assertThrows(
                IllegalArgumentException.class,
                () -> random(overLimit, DataType.FLOAT32, 0.0d, 1.0d, source));
        IllegalArgumentException typeFailure = assertThrows(
                IllegalArgumentException.class,
                () -> random(Shape.scalar(), DataType.INT64, Double.NaN, -1.0d, source));
        IllegalArgumentException meanFailure = assertThrows(
                IllegalArgumentException.class,
                () -> random(Shape.scalar(), DataType.FLOAT64, Double.POSITIVE_INFINITY, 1.0d, source));
        IllegalArgumentException negativeDeviation = assertThrows(
                IllegalArgumentException.class,
                () -> random(Shape.scalar(), DataType.FLOAT32, 0.0d, -0.25d, source));
        IllegalArgumentException infiniteDeviation = assertThrows(
                IllegalArgumentException.class,
                () -> random(Shape.scalar(), DataType.BFLOAT16, 0.0d, Double.NaN, source));

        assertAll(
                () -> assertEquals(
                        "random tensor creation requires a fully static shape: " + dynamic,
                        dynamicFailure.getMessage()),
                () -> assertEquals("long overflow", overflowFailure.getMessage()),
                () -> assertEquals(
                        "random tensor element count exceeds Java array limit: required=2147483648, maximum=2147483647",
                        limitFailure.getMessage()),
                () -> assertEquals(
                        "random normal creation requires floating data type: INT64",
                        typeFailure.getMessage()),
                () -> assertEquals(
                        "random normal mean must be finite: Infinity", meanFailure.getMessage()),
                () -> assertEquals(
                        "random normal standard deviation must be finite and non-negative: -0.25",
                        negativeDeviation.getMessage()),
                () -> assertEquals(
                        "random normal standard deviation must be finite and non-negative: NaN",
                        infiniteDeviation.getMessage()),
                () -> assertEquals(0, source.calls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void bothSignedZeroDeviationsAreAcceptedAndStillConsumeEverySample() {
        ScriptedGenerator positive = new ScriptedGenerator(1.0d, -2.0d);
        ScriptedGenerator negative = new ScriptedGenerator(1.0d, -2.0d);

        Tensor positiveZero = random(
                Shape.of(2), DataType.FLOAT64, 3.0d, 0.0d, positive);
        Tensor negativeZero = random(
                Shape.of(2), DataType.FLOAT64, 3.0d, -0.0d, negative);

        assertAll(
                () -> assertArrayEquals(new double[] {3.0d, 3.0d},
                        heapArray(positiveZero, double[].class)),
                () -> assertArrayEquals(new double[] {3.0d, 3.0d},
                        heapArray(negativeZero, double[].class)),
                () -> assertEquals(2, positive.calls()),
                () -> assertEquals(2, negative.calls()));
    }

    @Test
    void equivalentSeededGeneratorsProduceEquivalentIndependentResults() {
        Random firstSource = new Random(0x5eedL);
        Random secondSource = new Random(0x5eedL);

        Tensor first = TensorFactory.randomNormal(
                Shape.of(8),
                DataType.FLOAT32,
                -0.5d,
                1.25d,
                firstSource,
                Optional.empty(),
                false);
        Tensor second = TensorFactory.randomNormal(
                Shape.of(8),
                DataType.FLOAT32,
                -0.5d,
                1.25d,
                secondSource,
                Optional.empty(),
                false);

        assertAll(
                () -> assertArrayEquals(
                        heapArray(first, float[].class), heapArray(second, float[].class)),
                () -> assertNotSame(
                        heapArray(first, float[].class), heapArray(second, float[].class)),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertEquals(firstSource.nextGaussian(), secondSource.nextGaussian()));
    }

    @Test
    void generatorFailureConsumesOnlyPrecedingCallsAndNoIdentifier()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        ThrowingGenerator source = new ThrowingGenerator(2);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> TensorFactory.randomNormal(
                        Shape.of(4),
                        DataType.FLOAT64,
                        0.0d,
                        1.0d,
                        source,
                        Optional.empty(),
                        false));

        assertAll(
                () -> assertEquals("scripted generator failure", failure.getMessage()),
                () -> assertEquals(2, source.calls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void blankLabelAndPermanentExhaustionOccurAfterAllDrawsWithDocumentedIdEffects()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long before = next.get();
        ScriptedGenerator blankSource = new ScriptedGenerator(0.0d, 1.0d, 2.0d);

        IllegalArgumentException blank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.randomNormal(
                        Shape.of(3),
                        DataType.FLOAT32,
                        1.0d,
                        2.0d,
                        blankSource,
                        Optional.of(" \t\n "),
                        false));
        assertAll(
                () -> assertEquals("label must not be blank", blank.getMessage()),
                () -> assertEquals(3, blankSource.calls()),
                () -> assertEquals(before + 1, next.get()));

        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        ScriptedGenerator exhaustedSource = new ScriptedGenerator(-1.0d, 1.0d);
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            IllegalStateException exhausted = assertThrows(
                    IllegalStateException.class,
                    () -> TensorFactory.randomNormal(
                            Shape.of(2),
                            DataType.BFLOAT16,
                            0.0d,
                            1.0d,
                            exhaustedSource,
                            Optional.empty(),
                            true));

            assertAll(
                    () -> assertEquals(
                            "tensor identifier space exhausted", exhausted.getMessage()),
                    () -> assertEquals(2, exhaustedSource.calls()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    private static Tensor random(
            Shape shape,
            DataType dataType,
            double mean,
            double deviation,
            RandomGenerator source) {
        return TensorFactory.randomNormal(
                shape, dataType, mean, deviation, source, Optional.empty(), false);
    }

    private static void assertDescriptor(
            Tensor tensor, Shape shape, DataType dataType, boolean requiresGrad) {
        assertAll(
                () -> assertSame(shape, tensor.descriptor().shape()),
                () -> assertSame(dataType, tensor.descriptor().dataType()),
                () -> assertEquals(LayoutKind.DENSE_CONTIGUOUS,
                        tensor.descriptor().layout().orElseThrow().kind()),
                () -> assertFalse(tensor.descriptor().layout().orElseThrow().isView()),
                () -> assertEquals(requiresGrad, tensor.descriptor().requiresGrad()),
                () -> assertEquals(
                        shape.knownElementCount().orElseThrow(),
                        tensor.hostStorage().orElseThrow().elementCapacity()));
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

    private static final class ScriptedGenerator implements RandomGenerator {
        private final double[] values;
        private int calls;

        private ScriptedGenerator(double... values) {
            this.values = values.clone();
        }

        @Override
        public long nextLong() {
            throw new AssertionError("nextLong must not be called");
        }

        @Override
        public double nextGaussian() {
            if (calls >= values.length) {
                throw new AssertionError("unexpected Gaussian call " + calls);
            }
            return values[calls++];
        }

        private int calls() {
            return calls;
        }
    }

    private static final class ThrowingGenerator implements RandomGenerator {
        private final int throwOnCall;
        private int calls;

        private ThrowingGenerator(int throwOnCall) {
            this.throwOnCall = throwOnCall;
        }

        @Override
        public long nextLong() {
            throw new AssertionError("nextLong must not be called");
        }

        @Override
        public double nextGaussian() {
            calls++;
            if (calls == throwOnCall) {
                throw new IllegalStateException("scripted generator failure");
            }
            return 0.25d;
        }

        private int calls() {
            return calls;
        }
    }
}
