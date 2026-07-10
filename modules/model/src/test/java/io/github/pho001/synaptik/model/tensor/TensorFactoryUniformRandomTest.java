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
class TensorFactoryUniformRandomTest {
    @Test
    void publicOwnerKeepsExactlyTheFiveRandomEntriesWithoutState()
            throws ReflectiveOperationException {
        Method normalEntry = TensorRandoms.class.getDeclaredMethod(
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
        Method bernoulliEntry = TensorRandoms.class.getDeclaredMethod(
                "randomBernoulli",
                Shape.class,
                double.class,
                RandomGenerator.class,
                Optional.class);

        assertAll(
                () -> assertTrue(Modifier.isFinal(TensorRandoms.class.getModifiers())),
                () -> assertTrue(Modifier.isPublic(TensorRandoms.class.getModifiers())),
                () -> assertEquals(0, TensorRandoms.class.getDeclaredFields().length),
                () -> assertEquals(1, TensorRandoms.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(
                        TensorRandoms.class.getDeclaredConstructors()[0].getModifiers())),
                () -> assertEquals(
                        5,
                        Arrays.stream(TensorRandoms.class.getDeclaredMethods())
                                .filter(method -> !Modifier.isPrivate(method.getModifiers()))
                                .count()),
                () -> assertEquals(
                        java.util.Set.of(
                                normalEntry,
                                uniformEntry,
                                int32Entry,
                                int64Entry,
                                bernoulliEntry),
                        Arrays.stream(TensorRandoms.class.getDeclaredMethods())
                                .filter(method -> !Modifier.isPrivate(method.getModifiers()))
                                .collect(java.util.stream.Collectors.toSet())),
                () -> assertTrue(Arrays.stream(TensorRandoms.class.getDeclaredMethods())
                        .filter(method -> !Modifier.isPrivate(method.getModifiers()))
                        .allMatch(method -> Modifier.isPublic(method.getModifiers())
                                && Modifier.isStatic(method.getModifiers()))),
                () -> assertTrue(Arrays.stream(TensorFactory.class.getDeclaredFields())
                        .noneMatch(field -> RandomGenerator.class.isAssignableFrom(field.getType()))));
    }

    @Test
    void passesExactBoundsAndConvertsScriptedSamplesForAllFloatingCarriersInOrder() {
        double lower = -0x1.0000000000001p20;
        double upper = 0x1.0000000000001p20;
        double[] samples = {lower, -0.0d, 0x1.0000000000001p-20, Math.nextDown(upper)};
        BoundedScriptedGenerator float64Source = new BoundedScriptedGenerator(samples);
        BoundedScriptedGenerator float32Source = new BoundedScriptedGenerator(samples);
        BoundedScriptedGenerator bfloat16Source = new BoundedScriptedGenerator(samples);
        Shape matrix = Shape.of(2, 2);
        Shape vector = Shape.of(4);

        Tensor float64 = TensorRandoms.randomUniform(
                matrix,
                DataType.FLOAT64,
                lower,
                upper,
                float64Source,
                Optional.of("  uniform  "),
                true);
        Tensor float32 = TensorRandoms.randomUniform(
                vector,
                DataType.FLOAT32,
                lower,
                upper,
                float32Source,
                Optional.empty(),
                true);
        Tensor bfloat16 = TensorRandoms.randomUniform(
                vector,
                DataType.BFLOAT16,
                lower,
                upper,
                bfloat16Source,
                Optional.empty(),
                true);

        float[] expected32 = new float[samples.length];
        short[] expectedBfloat16 = new short[samples.length];
        for (int index = 0; index < samples.length; index++) {
            expected32[index] = (float) samples[index];
            expectedBfloat16[index] = BFloat16Bits.fromFloat((float) samples[index]);
        }

        assertAll(
                () -> assertArrayEquals(samples, heapArray(float64, double[].class)),
                () -> assertArrayEquals(expected32, heapArray(float32, float[].class)),
                () -> assertArrayEquals(expectedBfloat16, heapArray(bfloat16, short[].class)),
                () -> float64Source.assertEveryCallUsed(lower, upper),
                () -> float32Source.assertEveryCallUsed(lower, upper),
                () -> bfloat16Source.assertEveryCallUsed(lower, upper),
                () -> assertEquals(Optional.of("uniform"), float64.label()),
                () -> assertDescriptor(float64, matrix, DataType.FLOAT64, true),
                () -> assertDescriptor(float32, vector, DataType.FLOAT32, true),
                () -> assertDescriptor(bfloat16, vector, DataType.BFLOAT16, true),
                () -> assertNotSame(float64.hostStorage().orElseThrow(),
                        float32.hostStorage().orElseThrow()),
                () -> assertNotEquals(float64.id(), float32.id()));
    }

    @Test
    void scalarConsumesOneBoundedCallAndEmptyShapeConsumesNone() {
        BoundedScriptedGenerator scalarSource = new BoundedScriptedGenerator(0.25d);
        BoundedScriptedGenerator emptySource = new BoundedScriptedGenerator();

        Tensor scalar = uniform(
                Shape.scalar(), DataType.FLOAT64, -1.0d, 1.0d, scalarSource);
        Shape emptyShape = Shape.of(2, 0, 3);
        Tensor empty = uniform(
                emptyShape, DataType.FLOAT32, -1.0d, 1.0d, emptySource);

        assertAll(
                () -> assertArrayEquals(new double[] {0.25d}, heapArray(scalar, double[].class)),
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
        BoundedScriptedGenerator source = new BoundedScriptedGenerator(0.0d);
        Shape shape = Shape.scalar();

        NullPointerException shapeFailure = assertThrows(
                NullPointerException.class,
                () -> TensorRandoms.randomUniform(
                        null, null, 0.0d, 1.0d, null, null, false));
        NullPointerException typeFailure = assertThrows(
                NullPointerException.class,
                () -> TensorRandoms.randomUniform(
                        shape, null, 0.0d, 1.0d, null, null, false));
        NullPointerException sourceFailure = assertThrows(
                NullPointerException.class,
                () -> TensorRandoms.randomUniform(
                        shape, DataType.FLOAT32, 0.0d, 1.0d, null, null, false));
        NullPointerException labelFailure = assertThrows(
                NullPointerException.class,
                () -> TensorRandoms.randomUniform(
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
    void metadataValidationUsesExactOrderMessagesWithoutDrawingOrAllocatingAnIdentifier()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        BoundedScriptedGenerator source = new BoundedScriptedGenerator(0.0d);
        Shape dynamic = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(2));
        Shape overflow = Shape.of(Long.MAX_VALUE, 2);
        Shape overLimit = Shape.of((long) Integer.MAX_VALUE + 1);

        IllegalArgumentException dynamicFailure = assertThrows(
                IllegalArgumentException.class,
                () -> uniform(dynamic, DataType.INT32, Double.NaN, Double.NaN, source));
        ArithmeticException overflowFailure = assertThrows(
                ArithmeticException.class,
                () -> uniform(overflow, DataType.FLOAT32, 0.0d, 1.0d, source));
        IllegalArgumentException limitFailure = assertThrows(
                IllegalArgumentException.class,
                () -> uniform(overLimit, DataType.FLOAT32, 0.0d, 1.0d, source));
        IllegalArgumentException typeFailure = assertThrows(
                IllegalArgumentException.class,
                () -> uniform(Shape.scalar(), DataType.INT64, Double.NaN, Double.NaN, source));

        assertAll(
                () -> assertEquals(
                        "uniform random tensor creation requires a fully static shape: " + dynamic,
                        dynamicFailure.getMessage()),
                () -> assertEquals("long overflow", overflowFailure.getMessage()),
                () -> assertEquals(
                        "uniform random tensor element count exceeds Java array limit: required=2147483648, maximum=2147483647",
                        limitFailure.getMessage()),
                () -> assertEquals(
                        "uniform random creation requires floating data type: INT64",
                        typeFailure.getMessage()),
                () -> assertEquals(0, source.calls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void boundValidationUsesFiniteThenStrictOrderWithExactMessagesAndNoDraws()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        BoundedScriptedGenerator source = new BoundedScriptedGenerator(0.0d);

        IllegalArgumentException lowerInfinity = assertThrows(
                IllegalArgumentException.class,
                () -> uniform(Shape.scalar(), DataType.FLOAT64,
                        Double.NEGATIVE_INFINITY, Double.NaN, source));
        IllegalArgumentException lowerNaN = assertThrows(
                IllegalArgumentException.class,
                () -> uniform(Shape.scalar(), DataType.FLOAT32, Double.NaN, 1.0d, source));
        IllegalArgumentException upperInfinity = assertThrows(
                IllegalArgumentException.class,
                () -> uniform(Shape.scalar(), DataType.BFLOAT16,
                        0.0d, Double.POSITIVE_INFINITY, source));
        IllegalArgumentException upperNaN = assertThrows(
                IllegalArgumentException.class,
                () -> uniform(Shape.scalar(), DataType.FLOAT64, 0.0d, Double.NaN, source));
        IllegalArgumentException equal = assertThrows(
                IllegalArgumentException.class,
                () -> uniform(Shape.scalar(), DataType.FLOAT64, 1.0d, 1.0d, source));
        IllegalArgumentException reversed = assertThrows(
                IllegalArgumentException.class,
                () -> uniform(Shape.scalar(), DataType.FLOAT64, 2.0d, -2.0d, source));
        IllegalArgumentException signedZero = assertThrows(
                IllegalArgumentException.class,
                () -> uniform(Shape.scalar(), DataType.FLOAT64, -0.0d, 0.0d, source));

        assertAll(
                () -> assertEquals(
                        "uniform random lower bound must be finite: -Infinity",
                        lowerInfinity.getMessage()),
                () -> assertEquals(
                        "uniform random lower bound must be finite: NaN", lowerNaN.getMessage()),
                () -> assertEquals(
                        "uniform random upper bound must be finite: Infinity",
                        upperInfinity.getMessage()),
                () -> assertEquals(
                        "uniform random upper bound must be finite: NaN", upperNaN.getMessage()),
                () -> assertEquals(
                        "uniform random lower bound must be less than upper bound: lower=1.0, upper=1.0",
                        equal.getMessage()),
                () -> assertEquals(
                        "uniform random lower bound must be less than upper bound: lower=2.0, upper=-2.0",
                        reversed.getMessage()),
                () -> assertEquals(
                        "uniform random lower bound must be less than upper bound: lower=-0.0, upper=0.0",
                        signedZero.getMessage()),
                () -> assertEquals(0, source.calls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void veryNarrowValidIntervalPreservesBinary64HalfOpenSampleButMayRoundToNarrowedUpper() {
        double lower = 1.0d;
        double upper = Math.nextUp(lower);
        BoundedScriptedGenerator float32Source = new BoundedScriptedGenerator(lower);
        BoundedScriptedGenerator bfloat16Source = new BoundedScriptedGenerator(lower);

        Tensor float32 = uniform(
                Shape.scalar(), DataType.FLOAT32, lower, upper, float32Source);
        Tensor bfloat16 = uniform(
                Shape.scalar(), DataType.BFLOAT16, lower, upper, bfloat16Source);

        assertAll(
                () -> assertTrue(lower < upper),
                () -> assertEquals((float) upper, heapArray(float32, float[].class)[0]),
                () -> assertEquals(
                        BFloat16Bits.fromFloat((float) upper),
                        heapArray(bfloat16, short[].class)[0]),
                () -> float32Source.assertEveryCallUsed(lower, upper),
                () -> bfloat16Source.assertEveryCallUsed(lower, upper));
    }

    @Test
    void equivalentSeededSourcesProduceEquivalentIndependentResults() {
        Random firstSource = new Random(0x5eedL);
        Random secondSource = new Random(0x5eedL);

        Tensor first = TensorRandoms.randomUniform(
                Shape.of(8),
                DataType.FLOAT32,
                -0.5d,
                1.25d,
                firstSource,
                Optional.empty(),
                false);
        Tensor second = TensorRandoms.randomUniform(
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
                () -> assertEquals(firstSource.nextDouble(), secondSource.nextDouble()));
    }

    @Test
    void nonConformingGeneratorResultIsConvertedWithoutPostValidation() {
        BoundedScriptedGenerator source = new BoundedScriptedGenerator(Double.NaN);

        Tensor tensor = uniform(
                Shape.scalar(), DataType.FLOAT64, -1.0d, 1.0d, source);

        assertAll(
                () -> assertTrue(Double.isNaN(heapArray(tensor, double[].class)[0])),
                () -> assertEquals(1, source.calls()));
    }

    @Test
    void generatorFailureLeavesEarlierCallsConsumedAndAllocatesNoIdentifier()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        ThrowingBoundedGenerator source = new ThrowingBoundedGenerator(2);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> uniform(Shape.of(4), DataType.FLOAT64, -2.0d, 3.0d, source));

        assertAll(
                () -> assertEquals("scripted generator failure", failure.getMessage()),
                () -> assertEquals(2, source.calls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void blankLabelAndPermanentExhaustionFollowAllDrawsWithDocumentedIdEffects()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long before = next.get();
        BoundedScriptedGenerator blankSource = new BoundedScriptedGenerator(0.0d, 0.5d, 0.75d);

        IllegalArgumentException blank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorRandoms.randomUniform(
                        Shape.of(3),
                        DataType.FLOAT32,
                        0.0d,
                        1.0d,
                        blankSource,
                        Optional.of(" \t\n "),
                        false));
        assertAll(
                () -> assertEquals("label must not be blank", blank.getMessage()),
                () -> assertEquals(3, blankSource.calls()),
                () -> assertEquals(before + 1, next.get()));

        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        BoundedScriptedGenerator exhaustedSource = new BoundedScriptedGenerator(-1.0d, 1.0d);
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            IllegalStateException exhausted = assertThrows(
                    IllegalStateException.class,
                    () -> TensorRandoms.randomUniform(
                            Shape.of(2),
                            DataType.BFLOAT16,
                            -2.0d,
                            2.0d,
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

    private static Tensor uniform(
            Shape shape,
            DataType dataType,
            double lower,
            double upper,
            RandomGenerator source) {
        return TensorRandoms.randomUniform(
                shape, dataType, lower, upper, source, Optional.empty(), false);
    }

    private static void assertDescriptor(
            Tensor tensor, Shape shape, DataType dataType, boolean requiresGrad) {
        assertAll(
                () -> assertSame(shape, tensor.descriptor().shape()),
                () -> assertSame(dataType, tensor.descriptor().dataType()),
                () -> assertEquals(
                        LayoutKind.DENSE_CONTIGUOUS,
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

    private static class BoundedScriptedGenerator implements RandomGenerator {
        private final double[] values;
        private final double[] origins;
        private final double[] bounds;
        private int calls;

        private BoundedScriptedGenerator(double... values) {
            this.values = values.clone();
            this.origins = new double[values.length];
            this.bounds = new double[values.length];
        }

        @Override
        public long nextLong() {
            throw new AssertionError("nextLong must not be called");
        }

        @Override
        public double nextDouble() {
            throw new AssertionError("unbounded nextDouble must not be called");
        }

        @Override
        public double nextDouble(double randomNumberOrigin, double randomNumberBound) {
            if (calls >= values.length) {
                throw new AssertionError("unexpected bounded call " + calls);
            }
            origins[calls] = randomNumberOrigin;
            bounds[calls] = randomNumberBound;
            return values[calls++];
        }

        private int calls() {
            return calls;
        }

        private void assertEveryCallUsed(double expectedOrigin, double expectedBound) {
            assertEquals(values.length, calls);
            for (int index = 0; index < calls; index++) {
                assertEquals(expectedOrigin, origins[index]);
                assertEquals(expectedBound, bounds[index]);
            }
        }
    }

    private static final class ThrowingBoundedGenerator implements RandomGenerator {
        private final int throwOnCall;
        private int calls;

        private ThrowingBoundedGenerator(int throwOnCall) {
            this.throwOnCall = throwOnCall;
        }

        @Override
        public long nextLong() {
            throw new AssertionError("nextLong must not be called");
        }

        @Override
        public double nextDouble() {
            throw new AssertionError("unbounded nextDouble must not be called");
        }

        @Override
        public double nextDouble(double randomNumberOrigin, double randomNumberBound) {
            calls++;
            if (calls == throwOnCall) {
                throw new IllegalStateException("scripted generator failure");
            }
            return randomNumberOrigin;
        }

        private int calls() {
            return calls;
        }
    }
}
