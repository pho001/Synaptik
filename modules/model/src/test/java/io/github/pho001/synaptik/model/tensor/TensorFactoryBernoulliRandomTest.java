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

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Field;
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
class TensorFactoryBernoulliRandomTest {
    @Test
    void appliesStrictComparisonAndStoresCanonicalBoolBytesInRowMajorOrder() {
        ScriptedGenerator source = new ScriptedGenerator(
                Double.NEGATIVE_INFINITY,
                Math.nextDown(0.25d),
                0.25d,
                Math.nextUp(0.25d),
                Double.POSITIVE_INFINITY,
                Double.NaN);
        Shape shape = Shape.of(2, 3);

        Tensor tensor = TensorFactory.randomBernoulli(
                shape, 0.25d, source, Optional.of("  mask  "));

        assertAll(
                () -> assertArrayEquals(
                        new byte[] {1, 1, 0, 0, 0, 0}, heapArray(tensor)),
                () -> assertEquals(6, source.unboundedDoubleCalls()),
                () -> assertEquals(0, source.boundedDoubleCalls()),
                () -> assertEquals(0, source.booleanCalls()),
                () -> assertEquals(Optional.of("mask"), tensor.label()),
                () -> assertDescriptor(tensor, shape));
    }

    @Test
    void endpointsStillConsumeOneUnboundedDrawPerElementAndScalarAndEmptyStayExact() {
        ScriptedGenerator positiveZeroSource = new ScriptedGenerator(0.0d, 0.5d, Math.nextDown(1.0d));
        ScriptedGenerator negativeZeroSource = new ScriptedGenerator(0.0d, 0.5d, Math.nextDown(1.0d));
        ScriptedGenerator oneSource = new ScriptedGenerator(0.0d, 0.5d, Math.nextDown(1.0d));
        ScriptedGenerator scalarSource = new ScriptedGenerator(0.0d);
        ScriptedGenerator emptySource = new ScriptedGenerator();
        Shape vector = Shape.of(3);
        Shape emptyShape = Shape.of(2, 0, 3);

        Tensor positiveZero = bernoulli(vector, 0.0d, positiveZeroSource);
        Tensor negativeZero = bernoulli(vector, -0.0d, negativeZeroSource);
        Tensor one = bernoulli(vector, 1.0d, oneSource);
        Tensor scalar = bernoulli(Shape.scalar(), 1.0d, scalarSource);
        Tensor empty = bernoulli(emptyShape, 0.5d, emptySource);

        assertAll(
                () -> assertArrayEquals(new byte[] {0, 0, 0}, heapArray(positiveZero)),
                () -> assertArrayEquals(new byte[] {0, 0, 0}, heapArray(negativeZero)),
                () -> assertArrayEquals(new byte[] {1, 1, 1}, heapArray(one)),
                () -> assertArrayEquals(new byte[] {1}, heapArray(scalar)),
                () -> assertArrayEquals(new byte[0], heapArray(empty)),
                () -> assertEquals(3, positiveZeroSource.unboundedDoubleCalls()),
                () -> assertEquals(3, negativeZeroSource.unboundedDoubleCalls()),
                () -> assertEquals(3, oneSource.unboundedDoubleCalls()),
                () -> assertEquals(1, scalarSource.unboundedDoubleCalls()),
                () -> assertEquals(0, emptySource.unboundedDoubleCalls()),
                () -> assertDescriptor(scalar, Shape.scalar()),
                () -> assertDescriptor(empty, emptyShape));
    }

    @Test
    void nullValidationUsesPublicOrderWithoutDrawingOrAllocatingAnIdentifier()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Shape shape = Shape.scalar();
        ScriptedGenerator source = new ScriptedGenerator(0.5d);

        NullPointerException shapeFailure = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.randomBernoulli(null, 0.5d, null, null));
        NullPointerException sourceFailure = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.randomBernoulli(shape, 0.5d, null, null));
        NullPointerException labelFailure = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.randomBernoulli(shape, 0.5d, source, null));

        assertAll(
                () -> assertEquals("shape", shapeFailure.getMessage()),
                () -> assertEquals("randomGenerator", sourceFailure.getMessage()),
                () -> assertEquals("label", labelFailure.getMessage()),
                () -> assertEquals(0, source.unboundedDoubleCalls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void shapeAndCountValidationPrecedesProbabilityWithoutDrawingOrAllocatingAnIdentifier()
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
                () -> bernoulli(dynamic, Double.NaN, source));
        ArithmeticException overflowFailure = assertThrows(
                ArithmeticException.class,
                () -> bernoulli(overflow, Double.NaN, source));
        IllegalArgumentException limitFailure = assertThrows(
                IllegalArgumentException.class,
                () -> bernoulli(overLimit, Double.NaN, source));

        assertAll(
                () -> assertEquals(
                        "bernoulli random tensor creation requires a fully static shape: " + dynamic,
                        dynamicFailure.getMessage()),
                () -> assertEquals("long overflow", overflowFailure.getMessage()),
                () -> assertEquals(
                        "bernoulli random tensor element count exceeds Java array limit: required=2147483648, maximum=2147483647",
                        limitFailure.getMessage()),
                () -> assertEquals(0, source.unboundedDoubleCalls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void probabilityValidationIsFiniteClosedAndAppliesToEmptyOutput()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        ScriptedGenerator source = new ScriptedGenerator();
        Shape empty = Shape.of(0);
        double[] invalid = {
            Double.NaN,
            Double.NEGATIVE_INFINITY,
            Double.POSITIVE_INFINITY,
            Math.nextDown(0.0d),
            Math.nextUp(1.0d)
        };

        for (double probability : invalid) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> bernoulli(empty, probability, source));
            assertEquals(
                    "bernoulli probability must be finite and in [0.0, 1.0]: " + probability,
                    failure.getMessage());
        }

        assertAll(
                () -> assertEquals(0, source.unboundedDoubleCalls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void equivalentSeededSourcesProduceEquivalentIndependentResultsWithoutRetention() {
        Random firstSource = new Random(0x5eedL);
        Random secondSource = new Random(0x5eedL);

        Tensor first = bernoulli(Shape.of(16), 0.375d, firstSource);
        Tensor second = bernoulli(Shape.of(16), 0.375d, secondSource);

        assertAll(
                () -> assertArrayEquals(heapArray(first), heapArray(second)),
                () -> assertNotSame(heapArray(first), heapArray(second)),
                () -> assertNotSame(
                        first.hostStorage().orElseThrow(), second.hostStorage().orElseThrow()),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertEquals(firstSource.nextLong(), secondSource.nextLong()),
                () -> assertTrue(Arrays.stream(TensorFactory.class.getDeclaredFields())
                        .noneMatch(field -> RandomGenerator.class.isAssignableFrom(field.getType()))),
                () -> assertTrue(Arrays.stream(TensorRandoms.class.getDeclaredFields())
                        .noneMatch(field -> RandomGenerator.class.isAssignableFrom(field.getType()))));
    }

    @Test
    void generatorFailureLeavesEarlierCallsConsumedAndAllocatesNoIdentifier()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        ThrowingGenerator source = new ThrowingGenerator(2);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> bernoulli(Shape.of(4), 0.5d, source));

        assertAll(
                () -> assertEquals("scripted generator failure", failure.getMessage()),
                () -> assertEquals(2, source.unboundedDoubleCalls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void blankLabelAndPermanentExhaustionFollowAllDrawsWithDocumentedIdEffects()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long before = next.get();
        ScriptedGenerator blankSource = new ScriptedGenerator(0.1d, 0.2d, 0.3d);

        IllegalArgumentException blank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.randomBernoulli(
                        Shape.of(3), 0.5d, blankSource, Optional.of(" \t\n ")));
        assertAll(
                () -> assertEquals("label must not be blank", blank.getMessage()),
                () -> assertEquals(3, blankSource.unboundedDoubleCalls()),
                () -> assertEquals(before + 1, next.get()));

        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        ScriptedGenerator exhaustedSource = new ScriptedGenerator(0.25d, 0.75d);
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            IllegalStateException exhausted = assertThrows(
                    IllegalStateException.class,
                    () -> bernoulli(Shape.of(2), 0.5d, exhaustedSource));

            assertAll(
                    () -> assertEquals(
                            "tensor identifier space exhausted", exhausted.getMessage()),
                    () -> assertEquals(2, exhaustedSource.unboundedDoubleCalls()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    private static Tensor bernoulli(
            Shape shape, double probability, RandomGenerator randomGenerator) {
        return TensorFactory.randomBernoulli(
                shape, probability, randomGenerator, Optional.empty());
    }

    private static void assertDescriptor(Tensor tensor, Shape shape) {
        assertAll(
                () -> assertSame(shape, tensor.descriptor().shape()),
                () -> assertSame(DataType.BOOL, tensor.descriptor().dataType()),
                () -> assertEquals(
                        LayoutKind.DENSE_CONTIGUOUS,
                        tensor.descriptor().layout().orElseThrow().kind()),
                () -> assertFalse(tensor.descriptor().layout().orElseThrow().isView()),
                () -> assertFalse(tensor.descriptor().requiresGrad()),
                () -> assertEquals(
                        shape.knownElementCount().orElseThrow(),
                        tensor.hostStorage().orElseThrow().elementCapacity()));
    }

    private static byte[] heapArray(Tensor tensor) {
        return (byte[]) tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow();
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

    private static class ScriptedGenerator implements RandomGenerator {
        private final double[] values;
        private int index;
        private int boundedDoubleCalls;
        private int booleanCalls;

        ScriptedGenerator(double... values) {
            this.values = values.clone();
        }

        @Override
        public long nextLong() {
            throw new AssertionError("unexpected nextLong call");
        }

        @Override
        public double nextDouble() {
            if (index >= values.length) {
                throw new AssertionError("no scripted draw at index " + index);
            }
            return values[index++];
        }

        @Override
        public double nextDouble(double bound) {
            boundedDoubleCalls++;
            throw new AssertionError("unexpected bounded nextDouble call");
        }

        @Override
        public double nextDouble(double origin, double bound) {
            boundedDoubleCalls++;
            throw new AssertionError("unexpected bounded nextDouble call");
        }

        @Override
        public boolean nextBoolean() {
            booleanCalls++;
            throw new AssertionError("unexpected nextBoolean call");
        }

        int unboundedDoubleCalls() {
            return index;
        }

        int boundedDoubleCalls() {
            return boundedDoubleCalls;
        }

        int booleanCalls() {
            return booleanCalls;
        }
    }

    private static final class ThrowingGenerator extends ScriptedGenerator {
        private final int successfulCalls;

        ThrowingGenerator(int successfulCalls) {
            super(new double[successfulCalls]);
            this.successfulCalls = successfulCalls;
        }

        @Override
        public double nextDouble() {
            if (unboundedDoubleCalls() == successfulCalls) {
                throw new IllegalStateException("scripted generator failure");
            }
            return super.nextDouble();
        }
    }
}
