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
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.random.RandomGenerator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class TensorFactoryIntegralRandomTest {
    @Test
    void passesExactBoundsAndStoresDirectScriptedValuesForBothCarriersInOrder() {
        int intOrigin = -17;
        int intBound = 29;
        long longOrigin = -9_000_000_000L;
        long longBound = 11_000_000_000L;
        int[] intValues = {intOrigin, -1, 0, intBound - 1};
        long[] longValues = {longOrigin, -1L, 0L, longBound - 1};
        ScriptedIntegralGenerator intSource = ScriptedIntegralGenerator.forInts(intValues);
        ScriptedIntegralGenerator longSource = ScriptedIntegralGenerator.forLongs(longValues);
        Shape matrix = Shape.of(2, 2);
        Shape vector = Shape.of(4);

        Tensor int32 = TensorFactory.randomInt(
                matrix, intOrigin, intBound, intSource, Optional.of("  indices  "));
        Tensor int64 = TensorFactory.randomInt(
                vector, longOrigin, longBound, longSource, Optional.empty());

        assertAll(
                () -> assertArrayEquals(intValues, heapArray(int32, int[].class)),
                () -> assertArrayEquals(longValues, heapArray(int64, long[].class)),
                () -> intSource.assertEveryIntCallUsed(intOrigin, intBound),
                () -> longSource.assertEveryLongCallUsed(longOrigin, longBound),
                () -> assertEquals(Optional.of("indices"), int32.label()),
                () -> assertDescriptor(int32, matrix, DataType.INT32),
                () -> assertDescriptor(int64, vector, DataType.INT64),
                () -> assertNotSame(
                        int32.hostStorage().orElseThrow(), int64.hostStorage().orElseThrow()),
                () -> assertNotEquals(int32.id(), int64.id()));
    }

    @Test
    void supportsNegativeMixedSignAndPrimitiveBoundaryAdjacentIntervals() {
        ScriptedIntegralGenerator negativeInts =
                ScriptedIntegralGenerator.forInts(-10, -7, -4);
        ScriptedIntegralGenerator mixedLongs =
                ScriptedIntegralGenerator.forLongs(-3L, 0L, 4L);
        ScriptedIntegralGenerator minimumInts =
                ScriptedIntegralGenerator.forInts(Integer.MIN_VALUE, Integer.MIN_VALUE + 1);
        ScriptedIntegralGenerator maximumLongs =
                ScriptedIntegralGenerator.forLongs(Long.MAX_VALUE - 2, Long.MAX_VALUE - 1);

        Tensor negative = intRandom(Shape.of(3), -10, -3, negativeInts);
        Tensor mixed = longRandom(Shape.of(3), -3L, 5L, mixedLongs);
        Tensor minimum = intRandom(
                Shape.of(2), Integer.MIN_VALUE, Integer.MIN_VALUE + 2, minimumInts);
        Tensor maximum = longRandom(
                Shape.of(2), Long.MAX_VALUE - 2, Long.MAX_VALUE, maximumLongs);

        assertAll(
                () -> assertArrayEquals(new int[] {-10, -7, -4},
                        heapArray(negative, int[].class)),
                () -> assertArrayEquals(new long[] {-3, 0, 4},
                        heapArray(mixed, long[].class)),
                () -> assertArrayEquals(
                        new int[] {Integer.MIN_VALUE, Integer.MIN_VALUE + 1},
                        heapArray(minimum, int[].class)),
                () -> assertArrayEquals(
                        new long[] {Long.MAX_VALUE - 2, Long.MAX_VALUE - 1},
                        heapArray(maximum, long[].class)),
                () -> minimumInts.assertEveryIntCallUsed(
                        Integer.MIN_VALUE, Integer.MIN_VALUE + 2),
                () -> maximumLongs.assertEveryLongCallUsed(
                        Long.MAX_VALUE - 2, Long.MAX_VALUE));
    }

    @Test
    void scalarConsumesOneBoundedCallAndEmptyShapesConsumeNone() {
        ScriptedIntegralGenerator scalarInt = ScriptedIntegralGenerator.forInts(7);
        ScriptedIntegralGenerator scalarLong = ScriptedIntegralGenerator.forLongs(-2L);
        ScriptedIntegralGenerator emptyInt = ScriptedIntegralGenerator.forInts();
        ScriptedIntegralGenerator emptyLong = ScriptedIntegralGenerator.forLongs();
        Shape emptyShape = Shape.of(2, 0, 3);

        Tensor intScalar = intRandom(Shape.scalar(), 0, 10, scalarInt);
        Tensor longScalar = longRandom(Shape.scalar(), -5L, 5L, scalarLong);
        Tensor intEmpty = intRandom(emptyShape, 0, 1, emptyInt);
        Tensor longEmpty = longRandom(emptyShape, -1L, 0L, emptyLong);

        assertAll(
                () -> assertArrayEquals(new int[] {7}, heapArray(intScalar, int[].class)),
                () -> assertArrayEquals(new long[] {-2}, heapArray(longScalar, long[].class)),
                () -> assertEquals(1, scalarInt.intCalls()),
                () -> assertEquals(1, scalarLong.longCalls()),
                () -> assertEquals(0, emptyInt.intCalls()),
                () -> assertEquals(0, emptyLong.longCalls()),
                () -> assertEquals(0, heapArray(intEmpty, int[].class).length),
                () -> assertEquals(0, heapArray(longEmpty, long[].class).length),
                () -> assertDescriptor(intEmpty, emptyShape, DataType.INT32),
                () -> assertDescriptor(longEmpty, emptyShape, DataType.INT64));
    }

    @Test
    void nullValidationUsesPublicOrderWithoutDrawingOrAllocatingAnIdentifier()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Shape shape = Shape.scalar();
        ScriptedIntegralGenerator intSource = ScriptedIntegralGenerator.forInts(0);
        ScriptedIntegralGenerator longSource = ScriptedIntegralGenerator.forLongs(0L);

        NullPointerException intShape = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.randomInt(null, 0, 1, null, null));
        NullPointerException intSourceFailure = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.randomInt(shape, 0, 1, null, null));
        NullPointerException intLabel = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.randomInt(shape, 0, 1, intSource, null));
        NullPointerException longShape = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.randomInt(null, 0L, 1L, null, null));
        NullPointerException longSourceFailure = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.randomInt(shape, 0L, 1L, null, null));
        NullPointerException longLabel = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.randomInt(shape, 0L, 1L, longSource, null));

        assertAll(
                () -> assertEquals("shape", intShape.getMessage()),
                () -> assertEquals("randomGenerator", intSourceFailure.getMessage()),
                () -> assertEquals("label", intLabel.getMessage()),
                () -> assertEquals("shape", longShape.getMessage()),
                () -> assertEquals("randomGenerator", longSourceFailure.getMessage()),
                () -> assertEquals("label", longLabel.getMessage()),
                () -> assertEquals(0, intSource.intCalls()),
                () -> assertEquals(0, longSource.longCalls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void shapeAndCountValidationPrecedesBoundsWithoutDrawingOrAllocatingAnIdentifier()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        ScriptedIntegralGenerator source = ScriptedIntegralGenerator.forInts(0);
        Shape dynamic = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(2));
        Shape overflow = Shape.of(Long.MAX_VALUE, 2);
        Shape overLimit = Shape.of((long) Integer.MAX_VALUE + 1);

        IllegalArgumentException dynamicFailure = assertThrows(
                IllegalArgumentException.class,
                () -> intRandom(dynamic, 4, 4, source));
        ArithmeticException overflowFailure = assertThrows(
                ArithmeticException.class,
                () -> intRandom(overflow, 4, 4, source));
        IllegalArgumentException limitFailure = assertThrows(
                IllegalArgumentException.class,
                () -> intRandom(overLimit, 4, 4, source));

        assertAll(
                () -> assertEquals(
                        "integral random tensor creation requires a fully static shape: " + dynamic,
                        dynamicFailure.getMessage()),
                () -> assertEquals("long overflow", overflowFailure.getMessage()),
                () -> assertEquals(
                        "integral random tensor element count exceeds Java array limit: required=2147483648, maximum=2147483647",
                        limitFailure.getMessage()),
                () -> assertEquals(0, source.intCalls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void strictBoundsAreValidatedForBothCarriersIncludingEmptyOutput() throws Exception {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Shape empty = Shape.of(0);
        ScriptedIntegralGenerator intSource = ScriptedIntegralGenerator.forInts();
        ScriptedIntegralGenerator longSource = ScriptedIntegralGenerator.forLongs();

        IllegalArgumentException intEqual = assertThrows(
                IllegalArgumentException.class, () -> intRandom(empty, 5, 5, intSource));
        IllegalArgumentException intReversed = assertThrows(
                IllegalArgumentException.class, () -> intRandom(empty, 8, -2, intSource));
        IllegalArgumentException longEqual = assertThrows(
                IllegalArgumentException.class, () -> longRandom(empty, -4L, -4L, longSource));
        IllegalArgumentException longReversed = assertThrows(
                IllegalArgumentException.class, () -> longRandom(empty, 7L, -9L, longSource));

        assertAll(
                () -> assertEquals(
                        "integral random origin must be less than bound: origin=5, bound=5",
                        intEqual.getMessage()),
                () -> assertEquals(
                        "integral random origin must be less than bound: origin=8, bound=-2",
                        intReversed.getMessage()),
                () -> assertEquals(
                        "integral random origin must be less than bound: origin=-4, bound=-4",
                        longEqual.getMessage()),
                () -> assertEquals(
                        "integral random origin must be less than bound: origin=7, bound=-9",
                        longReversed.getMessage()),
                () -> assertEquals(0, intSource.intCalls()),
                () -> assertEquals(0, longSource.longCalls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void nonConformingBoundedResultsAreCopiedWithoutPostValidation() {
        ScriptedIntegralGenerator intSource = ScriptedIntegralGenerator.forInts(10);
        ScriptedIntegralGenerator longSource = ScriptedIntegralGenerator.forLongs(-6L);

        Tensor intTensor = intRandom(Shape.scalar(), 0, 10, intSource);
        Tensor longTensor = longRandom(Shape.scalar(), -5L, 5L, longSource);

        assertAll(
                () -> assertArrayEquals(new int[] {10}, heapArray(intTensor, int[].class)),
                () -> assertArrayEquals(new long[] {-6}, heapArray(longTensor, long[].class)),
                () -> intSource.assertEveryIntCallUsed(0, 10),
                () -> longSource.assertEveryLongCallUsed(-5L, 5L));
    }

    @Test
    void equivalentSeededSourcesProduceEquivalentIndependentResults() {
        Random firstIntSource = new Random(0x5eedL);
        Random secondIntSource = new Random(0x5eedL);
        Random firstLongSource = new Random(0x1eedL);
        Random secondLongSource = new Random(0x1eedL);

        Tensor firstInt = intRandom(Shape.of(8), -50, 75, firstIntSource);
        Tensor secondInt = intRandom(Shape.of(8), -50, 75, secondIntSource);
        Tensor firstLong = longRandom(
                Shape.of(8), -5_000_000_000L, 7_000_000_000L, firstLongSource);
        Tensor secondLong = longRandom(
                Shape.of(8), -5_000_000_000L, 7_000_000_000L, secondLongSource);

        assertAll(
                () -> assertArrayEquals(
                        heapArray(firstInt, int[].class), heapArray(secondInt, int[].class)),
                () -> assertArrayEquals(
                        heapArray(firstLong, long[].class), heapArray(secondLong, long[].class)),
                () -> assertNotSame(
                        heapArray(firstInt, int[].class), heapArray(secondInt, int[].class)),
                () -> assertNotSame(
                        heapArray(firstLong, long[].class), heapArray(secondLong, long[].class)),
                () -> assertNotEquals(firstInt.id(), secondInt.id()),
                () -> assertNotEquals(firstLong.id(), secondLong.id()),
                () -> assertEquals(firstIntSource.nextLong(), secondIntSource.nextLong()),
                () -> assertEquals(firstLongSource.nextLong(), secondLongSource.nextLong()));
    }

    @Test
    void generatorFailureLeavesEarlierCallsConsumedAndAllocatesNoIdentifier()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        ThrowingIntegralGenerator intSource = new ThrowingIntegralGenerator(2);
        ThrowingIntegralGenerator longSource = new ThrowingIntegralGenerator(3);

        IllegalStateException intFailure = assertThrows(
                IllegalStateException.class,
                () -> intRandom(Shape.of(4), -2, 3, intSource));
        IllegalStateException longFailure = assertThrows(
                IllegalStateException.class,
                () -> longRandom(Shape.of(4), -2L, 3L, longSource));

        assertAll(
                () -> assertEquals("scripted generator failure", intFailure.getMessage()),
                () -> assertEquals("scripted generator failure", longFailure.getMessage()),
                () -> assertEquals(2, intSource.intCalls()),
                () -> assertEquals(3, longSource.longCalls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void blankLabelAndPermanentExhaustionFollowAllCallsWithDocumentedIdEffects()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long before = next.get();
        ScriptedIntegralGenerator blankSource = ScriptedIntegralGenerator.forInts(1, 2, 3);

        IllegalArgumentException blank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.randomInt(
                        Shape.of(3), 0, 4, blankSource, Optional.of(" \t\n ")));
        assertAll(
                () -> assertEquals("label must not be blank", blank.getMessage()),
                () -> assertEquals(3, blankSource.intCalls()),
                () -> assertEquals(before + 1, next.get()));

        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        ScriptedIntegralGenerator exhaustedSource =
                ScriptedIntegralGenerator.forLongs(-1L, 1L);
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            IllegalStateException exhausted = assertThrows(
                    IllegalStateException.class,
                    () -> TensorFactory.randomInt(
                            Shape.of(2), -2L, 2L, exhaustedSource, Optional.empty()));

            assertAll(
                    () -> assertEquals(
                            "tensor identifier space exhausted", exhausted.getMessage()),
                    () -> assertEquals(2, exhaustedSource.longCalls()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    private static Tensor intRandom(
            Shape shape, int origin, int bound, RandomGenerator source) {
        return TensorFactory.randomInt(shape, origin, bound, source, Optional.empty());
    }

    private static Tensor longRandom(
            Shape shape, long origin, long bound, RandomGenerator source) {
        return TensorFactory.randomInt(shape, origin, bound, source, Optional.empty());
    }

    private static void assertDescriptor(Tensor tensor, Shape shape, DataType dataType) {
        assertAll(
                () -> assertSame(shape, tensor.descriptor().shape()),
                () -> assertSame(dataType, tensor.descriptor().dataType()),
                () -> assertEquals(
                        LayoutKind.DENSE_CONTIGUOUS,
                        tensor.descriptor().layout().orElseThrow().kind()),
                () -> assertFalse(tensor.descriptor().layout().orElseThrow().isView()),
                () -> assertFalse(tensor.descriptor().requiresGrad()),
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

    private static final class ScriptedIntegralGenerator implements RandomGenerator {
        private final int[] intValues;
        private final long[] longValues;
        private final int[] intOrigins;
        private final int[] intBounds;
        private final long[] longOrigins;
        private final long[] longBounds;
        private int intCalls;
        private int longCalls;

        private ScriptedIntegralGenerator(int[] intValues, long[] longValues) {
            this.intValues = intValues.clone();
            this.longValues = longValues.clone();
            this.intOrigins = new int[intValues.length];
            this.intBounds = new int[intValues.length];
            this.longOrigins = new long[longValues.length];
            this.longBounds = new long[longValues.length];
        }

        private static ScriptedIntegralGenerator forInts(int... values) {
            return new ScriptedIntegralGenerator(values, new long[0]);
        }

        private static ScriptedIntegralGenerator forLongs(long... values) {
            return new ScriptedIntegralGenerator(new int[0], values);
        }

        @Override
        public long nextLong() {
            throw new AssertionError("unbounded nextLong must not be called");
        }

        @Override
        public int nextInt() {
            throw new AssertionError("unbounded nextInt must not be called");
        }

        @Override
        public int nextInt(int randomNumberOrigin, int randomNumberBound) {
            if (intCalls >= intValues.length) {
                throw new AssertionError("unexpected bounded int call " + intCalls);
            }
            intOrigins[intCalls] = randomNumberOrigin;
            intBounds[intCalls] = randomNumberBound;
            return intValues[intCalls++];
        }

        @Override
        public long nextLong(long randomNumberOrigin, long randomNumberBound) {
            if (longCalls >= longValues.length) {
                throw new AssertionError("unexpected bounded long call " + longCalls);
            }
            longOrigins[longCalls] = randomNumberOrigin;
            longBounds[longCalls] = randomNumberBound;
            return longValues[longCalls++];
        }

        private int intCalls() {
            return intCalls;
        }

        private int longCalls() {
            return longCalls;
        }

        private void assertEveryIntCallUsed(int expectedOrigin, int expectedBound) {
            assertEquals(intValues.length, intCalls);
            for (int index = 0; index < intCalls; index++) {
                assertEquals(expectedOrigin, intOrigins[index]);
                assertEquals(expectedBound, intBounds[index]);
            }
        }

        private void assertEveryLongCallUsed(long expectedOrigin, long expectedBound) {
            assertEquals(longValues.length, longCalls);
            for (int index = 0; index < longCalls; index++) {
                assertEquals(expectedOrigin, longOrigins[index]);
                assertEquals(expectedBound, longBounds[index]);
            }
        }
    }

    private static final class ThrowingIntegralGenerator implements RandomGenerator {
        private final int throwOnCall;
        private int intCalls;
        private int longCalls;

        private ThrowingIntegralGenerator(int throwOnCall) {
            this.throwOnCall = throwOnCall;
        }

        @Override
        public long nextLong() {
            throw new AssertionError("unbounded nextLong must not be called");
        }

        @Override
        public int nextInt() {
            throw new AssertionError("unbounded nextInt must not be called");
        }

        @Override
        public int nextInt(int randomNumberOrigin, int randomNumberBound) {
            intCalls++;
            if (intCalls == throwOnCall) {
                throw new IllegalStateException("scripted generator failure");
            }
            return randomNumberOrigin;
        }

        @Override
        public long nextLong(long randomNumberOrigin, long randomNumberBound) {
            longCalls++;
            if (longCalls == throwOnCall) {
                throw new IllegalStateException("scripted generator failure");
            }
            return randomNumberOrigin;
        }

        private int intCalls() {
            return intCalls;
        }

        private int longCalls() {
            return longCalls;
        }
    }
}
