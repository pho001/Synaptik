package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.initialization.ParameterInitializers;
import java.lang.reflect.Field;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class GruCellInitializationTest {
    @Test
    void concurrentCompatibleFirstCallsShareOnePublishedPackedGroup() throws Exception {
        GruCell cell = new GruCell(
                3, false, DataType.FLOAT32, ParameterInitialization.zeros(), 1L);
        Tensor input = TensorFactory.zeros(
                Shape.of(2, 4), DataType.FLOAT32, Optional.empty(), false);
        Tensor hidden = TensorFactory.zeros(
                Shape.of(2, 3), DataType.FLOAT32, Optional.empty(), false);
        CountDownLatch ready = new CountDownLatch(2);
        CountDownLatch start = new CountDownLatch(1);
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            Future<Tensor> first = executor.submit(() -> {
                ready.countDown();
                start.await();
                return cell.forward(input, hidden);
            });
            Future<Tensor> second = executor.submit(() -> {
                ready.countDown();
                start.await();
                return cell.forward(input, hidden);
            });
            ready.await();
            start.countDown();
            Tensor firstResult = first.get();
            Tensor secondResult = second.get();

            assertAll(
                    () -> assertNotSame(firstResult, secondResult),
                    () -> assertEquals(List.of("inputWeight", "hiddenWeight"),
                            cell.parameters().stream().map(value -> value.name()).toList()));
        } finally {
            executor.shutdownNow();
        }
    }

    @Test
    void automaticRandomPolicyMatchesTheExactStandardAdvancedPackedStream() {
        long seed = 72L;
        ParameterInitialization policy = ParameterInitialization.kaimingReluNormal();
        RandomGenerator expectedSource = RandomGeneratorFactory.<RandomGenerator>of(
                "L64X128MixRandom").create(seed);
        Tensor expectedInput = ParameterInitializers.initialize(
                Shape.of(9, 4), DataType.FLOAT32, policy, expectedSource);
        Tensor expectedHidden = ParameterInitializers.initialize(
                Shape.of(9, 3), DataType.FLOAT32, policy, expectedSource);
        GruCell cell = new GruCell(3, false, DataType.FLOAT32, policy, seed);

        cell.forward(
                TensorFactory.zeros(Shape.of(2, 4), DataType.FLOAT32, Optional.empty(), false),
                TensorFactory.zeros(Shape.of(2, 3), DataType.FLOAT32, Optional.empty(), false));

        assertAll(
                () -> assertArrayEquals(floatValues(expectedInput),
                        floatValues(cell.inputWeight().value())),
                () -> assertArrayEquals(floatValues(expectedHidden),
                        floatValues(cell.hiddenWeight().value())));
    }

    @Test
    void strictLoadBindsExactAutomaticPackedGroupWithoutInitializationEffects()
            throws ReflectiveOperationException {
        GruCell donor = new GruCell(4, 3, true, DataType.FLOAT32,
                new BoundedSource(63, 0.0d));
        GruCell target = new GruCell(
                3, true, DataType.FLOAT32, ParameterInitialization.glorotNormal(), 45L);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        target.loadStateDictionary(donor.stateDictionary());

        assertAll(
                () -> assertEquals(before, ids.get()),
                () -> assertSame(donor.inputWeight().value(), target.inputWeight().value()),
                () -> assertSame(donor.hiddenWeight().value(), target.hiddenWeight().value()),
                () -> assertSame(donor.bias().orElseThrow().value(),
                        target.bias().orElseThrow().value()),
                () -> assertEquals(donor.stateDictionary(), target.stateDictionary()));

        GruCell incompatible = new GruCell(
                4, true, DataType.FLOAT32, ParameterInitialization.zeros(), 1L);
        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> incompatible.loadStateDictionary(donor.stateDictionary())),
                () -> assertThrows(IllegalStateException.class, incompatible::parameters));
    }

    @Test
    void automaticConstantPolicyPublishesPackedGroupOnFirstForwardOnly()
            throws ReflectiveOperationException {
        AtomicLong ids = nextTensorIdState();
        long beforeConstruction = ids.get();
        GruCell cell = new GruCell(
                3, true, DataType.FLOAT32, ParameterInitialization.ones(), 92L);

        assertAll(
                () -> assertEquals(beforeConstruction, ids.get()),
                () -> assertThrows(IllegalStateException.class, cell::parameters));

        Tensor input = TensorFactory.zeros(
                Shape.of(2, 4), DataType.FLOAT32, Optional.empty(), false);
        Tensor hidden = TensorFactory.zeros(
                Shape.of(2, 3), DataType.FLOAT32, Optional.empty(), false);
        long beforeBinding = ids.get();
        Tensor result = cell.forward(input, hidden);

        assertAll(
                () -> assertEquals(Shape.of(9, 4), cell.inputWeight().value().descriptor().shape()),
                () -> assertEquals(Shape.of(9, 3), cell.hiddenWeight().value().descriptor().shape()),
                () -> assertEquals(Shape.of(9), cell.bias().orElseThrow().value()
                        .descriptor().shape()),
                () -> assertEquals(beforeBinding, cell.inputWeight().value().id().value()),
                () -> assertEquals(beforeBinding + 1, cell.hiddenWeight().value().id().value()),
                () -> assertEquals(beforeBinding + 2,
                        cell.bias().orElseThrow().value().id().value()),
                () -> assertTrue(result.id().value() > beforeBinding + 2),
                () -> assertAllOne(cell.inputWeight().value()),
                () -> assertAllOne(cell.hiddenWeight().value()),
                () -> assertAllZero(cell.bias().orElseThrow().value()));
    }

    @Test
    void initializesPackedWeightsThenZeroBiasForEveryFloatingType()
            throws ReflectiveOperationException {
        for (DataType dataType : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            BoundedSource source = new BoundedSource(45, 0.0d);
            AtomicLong ids = nextTensorIdState();
            long before = ids.get();

            GruCell cell = new GruCell(2, 3, true, dataType, source);
            Tensor inputWeight = cell.inputWeight().value();
            Tensor hiddenWeight = cell.hiddenWeight().value();
            Tensor bias = cell.bias().orElseThrow().value();

            assertAll(
                    () -> assertEquals(before, inputWeight.id().value()),
                    () -> assertEquals(before + 1, hiddenWeight.id().value()),
                    () -> assertEquals(before + 2, bias.id().value()),
                    () -> assertEquals(before + 3, ids.get()),
                    () -> assertSame(dataType, inputWeight.descriptor().dataType()),
                    () -> assertSame(dataType, hiddenWeight.descriptor().dataType()),
                    () -> assertSame(dataType, bias.descriptor().dataType()),
                    () -> assertEquals(Shape.of(9, 2), inputWeight.descriptor().shape()),
                    () -> assertEquals(Shape.of(9, 3), hiddenWeight.descriptor().shape()),
                    () -> assertEquals(Shape.of(9), bias.descriptor().shape()),
                    () -> assertTrue(inputWeight.descriptor().requiresGrad()),
                    () -> assertTrue(hiddenWeight.descriptor().requiresGrad()),
                    () -> assertTrue(bias.descriptor().requiresGrad()),
                    () -> assertTrue(inputWeight.provenance().isEmpty()),
                    () -> assertTrue(hiddenWeight.provenance().isEmpty()),
                    () -> assertTrue(bias.provenance().isEmpty()),
                    () -> assertEquals(
                            List.of("inputWeight", "hiddenWeight", "bias"),
                            cell.parameters().stream().map(parameter -> parameter.name()).toList()),
                    () -> source.assertRange(
                            0, 18, -Math.sqrt(6.0d / 11.0d), Math.sqrt(6.0d / 11.0d)),
                    () -> source.assertRange(
                            18, 45, -Math.sqrt(6.0d / 12.0d), Math.sqrt(6.0d / 12.0d)),
                    () -> assertAllZero(inputWeight),
                    () -> assertAllZero(hiddenWeight),
                    () -> assertAllZero(bias));
        }
    }

    @Test
    void initializedNoBiasUsesBothWeightsAndNoAdditionalDrawOrIdentifier()
            throws ReflectiveOperationException {
        BoundedSource source = new BoundedSource(72, 0.0d);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        GruCell cell = new GruCell(2, 4, false, DataType.FLOAT32, source);

        assertAll(
                () -> assertEquals(before, cell.inputWeight().value().id().value()),
                () -> assertEquals(before + 1, cell.hiddenWeight().value().id().value()),
                () -> assertEquals(before + 2, ids.get()),
                () -> assertEquals(72, source.calls()),
                () -> assertTrue(cell.bias().isEmpty()),
                () -> assertEquals(
                        List.of("inputWeight", "hiddenWeight"),
                        cell.parameters().stream().map(parameter -> parameter.name()).toList()));
    }

    @Test
    void callerValidationAndEveryRequestedCountPrecedeDrawsAndIdentifiers()
            throws ReflectiveOperationException {
        BoundedSource source = new BoundedSource(1, 0.0d);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertEquals(
                        "dataType",
                        assertThrows(
                                NullPointerException.class,
                                () -> new GruCell(0, 0, true, null, null)).getMessage()),
                () -> assertEquals(
                        "randomGenerator",
                        assertThrows(
                                NullPointerException.class,
                                () -> new GruCell(0, 0, true, DataType.FLOAT32, null)).getMessage()),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new GruCell(0, 0, true, DataType.INT32, source))
                        .getMessage().contains("inputSize")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new GruCell(1, 0, true, DataType.INT32, source))
                        .getMessage().contains("hiddenSize")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new GruCell(1, 1, true, DataType.INT32, source))
                        .getMessage().contains("floating")),
                () -> assertThrows(
                        ArithmeticException.class,
                        () -> new GruCell(
                                1, Long.MAX_VALUE / 3L + 1L,
                                false, DataType.FLOAT32, source)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new GruCell(30_000, 30_000, false, DataType.FLOAT32, source)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new GruCell(1, 30_000, false, DataType.FLOAT32, source)),
                () -> assertEquals(0, source.calls()),
                () -> assertEquals(before, ids.get()));
    }

    @Test
    void sourceFailureInInputWeightCreatesNoTensorAndPreservesCompletedCalls()
            throws ReflectiveOperationException {
        BoundedSource source = new BoundedSource(45, 0.0d).throwOnCall(2);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertThrows(
                DeliberateSourceFailure.class,
                () -> new GruCell(2, 3, true, DataType.FLOAT64, source));

        assertAll(
                () -> assertEquals(3, source.calls()),
                () -> assertEquals(before, ids.get()));
    }

    @Test
    void sourceFailureInHiddenWeightKeepsCompletedInputTensorAndIdentifier()
            throws ReflectiveOperationException {
        BoundedSource source = new BoundedSource(45, 0.0d).throwOnCall(20);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertThrows(
                DeliberateSourceFailure.class,
                () -> new GruCell(2, 3, true, DataType.FLOAT64, source));

        assertAll(
                () -> assertEquals(21, source.calls()),
                () -> assertEquals(before + 1, ids.get()));
    }

    @Test
    void identifierFailureKeepsCompletedWeightDrawsWithoutReturningACell()
            throws ReflectiveOperationException {
        BoundedSource source = new BoundedSource(45, 0.0d);
        AtomicLong ids = nextTensorIdState();
        AtomicBoolean maximumClaimed = maximumTensorIdClaimedState();
        long saved = ids.get();
        boolean savedMaximumClaimed = maximumClaimed.get();
        try {
            ids.set(Long.MAX_VALUE);
            maximumClaimed.set(false);
            IllegalStateException failure = assertThrows(
                    IllegalStateException.class,
                    () -> new GruCell(2, 3, true, DataType.FLOAT32, source));
            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(45, source.calls()),
                    () -> assertEquals(Long.MAX_VALUE, ids.get()),
                    () -> assertTrue(maximumClaimed.get()));
        } finally {
            ids.set(saved);
            maximumClaimed.set(savedMaximumClaimed);
        }
    }

    private static void assertAllZero(Tensor tensor) {
        Object array = tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow();
        switch (tensor.descriptor().dataType()) {
            case FLOAT64 -> assertArrayEquals(
                    new double[java.lang.reflect.Array.getLength(array)], (double[]) array);
            case FLOAT32 -> assertArrayEquals(
                    new float[java.lang.reflect.Array.getLength(array)], (float[]) array);
            case BFLOAT16 -> assertArrayEquals(
                    new short[java.lang.reflect.Array.getLength(array)], (short[]) array);
            default -> throw new AssertionError("unexpected data type");
        }
    }

    private static float[] floatValues(Tensor tensor) {
        return (float[]) tensor.hostStorage().orElseThrow().segment()
                .heapBase().orElseThrow();
    }

    private static void assertAllOne(Tensor tensor) {
        Object array = tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow();
        switch (tensor.descriptor().dataType()) {
            case FLOAT64 -> assertArrayEquals(
                    java.util.Collections.nCopies(
                                    java.lang.reflect.Array.getLength(array), 1.0d)
                            .stream().mapToDouble(Double::doubleValue).toArray(),
                    (double[]) array);
            case FLOAT32 -> {
                float[] expected = new float[java.lang.reflect.Array.getLength(array)];
                java.util.Arrays.fill(expected, 1.0f);
                assertArrayEquals(expected, (float[]) array);
            }
            case BFLOAT16 -> {
                short[] expected = new short[java.lang.reflect.Array.getLength(array)];
                java.util.Arrays.fill(expected,
                        io.github.pho001.synaptik.model.datatype.BFloat16Bits.fromFloat(1.0f));
                assertArrayEquals(expected, (short[]) array);
            }
            default -> throw new AssertionError("unexpected data type");
        }
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

    private static final class BoundedSource implements RandomGenerator {
        private final double value;
        private final double[] origins;
        private final double[] bounds;
        private int calls;
        private int throwingCall = -1;

        private BoundedSource(int capacity, double value) {
            this.value = value;
            origins = new double[capacity];
            bounds = new double[capacity];
        }

        private BoundedSource throwOnCall(int call) {
            throwingCall = call;
            return this;
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
        public double nextDouble(double origin, double bound) {
            int call = calls++;
            if (call < origins.length) {
                origins[call] = origin;
                bounds[call] = bound;
            }
            if (call == throwingCall) {
                throw new DeliberateSourceFailure();
            }
            return value;
        }

        private int calls() {
            return calls;
        }

        private void assertRange(int from, int to, double origin, double bound) {
            assertEquals(origins.length, calls);
            for (int index = from; index < to; index++) {
                assertEquals(origin, origins[index]);
                assertEquals(bound, bounds[index]);
            }
        }
    }

    private static final class DeliberateSourceFailure extends RuntimeException {
    }
}
