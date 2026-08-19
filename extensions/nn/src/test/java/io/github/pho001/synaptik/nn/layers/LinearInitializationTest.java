package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.initialization.ParameterInitializers;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class LinearInitializationTest {
    private static final RandomGeneratorFactory<RandomGenerator> FACTORY =
            RandomGeneratorFactory.of("L64X128MixRandom");

    @Test
    void initializesGlorotUniformWeightThenDeterministicZeroBiasForEveryFloatingType()
            throws ReflectiveOperationException {
        for (DataType dataType : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            BoundedSource source = new BoundedSource(6, 0.0d);
            AtomicLong next = nextTensorIdState();
            long before = next.get();
            double expectedBound = Math.sqrt(6.0d / ((double) 2L + (double) 3L));

            Linear layer = new Linear(2, 3, true, dataType, source);
            Tensor weight = layer.weight().value();
            Tensor bias = layer.bias().orElseThrow().value();

            assertAll(
                    () -> assertEquals(before, weight.id().value()),
                    () -> assertEquals(before + 1, bias.id().value()),
                    () -> assertEquals(before + 2, next.get()),
                    () -> assertSame(dataType, weight.descriptor().dataType()),
                    () -> assertSame(dataType, bias.descriptor().dataType()),
                    () -> assertEquals(Shape.of(3, 2), weight.descriptor().shape()),
                    () -> assertEquals(Shape.of(3), bias.descriptor().shape()),
                    () -> assertTrue(weight.descriptor().requiresGrad()),
                    () -> assertTrue(bias.descriptor().requiresGrad()),
                    () -> assertTrue(weight.provenance().isEmpty()),
                    () -> assertTrue(bias.provenance().isEmpty()),
                    () -> assertEquals(List.of("weight", "bias"), layer.parameters().stream()
                            .map(parameter -> parameter.name())
                            .toList()),
                    () -> source.assertCalls(-expectedBound, expectedBound),
                    () -> assertAllZero(weight),
                    () -> assertAllZero(bias));
        }
    }

    @Test
    void initializedNoBiasConsumesOnlyWeightDrawsAndOneIdentifier()
            throws ReflectiveOperationException {
        BoundedSource source = new BoundedSource(8, 0.0d);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        Linear layer = new Linear(4, 2, false, DataType.FLOAT32, source);

        assertAll(
                () -> assertEquals(before, layer.weight().value().id().value()),
                () -> assertEquals(before + 1, next.get()),
                () -> assertEquals(8, source.calls()),
                () -> assertTrue(layer.bias().isEmpty()),
                () -> assertEquals(List.of("weight"), layer.parameters().stream()
                        .map(parameter -> parameter.name())
                        .toList()));
    }

    @Test
    void callerControlledValidationFailsBeforeDrawOrIdentifierAllocation()
            throws ReflectiveOperationException {
        BoundedSource source = new BoundedSource(1, 0.0d);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        NullPointerException typeNull = assertThrows(
                NullPointerException.class,
                () -> new Linear(0, 0, true, null, null));
        NullPointerException sourceNull = assertThrows(
                NullPointerException.class,
                () -> new Linear(0, 0, true, DataType.FLOAT32, null));
        IllegalArgumentException inFeatures = assertThrows(
                IllegalArgumentException.class,
                () -> new Linear(0, 0, true, DataType.INT32, source));
        IllegalArgumentException outFeatures = assertThrows(
                IllegalArgumentException.class,
                () -> new Linear(1, 0, true, DataType.INT32, source));
        IllegalArgumentException type = assertThrows(
                IllegalArgumentException.class,
                () -> new Linear(1, 1, true, DataType.INT32, source));

        assertAll(
                () -> assertEquals("dataType", typeNull.getMessage()),
                () -> assertEquals("randomGenerator", sourceNull.getMessage()),
                () -> assertTrue(inFeatures.getMessage().contains("inFeatures")),
                () -> assertTrue(outFeatures.getMessage().contains("outFeatures")),
                () -> assertTrue(type.getMessage().contains("floating")),
                () -> assertEquals(0, source.calls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void arrayLimitFailureConsumesNoDrawOrIdentifier() throws ReflectiveOperationException {
        BoundedSource source = new BoundedSource(1, 0.0d);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertThrows(
                IllegalArgumentException.class,
                () -> new Linear(46_341, 46_341, false, DataType.FLOAT64, source));

        assertAll(
                () -> assertEquals(0, source.calls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void sourceFailureKeepsCompletedDrawsAndCreatesNoTensorIdentifier()
            throws ReflectiveOperationException {
        BoundedSource source = new BoundedSource(6, 0.0d).throwOnCall(2);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertThrows(
                DeliberateSourceFailure.class,
                () -> new Linear(2, 3, true, DataType.FLOAT64, source));

        assertAll(
                () -> assertEquals(3, source.calls()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void automaticConstructionCreatesNoTensorAndFirstForwardPublishesBeforeExpression()
            throws ReflectiveOperationException {
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false));
        AtomicLong next = nextTensorIdState();
        long beforeConstruction = next.get();

        Linear layer = new Linear(
                4,
                true,
                DataType.FLOAT32,
                ParameterInitialization.glorotUniform(),
                FACTORY,
                73L);

        assertEquals(beforeConstruction, next.get());
        Tensor result = layer.forward(input);
        Tensor weight = layer.weight().value();
        Tensor bias = layer.bias().orElseThrow().value();
        Tensor product = result.provenance().orElseThrow().inputs().getFirst();
        Tensor transposed = product.provenance().orElseThrow().inputs().get(1);

        assertAll(
                () -> assertEquals(beforeConstruction, weight.id().value()),
                () -> assertEquals(beforeConstruction + 1, bias.id().value()),
                () -> assertEquals(beforeConstruction + 2, transposed.id().value()),
                () -> assertEquals(beforeConstruction + 3, product.id().value()),
                () -> assertEquals(beforeConstruction + 4, result.id().value()),
                () -> assertEquals(beforeConstruction + 5, next.get()),
                () -> assertSame(weight,
                        transposed.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(bias, result.provenance().orElseThrow().inputs().get(1)));
    }

    @Test
    void automaticPoliciesMatchExistingInitializersAndLaterForwardsCreateNoParameterTensor() {
        long seed = 91L;
        for (ParameterInitialization initialization : List.of(
                ParameterInitialization.glorotNormal(),
                ParameterInitialization.glorotUniform(),
                ParameterInitialization.kaimingReluNormal(),
                ParameterInitialization.kaimingReluUniform(),
                ParameterInitialization.normal(0.25d, 0.5d),
                ParameterInitialization.uniform(-0.5d, 0.75d),
                ParameterInitialization.zeros(),
                ParameterInitialization.ones())) {
            Linear layer = new Linear(3, false, DataType.FLOAT64, initialization, FACTORY, seed);
            Tensor firstInput = TensorFactory.create(new TensorDescriptor(
                    DataType.FLOAT64, Shape.of(4, 2), Optional.empty(), false));
            layer.forward(firstInput);
            Tensor actual = layer.weight().value();
            Tensor expected = initialization.requiresRandomGenerator()
                    ? ParameterInitializers.initialize(
                            Shape.of(3, 2), DataType.FLOAT64, initialization, FACTORY.create(seed))
                    : ParameterInitializers.initialize(
                            Shape.of(3, 2), DataType.FLOAT64, initialization);
            assertArrayEquals(storage(actual), storage(expected));

            Tensor stableWeight = layer.weight().value();
            layer.forward(TensorFactory.create(new TensorDescriptor(
                    DataType.FLOAT64,
                    Shape.ofDimensions(new DynamicDimension("B"), new StaticDimension(2)),
                    Optional.empty(),
                    false)));
            assertSame(stableWeight, layer.weight().value());
        }
    }

    @Test
    void automaticValidationPrecedesParameterAndExpressionIdentifiers()
            throws ReflectiveOperationException {
        Linear layer = new Linear(
                4,
                false,
                DataType.FLOAT32,
                ParameterInitialization.kaimingReluNormal(),
                FACTORY,
                13L);
        Tensor wrongType = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT64, Shape.of(2, 3), Optional.empty(), false));
        Tensor scalar = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.scalar(), Optional.empty(), false));
        Tensor dynamicFinal = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(2), new DynamicDimension("F")),
                Optional.empty(),
                false));
        Tensor zeroFinal = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 0), Optional.empty(), false));
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () -> layer.forward(wrongType)),
                () -> assertThrows(IllegalArgumentException.class, () -> layer.forward(scalar)),
                () -> assertThrows(IllegalArgumentException.class, () -> layer.forward(dynamicFinal)),
                () -> assertThrows(IllegalArgumentException.class, () -> layer.forward(zeroFinal)),
                () -> assertEquals(before, next.get()),
                () -> assertThrows(IllegalStateException.class, layer::weight));
    }

    @Test
    void compatibleConcurrentFirstForwardsPublishOneParameterSet() throws Exception {
        Linear layer = new Linear(
                5,
                true,
                DataType.FLOAT32,
                ParameterInitialization.glorotNormal(),
                FACTORY,
                101L);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(3, 2), Optional.empty(), false));
        int callers = 8;
        CountDownLatch ready = new CountDownLatch(callers);
        CountDownLatch start = new CountDownLatch(1);

        try (var executor = Executors.newFixedThreadPool(callers)) {
            List<Future<Tensor>> futures = new ArrayList<>();
            for (int index = 0; index < callers; index++) {
                futures.add(executor.submit(() -> {
                    ready.countDown();
                    start.await();
                    return layer.forward(input);
                }));
            }
            ready.await();
            start.countDown();
            List<Tensor> results = new ArrayList<>();
            for (Future<Tensor> future : futures) {
                results.add(future.get());
            }

            Tensor weight = layer.weight().value();
            Tensor bias = layer.bias().orElseThrow().value();
            assertAll(
                    () -> assertEquals(2, layer.parameters().size()),
                    () -> assertTrue(results.stream().allMatch(result -> {
                        Tensor product = result.provenance().orElseThrow().inputs().getFirst();
                        Tensor transposed = product.provenance().orElseThrow().inputs().get(1);
                        return transposed.provenance().orElseThrow().inputs().getFirst() == weight
                                && result.provenance().orElseThrow().inputs().get(1) == bias;
                    })));
        }
    }

    @Test
    void incompatibleConcurrentFirstForwardsLetOneSchemaWin() throws Exception {
        Linear layer = new Linear(
                5,
                false,
                DataType.FLOAT32,
                ParameterInitialization.glorotUniform(),
                FACTORY,
                151L);
        Tensor widthTwo = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(3, 2), Optional.empty(), false));
        Tensor widthThree = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(3, 3), Optional.empty(), false));
        CountDownLatch ready = new CountDownLatch(2);
        CountDownLatch start = new CountDownLatch(1);

        try (var executor = Executors.newFixedThreadPool(2)) {
            Future<Tensor> first = executor.submit(() -> {
                ready.countDown();
                start.await();
                return layer.forward(widthTwo);
            });
            Future<Tensor> second = executor.submit(() -> {
                ready.countDown();
                start.await();
                return layer.forward(widthThree);
            });
            ready.await();
            start.countDown();

            int successes = 0;
            int failures = 0;
            for (Future<Tensor> future : List.of(first, second)) {
                try {
                    future.get();
                    successes++;
                } catch (ExecutionException exception) {
                    assertTrue(exception.getCause() instanceof IllegalArgumentException);
                    failures++;
                }
            }
            long inferred = layer.weight().value().descriptor().shape()
                    .dimension(1).staticSize().orElseThrow();
            int successfulCalls = successes;
            int failedCalls = failures;
            assertAll(
                    () -> assertEquals(1, successfulCalls),
                    () -> assertEquals(1, failedCalls),
                    () -> assertTrue(inferred == 2 || inferred == 3));
        }
    }

    @Test
    void automaticConstructorValidatesInSpecifiedOrderWithoutTensorEffects()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertAll(
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new Linear(0, true, null, null, null, 1L))
                        .getMessage().contains("outFeatures")),
                () -> assertEquals("dataType", assertThrows(
                                NullPointerException.class,
                                () -> new Linear(1, true, null, null, null, 1L))
                        .getMessage()),
                () -> assertEquals("weightInitialization", assertThrows(
                                NullPointerException.class,
                                () -> new Linear(1, true, DataType.FLOAT32, null, null, 1L))
                        .getMessage()),
                () -> assertEquals("randomGeneratorFactory", assertThrows(
                                NullPointerException.class,
                                () -> new Linear(
                                        1,
                                        true,
                                        DataType.FLOAT32,
                                        ParameterInitialization.glorotUniform(),
                                        null,
                                        1L))
                        .getMessage()),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new Linear(
                                        1,
                                        true,
                                        DataType.INT32,
                                        ParameterInitialization.glorotUniform(),
                                        FACTORY,
                                        1L))
                        .getMessage().contains("floating")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new Linear(
                                        1,
                                        true,
                                        DataType.FLOAT32,
                                        ParameterInitialization.glorotUniform(),
                                        RandomGeneratorFactory.of("SecureRandom"),
                                        1L))
                        .getMessage().contains("deterministic")),
                () -> assertEquals(before, next.get()));
    }

    private static double[] storage(Tensor tensor) {
        return (double[]) tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow();
    }

    private static void assertAllZero(Tensor tensor) {
        Object array = tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow();
        switch (tensor.descriptor().dataType()) {
            case FLOAT64 -> assertArrayEquals(new double[java.lang.reflect.Array.getLength(array)],
                    (double[]) array);
            case FLOAT32 -> assertArrayEquals(new float[java.lang.reflect.Array.getLength(array)],
                    (float[]) array);
            case BFLOAT16 -> assertArrayEquals(new short[java.lang.reflect.Array.getLength(array)],
                    (short[]) array);
            default -> throw new AssertionError("unexpected data type");
        }
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static final class BoundedSource implements RandomGenerator {
        private final int expectedCalls;
        private final double value;
        private final double[] origins;
        private final double[] bounds;
        private int calls;
        private int throwingCall = -1;

        private BoundedSource(int expectedCalls, double value) {
            this.expectedCalls = expectedCalls;
            this.value = value;
            origins = new double[expectedCalls];
            bounds = new double[expectedCalls];
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

        private void assertCalls(double expectedOrigin, double expectedBound) {
            assertEquals(expectedCalls, calls);
            for (int index = 0; index < expectedCalls; index++) {
                assertEquals(expectedOrigin, origins[index]);
                assertEquals(expectedBound, bounds[index]);
            }
        }
    }

    private static final class DeliberateSourceFailure extends RuntimeException {
    }
}
