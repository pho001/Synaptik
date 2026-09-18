package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.*;

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
import java.util.concurrent.ExecutionException;
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
class ConvolutionInitializationTest {
    private static final RandomGeneratorFactory<RandomGenerator> FACTORY =
            RandomGeneratorFactory.of("L64X128MixRandom");
    private static final List<ParameterInitialization> POLICIES = List.of(
            ParameterInitialization.glorotNormal(),
            ParameterInitialization.glorotUniform(),
            ParameterInitialization.kaimingReluNormal(),
            ParameterInitialization.kaimingReluUniform(),
            ParameterInitialization.normal(0.25d, 0.5d),
            ParameterInitialization.uniform(-0.75d, 0.5d),
            ParameterInitialization.zeros(),
            ParameterInitialization.ones());

    @Test
    void allPoliciesAreDeterministicAcrossRanksAndFloatingTypes() {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            for (ParameterInitialization policy : POLICIES) {
                Conv1d first1 = new Conv1d(4, 3, 1, 0, 1, 2, true, type, policy, 41L);
                Conv1d second1 = new Conv1d(4, 3, 1, 0, 1, 2, true, type, policy, 41L);
                first1.forward(tensor(Shape.of(1, 4, 7), type));
                second1.forward(tensor(Shape.of(2, 4, 9), type));
                assertSameStorage(first1.weight().value(), second1.weight().value());
                assertAllZero(first1.bias().orElseThrow().value());

                Conv2d first2 = new Conv2d(4, 2, 3, 1, 1, 0, 0, 1, 1, 2, false, type, policy, 42L);
                Conv2d second2 = new Conv2d(4, 2, 3, 1, 1, 0, 0, 1, 1, 2, false, type, policy, 42L);
                first2.forward(tensor(Shape.of(1, 4, 5, 6), type));
                second2.forward(tensor(Shape.of(3, 4, 7, 8), type));
                assertSameStorage(first2.weight().value(), second2.weight().value());

                Conv3d first3 = new Conv3d(4, 2, 2, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, false, type, policy, 43L);
                Conv3d second3 = new Conv3d(4, 2, 2, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, false, type, policy, 43L);
                first3.forward(tensor(Shape.of(1, 4, 4, 5, 6), type));
                second3.forward(tensor(Shape.of(2, 4, 6, 7, 8), type));
                assertSameStorage(first3.weight().value(), second3.weight().value());
            }
        }
    }

    @Test
    void fanPoliciesMatchIndependentGroupedCleanJavaOracleAcrossRanksAndTypes() {
        for (ParameterInitialization policy : POLICIES.subList(0, 4)) {
            assertFanOracle(
                    new Conv1d(12, 5, 1, 0, 1, 3, false, DataType.FLOAT64, policy, 101L),
                    tensor(Shape.of(2, 6, 9), DataType.FLOAT64), Shape.of(12, 2, 5),
                    DataType.FLOAT64, policy, 101L, 10L, 20L);
            assertFanOracle(
                    new Conv2d(8, 2, 3, 1, 1, 0, 0, 1, 1, 2, false,
                            DataType.FLOAT32, policy, 102L),
                    tensor(Shape.of(2, 6, 5, 7), DataType.FLOAT32), Shape.of(8, 3, 2, 3),
                    DataType.FLOAT32, policy, 102L, 18L, 24L);
            assertFanOracle(
                    new Conv3d(8, 2, 2, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1, 4, false,
                            DataType.BFLOAT16, policy, 103L),
                    tensor(Shape.of(2, 12, 4, 5, 6), DataType.BFLOAT16),
                    Shape.of(8, 3, 2, 2, 3), DataType.BFLOAT16, policy, 103L, 36L, 24L);
        }
    }

    @Test
    void constructionAndInvalidPreflightConsumeNoTensorIdentifierOrPublishState()
            throws ReflectiveOperationException {
        Tensor dynamicChannels = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(1), new DynamicDimension("C"),
                        new StaticDimension(7)), Optional.empty(), false));
        Tensor zeroChannels = tensor(Shape.of(1, 0, 7), DataType.FLOAT32);
        Tensor invalidGeometry = tensor(Shape.of(1, 4, 2), DataType.FLOAT32);
        Tensor aboveArrayLimit = tensor(Shape.of(1, 600_000_000, 3), DataType.FLOAT32);
        Tensor geometryOverflow = tensor(Shape.of(1, 4, Long.MAX_VALUE), DataType.FLOAT32);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();
        Conv1d layer = new Conv1d(4, 3, 1, 0, 1, 1, true, DataType.FLOAT32,
                ParameterInitialization.glorotUniform(), 17L);
        Conv1d padded = new Conv1d(4, 3, 1, 1, 1, 1, false, DataType.FLOAT32,
                ParameterInitialization.zeros(), 18L);

        assertAll(
                () -> assertEquals(before, ids.get()),
                () -> assertThrows(IllegalArgumentException.class, () -> layer.forward(dynamicChannels)),
                () -> assertThrows(IllegalArgumentException.class, () -> layer.forward(zeroChannels)),
                () -> assertThrows(IllegalArgumentException.class, () -> layer.forward(invalidGeometry)),
                () -> assertThrows(IllegalArgumentException.class, () -> layer.forward(aboveArrayLimit)),
                () -> assertThrows(ArithmeticException.class, () -> padded.forward(geometryOverflow)),
                () -> assertEquals(before, ids.get()),
                () -> assertThrows(IllegalStateException.class, layer::weight),
                () -> assertThrows(IllegalStateException.class, layer::stateDictionary),
                () -> assertThrows(IllegalStateException.class, padded::weight));
    }

    @Test
    void compatibleConcurrentFirstCallsPublishOneCompleteGroupUsedByEveryResult()
            throws Exception {
        Conv2d layer = new Conv2d(8, 3, 3, 1, 1, 1, 1, 1, 1, 2, true,
                DataType.FLOAT32, ParameterInitialization.glorotUniform(), 91L);
        Tensor input = tensor(Shape.of(2, 4, 7, 7), DataType.FLOAT32);
        CountDownLatch ready = new CountDownLatch(8);
        CountDownLatch start = new CountDownLatch(1);
        List<Tensor> results = new ArrayList<>();
        try (var executor = Executors.newFixedThreadPool(8)) {
            List<Future<Tensor>> futures = new ArrayList<>();
            for (int index = 0; index < 8; index++) {
                futures.add(executor.submit(() -> {
                    ready.countDown();
                    start.await();
                    return layer.forward(input);
                }));
            }
            ready.await();
            start.countDown();
            for (Future<Tensor> future : futures) results.add(future.get());
        }
        Tensor weight = layer.weight().value();
        Tensor bias = layer.bias().orElseThrow().value();
        assertAll(
                () -> assertEquals(8, results.stream().map(Tensor::id).distinct().count()),
                () -> assertEquals(List.of("weight", "bias"),
                        layer.parameters().stream().map(parameter -> parameter.name()).toList()),
                () -> assertTrue(results.stream().allMatch(result -> {
                    List<Tensor> inputs = result.provenance().orElseThrow().inputs();
                    return inputs.get(1) == weight && inputs.get(2) == bias;
                })));
    }

    @Test
    void incompatibleConcurrentFirstCallsLeaveExactlyOneCompleteWinningSchema()
            throws Exception {
        Conv2d layer = new Conv2d(8, 3, 3, 1, 1, 1, 1, 1, 1, 2, true,
                DataType.FLOAT32, ParameterInitialization.glorotUniform(), 92L);
        Tensor fourChannels = tensor(Shape.of(2, 4, 7, 7), DataType.FLOAT32);
        Tensor sixChannels = tensor(Shape.of(2, 6, 7, 7), DataType.FLOAT32);
        CountDownLatch ready = new CountDownLatch(2);
        CountDownLatch start = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(2)) {
            Future<Tensor> first = executor.submit(() -> raceForward(layer, fourChannels, ready, start));
            Future<Tensor> second = executor.submit(() -> raceForward(layer, sixChannels, ready, start));
            ready.await();
            start.countDown();
            int successes = 0;
            int failures = 0;
            for (Future<Tensor> future : List.of(first, second)) {
                try {
                    Tensor result = future.get();
                    assertSame(layer.weight().value(), result.provenance().orElseThrow().inputs().get(1));
                    assertSame(layer.bias().orElseThrow().value(), result.provenance().orElseThrow().inputs().get(2));
                    successes++;
                } catch (ExecutionException exception) {
                    assertInstanceOf(IllegalArgumentException.class, exception.getCause());
                    failures++;
                }
            }
            assertEquals(1, successes);
            assertEquals(1, failures);
        }
        long channelsPerGroup = layer.weight().value().descriptor().shape()
                .dimension(1).staticSize().orElseThrow();
        assertAll(
                () -> assertTrue(channelsPerGroup == 2L || channelsPerGroup == 3L),
                () -> assertEquals(List.of("weight", "bias"), layer.stateDictionary().entries()
                        .stream().map(entry -> entry.path()).toList()));
    }

    @Test
    void identifierFailureAfterWeightCreationPublishesNothingAndSeededRetryIsExact()
            throws Exception {
        Tensor input = tensor(Shape.of(1, 4, 7, 7), DataType.FLOAT32);
        Conv2d reference = new Conv2d(8, 3, 3, 1, 1, 1, 1, 1, 1, 2, true,
                DataType.FLOAT32, ParameterInitialization.glorotUniform(), 211L);
        reference.forward(input);
        Conv2d layer = new Conv2d(8, 3, 3, 1, 1, 1, 1, 1, 1, 2, true,
                DataType.FLOAT32, ParameterInitialization.glorotUniform(), 211L);
        AtomicLong ids = nextTensorIdState();
        AtomicBoolean maximumClaimed = maximumTensorIdClaimedState();
        long savedNext = ids.get();
        boolean savedClaimed = maximumClaimed.get();
        try {
            ids.set(Long.MAX_VALUE);
            maximumClaimed.set(false);
            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> layer.forward(input));
            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, ids.get()),
                    () -> assertTrue(maximumClaimed.get()),
                    () -> assertThrows(IllegalStateException.class, layer::weight),
                    () -> assertThrows(IllegalStateException.class, () -> layer.bias().orElseThrow()),
                    () -> assertThrows(IllegalStateException.class, layer::stateDictionary));
        } finally {
            ids.set(savedNext);
            maximumClaimed.set(savedClaimed);
        }

        Tensor result = layer.forward(input);
        assertAll(
                () -> assertSameStorage(reference.weight().value(), layer.weight().value()),
                () -> assertAllZero(layer.bias().orElseThrow().value()),
                () -> assertSame(layer.weight().value(), result.provenance().orElseThrow().inputs().get(1)),
                () -> assertSame(layer.bias().orElseThrow().value(), result.provenance().orElseThrow().inputs().get(2)),
                () -> assertEquals(List.of("weight", "bias"), layer.stateDictionary().entries()
                        .stream().map(entry -> entry.path()).toList()));
    }

    @Test
    void checkedKernelFanEffectiveKernelAndPaddingOverflowNeedNoAllocation()
            throws ReflectiveOperationException {
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();
        assertAll(
                () -> assertThrows(ArithmeticException.class, () -> new Conv3d(
                        2, Long.MAX_VALUE, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                        false, DataType.FLOAT32, ParameterInitialization.zeros(), 0L)),
                () -> assertThrows(ArithmeticException.class, () -> new Conv1d(
                        Long.MAX_VALUE, 3, 1, 0, 1, 1, false, DataType.FLOAT32,
                        ParameterInitialization.zeros(), 0L)),
                () -> assertThrows(ArithmeticException.class, () -> new Conv1d(
                        1, Long.MAX_VALUE, 1, 0, 2, 1, false, DataType.FLOAT32,
                        ParameterInitialization.zeros(), 0L)),
                () -> assertThrows(ArithmeticException.class, () -> new Conv2d(
                        2, 1, 1, 1, 1, Long.MAX_VALUE, 0, 1, 1, 1, false,
                        DataType.FLOAT32, ParameterInitialization.zeros(), 0L)),
                () -> assertEquals(before, ids.get()));
    }

    @Test
    void constructorsValidateInRankOrderBeforeReservationOrTensorEffects()
            throws ReflectiveOperationException {
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();
        assertAll(
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> new Conv1d(0, 0, 0, -1, 0, 0, false, null, null, 0L))
                        .getMessage().contains("outChannels")),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> new Conv2d(2, 0, 0, 0, 0, -1, -1, 0, 0, 0, false,
                                null, null, 0L)).getMessage().contains("kernelHeight")),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> new Conv3d(2, 1, 1, 1, 0, 0, 0, -1, -1, -1,
                                0, 0, 0, 0, false, null, null, 0L))
                        .getMessage().contains("strideDepth")),
                () -> assertEquals("dataType", assertThrows(NullPointerException.class,
                        () -> new Conv1d(2, 1, 1, 0, 1, 1, false, null, null, 0L))
                        .getMessage()),
                () -> assertEquals("weightInitialization", assertThrows(NullPointerException.class,
                        () -> new Conv2d(2, 1, 1, 1, 1, 0, 0, 1, 1, 1, false,
                                DataType.FLOAT32, null, 0L)).getMessage()),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> new Conv3d(2, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                                1, 1, 1, 1, false, DataType.INT32,
                                ParameterInitialization.zeros(), 0L))
                        .getMessage().contains("floating")),
                () -> assertEquals(before, ids.get()));
    }

    private static Tensor raceForward(
            Conv2d layer, Tensor input, CountDownLatch ready, CountDownLatch start)
            throws InterruptedException {
        ready.countDown();
        start.await();
        return layer.forward(input);
    }

    private static void assertFanOracle(
            Object layer, Tensor input, Shape weightShape, DataType type,
            ParameterInitialization policy, long seed, long fanIn, long fanOut) {
        Tensor actual;
        if (layer instanceof Conv1d conv1d) {
            conv1d.forward(input);
            actual = conv1d.weight().value();
        } else if (layer instanceof Conv2d conv2d) {
            conv2d.forward(input);
            actual = conv2d.weight().value();
        } else {
            Conv3d conv3d = (Conv3d) layer;
            conv3d.forward(input);
            actual = conv3d.weight().value();
        }
        RandomGenerator source = FACTORY.create(seed);
        Tensor expected;
        if (policy.equals(ParameterInitialization.glorotNormal())) {
            expected = ParameterInitializers.normal(weightShape, type, 0.0d,
                    Math.sqrt(2.0d / ((double) fanIn + (double) fanOut)), source);
        } else if (policy.equals(ParameterInitialization.glorotUniform())) {
            double bound = Math.sqrt(6.0d / ((double) fanIn + (double) fanOut));
            expected = ParameterInitializers.uniform(weightShape, type, -bound, bound, source);
        } else if (policy.equals(ParameterInitialization.kaimingReluNormal())) {
            expected = ParameterInitializers.normal(weightShape, type, 0.0d,
                    Math.sqrt(2.0d / (double) fanIn), source);
        } else {
            double bound = Math.sqrt(6.0d / (double) fanIn);
            expected = ParameterInitializers.uniform(weightShape, type, -bound, bound, source);
        }
        assertSameStorage(expected, actual);
    }

    private static Tensor tensor(Shape shape, DataType type) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.empty(), false));
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

    private static void assertSameStorage(Tensor expected, Tensor actual) {
        Object left = expected.hostStorage().orElseThrow().segment().heapBase().orElseThrow();
        Object right = actual.hostStorage().orElseThrow().segment().heapBase().orElseThrow();
        switch (expected.descriptor().dataType()) {
            case FLOAT64 -> assertArrayEquals((double[]) left, (double[]) right);
            case FLOAT32 -> assertArrayEquals((float[]) left, (float[]) right);
            case BFLOAT16 -> assertArrayEquals((short[]) left, (short[]) right);
            default -> fail("unexpected type");
        }
    }

    private static void assertAllZero(Tensor tensor) {
        Object array = tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow();
        switch (tensor.descriptor().dataType()) {
            case FLOAT64 -> assertArrayEquals(new double[java.lang.reflect.Array.getLength(array)], (double[]) array);
            case FLOAT32 -> assertArrayEquals(new float[java.lang.reflect.Array.getLength(array)], (float[]) array);
            case BFLOAT16 -> assertArrayEquals(new short[java.lang.reflect.Array.getLength(array)], (short[]) array);
            default -> fail("unexpected type");
        }
    }
}
