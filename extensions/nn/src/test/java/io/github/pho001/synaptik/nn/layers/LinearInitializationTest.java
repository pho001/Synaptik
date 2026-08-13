package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.reflect.Field;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.random.RandomGenerator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class LinearInitializationTest {
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
