package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormInferenceAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.ForwardContext;
import io.github.pho001.synaptik.nn.module.ForwardMode;
import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class BatchNormInitializationTest {
    @Test
    void initializesOneZeroZeroOneInExactOrderForEveryFloatingType()
            throws ReflectiveOperationException {
        for (DataType dataType : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            ScalarValue momentum = momentum(dataType);
            ScalarValue epsilon = epsilon(dataType);
            AtomicLong next = nextTensorIdState();
            long before = next.get();

            BatchNorm layer = new BatchNorm(3, 1, dataType, momentum, epsilon);
            Tensor scale = layer.scale().value();
            Tensor bias = layer.bias().value();
            Tensor mean = layer.runningMean().value();
            Tensor variance = layer.runningVariance().value();

            assertAll(
                    () -> assertEquals(before, scale.id().value()),
                    () -> assertEquals(before + 1, bias.id().value()),
                    () -> assertEquals(before + 2, mean.id().value()),
                    () -> assertEquals(before + 3, variance.id().value()),
                    () -> assertEquals(before + 4, next.get()),
                    () -> assertSame(dataType, scale.descriptor().dataType()),
                    () -> assertSame(dataType, bias.descriptor().dataType()),
                    () -> assertSame(dataType, mean.descriptor().dataType()),
                    () -> assertSame(dataType, variance.descriptor().dataType()),
                    () -> assertEquals(Shape.of(3), scale.descriptor().shape()),
                    () -> assertEquals(Shape.of(3), bias.descriptor().shape()),
                    () -> assertEquals(Shape.of(3), mean.descriptor().shape()),
                    () -> assertEquals(Shape.of(3), variance.descriptor().shape()),
                    () -> assertSame(scale.descriptor().shape(), bias.descriptor().shape()),
                    () -> assertSame(scale.descriptor().shape(), mean.descriptor().shape()),
                    () -> assertSame(scale.descriptor().shape(), variance.descriptor().shape()),
                    () -> assertTrue(scale.descriptor().requiresGrad()),
                    () -> assertTrue(bias.descriptor().requiresGrad()),
                    () -> assertFalse(mean.descriptor().requiresGrad()),
                    () -> assertFalse(variance.descriptor().requiresGrad()),
                    () -> assertTrue(scale.provenance().isEmpty()),
                    () -> assertTrue(bias.provenance().isEmpty()),
                    () -> assertTrue(mean.provenance().isEmpty()),
                    () -> assertTrue(variance.provenance().isEmpty()),
                    () -> assertEquals(List.of("scale", "bias"),
                            layer.parameters().stream().map(parameter -> parameter.name()).toList()),
                    () -> assertEquals(List.of("runningMean", "runningVariance"),
                            layer.buffers().stream().map(buffer -> buffer.name()).toList()),
                    () -> assertAllOne(scale),
                    () -> assertAllZero(bias),
                    () -> assertAllZero(mean),
                    () -> assertAllOne(variance));

            Tensor input = TensorFactory.zeros(
                    Shape.of(2, 3), dataType, java.util.Optional.empty(), false);
            Tensor output = layer.forward(
                    input, new ForwardContext(ForwardMode.EVALUATION));
            BatchNormInferenceAttrs attrs = (BatchNormInferenceAttrs) output.provenance()
                    .orElseThrow().operation().attrs();
            assertAll(
                    () -> assertEquals(1, attrs.channelAxis()),
                    () -> assertSame(epsilon, attrs.epsilon()));
        }
    }

    @Test
    void callerControlledValidationPrecedesEveryTensorIdentifierAndInitializerSideEffect()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertAll(
                () -> assertTrue(initializationFailure(
                                3, -1, null, null, null)
                        .contains("channelAxis")),
                () -> assertEquals("dataType", nullFailure(
                        3, 0, null, null, null)),
                () -> assertEquals("momentum", nullFailure(
                        3, 0, DataType.FLOAT32, null, null)),
                () -> assertEquals("epsilon", nullFailure(
                        3, 0, DataType.FLOAT32, ScalarValue.float32(0.5f), null)),
                () -> assertTrue(initializationFailure(
                                0,
                                0,
                                DataType.FLOAT32,
                                ScalarValue.float32(0.5f),
                                ScalarValue.float32(1.0e-5f))
                        .contains("featureCount")),
                () -> assertTrue(initializationFailure(
                                -1,
                                0,
                                DataType.FLOAT32,
                                ScalarValue.float32(0.5f),
                                ScalarValue.float32(1.0e-5f))
                        .contains("featureCount")),
                () -> assertTrue(initializationFailure(
                                3,
                                0,
                                DataType.INT32,
                                ScalarValue.int32(0),
                                ScalarValue.int32(1))
                        .contains("floating data type")),
                () -> assertTrue(initializationFailure(
                                3,
                                0,
                                DataType.FLOAT32,
                                ScalarValue.float32(Float.NaN),
                                ScalarValue.float32(1.0e-5f))
                        .contains("momentum")),
                () -> assertTrue(initializationFailure(
                                3,
                                0,
                                DataType.FLOAT32,
                                ScalarValue.float32(0.5f),
                                ScalarValue.float32(-0.0f))
                        .contains("epsilon")),
                () -> assertTrue(initializationFailure(
                                3,
                                0,
                                DataType.FLOAT32,
                                ScalarValue.float64(0.5),
                                ScalarValue.float64(1.0e-5))
                        .contains("momentum data type")),
                () -> assertTrue(initializationFailure(
                                3,
                                0,
                                DataType.FLOAT32,
                                ScalarValue.float32(0.5f),
                                ScalarValue.float64(1.0e-5))
                        .contains("epsilon data type")),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void firstInitializerArrayLimitFailureConsumesNoTensorIdentifier()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertThrows(IllegalArgumentException.class,
                () -> new BatchNorm(
                        (long) Integer.MAX_VALUE + 1L,
                        0,
                        DataType.FLOAT32,
                        ScalarValue.float32(0.5f),
                        ScalarValue.float32(1.0e-5f)));

        assertEquals(before, next.get());
    }

    @Test
    void lateIdentifierExhaustionKeepsCreationEffectsWithoutReturningPartialLayer()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        AtomicBoolean maximumClaimed = maximumTensorIdClaimedState();
        long savedNext = next.get();
        boolean savedClaimed = maximumClaimed.get();
        try {
            next.set(Long.MAX_VALUE - 2);
            maximumClaimed.set(false);

            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> new BatchNorm(
                            1,
                            0,
                            DataType.FLOAT32,
                            ScalarValue.float32(0.5f),
                            ScalarValue.float32(1.0e-5f)));

            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(maximumClaimed.get()));
        } finally {
            next.set(savedNext);
            maximumClaimed.set(savedClaimed);
        }
    }

    private static String initializationFailure(
            long featureCount,
            int channelAxis,
            DataType dataType,
            ScalarValue momentum,
            ScalarValue epsilon) {
        return assertThrows(IllegalArgumentException.class,
                () -> new BatchNorm(
                        featureCount, channelAxis, dataType, momentum, epsilon))
                .getMessage();
    }

    private static String nullFailure(
            long featureCount,
            int channelAxis,
            DataType dataType,
            ScalarValue momentum,
            ScalarValue epsilon) {
        return assertThrows(NullPointerException.class,
                () -> new BatchNorm(
                        featureCount, channelAxis, dataType, momentum, epsilon))
                .getMessage();
    }

    private static ScalarValue momentum(DataType dataType) {
        return switch (dataType) {
            case BFLOAT16 -> ScalarValue.bfloat16(0.25f);
            case FLOAT32 -> ScalarValue.float32(0.25f);
            case FLOAT64 -> ScalarValue.float64(0.25);
            default -> throw new IllegalArgumentException("floating data type required");
        };
    }

    private static ScalarValue epsilon(DataType dataType) {
        return switch (dataType) {
            case BFLOAT16 -> ScalarValue.bfloat16(1.0e-2f);
            case FLOAT32 -> ScalarValue.float32(1.0e-5f);
            case FLOAT64 -> ScalarValue.float64(1.0e-5);
            default -> throw new IllegalArgumentException("floating data type required");
        };
    }

    private static void assertAllOne(Tensor tensor) {
        Object array = heapArray(tensor);
        switch (tensor.descriptor().dataType()) {
            case FLOAT64 -> {
                double[] expected = new double[java.lang.reflect.Array.getLength(array)];
                Arrays.fill(expected, 1.0d);
                assertArrayEquals(expected, (double[]) array);
            }
            case FLOAT32 -> {
                float[] expected = new float[java.lang.reflect.Array.getLength(array)];
                Arrays.fill(expected, 1.0f);
                assertArrayEquals(expected, (float[]) array);
            }
            case BFLOAT16 -> {
                short[] expected = new short[java.lang.reflect.Array.getLength(array)];
                Arrays.fill(expected, (short) 0x3F80);
                assertArrayEquals(expected, (short[]) array);
            }
            default -> throw new AssertionError("unexpected data type");
        }
    }

    private static void assertAllZero(Tensor tensor) {
        Object array = heapArray(tensor);
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

    private static Object heapArray(Tensor tensor) {
        return tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow();
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumTensorIdClaimedState()
            throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }
}
