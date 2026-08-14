package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.normalization.AffineLayerNormAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.reflect.Field;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class LayerNormInitializationTest {
    @Test
    void initializesExactOneScaleThenZeroBiasForEveryFloatingType()
            throws ReflectiveOperationException {
        for (DataType dataType : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            Shape normalizedShape = Shape.of(2, 3);
            ScalarValue epsilon = epsilon(dataType);
            AtomicLong next = nextTensorIdState();
            long before = next.get();

            LayerNorm layer = new LayerNorm(normalizedShape, dataType, epsilon);
            Tensor scale = layer.scale().value();
            Tensor bias = layer.bias().value();

            assertAll(
                    () -> assertEquals(before, scale.id().value()),
                    () -> assertEquals(before + 1, bias.id().value()),
                    () -> assertEquals(before + 2, next.get()),
                    () -> assertSame(dataType, scale.descriptor().dataType()),
                    () -> assertSame(dataType, bias.descriptor().dataType()),
                    () -> assertSame(normalizedShape, scale.descriptor().shape()),
                    () -> assertSame(normalizedShape, bias.descriptor().shape()),
                    () -> assertTrue(scale.descriptor().requiresGrad()),
                    () -> assertTrue(bias.descriptor().requiresGrad()),
                    () -> assertTrue(scale.provenance().isEmpty()),
                    () -> assertTrue(bias.provenance().isEmpty()),
                    () -> assertEquals(
                            List.of("scale", "bias"),
                            layer.parameters().stream().map(parameter -> parameter.name()).toList()),
                    () -> assertAllOne(scale),
                    () -> assertAllZero(bias));

            Tensor input = TensorFactory.zeros(
                    Shape.of(4, 2, 3), dataType, java.util.Optional.empty(), false);
            AffineLayerNormAttrs attrs = (AffineLayerNormAttrs) layer.forward(input)
                    .provenance().orElseThrow().operation().attrs();
            assertAll(
                    () -> assertSame(normalizedShape, attrs.normalizedShape()),
                    () -> assertSame(epsilon, attrs.epsilon()));
        }
    }

    @Test
    void callerControlledValidationFailsBeforeTensorIdentifierAllocation()
            throws ReflectiveOperationException {
        Shape dynamic = Shape.ofDimensions(new DynamicDimension("N"));
        Shape zeroFirst = Shape.of(0, 0);
        Shape zeroSecond = Shape.of(2, 0);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertAll(
                () -> assertEquals(
                        "normalizedShape",
                        assertThrows(
                                        NullPointerException.class,
                                        () -> new LayerNorm((Shape) null, null, null))
                                .getMessage()),
                () -> assertEquals(
                        "dataType",
                        assertThrows(
                                        NullPointerException.class,
                                        () -> new LayerNorm(Shape.scalar(), null, null))
                                .getMessage()),
                () -> assertEquals(
                        "epsilon",
                        assertThrows(
                                        NullPointerException.class,
                                        () -> new LayerNorm(
                                                Shape.scalar(), DataType.INT32, null))
                                .getMessage()),
                () -> assertTrue(failure(Shape.scalar(), DataType.FLOAT32, epsilon(DataType.FLOAT32))
                        .contains("positive rank")),
                () -> assertTrue(failure(dynamic, DataType.FLOAT32, epsilon(DataType.FLOAT32))
                        .contains("fully static")),
                () -> assertTrue(failure(zeroFirst, DataType.FLOAT32, epsilon(DataType.FLOAT32))
                        .contains("axis 0")),
                () -> assertTrue(failure(zeroSecond, DataType.FLOAT32, epsilon(DataType.FLOAT32))
                        .contains("axis 1")),
                () -> assertTrue(failure(Shape.of(2), DataType.INT64, ScalarValue.int64(1))
                        .contains("floating data type")),
                () -> assertTrue(failure(
                                Shape.of(2), DataType.FLOAT32, ScalarValue.float32(-0.0f))
                        .contains("finite and positive")),
                () -> assertTrue(failure(
                                Shape.of(2), DataType.FLOAT32, ScalarValue.float64(1.0e-5))
                        .contains("epsilon data type")),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void initializerArrayLimitFailureConsumesNoTensorIdentifier()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertThrows(
                IllegalArgumentException.class,
                () -> new LayerNorm(
                        Shape.of(46_341, 46_341),
                        DataType.FLOAT64,
                        ScalarValue.float64(1.0e-5)));

        assertEquals(before, next.get());
    }

    private static String failure(
            Shape normalizedShape, DataType dataType, ScalarValue epsilon) {
        return assertThrows(
                        IllegalArgumentException.class,
                        () -> new LayerNorm(normalizedShape, dataType, epsilon))
                .getMessage();
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
                java.util.Arrays.fill(expected, 1.0d);
                assertArrayEquals(expected, (double[]) array);
            }
            case FLOAT32 -> {
                float[] expected = new float[java.lang.reflect.Array.getLength(array)];
                java.util.Arrays.fill(expected, 1.0f);
                assertArrayEquals(expected, (float[]) array);
            }
            case BFLOAT16 -> {
                short[] expected = new short[java.lang.reflect.Array.getLength(array)];
                java.util.Arrays.fill(expected, (short) 0x3F80);
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
}
