package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Field;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorConv2dExpressionTest {
    private static final AtomicLong INPUT_IDS = new AtomicLong();

    

    @Test
    void derivesStaticFloorModeShapesAndPreservesBatchAndOutputChannelReferences() {
        Dimension batch = new StaticDimension(2);
        Dimension outputChannels = new StaticDimension(12);
        Tensor input = tensor(DataType.FLOAT32,
                Shape.ofDimensions(batch, new StaticDimension(8),
                        new StaticDimension(11), new StaticDimension(13)), true);
        Tensor weight = tensor(DataType.BFLOAT16,
                Shape.ofDimensions(outputChannels, new StaticDimension(4),
                        new StaticDimension(3), new StaticDimension(5)), false);
        Conv2dAttrs attrs = new Conv2dAttrs(2, 3, 1, 2, 2, 1, 2);

        Tensor result = input.conv2d(weight, attrs);
        List<Dimension> dimensions = result.descriptor().shape().dimensions();

        assertAll(
                () -> assertSame(batch, dimensions.get(0)),
                () -> assertSame(outputChannels, dimensions.get(1)),
                () -> assertEquals(new StaticDimension(5), dimensions.get(2)),
                () -> assertEquals(new StaticDimension(5), dimensions.get(3)),
                () -> assertSame(DataType.FLOAT32, result.descriptor().dataType()),
                () -> assertTrue(result.descriptor().requiresGrad()));
    }

    @Test
    void retainsCanonicalDynamicSpatialFormulasAndNeutralReferences() {
        Dimension height = new DynamicDimension("H");
        Dimension width = new DynamicDimension("W");
        Tensor input = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("N"), new StaticDimension(4),
                        height, width), false);
        Tensor unitWeight = tensor(DataType.FLOAT32, Shape.of(6, 4, 1, 1), false);

        Tensor neutral = input.conv2d(unitWeight, Conv2dAttrs.defaults());
        assertSame(height, neutral.descriptor().shape().dimension(2));
        assertSame(width, neutral.descriptor().shape().dimension(3));

        Conv2dAttrs attrs = new Conv2dAttrs(2, 3, 1, 0, 2, 1, 1);
        Tensor weight = tensor(DataType.FLOAT32, Shape.of(6, 4, 3, 5), false);
        Tensor result = input.conv2d(weight, attrs);
        Dimension expectedHeight = DimensionExpressions.addConstant(
                DimensionExpressions.floorDivide(
                        DimensionExpressions.addConstant(height, -3), 2), 1);
        Dimension expectedWidth = DimensionExpressions.addConstant(
                DimensionExpressions.floorDivide(
                        DimensionExpressions.addConstant(width, -5), 3), 1);
        assertAll(
                () -> assertEquals(expectedHeight, result.descriptor().shape().dimension(2)),
                () -> assertEquals(expectedWidth, result.descriptor().shape().dimension(3)));
    }

    @Test
    void acceptsDeferredChannelAndBiasRelations() {
        Dimension inputChannels = new DynamicDimension("Cin");
        Dimension outputChannels = new DynamicDimension("Cout");
        Dimension weightChannels = new DynamicDimension("Cpg");
        Dimension biasLength = new DynamicDimension("B");
        Tensor input = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(1), inputChannels,
                        new StaticDimension(5), new StaticDimension(5)), false);
        Tensor weight = tensor(DataType.FLOAT32,
                Shape.ofDimensions(outputChannels, weightChannels,
                        new StaticDimension(3), new StaticDimension(3)), false);
        Tensor bias = tensor(DataType.FLOAT32, Shape.ofDimensions(biasLength), false);

        Tensor result = input.conv2d(weight, bias,
                new Conv2dAttrs(1, 1, 0, 0, 1, 1, 4));
        assertSame(outputChannels, result.descriptor().shape().dimension(1));
    }

    @Test
    void promotesEveryOrderedFloatingCombinationIncludingBiasAndRejectsEveryOtherRole() {
        DataType[] floating = {DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64};
        for (DataType inputType : floating) {
            for (DataType weightType : floating) {
                for (DataType biasType : floating) {
                    Tensor result = valid(inputType).conv2d(
                            weight(weightType), bias(biasType), Conv2dAttrs.defaults());
                    assertSame(widest(inputType, weightType, biasType),
                            result.descriptor().dataType());
                }
            }
        }

        assertThrows(IllegalArgumentException.class,
                        () -> valid(DataType.INT32).conv2d(
                                weight(DataType.FLOAT32), Conv2dAttrs.defaults()));
        assertThrows(IllegalArgumentException.class,
                        () -> valid(DataType.FLOAT32).conv2d(
                                weight(DataType.BOOL), Conv2dAttrs.defaults()));
        assertThrows(IllegalArgumentException.class,
                        () -> valid(DataType.FLOAT32).conv2d(
                                weight(DataType.FLOAT32), bias(DataType.INT64),
                                Conv2dAttrs.defaults()));
    }

    @Test
    void rejectsAllTaskOwnedShapeRelationsWithExactMessages() {
        assertFailure(tensor(DataType.FLOAT32, Shape.of(2, 3, 4), false), weight(DataType.FLOAT32), null, Conv2dAttrs.defaults());
        assertFailure(valid(DataType.FLOAT32), tensor(DataType.FLOAT32, Shape.of(2, 3, 4), false), null, Conv2dAttrs.defaults());
        assertFailure(valid(DataType.FLOAT32), weight(DataType.FLOAT32), tensor(DataType.FLOAT32, Shape.of(4, 1), false), Conv2dAttrs.defaults());

        Dimension dynamicKernel = new DynamicDimension("Kh");
        assertFailure(valid(DataType.FLOAT32), tensor(DataType.FLOAT32,
                        Shape.ofDimensions(new StaticDimension(4), new StaticDimension(4),
                                dynamicKernel, new StaticDimension(3)), false), null, Conv2dAttrs.defaults());
        assertFailure(valid(DataType.FLOAT32), tensor(DataType.FLOAT32, Shape.of(4, 4, 0, 3), false), null, Conv2dAttrs.defaults());
        assertFailure(valid(DataType.FLOAT32), tensor(DataType.FLOAT32, Shape.of(4, 4, 3, 0), false), null, Conv2dAttrs.defaults());
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 5, 5, 5), false), tensor(DataType.FLOAT32, Shape.of(4, 2, 3, 3), false), null, new Conv2dAttrs(1, 1, 0, 0, 1, 1, 2));
        assertFailure(valid(DataType.FLOAT32), tensor(DataType.FLOAT32, Shape.of(5, 2, 3, 3), false), null, new Conv2dAttrs(1, 1, 0, 0, 1, 1, 2));
        assertFailure(valid(DataType.FLOAT32), tensor(DataType.FLOAT32, Shape.of(4, 3, 3, 3), false), null, new Conv2dAttrs(1, 1, 0, 0, 1, 1, 2));
        assertFailure(valid(DataType.FLOAT32), weight(DataType.FLOAT32), tensor(DataType.FLOAT32, Shape.of(3), false), Conv2dAttrs.defaults());
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 4, 2, 5), false), weight(DataType.FLOAT32), null, Conv2dAttrs.defaults());
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 4, 5, 2), false), weight(DataType.FLOAT32), null, Conv2dAttrs.defaults());
    }

    @Test
    void validatesNullsInOrderAndConsumesNoIdOnEveryLocalFailure() throws Exception {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Tensor input = valid(DataType.FLOAT32);
        Tensor weight = weight(DataType.FLOAT32);
        Tensor bias = bias(DataType.FLOAT32);

        assertThrows(NullPointerException.class,
                () -> TensorConv2dExpressions.apply(null, null, (Conv2dAttrs) null));
        assertThrows(NullPointerException.class,
                () -> TensorConv2dExpressions.apply(input, null, (Conv2dAttrs) null));
        assertThrows(NullPointerException.class,
                () -> TensorConv2dExpressions.apply(input, weight, (Conv2dAttrs) null));
        assertThrows(NullPointerException.class,
                () -> TensorConv2dExpressions.apply(input, weight, null, null));
        assertThrows(NullPointerException.class,
                () -> TensorConv2dExpressions.apply(input, weight, bias, null));
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 4, 2, 2), false), weight, null, Conv2dAttrs.defaults());
        assertEquals(before, next.get());
    }

    @Test
    void createsOneFreshUnlabeledStorageFreeExactOccurrencePerSuccess() throws Exception {
        Tensor input = valid(DataType.BFLOAT16);
        Tensor weight = weight(DataType.FLOAT32);
        Tensor bias = bias(DataType.FLOAT64, true);
        Conv2dAttrs attrs = new Conv2dAttrs(1, 1, 1, 1, 1, 1, 1);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        Tensor first = input.conv2d(weight, bias, attrs);
        Tensor second = input.conv2d(weight, bias, attrs);
        TensorProvenance provenance = first.provenance().orElseThrow();

        assertAll(
                () -> assertEquals(before + 2, next.get()),
                () -> assertNotSame(first, second),
                () -> assertFalse(first.label().isPresent()),
                () -> assertFalse(first.hostStorage().isPresent()),
                () -> assertTrue(first.descriptor().layout().isEmpty()),
                () -> assertTrue(first.descriptor().requiresGrad()),
                () -> assertSame(Conv2dKind.CONV2D, provenance.operation().kind()),
                () -> assertSame(attrs, provenance.operation().attrs()),
                () -> assertEquals(List.of(input, weight, bias), provenance.inputs()),
                () -> assertSame(input, provenance.inputs().get(0)),
                () -> assertSame(weight, provenance.inputs().get(1)),
                () -> assertSame(bias, provenance.inputs().get(2)),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertSame(first.descriptor(), provenance.outputDescriptor()));
    }

    @Test
    void usesCheckedGeometryArithmetic() {
        Tensor input = tensor(DataType.FLOAT32,
                Shape.of(1, 1, Long.MAX_VALUE, 5), false);
        Tensor weight = tensor(DataType.FLOAT32, Shape.of(1, 1, 1, 1), false);
        assertThrows(ArithmeticException.class,
                () -> input.conv2d(weight,
                        new Conv2dAttrs(1, 1, 1, 0, 1, 1, 1)));
        assertThrows(ArithmeticException.class,
                () -> input.conv2d(weight,
                        new Conv2dAttrs(1, 1, Long.MAX_VALUE, 0, 1, 1, 1)));
    }

    private static void assertFailure(Tensor input, Tensor weight, Tensor bias, Conv2dAttrs attrs) { assertThrows(IllegalArgumentException.class,
            () -> {
                if (bias == null) {
                    input.conv2d(weight, attrs);
                } else {
                    input.conv2d(weight, bias, attrs);
                }
            }); }

    private static Tensor valid(DataType type) {
        return tensor(type, Shape.of(2, 4, 7, 7), false);
    }

    private static Tensor weight(DataType type) {
        return tensor(type, Shape.of(4, 4, 3, 3), false);
    }

    private static Tensor bias(DataType type) {
        return bias(type, false);
    }

    private static Tensor bias(DataType type, boolean requiresGrad) {
        return tensor(type, Shape.of(4), requiresGrad);
    }

    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return new Tensor(
                new TensorId(INPUT_IDS.getAndIncrement()),
                new TensorDescriptor(type, shape, Optional.empty(), requiresGrad),
                Optional.empty(), Optional.empty(), Optional.empty());
    }

    private static DataType widest(DataType first, DataType second, DataType third) {
        if (first == DataType.FLOAT64 || second == DataType.FLOAT64 || third == DataType.FLOAT64) {
            return DataType.FLOAT64;
        }
        if (first == DataType.FLOAT32 || second == DataType.FLOAT32 || third == DataType.FLOAT32) {
            return DataType.FLOAT32;
        }
        return DataType.BFLOAT16;
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }
}
