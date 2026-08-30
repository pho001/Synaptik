package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv1dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

final class TensorConv1dExpressionTest {
    private static final AtomicLong INPUT_IDS = new AtomicLong(1_000_000);

    @Test
    void exposesOnlyRequiredHelperAndReceiverSurface() throws Exception {
        Method unbiased = Tensor.class.getDeclaredMethod(
                "conv1d", Tensor.class, Conv1dAttrs.class);
        Method biased = Tensor.class.getDeclaredMethod(
                "conv1d", Tensor.class, Tensor.class, Conv1dAttrs.class);
        Set<String> names = Arrays.stream(TensorConv1dExpressions.class.getDeclaredMethods())
                .map(Method::getName).collect(Collectors.toSet());

        assertAll(
                () -> assertTrue(Modifier.isFinal(TensorConv1dExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorConv1dExpressions.class.getModifiers())),
                () -> assertEquals(0, TensorConv1dExpressions.class.getDeclaredFields().length),
                () -> assertEquals(1, TensorConv1dExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(
                        TensorConv1dExpressions.class.getDeclaredConstructors()[0].getModifiers())),
                () -> assertEquals(Set.of("apply", "validate", "requireFloating", "requireRank",
                        "requirePositiveStaticKernel", "validateDivisible", "outputWidth"), names),
                () -> assertEquals(Tensor.class, unbiased.getReturnType()),
                () -> assertEquals(Tensor.class, biased.getReturnType()),
                () -> assertFalse(Modifier.isStatic(unbiased.getModifiers())),
                () -> assertFalse(Modifier.isStatic(biased.getModifiers())),
                () -> assertEquals(208, Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers())).count()));
    }

    @Test
    void derivesStaticAndSymbolicShapesWithExactDimensionReferences() {
        Dimension batch = new StaticDimension(2);
        Dimension outputChannels = new StaticDimension(12);
        Tensor input = tensor(DataType.FLOAT32,
                Shape.ofDimensions(batch, new StaticDimension(8), new StaticDimension(13)), true);
        Tensor weight = tensor(DataType.BFLOAT16,
                Shape.ofDimensions(outputChannels, new StaticDimension(4),
                        new StaticDimension(5)), false);
        Tensor result = input.conv1d(weight, new Conv1dAttrs(3, 2, 1, 2));

        assertAll(
                () -> assertSame(batch, result.descriptor().shape().dimension(0)),
                () -> assertSame(outputChannels, result.descriptor().shape().dimension(1)),
                () -> assertEquals(new StaticDimension(5), result.descriptor().shape().dimension(2)),
                () -> assertSame(DataType.FLOAT32, result.descriptor().dataType()),
                () -> assertTrue(result.descriptor().requiresGrad()));

        Dimension width = new DynamicDimension("W");
        Tensor dynamic = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("N"), new DynamicDimension("Cin"), width),
                false);
        Tensor dynamicWeight = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("Cout"), new DynamicDimension("Cpg"),
                        new StaticDimension(3)), false);
        Tensor symbolic = dynamic.conv1d(dynamicWeight, new Conv1dAttrs(2, 1, 2, 4));
        Dimension expected = DimensionExpressions.addConstant(
                DimensionExpressions.floorDivide(
                        DimensionExpressions.addConstant(width, -3), 2), 1);
        assertEquals(expected, symbolic.descriptor().shape().dimension(2));
    }

    @Test
    void createsExactFourProducerBiasedCompositionAndFreshCanonicalWrappers() throws Exception {
        Tensor input = tensor(DataType.BFLOAT16, Shape.of(2, 4, 7), false);
        Tensor weight = tensor(DataType.FLOAT32, Shape.of(6, 2, 3), false);
        Tensor bias = tensor(DataType.FLOAT64, Shape.of(6), true);
        Conv1dAttrs attrs = new Conv1dAttrs(2, 1, 2, 2);
        AtomicLong next = nextIds();
        long before = next.get();

        Tensor first = input.conv1d(weight, bias, attrs);
        Tensor second = input.conv1d(weight, bias, attrs);
        TensorProvenance squeeze = first.provenance().orElseThrow();
        Tensor conv = squeeze.inputs().getFirst();
        TensorProvenance convolution = conv.provenance().orElseThrow();
        Tensor expandedInput = convolution.inputs().get(0);
        Tensor expandedWeight = convolution.inputs().get(1);

        assertAll(
                () -> assertEquals(before + 8, next.get()),
                () -> assertNotSame(first, second),
                () -> assertSame(AxisTransformKind.SQUEEZE, squeeze.operation().kind()),
                () -> assertEquals(new AxisTransformAttrs(2), squeeze.operation().attrs()),
                () -> assertSame(Conv2dKind.CONV2D, convolution.operation().kind()),
                () -> assertEquals(new Conv2dAttrs(1, 2, 0, 1, 1, 2, 2),
                        convolution.operation().attrs()),
                () -> assertNotSame(attrs, convolution.operation().attrs()),
                () -> assertEquals(List.of(expandedInput, expandedWeight, bias),
                        convolution.inputs()),
                () -> assertSame(AxisTransformKind.EXPAND_DIMS,
                        expandedInput.provenance().orElseThrow().operation().kind()),
                () -> assertSame(input,
                        expandedInput.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(weight,
                        expandedWeight.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(expandedInput,
                        expandedInput.provenance().orElseThrow().producer().output(0)),
                () -> assertSame(expandedWeight,
                        expandedWeight.provenance().orElseThrow().producer().output(0)),
                () -> assertSame(first, squeeze.producer().output(0)),
                () -> assertSame(conv, convolution.producer().output(0)),
                () -> assertFalse(first.label().isPresent()),
                () -> assertFalse(first.hostStorage().isPresent()),
                () -> assertTrue(first.descriptor().layout().isEmpty()),
                () -> assertSame(DataType.FLOAT64, first.descriptor().dataType()),
                () -> assertTrue(first.descriptor().requiresGrad()));
    }

    @Test
    void promotesAllFloatingOrdersAndRejectsNonFloatingRoles() {
        DataType[] floating = {DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64};
        for (DataType inputType : floating) {
            for (DataType weightType : floating) {
                for (DataType biasType : floating) {
                    Tensor result = valid(inputType).conv1d(
                            weight(weightType), bias(biasType), Conv1dAttrs.defaults());
                    assertSame(widest(inputType, weightType, biasType),
                            result.descriptor().dataType());
                }
            }
        }
        assertFailure(tensor(DataType.INT32, Shape.of(1, 4, 7), false), weight(DataType.FLOAT32),
                null, Conv1dAttrs.defaults(),
                "conv1d input must have a floating data type, but was INT32");
        assertFailure(valid(DataType.FLOAT32), weight(DataType.BOOL), null,
                Conv1dAttrs.defaults(),
                "conv1d weight must have a floating data type, but was BOOL");
        assertFailure(valid(DataType.FLOAT32), weight(DataType.FLOAT32), bias(DataType.INT64),
                Conv1dAttrs.defaults(),
                "conv1d bias must have a floating data type, but was INT64");
    }

    @Test
    void rejectsEveryOwnedShapeRelationWithExactMessagesAndNoIdentityUse() throws Exception {
        AtomicLong next = nextIds();
        long before = next.get();
        assertFailure(tensor(DataType.FLOAT32, Shape.of(4, 7), false), weight(DataType.FLOAT32),
                null, Conv1dAttrs.defaults(), "conv1d input rank must be 3: 2");
        assertFailure(valid(DataType.FLOAT32), tensor(DataType.FLOAT32, Shape.of(4, 3), false),
                null, Conv1dAttrs.defaults(), "conv1d weight rank must be 3: 2");
        assertFailure(valid(DataType.FLOAT32), weight(DataType.FLOAT32),
                tensor(DataType.FLOAT32, Shape.of(4, 1), false), Conv1dAttrs.defaults(),
                "conv1d bias rank must be 1: 2");
        Dimension dynamicKernel = new DynamicDimension("K");
        assertFailure(valid(DataType.FLOAT32), tensor(DataType.FLOAT32,
                        Shape.ofDimensions(new StaticDimension(4), new StaticDimension(4),
                                dynamicKernel), false), null, Conv1dAttrs.defaults(),
                "conv1d kernel width must be static: " + dynamicKernel);
        assertFailure(valid(DataType.FLOAT32), tensor(DataType.FLOAT32, Shape.of(4, 4, 0), false),
                null, Conv1dAttrs.defaults(),
                "conv1d kernel width must be positive: StaticDimension[size=0]");
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 5, 7), false),
                tensor(DataType.FLOAT32, Shape.of(4, 2, 3), false), null,
                new Conv1dAttrs(1, 0, 1, 2),
                "conv1d input channels must be divisible by groups: channels=StaticDimension[size=5], groups=2");
        assertFailure(valid(DataType.FLOAT32), tensor(DataType.FLOAT32, Shape.of(5, 2, 3), false),
                null, new Conv1dAttrs(1, 0, 1, 2),
                "conv1d output channels must be divisible by groups: channels=StaticDimension[size=5], groups=2");
        assertFailure(valid(DataType.FLOAT32), tensor(DataType.FLOAT32, Shape.of(4, 3, 3), false),
                null, new Conv1dAttrs(1, 0, 1, 2),
                "conv1d weight channels per group do not match input channels: weight=StaticDimension[size=3], groups=2, input=StaticDimension[size=4]");
        assertFailure(valid(DataType.FLOAT32), weight(DataType.FLOAT32),
                tensor(DataType.FLOAT32, Shape.of(3), false), Conv1dAttrs.defaults(),
                "conv1d bias length must match output channels: bias=StaticDimension[size=3], output=StaticDimension[size=4]");
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 4, 2), false), weight(DataType.FLOAT32),
                null, Conv1dAttrs.defaults(),
                "conv1d effective kernel does not fit padded width: input=StaticDimension[size=2], effectiveKernel=3, padding=0");
        assertEquals(before, next.get());
    }

    @Test
    void validatesNullsAndUsesCheckedGeometry() {
        Tensor input = valid(DataType.FLOAT32);
        Tensor weight = weight(DataType.FLOAT32);
        Tensor bias = bias(DataType.FLOAT32);
        assertEquals("input", assertThrows(NullPointerException.class,
                () -> TensorConv1dExpressions.apply(null, null, (Conv1dAttrs) null)).getMessage());
        assertEquals("weight", assertThrows(NullPointerException.class,
                () -> TensorConv1dExpressions.apply(input, null, (Conv1dAttrs) null)).getMessage());
        assertEquals("attrs", assertThrows(NullPointerException.class,
                () -> TensorConv1dExpressions.apply(input, weight, (Conv1dAttrs) null)).getMessage());
        assertEquals("bias", assertThrows(NullPointerException.class,
                () -> TensorConv1dExpressions.apply(input, weight, null, null)).getMessage());
        assertEquals("attrs", assertThrows(NullPointerException.class,
                () -> TensorConv1dExpressions.apply(input, weight, bias, null)).getMessage());
        assertThrows(ArithmeticException.class, () -> tensor(DataType.FLOAT32,
                Shape.of(1, 1, Long.MAX_VALUE), false).conv1d(
                        tensor(DataType.FLOAT32, Shape.of(1, 1, 1), false),
                        new Conv1dAttrs(1, 1, 1, 1)));
    }

    @Test
    void preservesDelegatedPartialIdentifierEffects() throws Exception {
        AtomicLong next = nextIds();
        long saved = next.get();
        try {
            next.set(Long.MAX_VALUE - 2);
            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> valid(DataType.FLOAT32).conv1d(
                            weight(DataType.FLOAT32), Conv1dAttrs.defaults()));
            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()));
        } finally {
            next.set(saved);
        }
    }

    private static void assertFailure(
            Tensor input, Tensor weight, Tensor bias, Conv1dAttrs attrs, String message) {
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class, () -> {
            if (bias == null) {
                input.conv1d(weight, attrs);
            } else {
                input.conv1d(weight, bias, attrs);
            }
        });
        assertEquals(message, failure.getMessage());
    }

    private static Tensor valid(DataType type) {
        return tensor(type, Shape.of(2, 4, 7), false);
    }

    private static Tensor weight(DataType type) {
        return tensor(type, Shape.of(4, 4, 3), false);
    }

    private static Tensor bias(DataType type) {
        return tensor(type, Shape.of(4), false);
    }

    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return new Tensor(new TensorId(INPUT_IDS.getAndIncrement()),
                new TensorDescriptor(type, shape, Optional.empty(), requiresGrad),
                Optional.empty(), Optional.empty(), Optional.empty());
    }

    private static DataType widest(DataType first, DataType second, DataType third) {
        if (first == DataType.FLOAT64 || second == DataType.FLOAT64 || third == DataType.FLOAT64) {
            return DataType.FLOAT64;
        }
        return first == DataType.FLOAT32 || second == DataType.FLOAT32 || third == DataType.FLOAT32
                ? DataType.FLOAT32 : DataType.BFLOAT16;
    }

    private static AtomicLong nextIds() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }
}
