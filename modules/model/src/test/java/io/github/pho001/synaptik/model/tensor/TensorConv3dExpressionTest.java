package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
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
import org.junit.jupiter.api.Test;

class TensorConv3dExpressionTest {
    private static final AtomicLong INPUT_IDS = new AtomicLong();

    @Test
    void exposesOnlyTheRequiredHelperAndReceiverSurface() throws ReflectiveOperationException {
        int modifiers = TensorConv3dExpressions.class.getModifiers();
        var constructors = TensorConv3dExpressions.class.getDeclaredConstructors();
        Set<String> methodNames = Arrays.stream(TensorConv3dExpressions.class.getDeclaredMethods())
                .map(Method::getName)
                .collect(java.util.stream.Collectors.toSet());
        Method unbiased = TensorConv3dExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, Tensor.class, Conv3dAttrs.class);
        Method biased = TensorConv3dExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, Tensor.class, Tensor.class, Conv3dAttrs.class);
        Method publicUnbiased = Tensor.class.getDeclaredMethod(
                "conv3d", Tensor.class, Conv3dAttrs.class);
        Method publicBiased = Tensor.class.getDeclaredMethod(
                "conv3d", Tensor.class, Tensor.class, Conv3dAttrs.class);

        assertAll(
                () -> assertTrue(Modifier.isFinal(modifiers)),
                () -> assertFalse(Modifier.isPublic(modifiers)),
                () -> assertFalse(Modifier.isProtected(modifiers)),
                () -> assertEquals(0, TensorConv3dExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorConv3dExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(Set.of("apply", "build", "requireFloating", "requireRank",
                        "requirePositiveStaticKernel", "validateDivisible", "outputDimension"),
                        methodNames),
                () -> assertTrue(Modifier.isStatic(unbiased.getModifiers())),
                () -> assertTrue(Modifier.isStatic(biased.getModifiers())),
                () -> assertEquals(Tensor.class, publicUnbiased.getReturnType()),
                () -> assertEquals(Tensor.class, publicBiased.getReturnType()),
                () -> assertFalse(Modifier.isStatic(publicUnbiased.getModifiers())),
                () -> assertFalse(Modifier.isStatic(publicBiased.getModifiers())),
                () -> assertEquals(206, Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers())).count()),
                () -> assertEquals(2, Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(method -> method.getName().equals("conv3d")).count()));
    }

    @Test
    void derivesStaticFloorModeShapesAndPreservesBatchAndOutputChannelReferences() {
        Dimension batch = new StaticDimension(2);
        Dimension outputChannels = new StaticDimension(12);
        Tensor input = tensor(DataType.FLOAT32,
                Shape.ofDimensions(batch, new StaticDimension(8), new StaticDimension(11),
                        new StaticDimension(13), new StaticDimension(17)), true);
        Tensor weight = tensor(DataType.BFLOAT16,
                Shape.ofDimensions(outputChannels, new StaticDimension(4),
                        new StaticDimension(3), new StaticDimension(5), new StaticDimension(3)),
                false);
        Conv3dAttrs attrs = new Conv3dAttrs(2, 3, 4, 1, 2, 1, 2, 1, 3, 2);

        Tensor result = input.conv3d(weight, attrs);
        List<Dimension> dimensions = result.descriptor().shape().dimensions();

        assertAll(
                () -> assertSame(batch, dimensions.get(0)),
                () -> assertSame(outputChannels, dimensions.get(1)),
                () -> assertEquals(new StaticDimension(5), dimensions.get(2)),
                () -> assertEquals(new StaticDimension(5), dimensions.get(3)),
                () -> assertEquals(new StaticDimension(4), dimensions.get(4)),
                () -> assertSame(DataType.FLOAT32, result.descriptor().dataType()),
                () -> assertTrue(result.descriptor().requiresGrad()));
    }

    @Test
    void retainsCanonicalSymbolicSpatialFormulasAndNeutralReferences() {
        Dimension depth = new DynamicDimension("D");
        Dimension height = new DynamicDimension("H");
        Dimension width = new DynamicDimension("W");
        Tensor input = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("N"), new StaticDimension(4),
                        depth, height, width), false);
        Tensor unitWeight = tensor(DataType.FLOAT32, Shape.of(6, 4, 1, 1, 1), false);

        Tensor neutral = input.conv3d(unitWeight, Conv3dAttrs.defaults());
        assertSame(depth, neutral.descriptor().shape().dimension(2));
        assertSame(height, neutral.descriptor().shape().dimension(3));
        assertSame(width, neutral.descriptor().shape().dimension(4));

        Conv3dAttrs attrs = new Conv3dAttrs(2, 3, 4, 1, 0, 2, 2, 1, 3, 1);
        Tensor weight = tensor(DataType.FLOAT32, Shape.of(6, 4, 3, 5, 3), false);
        Tensor result = input.conv3d(weight, attrs);
        Dimension expectedDepth = DimensionExpressions.addConstant(
                DimensionExpressions.floorDivide(
                        DimensionExpressions.addConstant(depth, -3), 2), 1);
        Dimension expectedHeight = DimensionExpressions.addConstant(
                DimensionExpressions.floorDivide(
                        DimensionExpressions.addConstant(height, -5), 3), 1);
        Dimension expectedWidth = DimensionExpressions.addConstant(
                DimensionExpressions.floorDivide(
                        DimensionExpressions.addConstant(width, -3), 4), 1);
        assertAll(
                () -> assertEquals(expectedDepth, result.descriptor().shape().dimension(2)),
                () -> assertEquals(expectedHeight, result.descriptor().shape().dimension(3)),
                () -> assertEquals(expectedWidth, result.descriptor().shape().dimension(4)));
    }

    @Test
    void acceptsDeferredChannelBiasAndSpatialRelations() {
        Dimension outputChannels = new DynamicDimension("Cout");
        Tensor input = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(1), new DynamicDimension("Cin"),
                        new DynamicDimension("D"), new DynamicDimension("H"),
                        new DynamicDimension("W")), false);
        Tensor weight = tensor(DataType.FLOAT32,
                Shape.ofDimensions(outputChannels, new DynamicDimension("Cpg"),
                        new StaticDimension(3), new StaticDimension(3), new StaticDimension(3)),
                false);
        Tensor bias = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("B")), false);

        Tensor result = input.conv3d(weight, bias,
                new Conv3dAttrs(1, 1, 1, 0, 0, 0, 1, 1, 1, 4));
        assertSame(outputChannels, result.descriptor().shape().dimension(1));
    }

    @Test
    void acceptsEmptyNeutralAxesAndZeroStaticNumerators() {
        Tensor empty = tensor(DataType.FLOAT32, Shape.of(0, 0, 1, 1, 1), false);
        Tensor weight = tensor(DataType.FLOAT32, Shape.of(0, 0, 1, 1, 1), false);
        assertEquals(Shape.of(0, 0, 1, 1, 1),
                empty.conv3d(weight, Conv3dAttrs.defaults()).descriptor().shape());

        Tensor exact = tensor(DataType.FLOAT32, Shape.of(1, 1, 3, 5, 7), false);
        Tensor exactWeight = tensor(DataType.FLOAT32, Shape.of(1, 1, 3, 5, 7), false);
        assertEquals(Shape.of(1, 1, 1, 1, 1),
                exact.conv3d(exactWeight, Conv3dAttrs.defaults()).descriptor().shape());
    }

    @Test
    void promotesEveryOrderedFloatingCombinationAndRejectsEveryNonFloatingRole() {
        DataType[] floating = {DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64};
        for (DataType inputType : floating) {
            for (DataType weightType : floating) {
                for (DataType biasType : floating) {
                    Tensor result = valid(inputType).conv3d(
                            weight(weightType), bias(biasType), Conv3dAttrs.defaults());
                    assertSame(widest(inputType, weightType, biasType),
                            result.descriptor().dataType());
                }
            }
        }

        assertEquals("conv3d input must have a floating data type, but was INT32",
                assertThrows(IllegalArgumentException.class,
                        () -> valid(DataType.INT32).conv3d(
                                weight(DataType.BOOL), Conv3dAttrs.defaults())).getMessage());
        assertEquals("conv3d weight must have a floating data type, but was BOOL",
                assertThrows(IllegalArgumentException.class,
                        () -> valid(DataType.FLOAT32).conv3d(
                                weight(DataType.BOOL), Conv3dAttrs.defaults())).getMessage());
        assertEquals("conv3d bias must have a floating data type, but was INT64",
                assertThrows(IllegalArgumentException.class,
                        () -> valid(DataType.FLOAT32).conv3d(
                                weight(DataType.FLOAT32), bias(DataType.INT64),
                                Conv3dAttrs.defaults())).getMessage());
    }

    @Test
    void rejectsRanksKernelsAndChannelRelationsWithExactMessages() {
        assertFailure(tensor(DataType.FLOAT32, Shape.of(2, 3, 4, 5), false),
                weight(DataType.FLOAT32), null, Conv3dAttrs.defaults(),
                "conv3d input rank must be 5: 4");
        assertFailure(valid(DataType.FLOAT32),
                tensor(DataType.FLOAT32, Shape.of(2, 3, 4, 5), false), null,
                Conv3dAttrs.defaults(), "conv3d weight rank must be 5: 4");
        assertFailure(valid(DataType.FLOAT32), weight(DataType.FLOAT32),
                tensor(DataType.FLOAT32, Shape.of(4, 1), false), Conv3dAttrs.defaults(),
                "conv3d bias rank must be 1: 2");

        assertKernelFailure(2, new DynamicDimension("Kd"),
                "conv3d kernel depth must be static: DynamicDimension[symbol=Kd]");
        assertKernelFailure(3, new DynamicDimension("Kh"),
                "conv3d kernel height must be static: DynamicDimension[symbol=Kh]");
        assertKernelFailure(4, new DynamicDimension("Kw"),
                "conv3d kernel width must be static: DynamicDimension[symbol=Kw]");
        assertKernelFailure(2, new StaticDimension(0),
                "conv3d kernel depth must be positive: StaticDimension[size=0]");
        assertKernelFailure(3, new StaticDimension(0),
                "conv3d kernel height must be positive: StaticDimension[size=0]");
        assertKernelFailure(4, new StaticDimension(0),
                "conv3d kernel width must be positive: StaticDimension[size=0]");

        Conv3dAttrs groupsTwo = new Conv3dAttrs(1, 1, 1, 0, 0, 0, 1, 1, 1, 2);
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 5, 5, 5, 5), false),
                tensor(DataType.FLOAT32, Shape.of(4, 2, 3, 3, 3), false), null, groupsTwo,
                "conv3d input channels must be divisible by groups: channels=StaticDimension[size=5], groups=2");
        assertFailure(valid(DataType.FLOAT32),
                tensor(DataType.FLOAT32, Shape.of(5, 2, 3, 3, 3), false), null, groupsTwo,
                "conv3d output channels must be divisible by groups: channels=StaticDimension[size=5], groups=2");
        assertFailure(valid(DataType.FLOAT32),
                tensor(DataType.FLOAT32, Shape.of(4, 3, 3, 3, 3), false), null, groupsTwo,
                "conv3d weight channels per group do not match input channels: weight=StaticDimension[size=3], groups=2, input=StaticDimension[size=4]");
        assertFailure(valid(DataType.FLOAT32), weight(DataType.FLOAT32),
                tensor(DataType.FLOAT32, Shape.of(3), false), Conv3dAttrs.defaults(),
                "conv3d bias length must match output channels: bias=StaticDimension[size=3], output=StaticDimension[size=4]");
    }

    @Test
    void rejectsSpatialGeometryInDepthHeightWidthOrderWithExactMessages() {
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 4, 2, 2, 2), false),
                weight(DataType.FLOAT32), null, Conv3dAttrs.defaults(),
                "conv3d effective kernel does not fit padded depth: input=StaticDimension[size=2], effectiveKernel=3, padding=0");
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 4, 3, 2, 2), false),
                weight(DataType.FLOAT32), null, Conv3dAttrs.defaults(),
                "conv3d effective kernel does not fit padded height: input=StaticDimension[size=2], effectiveKernel=3, padding=0");
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 4, 3, 3, 2), false),
                weight(DataType.FLOAT32), null, Conv3dAttrs.defaults(),
                "conv3d effective kernel does not fit padded width: input=StaticDimension[size=2], effectiveKernel=3, padding=0");
    }

    @Test
    void validatesNullsInDeclarationOrderAndConsumesNoIdOnLocalFailure() throws Exception {
        Tensor input = valid(DataType.FLOAT32);
        Tensor weight = weight(DataType.FLOAT32);
        Tensor bias = bias(DataType.FLOAT32);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertEquals("input", assertThrows(NullPointerException.class,
                () -> TensorConv3dExpressions.apply(null, null, (Conv3dAttrs) null)).getMessage());
        assertEquals("weight", assertThrows(NullPointerException.class,
                () -> TensorConv3dExpressions.apply(input, null, (Conv3dAttrs) null)).getMessage());
        assertEquals("attrs", assertThrows(NullPointerException.class,
                () -> TensorConv3dExpressions.apply(input, weight, (Conv3dAttrs) null)).getMessage());
        assertEquals("bias", assertThrows(NullPointerException.class,
                () -> TensorConv3dExpressions.apply(input, weight, null, null)).getMessage());
        assertEquals("attrs", assertThrows(NullPointerException.class,
                () -> TensorConv3dExpressions.apply(input, weight, bias, null)).getMessage());
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 4, 2, 2, 2), false),
                weight, null, Conv3dAttrs.defaults(),
                "conv3d effective kernel does not fit padded depth: input=StaticDimension[size=2], effectiveKernel=3, padding=0");
        assertEquals(before, next.get());
    }

    @Test
    void createsOneFreshCanonicalOccurrenceAndConsumesOneIdPerSuccess() throws Exception {
        Tensor input = valid(DataType.BFLOAT16);
        Tensor weight = weight(DataType.FLOAT32);
        Tensor bias = bias(DataType.FLOAT64, true);
        Conv3dAttrs attrs = new Conv3dAttrs(1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        Tensor first = input.conv3d(weight, bias, attrs);
        Tensor second = input.conv3d(weight, bias, attrs);
        TensorProvenance provenance = first.provenance().orElseThrow();
        TensorProducer producer = provenance.producer();

        assertAll(
                () -> assertEquals(before + 2, next.get()),
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(first.descriptor(), second.descriptor()),
                () -> assertNotSame(producer, second.provenance().orElseThrow().producer()),
                () -> assertNotSame(provenance.operation(),
                        second.provenance().orElseThrow().operation()),
                () -> assertFalse(first.label().isPresent()),
                () -> assertFalse(first.hostStorage().isPresent()),
                () -> assertTrue(first.descriptor().layout().isEmpty()),
                () -> assertTrue(first.descriptor().requiresGrad()),
                () -> assertSame(Conv3dKind.CONV3D, provenance.operation().kind()),
                () -> assertSame(attrs, provenance.operation().attrs()),
                () -> assertEquals(List.of(input, weight, bias), provenance.inputs()),
                () -> assertSame(input, provenance.inputs().get(0)),
                () -> assertSame(weight, provenance.inputs().get(1)),
                () -> assertSame(bias, provenance.inputs().get(2)),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertSame(first.descriptor(), provenance.outputDescriptor()),
                () -> assertSame(first.descriptor(), producer.outputDescriptors().get(0)),
                () -> assertSame(first, producer.output(0)));
    }

    @Test
    void usesCheckedGeometryAndGroupedChannelArithmetic() {
        Tensor hugeDepth = tensor(DataType.FLOAT32,
                Shape.of(1, 1, Long.MAX_VALUE, 5, 5), false);
        Tensor unitWeight = tensor(DataType.FLOAT32, Shape.of(1, 1, 1, 1, 1), false);
        assertThrows(ArithmeticException.class,
                () -> hugeDepth.conv3d(unitWeight,
                        new Conv3dAttrs(1, 1, 1, 1, 0, 0, 1, 1, 1, 1)));
        assertThrows(ArithmeticException.class,
                () -> hugeDepth.conv3d(unitWeight,
                        new Conv3dAttrs(1, 1, 1, Long.MAX_VALUE, 0, 0, 1, 1, 1, 1)));
        assertThrows(ArithmeticException.class,
                () -> tensor(DataType.FLOAT32,
                                Shape.of(1, Long.MAX_VALUE, 3, 3, 3), false).conv3d(
                        tensor(DataType.FLOAT32,
                                Shape.of(Long.MAX_VALUE, 2, 1, 1, 1), false),
                        new Conv3dAttrs(1, 1, 1, 0, 0, 0, 1, 1, 1, Long.MAX_VALUE)));
        assertThrows(ArithmeticException.class,
                () -> valid(DataType.FLOAT32).conv3d(
                        tensor(DataType.FLOAT32, Shape.of(4, 4, Long.MAX_VALUE, 1, 1), false),
                        new Conv3dAttrs(1, 1, 1, 0, 0, 0, 2, 1, 1, 1)));
    }

    private static void assertKernelFailure(int axis, Dimension kernel, String message) {
        Dimension[] dimensions = {
            new StaticDimension(4), new StaticDimension(4),
            new StaticDimension(3), new StaticDimension(3), new StaticDimension(3)
        };
        dimensions[axis] = kernel;
        assertFailure(valid(DataType.FLOAT32),
                tensor(DataType.FLOAT32, Shape.ofDimensions(dimensions), false),
                null, Conv3dAttrs.defaults(), message);
    }

    private static void assertFailure(
            Tensor input, Tensor weight, Tensor bias, Conv3dAttrs attrs, String message) {
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> {
                    if (bias == null) {
                        input.conv3d(weight, attrs);
                    } else {
                        input.conv3d(weight, bias, attrs);
                    }
                });
        assertEquals(message, failure.getMessage());
    }

    private static Tensor valid(DataType type) {
        return tensor(type, Shape.of(2, 4, 7, 7, 7), false);
    }

    private static Tensor weight(DataType type) {
        return tensor(type, Shape.of(4, 4, 3, 3, 3), false);
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
