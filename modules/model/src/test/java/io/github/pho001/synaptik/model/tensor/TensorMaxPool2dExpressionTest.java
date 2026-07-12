package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
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

class TensorMaxPool2dExpressionTest {
    private static final AtomicLong INPUT_IDS = new AtomicLong();

    @Test
    void exposesOnlyTheRequiredHelperAndReceiverSurface() throws ReflectiveOperationException {
        int modifiers = TensorMaxPool2dExpressions.class.getModifiers();
        var constructors = TensorMaxPool2dExpressions.class.getDeclaredConstructors();
        Set<String> methodNames = Arrays.stream(TensorMaxPool2dExpressions.class.getDeclaredMethods())
                .map(Method::getName)
                .collect(java.util.stream.Collectors.toSet());
        Method helper = TensorMaxPool2dExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, MaxPool2dAttrs.class);
        Method receiver = Tensor.class.getDeclaredMethod("maxPool2d", MaxPool2dAttrs.class);

        assertAll(
                () -> assertTrue(Modifier.isFinal(modifiers)),
                () -> assertFalse(Modifier.isPublic(modifiers)),
                () -> assertFalse(Modifier.isProtected(modifiers)),
                () -> assertEquals(0, TensorMaxPool2dExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorMaxPool2dExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(Set.of("apply", "outputDimension"), methodNames),
                () -> assertTrue(Modifier.isStatic(helper.getModifiers())),
                () -> assertFalse(Modifier.isPublic(helper.getModifiers())),
                () -> assertEquals(Tensor.class, receiver.getReturnType()),
                () -> assertFalse(Modifier.isStatic(receiver.getModifiers())));
    }

    @Test
    void derivesStaticFloorAndLiteralCeilShapesWhileRetainingNAndCReferences() {
        Dimension batch = new StaticDimension(2);
        Dimension channels = new StaticDimension(3);
        Tensor input = tensor(DataType.FLOAT32,
                Shape.ofDimensions(batch, channels, new StaticDimension(11),
                        new StaticDimension(13)), true);
        MaxPool2dAttrs floorAttrs = new MaxPool2dAttrs(3, 5, 2, 3, 1, 2, 2, 1, false);

        Tensor floor = input.maxPool2d(floorAttrs);
        List<Dimension> dimensions = floor.descriptor().shape().dimensions();
        assertAll(
                () -> assertSame(batch, dimensions.get(0)),
                () -> assertSame(channels, dimensions.get(1)),
                () -> assertEquals(new StaticDimension(5), dimensions.get(2)),
                () -> assertEquals(new StaticDimension(5), dimensions.get(3)),
                () -> assertSame(DataType.FLOAT32, floor.descriptor().dataType()),
                () -> assertTrue(floor.descriptor().requiresGrad()));

        Tensor terminal = tensor(DataType.FLOAT64, Shape.of(1, 1, 2, 2), false);
        MaxPool2dAttrs floorGrid = new MaxPool2dAttrs(1, 1, 3, 3, 0, 0, 1, 1, false);
        MaxPool2dAttrs ceilGrid = new MaxPool2dAttrs(1, 1, 3, 3, 0, 0, 1, 1, true);
        assertEquals(Shape.of(1, 1, 1, 1), terminal.maxPool2d(floorGrid).descriptor().shape());
        assertEquals(Shape.of(1, 1, 2, 2), terminal.maxPool2d(ceilGrid).descriptor().shape());
    }

    @Test
    void retainsCanonicalFloorAndCeilingSymbolicFormulas() {
        Dimension height = new DynamicDimension("H");
        Dimension width = new DynamicDimension("W");
        Tensor input = tensor(DataType.BFLOAT16,
                Shape.ofDimensions(new DynamicDimension("N"), new DynamicDimension("C"),
                        height, width), false);
        MaxPool2dAttrs floorAttrs = new MaxPool2dAttrs(3, 5, 2, 3, 1, 0, 2, 1, false);
        MaxPool2dAttrs ceilAttrs = new MaxPool2dAttrs(3, 5, 2, 3, 1, 0, 2, 1, true);

        Tensor floor = input.maxPool2d(floorAttrs);
        Tensor ceil = input.maxPool2d(ceilAttrs);
        Dimension heightNumerator = DimensionExpressions.addConstant(height, -3);
        Dimension widthNumerator = DimensionExpressions.addConstant(width, -5);
        assertAll(
                () -> assertEquals(DimensionExpressions.addConstant(
                                DimensionExpressions.floorDivide(heightNumerator, 2), 1),
                        floor.descriptor().shape().dimension(2)),
                () -> assertEquals(DimensionExpressions.addConstant(
                                DimensionExpressions.floorDivide(widthNumerator, 3), 1),
                        floor.descriptor().shape().dimension(3)),
                () -> assertEquals(DimensionExpressions.addConstant(
                                DimensionExpressions.ceilingDivide(heightNumerator, 2), 1),
                        ceil.descriptor().shape().dimension(2)),
                () -> assertEquals(DimensionExpressions.addConstant(
                                DimensionExpressions.ceilingDivide(widthNumerator, 3), 1),
                        ceil.descriptor().shape().dimension(3)));
    }

    @Test
    void acceptsAllFloatingTypesAndRejectsIntegralBoolAndWrongRank() {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            assertSame(type, valid(type).maxPool2d(unit()).descriptor().dataType());
        }
        assertFailure(valid(DataType.INT32), unit(),
                "maxPool2d input must have a floating data type, but was INT32");
        assertFailure(valid(DataType.INT64), unit(),
                "maxPool2d input must have a floating data type, but was INT64");
        assertFailure(valid(DataType.BOOL), unit(),
                "maxPool2d input must have a floating data type, but was BOOL");
        assertFailure(tensor(DataType.FLOAT32, Shape.of(2, 3, 4), false), unit(),
                "maxPool2d input rank must be 4: 3");
    }

    @Test
    void acceptsZeroBatchChannelAndPaddedSpatialAxesButRejectsUnfittedGeometry() {
        MaxPool2dAttrs padded = new MaxPool2dAttrs(3, 3, 1, 1, 2, 2, 1, 1, false);
        Tensor empty = tensor(DataType.FLOAT32, Shape.of(0, 0, 0, 0), false).maxPool2d(padded);
        assertEquals(Shape.of(0, 0, 2, 2), empty.descriptor().shape());

        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 1, 2, 5), false),
                new MaxPool2dAttrs(3, 1, 1, 1, 0, 0, 1, 1, false),
                "maxPool2d effective kernel does not fit padded height: input="
                        + "StaticDimension[size=2], effectiveKernel=3, padding=0");
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 1, 5, 2), false),
                new MaxPool2dAttrs(1, 3, 1, 1, 0, 0, 1, 1, false),
                "maxPool2d effective kernel does not fit padded width: input="
                        + "StaticDimension[size=2], effectiveKernel=3, padding=0");
    }

    @Test
    void validatesNullsAndLocalFailuresBeforeIdentifierAllocation() throws Exception {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Tensor input = valid(DataType.FLOAT32);

        assertEquals("input", assertThrows(NullPointerException.class,
                () -> TensorMaxPool2dExpressions.apply(null, null)).getMessage());
        assertEquals("attrs", assertThrows(NullPointerException.class,
                () -> TensorMaxPool2dExpressions.apply(input, null)).getMessage());
        assertFailure(valid(DataType.INT32), unit(),
                "maxPool2d input must have a floating data type, but was INT32");
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 1, 2, 2), false),
                new MaxPool2dAttrs(3, 3, 1, 1, 0, 0, 1, 1, false),
                "maxPool2d effective kernel does not fit padded height: input="
                        + "StaticDimension[size=2], effectiveKernel=3, padding=0");
        assertEquals(before, next.get());
    }

    @Test
    void createsOneFreshUnlabeledStorageFreeExactOccurrencePerSuccess() throws Exception {
        Tensor input = valid(DataType.FLOAT64, true);
        MaxPool2dAttrs attrs = new MaxPool2dAttrs(2, 3, 2, 1, 1, 0, 1, 2, true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        Tensor first = input.maxPool2d(attrs);
        Tensor second = input.maxPool2d(attrs);
        TensorProvenance provenance = first.provenance().orElseThrow();
        assertAll(
                () -> assertEquals(before + 2, next.get()),
                () -> assertNotSame(first, second),
                () -> assertFalse(first.label().isPresent()),
                () -> assertFalse(first.hostStorage().isPresent()),
                () -> assertTrue(first.descriptor().layout().isEmpty()),
                () -> assertTrue(first.descriptor().requiresGrad()),
                () -> assertSame(Pool2dKind.MAX_POOL2D, provenance.operation().kind()),
                () -> assertSame(attrs, provenance.operation().attrs()),
                () -> assertEquals(List.of(input), provenance.inputs()),
                () -> assertSame(input, provenance.inputs().get(0)),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertSame(first.descriptor(), provenance.outputDescriptor()));
    }

    @Test
    void usesCheckedLongGeometryArithmetic() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(1, 1, 5, 5), false);
        assertThrows(ArithmeticException.class, () -> input.maxPool2d(
                new MaxPool2dAttrs(Long.MAX_VALUE, 1, 1, 1, 0, 0, 2, 1, false)));
        assertThrows(ArithmeticException.class, () -> input.maxPool2d(
                new MaxPool2dAttrs(1, 1, 1, 1, Long.MAX_VALUE, 0, 1, 1, false)));
    }

    private static MaxPool2dAttrs unit() {
        return new MaxPool2dAttrs(1, 1, 1, 1, 0, 0, 1, 1, false);
    }

    private static void assertFailure(Tensor input, MaxPool2dAttrs attrs, String message) {
        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class, () -> input.maxPool2d(attrs));
        assertEquals(message, failure.getMessage());
    }

    private static Tensor valid(DataType type) {
        return valid(type, false);
    }

    private static Tensor valid(DataType type, boolean requiresGrad) {
        return tensor(type, Shape.of(2, 3, 7, 7), requiresGrad);
    }

    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return new Tensor(
                new TensorId(INPUT_IDS.getAndIncrement()),
                new TensorDescriptor(type, shape, Optional.empty(), requiresGrad),
                Optional.empty(), Optional.empty(), Optional.empty());
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }
}
