package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
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

final class TensorMaxPool3dExpressionTest {
    private static final AtomicLong INPUT_IDS = new AtomicLong(3_000_000);

    @Test
    void exposesOnlyTheRequiredFieldFreeHelperAndReceiverSurface() throws Exception {
        int modifiers = TensorMaxPool3dExpressions.class.getModifiers();
        var constructors = TensorMaxPool3dExpressions.class.getDeclaredConstructors();
        Set<String> methodNames = Arrays.stream(TensorMaxPool3dExpressions.class.getDeclaredMethods())
                .map(Method::getName).collect(Collectors.toSet());
        Method helper = TensorMaxPool3dExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, MaxPool3dAttrs.class);
        List<Method> receivers = Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> method.getName().equals("maxPool3d")).toList();

        assertAll(
                () -> assertTrue(Modifier.isFinal(modifiers)),
                () -> assertFalse(Modifier.isPublic(modifiers)),
                () -> assertFalse(Modifier.isProtected(modifiers)),
                () -> assertEquals(0, TensorMaxPool3dExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorMaxPool3dExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(Set.of("apply", "outputDimension"), methodNames),
                () -> assertTrue(Modifier.isStatic(helper.getModifiers())),
                () -> assertFalse(Modifier.isPublic(helper.getModifiers())),
                () -> assertEquals(1, receivers.size()),
                () -> assertEquals(Tensor.class, receivers.getFirst().getReturnType()),
                () -> assertEquals(List.of(MaxPool3dAttrs.class),
                        List.of(receivers.getFirst().getParameterTypes())),
                () -> assertFalse(Modifier.isStatic(receivers.getFirst().getModifiers())),
                () -> assertEquals(210, Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers())).count()));
    }

    @Test
    void derivesStaticFloorLiteralCeilAndTerminalAllPaddingGrids() {
        Dimension batch = new StaticDimension(2);
        Dimension channels = new StaticDimension(3);
        Tensor input = tensor(DataType.FLOAT32,
                Shape.ofDimensions(batch, channels, new StaticDimension(11),
                        new StaticDimension(13), new StaticDimension(17)), true);
        MaxPool3dAttrs attrs = new MaxPool3dAttrs(
                3, 5, 4, 2, 3, 4, 1, 2, 3, 2, 1, 2, false);

        Tensor floor = input.maxPool3d(attrs);
        List<Dimension> dimensions = floor.descriptor().shape().dimensions();
        assertAll(
                () -> assertSame(batch, dimensions.get(0)),
                () -> assertSame(channels, dimensions.get(1)),
                () -> assertEquals(new StaticDimension(5), dimensions.get(2)),
                () -> assertEquals(new StaticDimension(5), dimensions.get(3)),
                () -> assertEquals(new StaticDimension(5), dimensions.get(4)),
                () -> assertSame(DataType.FLOAT32, floor.descriptor().dataType()),
                () -> assertTrue(floor.descriptor().requiresGrad()));

        Tensor terminal = tensor(DataType.FLOAT64, Shape.of(1, 1, 2, 2, 2), false);
        MaxPool3dAttrs floorGrid = new MaxPool3dAttrs(
                1, 1, 1, 3, 3, 3, 0, 0, 0, 1, 1, 1, false);
        MaxPool3dAttrs ceilGrid = new MaxPool3dAttrs(
                1, 1, 1, 3, 3, 3, 0, 0, 0, 1, 1, 1, true);
        assertAll(
                () -> assertEquals(Shape.of(1, 1, 1, 1, 1),
                        terminal.maxPool3d(floorGrid).descriptor().shape()),
                () -> assertEquals(Shape.of(1, 1, 2, 2, 2),
                        terminal.maxPool3d(ceilGrid).descriptor().shape()));
    }

    @Test
    void retainsCanonicalFloorAndCeilingSymbolicFormulasAndExactNAndC() {
        Dimension batch = new DynamicDimension("N");
        Dimension channels = new DynamicDimension("C");
        Dimension depth = new DynamicDimension("D");
        Dimension height = new DynamicDimension("H");
        Dimension width = new DynamicDimension("W");
        Tensor input = tensor(DataType.BFLOAT16,
                Shape.ofDimensions(batch, channels, depth, height, width), false);
        MaxPool3dAttrs floorAttrs = new MaxPool3dAttrs(
                3, 5, 4, 2, 3, 4, 1, 2, 3, 2, 1, 2, false);
        MaxPool3dAttrs ceilAttrs = new MaxPool3dAttrs(
                3, 5, 4, 2, 3, 4, 1, 2, 3, 2, 1, 2, true);

        Tensor floor = input.maxPool3d(floorAttrs);
        Tensor ceil = input.maxPool3d(ceilAttrs);
        Dimension depthNumerator = DimensionExpressions.addConstant(depth, -3);
        Dimension heightNumerator = DimensionExpressions.addConstant(height, -1);
        Dimension widthNumerator = DimensionExpressions.addConstant(width, -1);
        assertAll(
                () -> assertSame(batch, floor.descriptor().shape().dimension(0)),
                () -> assertSame(channels, floor.descriptor().shape().dimension(1)),
                () -> assertEquals(output(depthNumerator, 2, false),
                        floor.descriptor().shape().dimension(2)),
                () -> assertEquals(output(heightNumerator, 3, false),
                        floor.descriptor().shape().dimension(3)),
                () -> assertEquals(output(widthNumerator, 4, false),
                        floor.descriptor().shape().dimension(4)),
                () -> assertEquals(output(depthNumerator, 2, true),
                        ceil.descriptor().shape().dimension(2)),
                () -> assertEquals(output(heightNumerator, 3, true),
                        ceil.descriptor().shape().dimension(3)),
                () -> assertEquals(output(widthNumerator, 4, true),
                        ceil.descriptor().shape().dimension(4)));
    }

    @Test
    void acceptsExactFloatingTypesAndEmptyAxesButRejectsTypeRankAndGeometryInAxisOrder()
            throws Exception {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            assertSame(type, valid(type).maxPool3d(unit()).descriptor().dataType());
        }
        MaxPool3dAttrs padded = new MaxPool3dAttrs(
                3, 3, 3, 1, 1, 1, 2, 2, 2, 1, 1, 1, false);
        assertEquals(Shape.of(0, 0, 2, 2, 2),
                tensor(DataType.FLOAT32, Shape.of(0, 0, 0, 0, 0), false)
                        .maxPool3d(padded).descriptor().shape());

        AtomicLong next = nextTensorIdState();
        long before = next.get();
        assertFailure(valid(DataType.INT32), unit(),
                "maxPool3d input must have a floating data type, but was INT32");
        assertFailure(valid(DataType.BOOL), unit(),
                "maxPool3d input must have a floating data type, but was BOOL");
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 1, 2, 2), false), unit(),
                "maxPool3d input rank must be 5: 4");
        MaxPool3dAttrs unfitted = new MaxPool3dAttrs(
                3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 1, 2, 2, 2), false), unfitted,
                "maxPool3d effective kernel does not fit padded depth: input="
                        + "StaticDimension[size=2], effectiveKernel=3, padding=0");
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 1, 3, 2, 2), false), unfitted,
                "maxPool3d effective kernel does not fit padded height: input="
                        + "StaticDimension[size=2], effectiveKernel=3, padding=0");
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 1, 3, 3, 2), false), unfitted,
                "maxPool3d effective kernel does not fit padded width: input="
                        + "StaticDimension[size=2], effectiveKernel=3, padding=0");
        assertEquals(before, next.get());
    }

    @Test
    void validatesNullAndCheckedArithmeticFailuresBeforeAllocation() throws Exception {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Tensor input = valid(DataType.FLOAT32);

        assertAll(
                () -> assertEquals("input", assertThrows(NullPointerException.class,
                        () -> TensorMaxPool3dExpressions.apply(null, null)).getMessage()),
                () -> assertEquals("attrs", assertThrows(NullPointerException.class,
                        () -> TensorMaxPool3dExpressions.apply(input, null)).getMessage()),
                () -> assertThrows(ArithmeticException.class, () -> input.maxPool3d(
                        new MaxPool3dAttrs(Long.MAX_VALUE, 1, 1, 1, 1, 1,
                                0, 0, 0, 2, 1, 1, false))),
                () -> assertThrows(ArithmeticException.class, () -> input.maxPool3d(
                        new MaxPool3dAttrs(1, 1, 1, 1, 1, 1,
                                Long.MAX_VALUE, 0, 0, 1, 1, 1, false))),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void createsOneFreshCanonicalExactOccurrencePerSuccess() throws Exception {
        Tensor input = valid(DataType.FLOAT64, true);
        MaxPool3dAttrs attrs = new MaxPool3dAttrs(
                2, 3, 4, 2, 1, 3, 1, 0, 2, 1, 2, 1, true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        Tensor first = input.maxPool3d(attrs);
        Tensor second = input.maxPool3d(attrs);
        TensorProvenance provenance = first.provenance().orElseThrow();
        assertAll(
                () -> assertEquals(before + 2, next.get()),
                () -> assertNotSame(first, second),
                () -> assertNotSame(first.descriptor(), second.descriptor()),
                () -> assertNotSame(provenance.producer(),
                        second.provenance().orElseThrow().producer()),
                () -> assertFalse(first.label().isPresent()),
                () -> assertFalse(first.hostStorage().isPresent()),
                () -> assertTrue(first.descriptor().layout().isEmpty()),
                () -> assertTrue(first.descriptor().requiresGrad()),
                () -> assertSame(Pool3dKind.MAX_POOL3D, provenance.operation().kind()),
                () -> assertSame(attrs, provenance.operation().attrs()),
                () -> assertEquals(List.of(input), provenance.inputs()),
                () -> assertSame(input, provenance.inputs().getFirst()),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertSame(first.descriptor(), provenance.outputDescriptor()),
                () -> assertSame(first, provenance.producer().output(0)));
    }

    private static Dimension output(Dimension numerator, long stride, boolean ceil) {
        Dimension quotient = ceil
                ? DimensionExpressions.ceilingDivide(numerator, stride)
                : DimensionExpressions.floorDivide(numerator, stride);
        return DimensionExpressions.addConstant(quotient, 1);
    }

    private static MaxPool3dAttrs unit() {
        return new MaxPool3dAttrs(1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
    }

    private static void assertFailure(Tensor input, MaxPool3dAttrs attrs, String message) {
        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class, () -> input.maxPool3d(attrs));
        assertEquals(message, failure.getMessage());
    }

    private static Tensor valid(DataType type) {
        return valid(type, false);
    }

    private static Tensor valid(DataType type, boolean requiresGrad) {
        return tensor(type, Shape.of(2, 3, 7, 8, 9), requiresGrad);
    }

    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return new Tensor(new TensorId(INPUT_IDS.getAndIncrement()),
                new TensorDescriptor(type, shape, Optional.empty(), requiresGrad),
                Optional.empty(), Optional.empty(), Optional.empty());
    }

    private static AtomicLong nextTensorIdState() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }
}
