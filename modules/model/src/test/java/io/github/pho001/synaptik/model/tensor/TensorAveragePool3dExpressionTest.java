package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
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

final class TensorAveragePool3dExpressionTest {
    private static final AtomicLong INPUT_IDS = new AtomicLong(4_000_000);

    @Test
    void exposesOnlyTheRequiredFieldFreeHelperAndCanonicalReceiverSurface() throws Exception {
        int modifiers = TensorAveragePool3dExpressions.class.getModifiers();
        var constructors = TensorAveragePool3dExpressions.class.getDeclaredConstructors();
        Set<String> methodNames = Arrays.stream(
                        TensorAveragePool3dExpressions.class.getDeclaredMethods())
                .map(Method::getName).collect(Collectors.toSet());
        Method helper = TensorAveragePool3dExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, AveragePool3dAttrs.class);
        List<Method> receivers = Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> method.getName().equals("averagePool3d")).toList();

        assertAll(
                () -> assertTrue(Modifier.isFinal(modifiers)),
                () -> assertFalse(Modifier.isPublic(modifiers)),
                () -> assertFalse(Modifier.isProtected(modifiers)),
                () -> assertEquals(0,
                        TensorAveragePool3dExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0,
                        TensorAveragePool3dExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(Set.of("apply", "outputDimension"), methodNames),
                () -> assertTrue(Modifier.isStatic(helper.getModifiers())),
                () -> assertFalse(Modifier.isPublic(helper.getModifiers())),
                () -> assertEquals(1, receivers.size()),
                () -> assertEquals(Tensor.class, receivers.getFirst().getReturnType()),
                () -> assertEquals(List.of(AveragePool3dAttrs.class),
                        List.of(receivers.getFirst().getParameterTypes())),
                () -> assertFalse(Modifier.isStatic(receivers.getFirst().getModifiers())),
                () -> assertTrue(Arrays.stream(Tensor.class.getDeclaredMethods())
                        .noneMatch(method -> method.getName().equals("avgPool3d"))));
    }

    @Test
    void derivesStaticFloorLiteralCeilAndTerminalAllPaddingGrids() {
        Dimension batch = new StaticDimension(2);
        Dimension channels = new StaticDimension(3);
        Tensor input = tensor(DataType.FLOAT32,
                Shape.ofDimensions(batch, channels, new StaticDimension(11),
                        new StaticDimension(13), new StaticDimension(17)), true);
        AveragePool3dAttrs attrs = new AveragePool3dAttrs(
                3, 5, 4, 2, 3, 4, 1, 2, 3, 2, 1, 2, false);

        Tensor floor = input.averagePool3d(attrs);
        assertAll(
                () -> assertEquals(Shape.ofDimensions(batch, channels,
                                new StaticDimension(5), new StaticDimension(5),
                                new StaticDimension(5)),
                        floor.descriptor().shape()),
                () -> assertSame(batch, floor.descriptor().shape().dimension(0)),
                () -> assertSame(channels, floor.descriptor().shape().dimension(1)),
                () -> assertSame(DataType.FLOAT32, floor.descriptor().dataType()),
                () -> assertTrue(floor.descriptor().requiresGrad()));

        Tensor terminal = tensor(DataType.FLOAT64, Shape.of(1, 1, 2, 2, 2), false);
        AveragePool3dAttrs floorGrid = new AveragePool3dAttrs(
                1, 1, 1, 3, 3, 3, 0, 0, 0, 1, 1, 1, false);
        AveragePool3dAttrs ceilGrid = new AveragePool3dAttrs(
                1, 1, 1, 3, 3, 3, 0, 0, 0, 1, 1, 1, true);
        assertAll(
                () -> assertEquals(Shape.of(1, 1, 1, 1, 1),
                        terminal.averagePool3d(floorGrid).descriptor().shape()),
                () -> assertEquals(Shape.of(1, 1, 2, 2, 2),
                        terminal.averagePool3d(ceilGrid).descriptor().shape()));
    }

    @Test
    void retainsCanonicalSymbolicGeometryAndExactNAndCReferences() {
        Dimension batch = new DynamicDimension("N");
        Dimension channels = new DynamicDimension("C");
        Dimension depth = new DynamicDimension("D");
        Dimension height = new DynamicDimension("H");
        Dimension width = new DynamicDimension("W");
        Tensor input = tensor(DataType.BFLOAT16,
                Shape.ofDimensions(batch, channels, depth, height, width), false);
        AveragePool3dAttrs floorAttrs = new AveragePool3dAttrs(
                3, 5, 4, 2, 3, 4, 1, 2, 3, 2, 1, 2, false);
        AveragePool3dAttrs ceilAttrs = new AveragePool3dAttrs(
                3, 5, 4, 2, 3, 4, 1, 2, 3, 2, 1, 2, true);
        Tensor floor = input.averagePool3d(floorAttrs);
        Tensor ceil = input.averagePool3d(ceilAttrs);

        assertAll(
                () -> assertSame(batch, floor.descriptor().shape().dimension(0)),
                () -> assertSame(channels, floor.descriptor().shape().dimension(1)),
                () -> assertEquals(output(DimensionExpressions.addConstant(depth, -3), 2, false),
                        floor.descriptor().shape().dimension(2)),
                () -> assertEquals(output(DimensionExpressions.addConstant(height, -1), 3, false),
                        floor.descriptor().shape().dimension(3)),
                () -> assertEquals(output(DimensionExpressions.addConstant(width, -1), 4, false),
                        floor.descriptor().shape().dimension(4)),
                () -> assertEquals(output(DimensionExpressions.addConstant(depth, -3), 2, true),
                        ceil.descriptor().shape().dimension(2)),
                () -> assertEquals(output(DimensionExpressions.addConstant(height, -1), 3, true),
                        ceil.descriptor().shape().dimension(3)),
                () -> assertEquals(output(DimensionExpressions.addConstant(width, -1), 4, true),
                        ceil.descriptor().shape().dimension(4)));
    }

    @Test
    void acceptsExactFloatingTypesAndEmptyAxesButRejectsTypeRankAndGeometryInAxisOrder()
            throws Exception {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            assertSame(type, valid(type).averagePool3d(unit()).descriptor().dataType());
        }
        AveragePool3dAttrs padded = new AveragePool3dAttrs(
                3, 3, 3, 1, 1, 1, 2, 2, 2, 1, 1, 1, false);
        assertEquals(Shape.of(0, 0, 2, 2, 2),
                tensor(DataType.FLOAT32, Shape.of(0, 0, 0, 0, 0), false)
                        .averagePool3d(padded).descriptor().shape());

        AtomicLong next = nextTensorIdState();
        long before = next.get();
        assertFailure(valid(DataType.INT64), unit(),
                "averagePool3d input must have a floating data type, but was INT64");
        assertFailure(valid(DataType.BOOL), unit(),
                "averagePool3d input must have a floating data type, but was BOOL");
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 1, 2, 2), false), unit(),
                "averagePool3d input rank must be 5: 4");
        AveragePool3dAttrs unfitted = new AveragePool3dAttrs(
                3, 3, 3, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 1, 2, 2, 2), false), unfitted,
                "averagePool3d effective kernel does not fit padded depth: input="
                        + "StaticDimension[size=2], effectiveKernel=3, padding=0");
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 1, 3, 2, 2), false), unfitted,
                "averagePool3d effective kernel does not fit padded height: input="
                        + "StaticDimension[size=2], effectiveKernel=3, padding=0");
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 1, 3, 3, 2), false), unfitted,
                "averagePool3d effective kernel does not fit padded width: input="
                        + "StaticDimension[size=2], effectiveKernel=3, padding=0");
        assertEquals(before, next.get());
    }

    @Test
    void validatesNullAndCheckedArithmeticFailuresBeforeAllocationWithoutMaterializingDivisor()
            throws Exception {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Tensor input = valid(DataType.FLOAT32);
        assertAll(
                () -> assertEquals("input", assertThrows(NullPointerException.class,
                        () -> TensorAveragePool3dExpressions.apply(null, null)).getMessage()),
                () -> assertEquals("attrs", assertThrows(NullPointerException.class,
                        () -> TensorAveragePool3dExpressions.apply(input, null)).getMessage()),
                () -> assertThrows(ArithmeticException.class, () -> input.averagePool3d(
                        new AveragePool3dAttrs(Long.MAX_VALUE, 1, 1, 1, 1, 1,
                                0, 0, 0, 2, 1, 1, false))),
                () -> assertThrows(ArithmeticException.class, () -> input.averagePool3d(
                        new AveragePool3dAttrs(1, 1, 1, 1, 1, 1,
                                Long.MAX_VALUE, 0, 0, 1, 1, 1, false))),
                () -> assertEquals(before, next.get()));

        AveragePool3dAttrs hugeDivisor = new AveragePool3dAttrs(
                Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE,
                1, 1, 1,
                Long.MAX_VALUE / 2, Long.MAX_VALUE / 2, Long.MAX_VALUE / 2,
                1, 1, 1, false);
        assertEquals(Shape.of(1, 1, 1, 1, 1),
                tensor(DataType.FLOAT32, Shape.of(1, 1, 1, 1, 1), false)
                        .averagePool3d(hugeDivisor).descriptor().shape());
    }

    @Test
    void createsOneFreshCanonicalExactOccurrencePerSuccess() throws Exception {
        Tensor input = valid(DataType.FLOAT64, true);
        AveragePool3dAttrs attrs = new AveragePool3dAttrs(
                2, 3, 4, 2, 1, 3, 1, 0, 2, 1, 2, 1, true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Tensor first = input.averagePool3d(attrs);
        Tensor second = input.averagePool3d(attrs);
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
                () -> assertSame(Pool3dKind.AVERAGE_POOL3D, provenance.operation().kind()),
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

    private static AveragePool3dAttrs unit() {
        return new AveragePool3dAttrs(1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
    }

    private static void assertFailure(Tensor input, AveragePool3dAttrs attrs, String message) {
        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class, () -> input.averagePool3d(attrs));
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
