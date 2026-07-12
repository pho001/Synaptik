package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
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

class TensorAveragePool2dExpressionTest {
    private static final AtomicLong INPUT_IDS = new AtomicLong();

    @Test
    void exposesOnlyTheRequiredHelperAndCanonicalReceiverSurface()
            throws ReflectiveOperationException {
        int modifiers = TensorAveragePool2dExpressions.class.getModifiers();
        var constructors = TensorAveragePool2dExpressions.class.getDeclaredConstructors();
        Set<String> methodNames = Arrays.stream(
                        TensorAveragePool2dExpressions.class.getDeclaredMethods())
                .map(Method::getName)
                .collect(java.util.stream.Collectors.toSet());
        Method helper = TensorAveragePool2dExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, AveragePool2dAttrs.class);
        List<Method> receivers = Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> method.getName().equals("averagePool2d"))
                .toList();

        assertAll(
                () -> assertTrue(Modifier.isFinal(modifiers)),
                () -> assertFalse(Modifier.isPublic(modifiers)),
                () -> assertFalse(Modifier.isProtected(modifiers)),
                () -> assertEquals(0,
                        TensorAveragePool2dExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0,
                        TensorAveragePool2dExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(Set.of("apply", "outputDimension"), methodNames),
                () -> assertTrue(Modifier.isStatic(helper.getModifiers())),
                () -> assertFalse(Modifier.isPublic(helper.getModifiers())),
                () -> assertEquals(1, receivers.size()),
                () -> assertEquals(Tensor.class, receivers.get(0).getReturnType()),
                () -> assertEquals(List.of(AveragePool2dAttrs.class),
                        List.of(receivers.get(0).getParameterTypes())),
                () -> assertFalse(Modifier.isStatic(receivers.get(0).getModifiers())),
                () -> assertTrue(Arrays.stream(Tensor.class.getDeclaredMethods())
                        .noneMatch(method -> method.getName().equals("avgPool2d"))));
    }

    @Test
    void derivesStaticFloorAndLiteralCeilShapesWhileRetainingNAndCReferences() {
        Dimension batch = new StaticDimension(2);
        Dimension channels = new StaticDimension(3);
        Tensor input = tensor(DataType.FLOAT32,
                Shape.ofDimensions(batch, channels, new StaticDimension(11),
                        new StaticDimension(13)), true);
        AveragePool2dAttrs floorAttrs = new AveragePool2dAttrs(
                3, 5, 2, 3, 1, 2, 2, 1, false);

        Tensor floor = input.averagePool2d(floorAttrs);
        List<Dimension> dimensions = floor.descriptor().shape().dimensions();
        assertAll(
                () -> assertSame(batch, dimensions.get(0)),
                () -> assertSame(channels, dimensions.get(1)),
                () -> assertEquals(new StaticDimension(5), dimensions.get(2)),
                () -> assertEquals(new StaticDimension(5), dimensions.get(3)),
                () -> assertSame(DataType.FLOAT32, floor.descriptor().dataType()),
                () -> assertTrue(floor.descriptor().requiresGrad()));

        Tensor terminal = tensor(DataType.FLOAT64, Shape.of(1, 1, 2, 2), false);
        AveragePool2dAttrs floorGrid = new AveragePool2dAttrs(
                1, 1, 3, 3, 0, 0, 1, 1, false);
        AveragePool2dAttrs ceilGrid = new AveragePool2dAttrs(
                1, 1, 3, 3, 0, 0, 1, 1, true);
        assertEquals(Shape.of(1, 1, 1, 1),
                terminal.averagePool2d(floorGrid).descriptor().shape());
        assertEquals(Shape.of(1, 1, 2, 2),
                terminal.averagePool2d(ceilGrid).descriptor().shape());
    }

    @Test
    void retainsCanonicalFloorAndCeilingSymbolicFormulas() {
        Dimension height = new DynamicDimension("H");
        Dimension width = new DynamicDimension("W");
        Dimension batch = new DynamicDimension("N");
        Dimension channels = new DynamicDimension("C");
        Tensor input = tensor(DataType.BFLOAT16,
                Shape.ofDimensions(batch, channels, height, width), false);
        AveragePool2dAttrs floorAttrs = new AveragePool2dAttrs(
                3, 5, 2, 3, 1, 0, 2, 1, false);
        AveragePool2dAttrs ceilAttrs = new AveragePool2dAttrs(
                3, 5, 2, 3, 1, 0, 2, 1, true);

        Tensor floor = input.averagePool2d(floorAttrs);
        Tensor ceil = input.averagePool2d(ceilAttrs);
        Dimension heightNumerator = DimensionExpressions.addConstant(height, -3);
        Dimension widthNumerator = DimensionExpressions.addConstant(width, -5);
        assertAll(
                () -> assertSame(batch, floor.descriptor().shape().dimension(0)),
                () -> assertSame(channels, floor.descriptor().shape().dimension(1)),
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
    void acceptsOnlyTheThreeFloatingTypesAndRetainsTheirExactType() {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            assertSame(type, valid(type).averagePool2d(unit()).descriptor().dataType());
        }
        assertFailure(valid(DataType.INT32), unit(),
                "averagePool2d input must have a floating data type, but was INT32");
        assertFailure(valid(DataType.INT64), unit(),
                "averagePool2d input must have a floating data type, but was INT64");
        assertFailure(valid(DataType.BOOL), unit(),
                "averagePool2d input must have a floating data type, but was BOOL");
        assertFailure(tensor(DataType.FLOAT32, Shape.of(2, 3, 4), false), unit(),
                "averagePool2d input rank must be 4: 3");
    }

    @Test
    void acceptsEmptyAxesAndAllPaddingSpatialWindowsButRejectsUnfittedGeometry() {
        AveragePool2dAttrs padded = new AveragePool2dAttrs(
                3, 3, 1, 1, 2, 2, 1, 1, false);
        Tensor empty = tensor(DataType.FLOAT32, Shape.of(0, 0, 0, 0), false)
                .averagePool2d(padded);
        assertEquals(Shape.of(0, 0, 2, 2), empty.descriptor().shape());

        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 1, 2, 5), false),
                new AveragePool2dAttrs(3, 1, 1, 1, 0, 0, 1, 1, false),
                "averagePool2d effective kernel does not fit padded height: input="
                        + "StaticDimension[size=2], effectiveKernel=3, padding=0");
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 1, 5, 2), false),
                new AveragePool2dAttrs(1, 3, 1, 1, 0, 0, 1, 1, false),
                "averagePool2d effective kernel does not fit padded width: input="
                        + "StaticDimension[size=2], effectiveKernel=3, padding=0");
    }

    @Test
    void validatesNullsAndLocalFailuresBeforeIdentifierAllocation() throws Exception {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        Tensor input = valid(DataType.FLOAT32);

        assertEquals("input", assertThrows(NullPointerException.class,
                () -> TensorAveragePool2dExpressions.apply(null, null)).getMessage());
        assertEquals("attrs", assertThrows(NullPointerException.class,
                () -> TensorAveragePool2dExpressions.apply(input, null)).getMessage());
        assertFailure(valid(DataType.INT32), unit(),
                "averagePool2d input must have a floating data type, but was INT32");
        assertFailure(tensor(DataType.FLOAT32, Shape.of(1, 1, 2, 2), false),
                new AveragePool2dAttrs(3, 3, 1, 1, 0, 0, 1, 1, false),
                "averagePool2d effective kernel does not fit padded height: input="
                        + "StaticDimension[size=2], effectiveKernel=3, padding=0");
        assertEquals(before, next.get());
    }

    @Test
    void createsOneFreshUnlabeledStorageFreeExactOccurrencePerSuccess() throws Exception {
        Tensor input = valid(DataType.FLOAT64, true);
        AveragePool2dAttrs attrs = new AveragePool2dAttrs(
                2, 3, 2, 1, 1, 0, 1, 2, true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        Tensor first = input.averagePool2d(attrs);
        Tensor second = input.averagePool2d(attrs);
        TensorProvenance provenance = first.provenance().orElseThrow();
        assertAll(
                () -> assertEquals(before + 2, next.get()),
                () -> assertNotSame(first, second),
                () -> assertFalse(first.label().isPresent()),
                () -> assertFalse(first.hostStorage().isPresent()),
                () -> assertTrue(first.descriptor().layout().isEmpty()),
                () -> assertTrue(first.descriptor().requiresGrad()),
                () -> assertSame(Pool2dKind.AVERAGE_POOL2D, provenance.operation().kind()),
                () -> assertSame(attrs, provenance.operation().attrs()),
                () -> assertEquals(List.of(input), provenance.inputs()),
                () -> assertSame(input, provenance.inputs().get(0)),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertSame(first.descriptor(), provenance.outputDescriptor()));
    }

    @Test
    void usesCheckedLongGeometryArithmeticWithoutMaterializingTheDivisor() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(1, 1, 5, 5), false);
        assertThrows(ArithmeticException.class, () -> input.averagePool2d(
                new AveragePool2dAttrs(Long.MAX_VALUE, 1, 1, 1, 0, 0, 2, 1, false)));
        assertThrows(ArithmeticException.class, () -> input.averagePool2d(
                new AveragePool2dAttrs(1, 1, 1, 1, Long.MAX_VALUE, 0, 1, 1, false)));

        AveragePool2dAttrs unmaterializedDivisor = new AveragePool2dAttrs(
                Long.MAX_VALUE, Long.MAX_VALUE, 1, 1,
                Long.MAX_VALUE / 2, Long.MAX_VALUE / 2, 1, 1, false);
        Tensor large = tensor(DataType.FLOAT32,
                Shape.of(1, 1, 1, 1), false);
        assertEquals(Shape.of(1, 1, 1, 1),
                large.averagePool2d(unmaterializedDivisor).descriptor().shape());
    }

    private static AveragePool2dAttrs unit() {
        return new AveragePool2dAttrs(1, 1, 1, 1, 0, 0, 1, 1, false);
    }

    private static void assertFailure(Tensor input, AveragePool2dAttrs attrs, String message) {
        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class, () -> input.averagePool2d(attrs));
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
