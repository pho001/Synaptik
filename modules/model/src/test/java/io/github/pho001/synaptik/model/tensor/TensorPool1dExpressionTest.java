package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool1dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool1dAttrs;
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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

final class TensorPool1dExpressionTest {
    private static final AtomicLong INPUT_IDS = new AtomicLong(2_000_000);

    @Test
    void exposesOnlyTheRequiredFieldFreeHelperAndReceiverSurface() throws Exception {
        Method maximum = Tensor.class.getDeclaredMethod("maxPool1d", MaxPool1dAttrs.class);
        Method average = Tensor.class.getDeclaredMethod("averagePool1d", AveragePool1dAttrs.class);
        Set<String> helperNames = Arrays.stream(TensorPool1dExpressions.class.getDeclaredMethods())
                .map(Method::getName).collect(Collectors.toSet());

        assertAll(
                () -> assertTrue(Modifier.isFinal(TensorPool1dExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorPool1dExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isProtected(
                        TensorPool1dExpressions.class.getModifiers())),
                () -> assertEquals(0, TensorPool1dExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorPool1dExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1,
                        TensorPool1dExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(TensorPool1dExpressions.class
                        .getDeclaredConstructors()[0].getModifiers())),
                () -> assertEquals(Set.of("apply", "validate", "outputWidth"), helperNames),
                () -> assertEquals(Tensor.class, maximum.getReturnType()),
                () -> assertEquals(Tensor.class, average.getReturnType()),
                () -> assertFalse(Modifier.isStatic(maximum.getModifiers())),
                () -> assertFalse(Modifier.isStatic(average.getModifiers())),
                () -> assertEquals(213, Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers())).count()));
    }

    @Test
    void derivesStaticFloorLiteralCeilAndTerminalAllPaddingWidths() {
        Dimension batch = new StaticDimension(2);
        Dimension channels = new StaticDimension(3);
        Tensor input = tensor(DataType.FLOAT32,
                Shape.ofDimensions(batch, channels, new StaticDimension(13)), true);
        MaxPool1dAttrs maximum = new MaxPool1dAttrs(5, 3, 2, 1, false);
        AveragePool1dAttrs average = new AveragePool1dAttrs(5, 3, 2, 1, false);

        Tensor maxResult = input.maxPool1d(maximum);
        Tensor averageResult = input.averagePool1d(average);
        assertAll(
                () -> assertEquals(new StaticDimension(5),
                        maxResult.descriptor().shape().dimension(2)),
                () -> assertEquals(new StaticDimension(5),
                        averageResult.descriptor().shape().dimension(2)),
                () -> assertSame(batch, maxResult.descriptor().shape().dimension(0)),
                () -> assertSame(channels, maxResult.descriptor().shape().dimension(1)),
                () -> assertSame(batch, averageResult.descriptor().shape().dimension(0)),
                () -> assertSame(channels, averageResult.descriptor().shape().dimension(1)),
                () -> assertSame(DataType.FLOAT32, maxResult.descriptor().dataType()),
                () -> assertSame(DataType.FLOAT32, averageResult.descriptor().dataType()),
                () -> assertTrue(maxResult.descriptor().requiresGrad()),
                () -> assertTrue(averageResult.descriptor().requiresGrad()));

        Tensor terminal = tensor(DataType.FLOAT64, Shape.of(1, 1, 3), false);
        assertAll(
                () -> assertEquals(Shape.of(1, 1, 2), terminal.maxPool1d(
                                new MaxPool1dAttrs(2, 3, 2, 1, false))
                        .descriptor().shape()),
                () -> assertEquals(Shape.of(1, 1, 3), terminal.maxPool1d(
                                new MaxPool1dAttrs(2, 3, 2, 1, true))
                        .descriptor().shape()),
                () -> assertEquals(Shape.of(1, 1, 2), terminal.averagePool1d(
                                new AveragePool1dAttrs(2, 3, 2, 1, false))
                        .descriptor().shape()),
                () -> assertEquals(Shape.of(1, 1, 3), terminal.averagePool1d(
                                new AveragePool1dAttrs(2, 3, 2, 1, true))
                        .descriptor().shape()));
    }

    @Test
    void retainsCanonicalFloorAndCeilingSymbolicWidthFormulas() {
        Dimension width = new DynamicDimension("W");
        Tensor input = tensor(DataType.BFLOAT16,
                Shape.ofDimensions(new DynamicDimension("N"), new DynamicDimension("C"), width),
                false);
        Dimension numerator = DimensionExpressions.addConstant(width, -3);
        Dimension expectedFloor = DimensionExpressions.addConstant(
                DimensionExpressions.floorDivide(numerator, 2), 1);
        Dimension expectedCeil = DimensionExpressions.addConstant(
                DimensionExpressions.ceilingDivide(numerator, 2), 1);

        assertAll(
                () -> assertEquals(expectedFloor, input.maxPool1d(
                                new MaxPool1dAttrs(3, 2, 1, 2, false))
                        .descriptor().shape().dimension(2)),
                () -> assertEquals(expectedCeil, input.maxPool1d(
                                new MaxPool1dAttrs(3, 2, 1, 2, true))
                        .descriptor().shape().dimension(2)),
                () -> assertEquals(expectedFloor, input.averagePool1d(
                                new AveragePool1dAttrs(3, 2, 1, 2, false))
                        .descriptor().shape().dimension(2)),
                () -> assertEquals(expectedCeil, input.averagePool1d(
                                new AveragePool1dAttrs(3, 2, 1, 2, true))
                        .descriptor().shape().dimension(2)));
    }

    @Test
    void mapsMaximumToExactVisiblePool2dCompositionAndFreshCanonicalWrappers() throws Exception {
        Tensor input = tensor(DataType.BFLOAT16, Shape.of(2, 4, 9), true);
        MaxPool1dAttrs attrs = new MaxPool1dAttrs(3, 2, 1, 2, true);
        AtomicLong next = nextIds();
        long before = next.get();

        Tensor first = input.maxPool1d(attrs);
        Tensor second = input.maxPool1d(attrs);
        assertMaximumComposition(first, input,
                new MaxPool2dAttrs(1, 3, 1, 2, 0, 1, 1, 2, true));
        Object mappedAttrs = first.provenance().orElseThrow().inputs().getFirst()
                .provenance().orElseThrow().operation().attrs();
        Object secondMappedAttrs = second.provenance().orElseThrow().inputs().getFirst()
                .provenance().orElseThrow().operation().attrs();

        assertAll(
                () -> assertEquals(before + 6, next.get()),
                () -> assertNotSame(first, second),
                () -> assertNotSame(attrs, mappedAttrs),
                () -> assertNotSame(mappedAttrs, secondMappedAttrs),
                () -> assertNotSame(first.provenance().orElseThrow().producer(),
                        second.provenance().orElseThrow().producer()),
                () -> assertSame(DataType.BFLOAT16, first.descriptor().dataType()),
                () -> assertTrue(first.descriptor().requiresGrad()),
                () -> assertFalse(first.label().isPresent()),
                () -> assertFalse(first.hostStorage().isPresent()),
                () -> assertTrue(first.descriptor().layout().isEmpty()));
    }

    @Test
    void mapsAverageToExactVisiblePool2dFixedCountCompositionAndFreshAttrs() {
        Tensor input = tensor(DataType.FLOAT64, Shape.of(2, 4, 9), false);
        AveragePool1dAttrs attrs = new AveragePool1dAttrs(3, 2, 1, 2, true);
        Tensor result = input.averagePool1d(attrs);
        Tensor second = input.averagePool1d(attrs);
        TensorProvenance squeeze = result.provenance().orElseThrow();
        Tensor pooled = squeeze.inputs().getFirst();
        TensorProvenance pooling = pooled.provenance().orElseThrow();
        Tensor expanded = pooling.inputs().getFirst();
        Object secondMappedAttrs = second.provenance().orElseThrow().inputs().getFirst()
                .provenance().orElseThrow().operation().attrs();

        assertAll(
                () -> assertNotSame(result, second),
                () -> assertNotSame(pooling.operation().attrs(), secondMappedAttrs),
                () -> assertSame(AxisTransformKind.SQUEEZE, squeeze.operation().kind()),
                () -> assertEquals(new AxisTransformAttrs(2), squeeze.operation().attrs()),
                () -> assertSame(Pool2dKind.AVERAGE_POOL2D, pooling.operation().kind()),
                () -> assertEquals(new AveragePool2dAttrs(1, 3, 1, 2, 0, 1, 1, 2, true),
                        pooling.operation().attrs()),
                () -> assertNotSame(attrs, pooling.operation().attrs()),
                () -> assertSame(AxisTransformKind.EXPAND_DIMS,
                        expanded.provenance().orElseThrow().operation().kind()),
                () -> assertEquals(new AxisTransformAttrs(2),
                        expanded.provenance().orElseThrow().operation().attrs()),
                () -> assertEquals(Shape.of(2, 4, 1, 9), expanded.descriptor().shape()),
                () -> assertEquals(Shape.of(2, 4, 1, 4), pooled.descriptor().shape()),
                () -> assertEquals(Shape.of(2, 4, 4), result.descriptor().shape()),
                () -> assertSame(input,
                        expanded.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(expanded, pooling.inputs().getFirst()),
                () -> assertSame(pooled, squeeze.inputs().getFirst()),
                () -> assertSame(expanded,
                        expanded.provenance().orElseThrow().producer().output(0)),
                () -> assertSame(pooled, pooling.producer().output(0)),
                () -> assertSame(result, squeeze.producer().output(0)),
                () -> assertEquals(0, pooling.outputIndex()),
                () -> assertEquals(0, squeeze.outputIndex()),
                () -> assertSame(DataType.FLOAT64, result.descriptor().dataType()),
                () -> assertFalse(result.descriptor().requiresGrad()));
    }

    @Test
    void acceptsEveryFloatingTypeAndRejectsTypeRankAndGeometryBeforeAllocation() throws Exception {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            assertSame(type, valid(type).maxPool1d(maxUnit()).descriptor().dataType());
            assertSame(type, valid(type).averagePool1d(averageUnit()).descriptor().dataType());
        }

        AtomicLong next = nextIds();
        long before = next.get();
        assertMaxFailure(valid(DataType.INT32), maxUnit(),
                "maxPool1d input must have a floating data type, but was INT32");
        assertAverageFailure(valid(DataType.BOOL), averageUnit(),
                "averagePool1d input must have a floating data type, but was BOOL");
        assertMaxFailure(tensor(DataType.FLOAT32, Shape.of(2, 3), false), maxUnit(),
                "maxPool1d input rank must be 3: 2");
        assertAverageFailure(tensor(DataType.FLOAT32, Shape.of(2, 3, 4, 5), false), averageUnit(),
                "averagePool1d input rank must be 3: 4");
        assertMaxFailure(tensor(DataType.FLOAT32, Shape.of(1, 1, 2), false),
                new MaxPool1dAttrs(3, 1, 0, 1, false),
                "maxPool1d effective kernel does not fit padded width: input="
                        + "StaticDimension[size=2], effectiveKernel=3, padding=0");
        assertAverageFailure(tensor(DataType.FLOAT32, Shape.of(1, 1, 2), false),
                new AveragePool1dAttrs(3, 1, 0, 1, false),
                "averagePool1d effective kernel does not fit padded width: input="
                        + "StaticDimension[size=2], effectiveKernel=3, padding=0");
        assertThrows(ArithmeticException.class, () -> valid(DataType.FLOAT32).maxPool1d(
                new MaxPool1dAttrs(Long.MAX_VALUE, 1, 0, 2, false)));
        assertThrows(ArithmeticException.class, () -> valid(DataType.FLOAT32).averagePool1d(
                new AveragePool1dAttrs(1, 1, Long.MAX_VALUE, 1, false)));
        assertEquals(before, next.get());
    }

    @Test
    void validatesHelperNullsBeforeAnyTensorIdentityUse() throws Exception {
        Tensor input = valid(DataType.FLOAT32);
        AtomicLong next = nextIds();
        long before = next.get();

        assertAll(
                () -> assertEquals("input", assertThrows(NullPointerException.class,
                        () -> TensorPool1dExpressions.apply(null, (MaxPool1dAttrs) null))
                        .getMessage()),
                () -> assertEquals("attrs", assertThrows(NullPointerException.class,
                        () -> TensorPool1dExpressions.apply(input, (MaxPool1dAttrs) null))
                        .getMessage()),
                () -> assertEquals("input", assertThrows(NullPointerException.class,
                        () -> TensorPool1dExpressions.apply(null, (AveragePool1dAttrs) null))
                        .getMessage()),
                () -> assertEquals("attrs", assertThrows(NullPointerException.class,
                        () -> TensorPool1dExpressions.apply(input, (AveragePool1dAttrs) null))
                        .getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void preservesDelegatedPartialIdentifierEffectsAfterCompositionStarts() throws Exception {
        AtomicLong next = nextIds();
        AtomicBoolean maximumClaimed = maximumIdClaimed();
        long savedNext = next.get();
        boolean savedClaimed = maximumClaimed.get();
        try {
            next.set(Long.MAX_VALUE - 1);
            maximumClaimed.set(false);
            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> valid(DataType.FLOAT32).maxPool1d(maxUnit()));
            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(maximumClaimed.get()));
        } finally {
            next.set(savedNext);
            maximumClaimed.set(savedClaimed);
        }
    }

    private static void assertMaximumComposition(
            Tensor result, Tensor input, MaxPool2dAttrs expectedAttrs) {
        TensorProvenance squeeze = result.provenance().orElseThrow();
        Tensor pooled = squeeze.inputs().getFirst();
        TensorProvenance pooling = pooled.provenance().orElseThrow();
        Tensor expanded = pooling.inputs().getFirst();
        TensorProvenance expansion = expanded.provenance().orElseThrow();

        assertAll(
                () -> assertSame(AxisTransformKind.SQUEEZE, squeeze.operation().kind()),
                () -> assertEquals(new AxisTransformAttrs(2), squeeze.operation().attrs()),
                () -> assertSame(Pool2dKind.MAX_POOL2D, pooling.operation().kind()),
                () -> assertEquals(expectedAttrs, pooling.operation().attrs()),
                () -> assertSame(AxisTransformKind.EXPAND_DIMS, expansion.operation().kind()),
                () -> assertEquals(new AxisTransformAttrs(2), expansion.operation().attrs()),
                () -> assertEquals(List.of(input), expansion.inputs()),
                () -> assertEquals(List.of(expanded), pooling.inputs()),
                () -> assertEquals(List.of(pooled), squeeze.inputs()),
                () -> assertEquals(Shape.of(2, 4, 1, 9), expanded.descriptor().shape()),
                () -> assertEquals(Shape.of(2, 4, 1, 4), pooled.descriptor().shape()),
                () -> assertEquals(Shape.of(2, 4, 4), result.descriptor().shape()),
                () -> assertSame(expanded, expansion.producer().output(0)),
                () -> assertSame(pooled, pooling.producer().output(0)),
                () -> assertSame(result, squeeze.producer().output(0)),
                () -> assertEquals(0, expansion.outputIndex()),
                () -> assertEquals(0, pooling.outputIndex()),
                () -> assertEquals(0, squeeze.outputIndex()));
    }

    private static MaxPool1dAttrs maxUnit() {
        return new MaxPool1dAttrs(1, 1, 0, 1, false);
    }

    private static AveragePool1dAttrs averageUnit() {
        return new AveragePool1dAttrs(1, 1, 0, 1, false);
    }

    private static void assertMaxFailure(Tensor input, MaxPool1dAttrs attrs, String message) {
        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class, () -> input.maxPool1d(attrs));
        assertEquals(message, failure.getMessage());
    }

    private static void assertAverageFailure(
            Tensor input, AveragePool1dAttrs attrs, String message) {
        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class, () -> input.averagePool1d(attrs));
        assertEquals(message, failure.getMessage());
    }

    private static Tensor valid(DataType type) {
        return tensor(type, Shape.of(2, 3, 7), false);
    }

    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return new Tensor(new TensorId(INPUT_IDS.getAndIncrement()),
                new TensorDescriptor(type, shape, Optional.empty(), requiresGrad),
                Optional.empty(), Optional.empty(), Optional.empty());
    }

    private static AtomicLong nextIds() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumIdClaimed() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }
}
