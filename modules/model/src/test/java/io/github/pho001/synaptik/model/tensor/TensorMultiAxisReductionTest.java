package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorMultiAxisReductionTest {
    private static final AtomicLong NEXT_INPUT_ID = new AtomicLong(190_000);

    @Test
    void exposesExactPackagePrivateEntrySurface() throws ReflectiveOperationException {
        Class<?> type = TensorMultiAxisReductionExpressions.class;
        var entries = Arrays.stream(type.getDeclaredMethods())
                .filter(method -> !Modifier.isPrivate(method.getModifiers()))
                .toList();
        assertAll(
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertFalse(Modifier.isPublic(type.getModifiers())),
                () -> assertEquals(Set.of(), Set.of(type.getInterfaces())),
                () -> assertEquals(0, type.getDeclaredFields().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertEquals(1, type.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(type.getDeclaredConstructors()[0].getModifiers())),
                () -> assertEquals(3, entries.size()));
        assertEntry("applyOrdinary", Tensor.class, AggregateReductionKind.class,
                int[].class, boolean.class);
        assertEntry("applyAdvanced", Tensor.class, AggregateReductionKind.class,
                int[].class, boolean.class);
        assertEntry("applyStatistical", Tensor.class, AggregateReductionKind.class,
                int[].class, boolean.class, long.class);
    }

    @Test
    void constructsEveryOrdinaryKindWithNormalizedOrderedAxesAndExactMetadata() {
        Dimension first = new DynamicDimension("batch");
        Dimension middle = new StaticDimension(3);
        Dimension last = DimensionExpressions.unknown(0, Optional.empty());
        Shape shape = Shape.ofDimensions(first, middle, last);

        for (AggregateReductionKind kind : List.of(
                AggregateReductionKind.SUM, AggregateReductionKind.MEAN,
                AggregateReductionKind.PROD, AggregateReductionKind.MIN,
                AggregateReductionKind.MAX)) {
            Tensor input = tensor(DataType.FLOAT32, shape, true);
            Tensor removed = TensorMultiAxisReductionExpressions.applyOrdinary(
                    input, kind, new int[] {-1, 0}, false);
            Tensor retained = TensorMultiAxisReductionExpressions.applyOrdinary(
                    input, kind, new int[] {2, -3}, true);
            assertExpression(removed, input, kind,
                    new MultiAxisReductionAttrs(List.of(2, 0), false));
            assertExpression(retained, input, kind,
                    new MultiAxisReductionAttrs(List.of(2, 0), true));
            assertAll(
                    () -> assertEquals(Shape.of(3), removed.descriptor().shape()),
                    () -> assertSame(middle, removed.descriptor().shape().dimensions().getFirst()),
                    () -> assertEquals(Shape.of(1, 3, 1), retained.descriptor().shape()),
                    () -> assertSame(middle, retained.descriptor().shape().dimensions().get(1)),
                    () -> assertSame(DataType.FLOAT32, removed.descriptor().dataType()),
                    () -> assertTrue(removed.descriptor().requiresGrad()),
                    () -> assertTrue(removed.descriptor().layout().isEmpty()),
                    () -> assertTrue(removed.label().isEmpty()),
                    () -> assertTrue(removed.hostStorage().isEmpty()));
        }

        Tensor bool = tensor(DataType.BOOL, shape, false);
        assertExpression(bool.all(2, 0), bool, AggregateReductionKind.ALL,
                new MultiAxisReductionAttrs(List.of(2, 0), false));
        assertExpression(bool.any(new int[] {0, 2}, true), bool, AggregateReductionKind.ANY,
                new MultiAxisReductionAttrs(List.of(0, 2), true));
    }

    @Test
    void constructsAdvancedFamiliesAndCorrectionDefaultsExactly() {
        Tensor input = tensor(DataType.FLOAT64, Shape.of(2, 3, 4), true);
        assertExpression(input.logSumExp(2, 0), input, AggregateReductionKind.LOG_SUM_EXP,
                new MultiAxisReductionAttrs(List.of(2, 0), false));
        assertExpression(input.l1Norm(new int[] {-1, 0}, true), input,
                AggregateReductionKind.L1_NORM,
                new MultiAxisReductionAttrs(List.of(2, 0), true));
        assertExpression(input.l2Norm(1), input, AggregateReductionKind.L2_NORM,
                new MultiAxisReductionAttrs(List.of(1), false));
        assertExpression(input.variance(2, 0), input, AggregateReductionKind.VARIANCE,
                new StatisticalReductionAttrs(List.of(2, 0), false, 0));
        assertExpression(input.variance(new int[] {1}, true, 2), input,
                AggregateReductionKind.VARIANCE,
                new StatisticalReductionAttrs(List.of(1), true, 2));
        assertExpression(input.standardDeviation(new int[] {0, -1}, true), input,
                AggregateReductionKind.STANDARD_DEVIATION,
                new StatisticalReductionAttrs(List.of(0, 2), true, 0));
        assertExpression(input.standardDeviation(new int[] {0}, false, 1), input,
                AggregateReductionKind.STANDARD_DEVIATION,
                new StatisticalReductionAttrs(List.of(0), false, 1));
    }

    @Test
    void treatsEmptyAxesAsFreshPointDomainsAndAllAxesAsScalarReduction() {
        Tensor vector = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        int[] axes = {};
        Tensor point = vector.sum(axes);
        Tensor retainedPoint = vector.sum(new int[0], true);
        Tensor allAxes = vector.sum(1, 0);

        assertAll(
                () -> assertEquals(vector.descriptor().shape(), point.descriptor().shape()),
                () -> assertEquals(vector.descriptor().shape(), retainedPoint.descriptor().shape()),
                () -> assertNotSame(vector, point),
                () -> assertNotSame(point, retainedPoint),
                () -> assertEquals(Shape.scalar(), allAxes.descriptor().shape()),
                () -> assertEquals(List.of(), ((MultiAxisReductionAttrs) point.provenance()
                        .orElseThrow().operation().attrs()).axes()));

        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), true);
        assertEquals(Shape.scalar(), scalar.l2Norm().descriptor().shape());
        assertThrows(IndexOutOfBoundsException.class, () -> scalar.sum(0));
    }

    @Test
    void validatesKindsTypesAxesDuplicatesAndCorrectionInRequiredOrder() {
        Tensor floating = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        Tensor integral = tensor(DataType.INT32, Shape.of(2, 3), false);
        Tensor bool = tensor(DataType.BOOL, Shape.of(2, 3), false);

        assertEquals("input", assertThrows(NullPointerException.class,
                () -> TensorMultiAxisReductionExpressions.applyOrdinary(
                        null, null, null, false)).getMessage());
        assertEquals("kind", assertThrows(NullPointerException.class,
                () -> TensorMultiAxisReductionExpressions.applyAdvanced(
                        floating, null, null, false)).getMessage());
        assertEquals("axes", assertThrows(NullPointerException.class,
                () -> TensorMultiAxisReductionExpressions.applyAdvanced(
                        floating, AggregateReductionKind.L1_NORM, null, false)).getMessage());
        assertEquals("kind must be LOG_SUM_EXP, L1_NORM, or L2_NORM, but was SUM",
                assertThrows(IllegalArgumentException.class,
                        () -> TensorMultiAxisReductionExpressions.applyAdvanced(
                                bool, AggregateReductionKind.SUM, new int[] {9}, false)).getMessage());
        assertEquals("input must have a floating data type, but was INT32",
                assertThrows(IllegalArgumentException.class,
                        () -> integral.variance(new int[] {9}, false, -1)).getMessage());
        assertEquals("correction must be non-negative: -1",
                assertThrows(IllegalArgumentException.class,
                        () -> floating.variance(new int[] {9}, false, -1)).getMessage());
        assertEquals("axes contains duplicate axis 1 at index 1",
                assertThrows(IllegalArgumentException.class,
                        () -> floating.sum(1, -1, 9)).getMessage());
        assertEquals("Axis 2 is outside shape rank 2",
                assertThrows(IndexOutOfBoundsException.class,
                        () -> floating.sum(2)).getMessage());
        assertEquals("input must have BOOL data type for ALL, but was FLOAT32",
                assertThrows(IllegalArgumentException.class,
                        () -> floating.all(0)).getMessage());
        assertEquals("input must have a numeric data type for SUM, but was BOOL",
                assertThrows(IllegalArgumentException.class,
                        () -> bool.sum(0)).getMessage());
    }

    @Test
    void validatesStaticStatisticalDomainsAndAcceptsSymbolicOrOverflowingCounts() {
        Tensor staticInput = tensor(DataType.FLOAT32, Shape.of(2, 3, 0), true);
        assertEquals("reduction domain count 6 must be greater than correction 6",
                assertThrows(IllegalArgumentException.class,
                        () -> staticInput.variance(new int[] {0, 1}, false, 6)).getMessage());
        assertEquals("reduction domain count 0 must be greater than correction 0",
                assertThrows(IllegalArgumentException.class,
                        () -> staticInput.standardDeviation(new int[] {0, 2}, false, 0)).getMessage());
        assertEquals("reduction domain count 1 must be greater than correction 1",
                assertThrows(IllegalArgumentException.class,
                        () -> staticInput.variance(new int[0], false, 1)).getMessage());

        Tensor symbolic = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("n"), new StaticDimension(2)), true);
        assertExpression(symbolic.variance(new int[] {0}, false, Long.MAX_VALUE), symbolic,
                AggregateReductionKind.VARIANCE,
                new StatisticalReductionAttrs(List.of(0), false, Long.MAX_VALUE));
        Tensor overflowing = tensor(DataType.FLOAT32, Shape.of(Long.MAX_VALUE, 2), true);
        assertExpression(overflowing.standardDeviation(
                        new int[] {0, 1}, false, Long.MAX_VALUE),
                overflowing, AggregateReductionKind.STANDARD_DEVIATION,
                new StatisticalReductionAttrs(List.of(0, 1), false, Long.MAX_VALUE));
    }

    @Test
    void preservesExactTypeEligibilityAndFreshProducerIdentity() {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            Tensor input = tensor(type, Shape.of(2), true);
            Tensor first = input.logSumExp(0);
            Tensor second = input.logSumExp(0);
            assertAll(
                    () -> assertSame(type, first.descriptor().dataType()),
                    () -> assertTrue(first.descriptor().requiresGrad()),
                    () -> assertNotSame(first, second),
                    () -> assertNotSame(first.provenance().orElseThrow().producer(),
                            second.provenance().orElseThrow().producer()),
                    () -> assertEquals(0, first.provenance().orElseThrow().outputIndex()));
        }
        for (DataType type : List.of(DataType.INT32, DataType.INT64)) {
            Tensor input = tensor(type, Shape.of(2), false);
            assertSame(type, input.sum(0).descriptor().dataType());
            assertThrows(IllegalArgumentException.class, () -> input.l1Norm(0));
        }
    }

    private static void assertEntry(String name, Class<?>... parameterTypes)
            throws ReflectiveOperationException {
        Method method = TensorMultiAxisReductionExpressions.class
                .getDeclaredMethod(name, parameterTypes);
        assertAll(
                () -> assertSame(Tensor.class, method.getReturnType()),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(method.getModifiers())),
                () -> assertFalse(Modifier.isPublic(method.getModifiers())),
                () -> assertFalse(Modifier.isProtected(method.getModifiers())));
    }

    private static void assertExpression(
            Tensor result, Tensor input, AggregateReductionKind kind, Object attrs) {
        TensorProvenance provenance = result.provenance().orElseThrow();
        assertAll(
                () -> assertSame(kind, provenance.operation().kind()),
                () -> assertEquals(attrs, provenance.operation().attrs()),
                () -> assertEquals(List.of(input), provenance.inputs()),
                () -> assertSame(result.descriptor(), provenance.outputDescriptor()),
                () -> assertEquals(0, provenance.outputIndex()));
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return new Tensor(
                new TensorId(NEXT_INPUT_ID.getAndIncrement()),
                new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad),
                Optional.empty(), Optional.empty(), Optional.empty());
    }
}
