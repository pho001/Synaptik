package io.github.pho001.synaptik.model.operation.normalization;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class SoftmaxSemanticsTest {
    private static final double EXAMPLE_TOLERANCE = 1.0e-8;

    @Test
    void declaresExactlyTheTwoOrderedSoftmaxKindsWithTypedIdentity() {
        OperationKind softmax = SoftmaxKind.SOFTMAX;
        OperationKind logSoftmax = SoftmaxKind.LOG_SOFTMAX;

        assertAll(
                () -> assertArrayEquals(
                        new SoftmaxKind[] {SoftmaxKind.SOFTMAX, SoftmaxKind.LOG_SOFTMAX},
                        SoftmaxKind.values()),
                () -> assertEquals("SOFTMAX", softmax.name()),
                () -> assertEquals("LOG_SOFTMAX", logSoftmax.name()),
                () -> assertEquals("SOFTMAX", softmax.toString()),
                () -> assertEquals("LOG_SOFTMAX", logSoftmax.toString()),
                () -> assertSame(SoftmaxKind.SOFTMAX, SoftmaxKind.valueOf("SOFTMAX")),
                () -> assertSame(
                        SoftmaxKind.LOG_SOFTMAX, SoftmaxKind.valueOf("LOG_SOFTMAX")),
                () -> assertInstanceOf(OperationKind.class, softmax),
                () -> assertNotEquals(softmax, logSoftmax));
    }

    @Test
    void exposesOnlyTheExactEnumShape() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(SoftmaxKind.class);
    }

    @Test
    void exposesOnlyTheExactAttributesRecordShape() {
        var components = SoftmaxAttrs.class.getRecordComponents();
        var constructors = SoftmaxAttrs.class.getDeclaredConstructors();
        var fields = SoftmaxAttrs.class.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.normalization",
                        SoftmaxAttrs.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(SoftmaxAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(SoftmaxAttrs.class.getModifiers())),
                () -> assertTrue(SoftmaxAttrs.class.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(SoftmaxAttrs.class.getInterfaces())),
                () -> assertEquals(
                        List.of("axis"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(
                        List.of(int.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        List.of(int.class),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(
                        List.of("axis"),
                        Arrays.stream(fields).map(field -> field.getName()).toList()),
                () -> assertTrue(Arrays.stream(fields).allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers())
                                && !Modifier.isStatic(field.getModifiers()))),
                () -> assertEquals(
                        List.of(
                                "axis():int",
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "toString():java.lang.String"),
                        Arrays.stream(SoftmaxAttrs.class.getDeclaredMethods())
                                .map(SoftmaxSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, SoftmaxAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void acceptsEveryStructuralAxisBoundary() {
        for (int axis : new int[] {0, 1, 37, Integer.MAX_VALUE}) {
            assertEquals(axis, new SoftmaxAttrs(axis).axis());
        }
    }

    @Test
    void rejectsEveryRepresentativeNegativeAxisWithTheExactMessage() {
        for (int axis : new int[] {-1, -2, -37, Integer.MIN_VALUE}) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class, () -> new SoftmaxAttrs(axis));

            assertEquals("axis must be non-negative: " + axis, failure.getMessage());
        }
    }

    @Test
    void usesGeneratedRecordValueSemanticsAndDiagnosticText() {
        var attrs = new SoftmaxAttrs(2);
        var equal = new SoftmaxAttrs(2);
        var different = new SoftmaxAttrs(1);

        assertAll(
                () -> assertEquals(attrs, equal),
                () -> assertEquals(attrs.hashCode(), equal.hashCode()),
                () -> assertNotEquals(attrs, different),
                () -> assertEquals("SoftmaxAttrs[axis=2]", attrs.toString()));
    }

    @Test
    void composesBothKindsWithTheExactAttributesReference() {
        SoftmaxAttrs attrs = new SoftmaxAttrs(1);
        Operation softmax = new Operation(SoftmaxKind.SOFTMAX, attrs);
        Operation logSoftmax = new Operation(SoftmaxKind.LOG_SOFTMAX, attrs);

        assertAll(
                () -> assertSame(SoftmaxKind.SOFTMAX, softmax.kind()),
                () -> assertSame(attrs, softmax.attrs()),
                () -> assertSame(SoftmaxKind.LOG_SOFTMAX, logSoftmax.kind()),
                () -> assertSame(attrs, logSoftmax.attrs()),
                () -> assertNotEquals(softmax, logSoftmax));
    }

    @Test
    void idealExampleSumsToOneAndRelatesLogSoftmaxByExponentiation() {
        double[] input = {1.0, 2.0, 3.0};
        double exponentialSum = Arrays.stream(input).map(Math::exp).sum();
        double[] softmax = Arrays.stream(input)
                .map(value -> Math.exp(value) / exponentialSum)
                .toArray();
        double logNormalizer = Math.log(exponentialSum);
        double[] logSoftmax = Arrays.stream(input)
                .map(value -> value - logNormalizer)
                .toArray();

        assertAll(
                () -> assertArrayEquals(
                        new double[] {0.09003057, 0.24472847, 0.66524096},
                        softmax,
                        EXAMPLE_TOLERANCE),
                () -> assertArrayEquals(
                        new double[] {-2.40760596, -1.40760596, -0.40760596},
                        logSoftmax,
                        EXAMPLE_TOLERANCE),
                () -> assertEquals(1.0, Arrays.stream(softmax).sum(), EXAMPLE_TOLERANCE),
                () -> assertArrayEquals(
                        softmax,
                        Arrays.stream(logSoftmax).map(Math::exp).toArray(),
                        EXAMPLE_TOLERANCE));
    }

    @Test
    void remainsDistinctFromAggregateAndScanSemanticsAndContainsNoCrossLayerState() {
        SoftmaxAttrs normalizationAttrs = new SoftmaxAttrs(0);
        AxisReductionAttrs reductionAttrs = new AxisReductionAttrs(0, false);
        CumulativeScanAttrs scanAttrs = new CumulativeScanAttrs(0, false, false);
        Operation normalization = new Operation(SoftmaxKind.SOFTMAX, normalizationAttrs);
        Operation reduction = new Operation(AggregateReductionKind.SUM, reductionAttrs);
        Operation scan = new Operation(CumulativeScanKind.CUM_SUM, scanAttrs);
        var componentTypes = Arrays.stream(SoftmaxAttrs.class.getRecordComponents())
                .map(component -> component.getType().getName())
                .toList();

        assertAll(
                () -> assertNotEquals(normalization, reduction),
                () -> assertNotEquals(normalization, scan),
                () -> assertNotEquals(normalizationAttrs, reductionAttrs),
                () -> assertNotEquals(normalizationAttrs, scanAttrs),
                () -> assertEquals(List.of("int"), componentTypes),
                () -> assertFalse(componentTypes.stream().anyMatch(name ->
                        name.contains("Tensor")
                                || name.contains("Shape")
                                || name.contains("DataType")
                                || name.contains("graph")
                                || name.contains("runtime")
                                || name.contains("backend"))));
    }

    private static String methodSignature(java.lang.reflect.Method method) {
        String parameters = Arrays.stream(method.getParameterTypes())
                .map(Class::getName)
                .collect(java.util.stream.Collectors.joining(","));
        return method.getName()
                + "("
                + parameters
                + "):"
                + method.getReturnType().getName();
    }
}
