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

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
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
    void stableFiniteContractExamplesDistinguishOverflowAndRoundedUnderflow() {
        double maximum = Math.max(1000, 1000);
        double shiftedSum = Math.exp(1000 - maximum) + Math.exp(1000 - maximum);
        assertEquals(0.5, Math.exp(1000 - maximum) / shiftedSum);
        assertEquals(-Math.log(2), (1000 - maximum) - Math.log(shiftedSum));
        assertEquals(Double.POSITIVE_INFINITY, Math.exp(1000));

        double small = -1000;
        double probability = Math.exp(small) / (Math.exp(0) + Math.exp(small));
        double logProbability = small - Math.log(Math.exp(0) + Math.exp(small));
        assertEquals(0, probability);
        assertEquals(-1000, logProbability);
        assertEquals(Double.NEGATIVE_INFINITY, Math.log(probability));
    }

    @Test
    void mixedInfinityPreservesLiteralSpecialValueClassInsteadOfFiniteShift() {
        double[] input = {Double.NEGATIVE_INFINITY, 1000};
        double sum = Math.exp(input[0]) + Math.exp(input[1]);
        assertEquals(0, Math.exp(input[0]) / sum);
        assertTrue(Double.isNaN(Math.exp(input[1]) / sum));
        assertEquals(Double.NEGATIVE_INFINITY, input[0] - Math.log(sum));
        assertEquals(Double.NEGATIVE_INFINITY, input[1] - Math.log(sum));
    }

    @Test
    void nanTakesPrecedenceOverPositiveInfinityForTheWholeSlice() {
        double[] input = {Double.NaN, Double.POSITIVE_INFINITY, 0};
        double sum = Arrays.stream(input).map(Math::exp).sum();
        assertTrue(Double.isNaN(sum));
        for (double value : input) {
            assertTrue(Double.isNaN(Math.exp(value) / sum));
            assertTrue(Double.isNaN(value - Math.log(sum)));
        }
    }

    @Test
    void positiveInfinityTiesAreNanOnlyAtWinningPositions() {
        double[] input = {Double.POSITIVE_INFINITY, -2, Double.POSITIVE_INFINITY};
        double sum = Arrays.stream(input).map(Math::exp).sum();
        assertEquals(Double.POSITIVE_INFINITY, sum);
        for (int position : new int[] {0, 2}) {
            assertTrue(Double.isNaN(Math.exp(input[position]) / sum));
            assertTrue(Double.isNaN(input[position] - Math.log(sum)));
        }
        assertEquals(0L, Double.doubleToRawLongBits(Math.exp(input[1]) / sum));
        assertEquals(Double.NEGATIVE_INFINITY, input[1] - Math.log(sum));
    }

    @Test
    void allNegativeInfinityProducesOnlyNanResults() {
        double[] input = {Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY};
        double sum = Arrays.stream(input).map(Math::exp).sum();
        assertEquals(0, sum);
        for (double value : input) {
            assertTrue(Double.isNaN(Math.exp(value) / sum));
            assertTrue(Double.isNaN(value - Math.log(sum)));
        }
    }

    @Test
    void emptyNonSelectedAxisRetainsAnEmptyOutputForBothKinds() {
        Shape emptyRows = Shape.of(0, 3);
        Tensor input = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, emptyRows, Optional.empty(), false));
        Tensor softmax = input.softmax(1);
        Tensor logSoftmax = input.logSoftmax(1);

        for (Tensor result : List.of(softmax, logSoftmax)) {
            assertSame(emptyRows, result.descriptor().shape());
            assertEquals(0, result.descriptor().shape().knownElementCount().orElseThrow());
            assertTrue(result.hostStorage().isEmpty());
            assertSame(input, result.provenance().orElseThrow().inputs().getFirst());
            assertEquals(new SoftmaxAttrs(1), result.provenance().orElseThrow().operation().attrs());
        }
        assertSame(SoftmaxKind.SOFTMAX, softmax.provenance().orElseThrow().operation().kind());
        assertSame(SoftmaxKind.LOG_SOFTMAX,
                logSoftmax.provenance().orElseThrow().operation().kind());
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
