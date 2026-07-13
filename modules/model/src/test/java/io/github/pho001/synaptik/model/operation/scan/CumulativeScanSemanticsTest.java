package io.github.pho001.synaptik.model.operation.scan;

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
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class CumulativeScanSemanticsTest {
    @Test
    void declaresExactlyTheCumulativeScanKindsInOrderWithTypedIdentity() {
        OperationKind cumulativeSum = CumulativeScanKind.CUM_SUM;
        OperationKind cumulativeProduct = CumulativeScanKind.CUM_PROD;
        OperationKind aggregateSum = AggregateReductionKind.SUM;

        assertAll(
                () -> assertArrayEquals(
                        new CumulativeScanKind[] {
                                CumulativeScanKind.CUM_SUM, CumulativeScanKind.CUM_PROD
                        },
                        CumulativeScanKind.values()),
                () -> assertEquals("CUM_SUM", cumulativeSum.name()),
                () -> assertEquals("CUM_SUM", cumulativeSum.toString()),
                () -> assertSame(
                        CumulativeScanKind.CUM_SUM,
                        CumulativeScanKind.valueOf("CUM_SUM")),
                () -> assertEquals("CUM_PROD", cumulativeProduct.name()),
                () -> assertEquals("CUM_PROD", cumulativeProduct.toString()),
                () -> assertSame(
                        CumulativeScanKind.CUM_PROD,
                        CumulativeScanKind.valueOf("CUM_PROD")),
                () -> assertInstanceOf(OperationKind.class, cumulativeSum),
                () -> assertInstanceOf(OperationKind.class, cumulativeProduct),
                () -> assertNotEquals(cumulativeSum, cumulativeProduct),
                () -> assertNotEquals(cumulativeSum, aggregateSum));
    }

    @Test
    void exposesOnlyTheExactEnumShape() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(CumulativeScanKind.class);
    }

    @Test
    void exposesOnlyTheExactAttributesRecordShape() {
        var components = CumulativeScanAttrs.class.getRecordComponents();
        var constructors = CumulativeScanAttrs.class.getDeclaredConstructors();
        var fields = CumulativeScanAttrs.class.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.scan",
                        CumulativeScanAttrs.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(CumulativeScanAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(CumulativeScanAttrs.class.getModifiers())),
                () -> assertTrue(CumulativeScanAttrs.class.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(CumulativeScanAttrs.class.getInterfaces())),
                () -> assertEquals(
                        List.of("axis", "exclusive", "reverse"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(
                        List.of(int.class, boolean.class, boolean.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        List.of(int.class, boolean.class, boolean.class),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(
                        List.of("axis", "exclusive", "reverse"),
                        Arrays.stream(fields).map(field -> field.getName()).toList()),
                () -> assertTrue(Arrays.stream(fields).allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers())
                                && !Modifier.isStatic(field.getModifiers()))),
                () -> assertEquals(
                        List.of(
                                "axis():int",
                                "equals(java.lang.Object):boolean",
                                "exclusive():boolean",
                                "hashCode():int",
                                "reverse():boolean",
                                "toString():java.lang.String"),
                        Arrays.stream(CumulativeScanAttrs.class.getDeclaredMethods())
                                .map(CumulativeScanSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, CumulativeScanAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void acceptsEveryStructuralAxisBoundaryAndAllFourModes() {
        for (int axis : new int[] {0, 1, 37, Integer.MAX_VALUE}) {
            for (boolean exclusive : new boolean[] {false, true}) {
                for (boolean reverse : new boolean[] {false, true}) {
                    CumulativeScanAttrs attrs =
                            new CumulativeScanAttrs(axis, exclusive, reverse);

                    assertAll(
                            () -> assertEquals(axis, attrs.axis()),
                            () -> assertEquals(exclusive, attrs.exclusive()),
                            () -> assertEquals(reverse, attrs.reverse()));
                }
            }
        }
    }

    @Test
    void rejectsEveryRepresentativeNegativeAxisWithTheExactMessage() {
        for (int axis : new int[] {-1, -2, -37, Integer.MIN_VALUE}) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new CumulativeScanAttrs(axis, true, true));

            assertEquals("axis must be non-negative: " + axis, failure.getMessage());
        }
    }

    @Test
    void usesGeneratedRecordValueSemanticsAndDiagnosticText() {
        var attrs = new CumulativeScanAttrs(2, true, false);
        var equal = new CumulativeScanAttrs(2, true, false);
        var differentAxis = new CumulativeScanAttrs(1, true, false);
        var differentExclusive = new CumulativeScanAttrs(2, false, false);
        var differentReverse = new CumulativeScanAttrs(2, true, true);

        assertAll(
                () -> assertEquals(attrs, equal),
                () -> assertEquals(attrs.hashCode(), equal.hashCode()),
                () -> assertNotEquals(attrs, differentAxis),
                () -> assertNotEquals(attrs, differentExclusive),
                () -> assertNotEquals(attrs, differentReverse),
                () -> assertEquals(
                        "CumulativeScanAttrs[axis=2, exclusive=true, reverse=false]",
                        attrs.toString()));
    }

    @Test
    void composesEveryModeWithTheExactKindAndAttributesReference() {
        for (boolean exclusive : new boolean[] {false, true}) {
            for (boolean reverse : new boolean[] {false, true}) {
                for (CumulativeScanKind kind : CumulativeScanKind.values()) {
                    CumulativeScanAttrs attrs =
                            new CumulativeScanAttrs(1, exclusive, reverse);
                    Operation operation = new Operation(kind, attrs);

                    assertAll(
                            () -> assertSame(kind, operation.kind()),
                            () -> assertSame(attrs, operation.attrs()));
                }
            }
        }
    }

    @Test
    void remainsDistinctFromAggregateReductionSemanticsAndContainsNoCrossLayerState() {
        CumulativeScanAttrs scanAttrs = new CumulativeScanAttrs(0, false, false);
        AxisReductionAttrs reductionAttrs = new AxisReductionAttrs(0, false);
        Operation scan = new Operation(CumulativeScanKind.CUM_SUM, scanAttrs);
        Operation reduction = new Operation(AggregateReductionKind.SUM, reductionAttrs);
        var componentTypes = Arrays.stream(CumulativeScanAttrs.class.getRecordComponents())
                .map(component -> component.getType().getName())
                .toList();

        assertAll(
                () -> assertNotEquals(scan, reduction),
                () -> assertNotEquals(scanAttrs, reductionAttrs),
                () -> assertEquals(List.of("int", "boolean", "boolean"), componentTypes),
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
