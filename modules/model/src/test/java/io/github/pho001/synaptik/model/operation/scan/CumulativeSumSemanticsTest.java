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

class CumulativeSumSemanticsTest {
    @Test
    void declaresExactlyTheCumulativeSumKindWithTypedIdentity() {
        OperationKind cumulativeSum = CumulativeSumKind.CUM_SUM;
        OperationKind aggregateSum = AggregateReductionKind.SUM;

        assertAll(
                () -> assertArrayEquals(
                        new CumulativeSumKind[] {CumulativeSumKind.CUM_SUM},
                        CumulativeSumKind.values()),
                () -> assertEquals("CUM_SUM", cumulativeSum.name()),
                () -> assertEquals("CUM_SUM", cumulativeSum.toString()),
                () -> assertSame(
                        CumulativeSumKind.CUM_SUM,
                        CumulativeSumKind.valueOf("CUM_SUM")),
                () -> assertInstanceOf(OperationKind.class, cumulativeSum),
                () -> assertNotEquals(cumulativeSum, aggregateSum));
    }

    @Test
    void exposesOnlyTheExactEnumShape() {
        var constructors = CumulativeSumKind.class.getDeclaredConstructors();
        var fields = CumulativeSumKind.class.getDeclaredFields();
        var methods = CumulativeSumKind.class.getDeclaredMethods();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.scan",
                        CumulativeSumKind.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(CumulativeSumKind.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(CumulativeSumKind.class.getModifiers())),
                () -> assertTrue(CumulativeSumKind.class.isEnum()),
                () -> assertEquals(
                        List.of(OperationKind.class),
                        Arrays.asList(CumulativeSumKind.class.getInterfaces())),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        List.of(String.class, int.class),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertTrue(Arrays.stream(fields)
                        .filter(field -> !field.isEnumConstant())
                        .allMatch(field -> field.isSynthetic()
                                && Modifier.isStatic(field.getModifiers()))),
                () -> assertTrue(Arrays.stream(fields)
                        .filter(field -> !Modifier.isStatic(field.getModifiers()))
                        .findAny()
                        .isEmpty()),
                () -> assertEquals(
                        List.of(
                                "valueOf(java.lang.String):io.github.pho001.synaptik.model.operation.scan.CumulativeSumKind",
                                "values():[Lio.github.pho001.synaptik.model.operation.scan.CumulativeSumKind;"),
                        Arrays.stream(methods)
                                .filter(method -> !method.isSynthetic())
                                .map(CumulativeSumSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertTrue(Arrays.stream(methods)
                        .filter(method -> !Modifier.isStatic(method.getModifiers()))
                        .findAny()
                        .isEmpty()),
                () -> assertEquals(0, CumulativeSumKind.class.getDeclaredClasses().length),
                () -> assertSame(CumulativeSumKind.class, CumulativeSumKind.CUM_SUM.getClass()));
    }

    @Test
    void exposesOnlyTheExactAttributesRecordShape() {
        var components = CumulativeSumAttrs.class.getRecordComponents();
        var constructors = CumulativeSumAttrs.class.getDeclaredConstructors();
        var fields = CumulativeSumAttrs.class.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.scan",
                        CumulativeSumAttrs.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(CumulativeSumAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(CumulativeSumAttrs.class.getModifiers())),
                () -> assertTrue(CumulativeSumAttrs.class.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(CumulativeSumAttrs.class.getInterfaces())),
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
                        Arrays.stream(CumulativeSumAttrs.class.getDeclaredMethods())
                                .map(CumulativeSumSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, CumulativeSumAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void acceptsEveryStructuralAxisBoundaryAndAllFourModes() {
        for (int axis : new int[] {0, 1, 37, Integer.MAX_VALUE}) {
            for (boolean exclusive : new boolean[] {false, true}) {
                for (boolean reverse : new boolean[] {false, true}) {
                    CumulativeSumAttrs attrs =
                            new CumulativeSumAttrs(axis, exclusive, reverse);

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
                    () -> new CumulativeSumAttrs(axis, true, true));

            assertEquals("axis must be non-negative: " + axis, failure.getMessage());
        }
    }

    @Test
    void usesGeneratedRecordValueSemanticsAndDiagnosticText() {
        var attrs = new CumulativeSumAttrs(2, true, false);
        var equal = new CumulativeSumAttrs(2, true, false);
        var differentAxis = new CumulativeSumAttrs(1, true, false);
        var differentExclusive = new CumulativeSumAttrs(2, false, false);
        var differentReverse = new CumulativeSumAttrs(2, true, true);

        assertAll(
                () -> assertEquals(attrs, equal),
                () -> assertEquals(attrs.hashCode(), equal.hashCode()),
                () -> assertNotEquals(attrs, differentAxis),
                () -> assertNotEquals(attrs, differentExclusive),
                () -> assertNotEquals(attrs, differentReverse),
                () -> assertEquals(
                        "CumulativeSumAttrs[axis=2, exclusive=true, reverse=false]",
                        attrs.toString()));
    }

    @Test
    void composesEveryModeWithTheExactKindAndAttributesReference() {
        for (boolean exclusive : new boolean[] {false, true}) {
            for (boolean reverse : new boolean[] {false, true}) {
                CumulativeSumAttrs attrs = new CumulativeSumAttrs(1, exclusive, reverse);
                Operation operation = new Operation(CumulativeSumKind.CUM_SUM, attrs);

                assertAll(
                        () -> assertSame(CumulativeSumKind.CUM_SUM, operation.kind()),
                        () -> assertSame(attrs, operation.attrs()));
            }
        }
    }

    @Test
    void remainsDistinctFromAggregateReductionSemanticsAndContainsNoCrossLayerState() {
        CumulativeSumAttrs scanAttrs = new CumulativeSumAttrs(0, false, false);
        AxisReductionAttrs reductionAttrs = new AxisReductionAttrs(0, false);
        Operation scan = new Operation(CumulativeSumKind.CUM_SUM, scanAttrs);
        Operation reduction = new Operation(AggregateReductionKind.SUM, reductionAttrs);
        var componentTypes = Arrays.stream(CumulativeSumAttrs.class.getRecordComponents())
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
