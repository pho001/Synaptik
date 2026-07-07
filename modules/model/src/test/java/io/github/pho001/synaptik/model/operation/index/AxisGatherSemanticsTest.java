package io.github.pho001.synaptik.model.operation.index;

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
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class AxisGatherSemanticsTest {
    @Test
    void declaresExactlyTheThreeOrderedAxisGatherKinds() {
        OperationKind gather = AxisGatherKind.GATHER;
        OperationKind gatherAxis = AxisGatherKind.GATHER_AXIS;
        OperationKind takeAlongAxis = AxisGatherKind.TAKE_ALONG_AXIS;

        assertAll(
                () -> assertArrayEquals(
                        new AxisGatherKind[] {
                            AxisGatherKind.GATHER,
                            AxisGatherKind.GATHER_AXIS,
                            AxisGatherKind.TAKE_ALONG_AXIS
                        },
                        AxisGatherKind.values()),
                () -> assertEquals("GATHER", gather.name()),
                () -> assertEquals("GATHER_AXIS", gatherAxis.name()),
                () -> assertEquals("TAKE_ALONG_AXIS", takeAlongAxis.name()),
                () -> assertSame(AxisGatherKind.GATHER, AxisGatherKind.valueOf("GATHER")),
                () -> assertSame(
                        AxisGatherKind.GATHER_AXIS,
                        AxisGatherKind.valueOf("GATHER_AXIS")),
                () -> assertSame(
                        AxisGatherKind.TAKE_ALONG_AXIS,
                        AxisGatherKind.valueOf("TAKE_ALONG_AXIS")),
                () -> assertInstanceOf(OperationKind.class, gather),
                () -> assertNotEquals(gather, gatherAxis),
                () -> assertNotEquals(gatherAxis, takeAlongAxis));
    }

    @Test
    void exposesOnlyTheExactEnumShape() {
        var constructors = AxisGatherKind.class.getDeclaredConstructors();
        var fields = AxisGatherKind.class.getDeclaredFields();
        var methods = AxisGatherKind.class.getDeclaredMethods();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.index",
                        AxisGatherKind.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(AxisGatherKind.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(AxisGatherKind.class.getModifiers())),
                () -> assertTrue(AxisGatherKind.class.isEnum()),
                () -> assertEquals(
                        List.of(OperationKind.class),
                        Arrays.asList(AxisGatherKind.class.getInterfaces())),
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
                                "valueOf(java.lang.String):io.github.pho001.synaptik.model.operation.index.AxisGatherKind",
                                "values():[Lio.github.pho001.synaptik.model.operation.index.AxisGatherKind;"),
                        Arrays.stream(methods)
                                .filter(method -> !method.isSynthetic())
                                .map(AxisGatherSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertTrue(Arrays.stream(methods)
                        .filter(method -> !Modifier.isStatic(method.getModifiers()))
                        .findAny()
                        .isEmpty()),
                () -> assertEquals(0, AxisGatherKind.class.getDeclaredClasses().length),
                () -> assertSame(AxisGatherKind.class, AxisGatherKind.GATHER.getClass()),
                () -> assertSame(AxisGatherKind.class, AxisGatherKind.GATHER_AXIS.getClass()),
                () -> assertSame(
                        AxisGatherKind.class, AxisGatherKind.TAKE_ALONG_AXIS.getClass()));
    }

    @Test
    void exposesOnlyTheExactAttributesRecordShape() {
        var components = IndexAxisAttrs.class.getRecordComponents();
        var constructors = IndexAxisAttrs.class.getDeclaredConstructors();
        var fields = IndexAxisAttrs.class.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.index",
                        IndexAxisAttrs.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(IndexAxisAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(IndexAxisAttrs.class.getModifiers())),
                () -> assertTrue(IndexAxisAttrs.class.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(IndexAxisAttrs.class.getInterfaces())),
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
                        Arrays.stream(IndexAxisAttrs.class.getDeclaredMethods())
                                .map(AxisGatherSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, IndexAxisAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void retainsZeroOrdinaryAndMaximumAxesUnchanged() {
        for (int axis : new int[] {0, 1, 37, Integer.MAX_VALUE}) {
            assertEquals(axis, new IndexAxisAttrs(axis).axis());
        }
    }

    @Test
    void rejectsRepresentativeNegativeAxesWithTheExactMessage() {
        for (int axis : new int[] {-1, -2, -37, Integer.MIN_VALUE}) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class, () -> new IndexAxisAttrs(axis));

            assertEquals("axis must be non-negative: " + axis, failure.getMessage());
        }
    }

    @Test
    void usesGeneratedRecordValueSemanticsAndDiagnosticText() {
        var attrs = new IndexAxisAttrs(2);
        var equal = new IndexAxisAttrs(2);
        var different = new IndexAxisAttrs(1);

        assertAll(
                () -> assertEquals(attrs, equal),
                () -> assertEquals(attrs.hashCode(), equal.hashCode()),
                () -> assertNotEquals(attrs, different),
                () -> assertEquals("IndexAxisAttrs[axis=2]", attrs.toString()));
    }

    @Test
    void composesEveryKindWithTheExactSharedAttributesReference() {
        IndexAxisAttrs attrs = new IndexAxisAttrs(1);
        Operation gather = new Operation(AxisGatherKind.GATHER, attrs);
        Operation gatherAxis = new Operation(AxisGatherKind.GATHER_AXIS, attrs);
        Operation takeAlongAxis = new Operation(AxisGatherKind.TAKE_ALONG_AXIS, attrs);

        assertAll(
                () -> assertSame(AxisGatherKind.GATHER, gather.kind()),
                () -> assertSame(attrs, gather.attrs()),
                () -> assertSame(AxisGatherKind.GATHER_AXIS, gatherAxis.kind()),
                () -> assertSame(attrs, gatherAxis.attrs()),
                () -> assertSame(AxisGatherKind.TAKE_ALONG_AXIS, takeAlongAxis.kind()),
                () -> assertSame(attrs, takeAlongAxis.attrs()),
                () -> assertNotEquals(gather, gatherAxis),
                () -> assertNotEquals(gatherAxis, takeAlongAxis));
    }

    @Test
    void keepsGatherShapeMeaningsAndScalarSelectIdentityDistinct() {
        Operation gather = new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1));
        Operation gatherAxis =
                new Operation(AxisGatherKind.GATHER_AXIS, new IndexAxisAttrs(1));
        Operation takeAlongAxis =
                new Operation(AxisGatherKind.TAKE_ALONG_AXIS, new IndexAxisAttrs(1));
        Operation scalarSelect =
                new Operation(SelectKind.SELECT, new SelectAttrs(1, 2L));

        assertAll(
                // data [2, 3, 4], axis 1, indices [2, 4] -> result [2, 4]
                () -> assertNotEquals(gather, gatherAxis),
                // data [2, 3, 4], axis 1, indices [5, 6] -> result [2, 5, 6, 4]
                () -> assertNotEquals(gatherAxis, takeAlongAxis),
                // data [2, 3, 4], axis 1, indices [2, 7, 4] -> result [2, 7, 4]
                () -> assertNotEquals(takeAlongAxis, gather),
                () -> assertNotEquals(gather, scalarSelect),
                () -> assertNotEquals(AxisGatherKind.GATHER, SelectKind.SELECT));
    }

    @Test
    void containsNoAliasOrOtherIndexFamilyKindsAndNoCrossLayerState() {
        List<String> kindNames = Arrays.stream(AxisGatherKind.values())
                .map(Enum::name)
                .toList();
        List<String> componentTypes = Arrays.stream(IndexAxisAttrs.class.getRecordComponents())
                .map(component -> component.getType().getName())
                .toList();

        assertAll(
                () -> assertEquals(
                        List.of("GATHER", "GATHER_AXIS", "TAKE_ALONG_AXIS"), kindNames),
                () -> assertFalse(kindNames.contains("TAKE")),
                () -> assertFalse(kindNames.stream().anyMatch(name ->
                        name.contains("ND")
                                || name.contains("SCATTER")
                                || name.contains("GRAD"))),
                () -> assertEquals(List.of("int"), componentTypes),
                () -> assertFalse(componentTypes.stream()
                        .anyMatch(AxisGatherSemanticsTest::isForbiddenComponentType)));
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

    private static boolean isForbiddenComponentType(String name) {
        return name.contains("Tensor")
                || name.contains("DataType")
                || name.contains("Shape")
                || name.contains("layout")
                || name.contains("provenance")
                || name.contains("graph")
                || name.contains("compiler")
                || name.contains("planning")
                || name.contains("prepare")
                || name.contains("runtime")
                || name.contains("backend");
    }
}
