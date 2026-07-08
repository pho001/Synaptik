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

class AxisScatterSemanticsTest {
    @Test
    void declaresExactlyTheThreeOrderedAxisScatterKinds() {
        OperationKind scatterAdd = AxisScatterKind.SCATTER_ADD;
        OperationKind scatterAxisAdd = AxisScatterKind.SCATTER_AXIS_ADD;
        OperationKind scatterElements = AxisScatterKind.SCATTER_ELEMENTS;

        assertAll(
                () -> assertArrayEquals(
                        new AxisScatterKind[] {
                            AxisScatterKind.SCATTER_ADD,
                            AxisScatterKind.SCATTER_AXIS_ADD,
                            AxisScatterKind.SCATTER_ELEMENTS
                        },
                        AxisScatterKind.values()),
                () -> assertEquals("SCATTER_ADD", scatterAdd.name()),
                () -> assertEquals("SCATTER_AXIS_ADD", scatterAxisAdd.name()),
                () -> assertEquals("SCATTER_ELEMENTS", scatterElements.name()),
                () -> assertSame(
                        AxisScatterKind.SCATTER_ADD,
                        AxisScatterKind.valueOf("SCATTER_ADD")),
                () -> assertSame(
                        AxisScatterKind.SCATTER_AXIS_ADD,
                        AxisScatterKind.valueOf("SCATTER_AXIS_ADD")),
                () -> assertSame(
                        AxisScatterKind.SCATTER_ELEMENTS,
                        AxisScatterKind.valueOf("SCATTER_ELEMENTS")),
                () -> assertInstanceOf(OperationKind.class, scatterAdd),
                () -> assertNotEquals(scatterAdd, scatterAxisAdd),
                () -> assertNotEquals(scatterAxisAdd, scatterElements));
    }

    @Test
    void exposesOnlyTheExactAxisScatterEnumShape() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(AxisScatterKind.class);

        assertAll(
                () -> assertSame(
                        AxisScatterKind.class, AxisScatterKind.SCATTER_ADD.getClass()),
                () -> assertSame(
                        AxisScatterKind.class, AxisScatterKind.SCATTER_AXIS_ADD.getClass()),
                () -> assertSame(
                        AxisScatterKind.class, AxisScatterKind.SCATTER_ELEMENTS.getClass()));
    }

    @Test
    void declaresExactlyTheFiveOrderedScatterReductions() {
        assertAll(
                () -> assertArrayEquals(
                        new ScatterReduction[] {
                            ScatterReduction.NONE,
                            ScatterReduction.ADD,
                            ScatterReduction.MUL,
                            ScatterReduction.MAX,
                            ScatterReduction.MIN
                        },
                        ScatterReduction.values()),
                () -> assertEquals("NONE", ScatterReduction.NONE.name()),
                () -> assertEquals("ADD", ScatterReduction.ADD.name()),
                () -> assertEquals("MUL", ScatterReduction.MUL.name()),
                () -> assertEquals("MAX", ScatterReduction.MAX.name()),
                () -> assertEquals("MIN", ScatterReduction.MIN.name()),
                () -> assertSame(
                        ScatterReduction.NONE, ScatterReduction.valueOf("NONE")),
                () -> assertSame(
                        ScatterReduction.MIN, ScatterReduction.valueOf("MIN")));
    }

    @Test
    void exposesOnlyTheExactScatterReductionEnumShape() {
        assertExactEnumShape(ScatterReduction.class);

        assertAll(
                () -> assertSame(ScatterReduction.class, ScatterReduction.NONE.getClass()),
                () -> assertSame(ScatterReduction.class, ScatterReduction.ADD.getClass()),
                () -> assertSame(ScatterReduction.class, ScatterReduction.MUL.getClass()),
                () -> assertSame(ScatterReduction.class, ScatterReduction.MAX.getClass()),
                () -> assertSame(ScatterReduction.class, ScatterReduction.MIN.getClass()));
    }

    @Test
    void exposesOnlyTheExactScatterElementsAttributesRecordShape() {
        var components = ScatterElementsAttrs.class.getRecordComponents();
        var constructors = ScatterElementsAttrs.class.getDeclaredConstructors();
        var fields = ScatterElementsAttrs.class.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.index",
                        ScatterElementsAttrs.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(ScatterElementsAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(ScatterElementsAttrs.class.getModifiers())),
                () -> assertTrue(ScatterElementsAttrs.class.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(ScatterElementsAttrs.class.getInterfaces())),
                () -> assertEquals(
                        List.of("axis", "reduction"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(
                        List.of(int.class, ScatterReduction.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        List.of(int.class, ScatterReduction.class),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(
                        List.of("axis", "reduction"),
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
                                "reduction():io.github.pho001.synaptik.model.operation.index.ScatterReduction",
                                "toString():java.lang.String"),
                        Arrays.stream(ScatterElementsAttrs.class.getDeclaredMethods())
                                .map(AxisScatterSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, ScatterElementsAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void retainsEveryReductionWithZeroOrdinaryAndMaximumAxes() {
        for (ScatterReduction reduction : ScatterReduction.values()) {
            for (int axis : new int[] {0, 1, 37, Integer.MAX_VALUE}) {
                var attrs = new ScatterElementsAttrs(axis, reduction);
                assertAll(
                        () -> assertEquals(axis, attrs.axis()),
                        () -> assertSame(reduction, attrs.reduction()));
            }
        }
    }

    @Test
    void validatesNegativeAxisBeforeNullReductionWithExactFailures() {
        for (int axis : new int[] {-1, -2, -37, Integer.MIN_VALUE}) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new ScatterElementsAttrs(axis, null));

            assertEquals("axis must be non-negative: " + axis, failure.getMessage());
        }

        NullPointerException failure = assertThrows(
                NullPointerException.class,
                () -> new ScatterElementsAttrs(0, null));
        assertEquals("reduction", failure.getMessage());
    }

    @Test
    void usesGeneratedRecordValueSemanticsAndDiagnosticText() {
        var attrs = new ScatterElementsAttrs(1, ScatterReduction.ADD);
        var equal = new ScatterElementsAttrs(1, ScatterReduction.ADD);
        var otherAxis = new ScatterElementsAttrs(0, ScatterReduction.ADD);
        var otherReduction = new ScatterElementsAttrs(1, ScatterReduction.NONE);

        assertAll(
                () -> assertEquals(attrs, equal),
                () -> assertEquals(attrs.hashCode(), equal.hashCode()),
                () -> assertNotEquals(attrs, otherAxis),
                () -> assertNotEquals(attrs, otherReduction),
                () -> assertEquals(
                        "ScatterElementsAttrs[axis=1, reduction=ADD]", attrs.toString()));
    }

    @Test
    void composesFixedAddKindsWithTheExactSharedAxisAttributesReference() {
        IndexAxisAttrs attrs = new IndexAxisAttrs(1);
        Operation scatterAdd = new Operation(AxisScatterKind.SCATTER_ADD, attrs);
        Operation scatterAxisAdd =
                new Operation(AxisScatterKind.SCATTER_AXIS_ADD, attrs);

        assertAll(
                () -> assertSame(AxisScatterKind.SCATTER_ADD, scatterAdd.kind()),
                () -> assertSame(attrs, scatterAdd.attrs()),
                () -> assertSame(AxisScatterKind.SCATTER_AXIS_ADD, scatterAxisAdd.kind()),
                () -> assertSame(attrs, scatterAxisAdd.attrs()),
                () -> assertNotEquals(scatterAdd, scatterAxisAdd));
    }

    @Test
    void composesScatterElementsWithEveryExactReductionAttributesReference() {
        for (ScatterReduction reduction : ScatterReduction.values()) {
            ScatterElementsAttrs attrs = new ScatterElementsAttrs(1, reduction);
            Operation operation =
                    new Operation(AxisScatterKind.SCATTER_ELEMENTS, attrs);

            assertAll(
                    () -> assertSame(AxisScatterKind.SCATTER_ELEMENTS, operation.kind()),
                    () -> assertSame(attrs, operation.attrs()),
                    () -> assertSame(reduction, attrs.reduction()));
        }
    }

    @Test
    void keepsAllScatterAndGatherShapeMeaningsDistinct() {
        Operation scatterAdd =
                new Operation(AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(1));
        Operation scatterAxisAdd =
                new Operation(AxisScatterKind.SCATTER_AXIS_ADD, new IndexAxisAttrs(1));
        Operation scatterElements = new Operation(
                AxisScatterKind.SCATTER_ELEMENTS,
                new ScatterElementsAttrs(1, ScatterReduction.ADD));
        Operation gather =
                new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1));
        Operation gatherAxis =
                new Operation(AxisGatherKind.GATHER_AXIS, new IndexAxisAttrs(1));
        Operation gatherNd =
                new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(0));

        assertAll(
                // data [2, 3, 4], axis 1, indices/updates [2, 4] -> result [2, 3, 4]
                () -> assertNotEquals(scatterAdd, scatterAxisAdd),
                // data [2, 3, 4], axis 1, indices [5, 6], updates [2, 5, 6, 4]
                // -> result [2, 3, 4]
                () -> assertNotEquals(scatterAxisAdd, scatterElements),
                // data [2, 3, 4], axis 1, indices/updates [2, 5, 4] -> result [2, 3, 4]
                () -> assertNotEquals(scatterElements, scatterAdd),
                () -> assertNotEquals(scatterAdd, gather),
                () -> assertNotEquals(scatterAxisAdd, gatherAxis),
                () -> assertNotEquals(scatterElements, gatherNd),
                () -> assertNotEquals(AxisScatterKind.SCATTER_ADD, AxisGatherKind.GATHER),
                () -> assertNotEquals(
                        AxisScatterKind.SCATTER_AXIS_ADD, AxisGatherKind.GATHER_AXIS),
                () -> assertNotEquals(
                        AxisScatterKind.SCATTER_ELEMENTS, GatherNdKind.GATHER_ND));
    }

    @Test
    void containsNoScatterNdGradientAliasDefaultOrCrossLayerState() {
        List<String> kindNames = Arrays.stream(AxisScatterKind.values())
                .map(Enum::name)
                .toList();
        List<String> reductionNames = Arrays.stream(ScatterReduction.values())
                .map(Enum::name)
                .toList();
        List<String> componentTypes = Arrays.stream(
                        ScatterElementsAttrs.class.getRecordComponents())
                .map(component -> component.getType().getName())
                .toList();

        assertAll(
                () -> assertEquals(
                        List.of("SCATTER_ADD", "SCATTER_AXIS_ADD", "SCATTER_ELEMENTS"),
                        kindNames),
                () -> assertFalse(kindNames.stream().anyMatch(name ->
                        name.contains("ND")
                                || name.contains("GRAD")
                                || name.contains("GATHER")
                                || name.contains("BACKWARD")
                                || name.contains("DEFAULT")
                                || name.contains("ALIAS"))),
                () -> assertEquals(List.of("NONE", "ADD", "MUL", "MAX", "MIN"), reductionNames),
                () -> assertFalse(reductionNames.stream().anyMatch(name ->
                        name.contains("DEFAULT")
                                || name.contains("OVERWRITE")
                                || name.contains("REPLACE"))),
                () -> assertEquals(
                        List.of("int", ScatterReduction.class.getName()), componentTypes),
                () -> assertFalse(componentTypes.stream()
                        .anyMatch(AxisScatterSemanticsTest::isForbiddenComponentType)));
    }

    private static void assertExactEnumShape(Class<? extends Enum<?>> enumType, Class<?>... interfaces) {
        var constructors = enumType.getDeclaredConstructors();
        var fields = enumType.getDeclaredFields();
        var methods = enumType.getDeclaredMethods();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.index",
                        enumType.getPackageName()),
                () -> assertTrue(Modifier.isPublic(enumType.getModifiers())),
                () -> assertTrue(Modifier.isFinal(enumType.getModifiers())),
                () -> assertTrue(enumType.isEnum()),
                () -> assertEquals(
                        Arrays.asList(interfaces),
                        Arrays.asList(enumType.getInterfaces())),
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
                        List.of("valueOf", "values"),
                        Arrays.stream(methods)
                                .filter(method -> !method.isSynthetic())
                                .map(method -> method.getName())
                                .sorted()
                                .toList()),
                () -> assertTrue(Arrays.stream(methods)
                        .filter(method -> !Modifier.isStatic(method.getModifiers()))
                        .findAny()
                        .isEmpty()),
                () -> assertEquals(0, enumType.getDeclaredClasses().length));
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
