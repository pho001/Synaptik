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

class GatherNdSemanticsTest {
    @Test
    void declaresExactlyTheGatherNdKind() {
        OperationKind kind = GatherNdKind.GATHER_ND;

        assertAll(
                () -> assertArrayEquals(
                        new GatherNdKind[] {GatherNdKind.GATHER_ND}, GatherNdKind.values()),
                () -> assertEquals("GATHER_ND", kind.name()),
                () -> assertSame(
                        GatherNdKind.GATHER_ND, GatherNdKind.valueOf("GATHER_ND")),
                () -> assertInstanceOf(OperationKind.class, kind));
    }

    @Test
    void exposesOnlyTheExactEnumShape() {
        var constructors = GatherNdKind.class.getDeclaredConstructors();
        var fields = GatherNdKind.class.getDeclaredFields();
        var methods = GatherNdKind.class.getDeclaredMethods();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.index",
                        GatherNdKind.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(GatherNdKind.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(GatherNdKind.class.getModifiers())),
                () -> assertTrue(GatherNdKind.class.isEnum()),
                () -> assertEquals(
                        List.of(OperationKind.class),
                        Arrays.asList(GatherNdKind.class.getInterfaces())),
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
                                "valueOf(java.lang.String):io.github.pho001.synaptik.model.operation.index.GatherNdKind",
                                "values():[Lio.github.pho001.synaptik.model.operation.index.GatherNdKind;"),
                        Arrays.stream(methods)
                                .filter(method -> !method.isSynthetic())
                                .map(GatherNdSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertTrue(Arrays.stream(methods)
                        .filter(method -> !Modifier.isStatic(method.getModifiers()))
                        .findAny()
                        .isEmpty()),
                () -> assertEquals(0, GatherNdKind.class.getDeclaredClasses().length),
                () -> assertSame(GatherNdKind.class, GatherNdKind.GATHER_ND.getClass()));
    }

    @Test
    void exposesOnlyTheExactAttributesRecordShape() {
        var components = GatherNdAttrs.class.getRecordComponents();
        var constructors = GatherNdAttrs.class.getDeclaredConstructors();
        var fields = GatherNdAttrs.class.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.index",
                        GatherNdAttrs.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(GatherNdAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(GatherNdAttrs.class.getModifiers())),
                () -> assertTrue(GatherNdAttrs.class.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(GatherNdAttrs.class.getInterfaces())),
                () -> assertEquals(
                        List.of("batchDimensions"),
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
                        List.of("batchDimensions"),
                        Arrays.stream(fields).map(field -> field.getName()).toList()),
                () -> assertTrue(Arrays.stream(fields).allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers())
                                && !Modifier.isStatic(field.getModifiers()))),
                () -> assertEquals(
                        List.of(
                                "batchDimensions():int",
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "toString():java.lang.String"),
                        Arrays.stream(GatherNdAttrs.class.getDeclaredMethods())
                                .map(GatherNdSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, GatherNdAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void retainsZeroOrdinaryAndMaximumBatchCountsUnchanged() {
        for (int batchDimensions : new int[] {0, 1, 37, Integer.MAX_VALUE}) {
            assertEquals(
                    batchDimensions,
                    new GatherNdAttrs(batchDimensions).batchDimensions());
        }
    }

    @Test
    void rejectsRepresentativeNegativeBatchCountsWithTheExactMessage() {
        for (int batchDimensions : new int[] {-1, -2, -37, Integer.MIN_VALUE}) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new GatherNdAttrs(batchDimensions));

            assertEquals(
                    "batchDimensions must be non-negative: " + batchDimensions,
                    failure.getMessage());
        }
    }

    @Test
    void usesGeneratedRecordValueSemanticsAndDiagnosticText() {
        var attrs = new GatherNdAttrs(1);
        var equal = new GatherNdAttrs(1);
        var different = new GatherNdAttrs(0);

        assertAll(
                () -> assertEquals(attrs, equal),
                () -> assertEquals(attrs.hashCode(), equal.hashCode()),
                () -> assertNotEquals(attrs, different),
                () -> assertEquals(
                        "GatherNdAttrs[batchDimensions=1]", attrs.toString()));
    }

    @Test
    void composesGatherNdWithTheExactAttributesReference() {
        GatherNdAttrs attrs = new GatherNdAttrs(1);
        Operation operation = new Operation(GatherNdKind.GATHER_ND, attrs);

        assertAll(
                () -> assertSame(GatherNdKind.GATHER_ND, operation.kind()),
                () -> assertSame(attrs, operation.attrs()),
                () -> assertEquals(
                        new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(1)), operation));
    }

    @Test
    void remainsDistinctFromAxisGatherAndScalarSelectSemantics() {
        Operation gatherNd =
                new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(1));
        Operation axisGather =
                new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1));
        Operation scalarSelect =
                new Operation(SelectKind.SELECT, new SelectAttrs(1, 2L));

        assertAll(
                // data [2, 3, 4], indices [5, 2], B=0, K=2 -> result [5, 4]
                () -> assertNotEquals(gatherNd, axisGather),
                // data [2, 3, 4], indices [2, 5, 1], B=1, K=1 -> result [2, 5, 4]
                () -> assertNotEquals(gatherNd, scalarSelect),
                // data [2, 3], indices [2], B=0, K=2 -> canonical scalar result []
                () -> assertNotEquals(GatherNdKind.GATHER_ND, AxisGatherKind.GATHER),
                () -> assertNotEquals(GatherNdKind.GATHER_ND, SelectKind.SELECT));
    }

    @Test
    void containsNoOtherKindsOrCrossLayerAttributeState() {
        List<String> kindNames = Arrays.stream(GatherNdKind.values())
                .map(Enum::name)
                .toList();
        List<String> componentTypes = Arrays.stream(GatherNdAttrs.class.getRecordComponents())
                .map(component -> component.getType().getName())
                .toList();

        assertAll(
                () -> assertEquals(List.of("GATHER_ND"), kindNames),
                () -> assertFalse(kindNames.stream().anyMatch(name ->
                        name.contains("SCATTER") || name.contains("GRAD"))),
                () -> assertEquals(List.of("int"), componentTypes),
                () -> assertFalse(componentTypes.stream()
                        .anyMatch(GatherNdSemanticsTest::isForbiddenComponentType)));
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
