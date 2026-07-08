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

class ScatterNdSemanticsTest {
    @Test
    void declaresExactlyTheScatterNdKind() {
        OperationKind kind = ScatterNdKind.SCATTER_ND;

        assertAll(
                () -> assertArrayEquals(
                        new ScatterNdKind[] {ScatterNdKind.SCATTER_ND}, ScatterNdKind.values()),
                () -> assertEquals("SCATTER_ND", kind.name()),
                () -> assertSame(
                        ScatterNdKind.SCATTER_ND, ScatterNdKind.valueOf("SCATTER_ND")),
                () -> assertInstanceOf(OperationKind.class, kind));
    }

    @Test
    void exposesOnlyTheExactEnumShape() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(ScatterNdKind.class);
    }

    @Test
    void exposesOnlyTheExactAttributesRecordShape() {
        var components = ScatterNdAttrs.class.getRecordComponents();
        var constructors = ScatterNdAttrs.class.getDeclaredConstructors();
        var fields = ScatterNdAttrs.class.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.index",
                        ScatterNdAttrs.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(ScatterNdAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(ScatterNdAttrs.class.getModifiers())),
                () -> assertTrue(ScatterNdAttrs.class.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(ScatterNdAttrs.class.getInterfaces())),
                () -> assertEquals(
                        List.of("batchDimensions", "reduction"),
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
                        List.of("batchDimensions", "reduction"),
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
                                "reduction():io.github.pho001.synaptik.model.operation.index.ScatterReduction",
                                "toString():java.lang.String"),
                        Arrays.stream(ScatterNdAttrs.class.getDeclaredMethods())
                                .map(ScatterNdSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, ScatterNdAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void retainsEveryReductionWithZeroOrdinaryAndMaximumBatchCounts() {
        for (ScatterReduction reduction : ScatterReduction.values()) {
            for (int batchDimensions : new int[] {0, 1, 37, Integer.MAX_VALUE}) {
                var attrs = new ScatterNdAttrs(batchDimensions, reduction);
                assertAll(
                        () -> assertEquals(batchDimensions, attrs.batchDimensions()),
                        () -> assertSame(reduction, attrs.reduction()));
            }
        }
    }

    @Test
    void validatesNegativeBatchCountBeforeNullReductionWithExactFailures() {
        for (int batchDimensions : new int[] {-1, -2, -37, Integer.MIN_VALUE}) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new ScatterNdAttrs(batchDimensions, null));

            assertEquals(
                    "batchDimensions must be non-negative: " + batchDimensions,
                    failure.getMessage());
        }

        NullPointerException failure = assertThrows(
                NullPointerException.class,
                () -> new ScatterNdAttrs(0, null));
        assertEquals("reduction", failure.getMessage());
    }

    @Test
    void usesGeneratedRecordValueSemanticsAndDiagnosticText() {
        var attrs = new ScatterNdAttrs(1, ScatterReduction.ADD);
        var equal = new ScatterNdAttrs(1, ScatterReduction.ADD);
        var otherBatch = new ScatterNdAttrs(0, ScatterReduction.ADD);
        var otherReduction = new ScatterNdAttrs(1, ScatterReduction.NONE);

        assertAll(
                () -> assertEquals(attrs, equal),
                () -> assertEquals(attrs.hashCode(), equal.hashCode()),
                () -> assertNotEquals(attrs, otherBatch),
                () -> assertNotEquals(attrs, otherReduction),
                () -> assertEquals(
                        "ScatterNdAttrs[batchDimensions=1, reduction=ADD]", attrs.toString()));
    }

    @Test
    void composesScatterNdWithEveryExactReductionAttributesReference() {
        for (ScatterReduction reduction : ScatterReduction.values()) {
            ScatterNdAttrs attrs = new ScatterNdAttrs(1, reduction);
            Operation operation = new Operation(ScatterNdKind.SCATTER_ND, attrs);

            assertAll(
                    () -> assertSame(ScatterNdKind.SCATTER_ND, operation.kind()),
                    () -> assertSame(attrs, operation.attrs()),
                    () -> assertSame(reduction, attrs.reduction()),
                    () -> assertEquals(
                            new Operation(
                                    ScatterNdKind.SCATTER_ND,
                                    new ScatterNdAttrs(1, reduction)),
                            operation));
        }
    }

    @Test
    void remainsDistinctFromGatherNdAndAxisScatterSemantics() {
        Operation scatterNd = new Operation(
                ScatterNdKind.SCATTER_ND,
                new ScatterNdAttrs(1, ScatterReduction.ADD));
        Operation gatherNd =
                new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(1));
        Operation axisScatter = new Operation(
                AxisScatterKind.SCATTER_ELEMENTS,
                new ScatterElementsAttrs(1, ScatterReduction.ADD));

        assertAll(
                // data [2, 3, 4], indices [5, 2], B=0, K=2, updates [5, 4]
                // -> result [2, 3, 4]
                () -> assertNotEquals(scatterNd, gatherNd),
                // data [2, 3, 4], indices [2, 5, 1], B=1, K=1, updates [2, 5, 4]
                // -> result [2, 3, 4]
                () -> assertNotEquals(scatterNd, axisScatter),
                // data [2, 3], indices [2], B=0, K=2, scalar updates [] -> result [2, 3]
                () -> assertNotEquals(ScatterNdKind.SCATTER_ND, GatherNdKind.GATHER_ND),
                () -> assertNotEquals(
                        ScatterNdKind.SCATTER_ND, AxisScatterKind.SCATTER_ELEMENTS));
    }

    @Test
    void containsNoOtherKindsOrCrossLayerAttributeState() {
        List<String> kindNames = Arrays.stream(ScatterNdKind.values())
                .map(Enum::name)
                .toList();
        List<String> componentTypes = Arrays.stream(ScatterNdAttrs.class.getRecordComponents())
                .map(component -> component.getType().getName())
                .toList();

        assertAll(
                () -> assertEquals(List.of("SCATTER_ND"), kindNames),
                () -> assertFalse(kindNames.stream().anyMatch(name ->
                        name.contains("GRAD")
                                || name.contains("BACKWARD")
                                || name.contains("DEFAULT")
                                || name.contains("ALIAS"))),
                () -> assertEquals(
                        List.of("int", ScatterReduction.class.getName()), componentTypes),
                () -> assertFalse(componentTypes.stream()
                        .anyMatch(ScatterNdSemanticsTest::isForbiddenComponentType)));
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
