package io.github.pho001.synaptik.model.operation.reduction;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class MaskedReductionAttrsTest {
    @Test
    void exposesExactlyTheRequiredRecordShape() {
        var type = MaskedReductionAttrs.class;
        var components = type.getRecordComponents();
        var constructors = type.getDeclaredConstructors();
        var fields = type.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.reduction",
                        type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class), Arrays.asList(type.getInterfaces())),
                () -> assertEquals(
                        List.of("axis", "maskInputAxes"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(
                        List.of(int.class, List.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(
                        List.of(int.class, List.class),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertEquals(
                        List.of("axis", "maskInputAxes"),
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
                                "maskInputAxes():java.util.List",
                                "toString():java.lang.String"),
                        Arrays.stream(type.getDeclaredMethods())
                                .map(MaskedReductionAttrsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, type.getDeclaredClasses().length));
    }

    @Test
    void acceptsEveryRequiredStructuralMappingBoundary() {
        var empty = new MaskedReductionAttrs(0, List.of());
        var singleton = new MaskedReductionAttrs(1, List.of(1));
        var contiguous = new MaskedReductionAttrs(1, List.of(0, 1));
        var gapped = new MaskedReductionAttrs(2, List.of(0, 2));
        var maximum = new MaskedReductionAttrs(Integer.MAX_VALUE, List.of(Integer.MAX_VALUE));

        assertAll(
                () -> assertEquals(0, empty.axis()),
                () -> assertEquals(List.of(), empty.maskInputAxes()),
                () -> assertEquals(List.of(1), singleton.maskInputAxes()),
                () -> assertEquals(List.of(0, 1), contiguous.maskInputAxes()),
                () -> assertEquals(List.of(0, 2), gapped.maskInputAxes()),
                () -> assertEquals(Integer.MAX_VALUE, maximum.axis()),
                () -> assertEquals(List.of(Integer.MAX_VALUE), maximum.maskInputAxes()));
    }

    @Test
    void rejectsNegativeAxisBeforeInspectingTheMapping() {
        for (int axis : new int[] {-1, -37, Integer.MIN_VALUE}) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new MaskedReductionAttrs(axis, null));

            assertEquals("axis must be non-negative: " + axis, failure.getMessage());
        }
    }

    @Test
    void rejectsNullMappingAndNullElementsWithExactIndexedMessages() {
        NullPointerException nullMapping = assertThrows(
                NullPointerException.class,
                () -> new MaskedReductionAttrs(0, null));
        NullPointerException firstNull = assertThrows(
                NullPointerException.class,
                () -> new MaskedReductionAttrs(0, Arrays.asList(null, -1)));
        NullPointerException laterNull = assertThrows(
                NullPointerException.class,
                () -> new MaskedReductionAttrs(0, Arrays.asList(0, null, -1)));

        assertAll(
                () -> assertEquals("maskInputAxes", nullMapping.getMessage()),
                () -> assertEquals("maskInputAxes[0]", firstNull.getMessage()),
                () -> assertEquals("maskInputAxes[1]", laterNull.getMessage()));
    }

    @Test
    void rejectsNegativeElementsBeforeComparingTheirOrder() {
        for (List<Integer> mapping : List.of(List.of(-1), List.of(0, -2))) {
            int index = mapping.size() - 1;
            int value = mapping.get(index);
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new MaskedReductionAttrs(0, mapping));

            assertEquals(
                    "maskInputAxes[" + index + "] must be non-negative: " + value,
                    failure.getMessage());
        }
    }

    @Test
    void rejectsRepeatedAndDescendingMappingsWithExactMessages() {
        IllegalArgumentException repeated = assertThrows(
                IllegalArgumentException.class,
                () -> new MaskedReductionAttrs(0, List.of(1, 1)));
        IllegalArgumentException descending = assertThrows(
                IllegalArgumentException.class,
                () -> new MaskedReductionAttrs(0, List.of(0, 3, 2)));

        assertAll(
                () -> assertEquals(
                        "maskInputAxes must be strictly increasing at index 1: previous=1, current=1",
                        repeated.getMessage()),
                () -> assertEquals(
                        "maskInputAxes must be strictly increasing at index 2: previous=3, current=2",
                        descending.getMessage()));
    }

    @Test
    void snapshotsCallerStateAndExposesAnImmutableList() {
        var callerMapping = new ArrayList<>(List.of(0, 2));
        var attrs = new MaskedReductionAttrs(1, callerMapping);

        callerMapping.set(0, 1);
        callerMapping.add(3);

        assertAll(
                () -> assertEquals(List.of(0, 2), attrs.maskInputAxes()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> attrs.maskInputAxes().add(4)));
    }

    @Test
    void usesGeneratedValueSemanticsAndDiagnosticText() {
        var attrs = new MaskedReductionAttrs(1, List.of(0, 2));
        var equal = new MaskedReductionAttrs(1, List.of(0, 2));
        var differentAxis = new MaskedReductionAttrs(2, List.of(0, 2));
        var differentMapping = new MaskedReductionAttrs(1, List.of(1, 2));

        assertAll(
                () -> assertEquals(attrs, equal),
                () -> assertEquals(attrs.hashCode(), equal.hashCode()),
                () -> assertNotEquals(attrs, differentAxis),
                () -> assertNotEquals(attrs, differentMapping),
                () -> assertEquals(
                        "MaskedReductionAttrs[axis=1, maskInputAxes=[0, 2]]",
                        attrs.toString()));
    }

    @Test
    void composesSumAndMeanWithTheExactAttributesReference() {
        var attrs = new MaskedReductionAttrs(1, List.of(0, 1));
        var sum = new Operation(AggregateReductionKind.SUM, attrs);
        var mean = new Operation(AggregateReductionKind.MEAN, attrs);

        assertAll(
                () -> assertSame(AggregateReductionKind.SUM, sum.kind()),
                () -> assertSame(attrs, sum.attrs()),
                () -> assertSame(AggregateReductionKind.MEAN, mean.kind()),
                () -> assertSame(attrs, mean.attrs()));
    }

    @Test
    void containsNoTensorShapeOrExecutionState() {
        var componentTypes = Arrays.stream(MaskedReductionAttrs.class.getRecordComponents())
                .map(component -> component.getType().getName())
                .toList();

        assertAll(
                () -> assertEquals(List.of("int", "java.util.List"), componentTypes),
                () -> assertFalse(componentTypes.stream().anyMatch(name ->
                        name.contains("Tensor")
                                || name.contains("Shape")
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
