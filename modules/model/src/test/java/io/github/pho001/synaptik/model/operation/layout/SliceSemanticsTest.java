package io.github.pho001.synaptik.model.operation.layout;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class SliceSemanticsTest {
    @Test
    void declaresExactlyTheSingleSliceKindWithoutSingleAxisAlias() {
        OperationKind slice = SliceKind.SLICE;

        assertAll(
                () -> assertArrayEquals(new SliceKind[] {SliceKind.SLICE}, SliceKind.values()),
                () -> assertEquals("SLICE", slice.name()),
                () -> assertSame(SliceKind.SLICE, SliceKind.valueOf("SLICE")),
                () -> assertInstanceOf(OperationKind.class, slice),
                () -> assertThrows(
                        IllegalArgumentException.class, () -> SliceKind.valueOf("SLICE_AXIS")));
    }

    @Test
    void exposesOnlyTheExactEnumShape() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(SliceKind.class);
    }

    @Test
    void exposesOnlyTheExactFourComponentRecordShape() {
        var components = SliceAttrs.class.getRecordComponents();
        var constructors = SliceAttrs.class.getDeclaredConstructors();
        var fields = SliceAttrs.class.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.layout",
                        SliceAttrs.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(SliceAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(SliceAttrs.class.getModifiers())),
                () -> assertTrue(SliceAttrs.class.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(SliceAttrs.class.getInterfaces())),
                () -> assertEquals(
                        List.of("starts", "ends", "axes", "steps"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(
                        List.of(List.class, List.class, List.class, List.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(
                        List.of(
                                "java.util.List<java.lang.Long>",
                                "java.util.List<java.lang.Long>",
                                "java.util.List<java.lang.Integer>",
                                "java.util.List<java.lang.Long>"),
                        Arrays.stream(components)
                                .map(component -> component.getGenericType().getTypeName())
                                .toList()),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        List.of(List.class, List.class, List.class, List.class),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(
                        List.of("starts", "ends", "axes", "steps"),
                        Arrays.stream(fields).map(field -> field.getName()).toList()),
                () -> assertTrue(Arrays.stream(fields).allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers())
                                && !Modifier.isStatic(field.getModifiers()))),
                () -> assertEquals(
                        List.of(
                                "axes():java.util.List",
                                "ends():java.util.List",
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "starts():java.util.List",
                                "steps():java.util.List",
                                "toString():java.lang.String"),
                        Arrays.stream(SliceAttrs.class.getDeclaredMethods())
                                .map(SliceSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, SliceAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void acceptsIdentitySingleAxisAndGeneralHalfOpenSlices() {
        SliceAttrs identity = new SliceAttrs(List.of(), List.of(), List.of(), List.of());
        SliceAttrs singleAxis =
                new SliceAttrs(List.of(1L), List.of(6L), List.of(1), List.of(1L));
        SliceAttrs general = new SliceAttrs(
                List.of(0L, 1L),
                List.of(3L, 6L),
                List.of(0, 1),
                List.of(1L, 2L));
        List<Long> selectedColumns = java.util.stream.LongStream.iterate(
                        general.starts().get(1),
                        coordinate -> coordinate < general.ends().get(1),
                        coordinate -> coordinate + general.steps().get(1))
                .boxed()
                .toList();

        assertAll(
                () -> assertEquals(List.of(), identity.starts()),
                () -> assertEquals(List.of(), identity.ends()),
                () -> assertEquals(List.of(), identity.axes()),
                () -> assertEquals(List.of(), identity.steps()),
                () -> assertEquals(List.of(1L), singleAxis.starts()),
                () -> assertEquals(List.of(6L), singleAxis.ends()),
                () -> assertEquals(List.of(1), singleAxis.axes()),
                () -> assertEquals(List.of(1L), singleAxis.steps()),
                () -> assertEquals(List.of(0L, 1L), general.starts()),
                () -> assertEquals(List.of(3L, 6L), general.ends()),
                () -> assertEquals(List.of(0, 1), general.axes()),
                () -> assertEquals(List.of(1L, 2L), general.steps()),
                () -> assertEquals(List.of(1L, 3L, 5L), selectedColumns));
    }

    @Test
    void snapshotsAllCallerListsAndExposesImmutableValues() {
        ArrayList<Long> callerStarts = new ArrayList<>(List.of(0L, 1L));
        ArrayList<Long> callerEnds = new ArrayList<>(List.of(3L, 6L));
        ArrayList<Integer> callerAxes = new ArrayList<>(List.of(0, 1));
        ArrayList<Long> callerSteps = new ArrayList<>(List.of(1L, 2L));
        SliceAttrs attrs = new SliceAttrs(callerStarts, callerEnds, callerAxes, callerSteps);

        callerStarts.set(0, 2L);
        callerEnds.set(0, 2L);
        callerAxes.set(0, 2);
        callerSteps.set(0, 3L);
        callerStarts.add(7L);
        callerEnds.add(8L);
        callerAxes.add(3);
        callerSteps.add(1L);

        assertAll(
                () -> assertEquals(List.of(0L, 1L), attrs.starts()),
                () -> assertEquals(List.of(3L, 6L), attrs.ends()),
                () -> assertEquals(List.of(0, 1), attrs.axes()),
                () -> assertEquals(List.of(1L, 2L), attrs.steps()),
                () -> assertNotSame(callerStarts, attrs.starts()),
                () -> assertNotSame(callerEnds, attrs.ends()),
                () -> assertNotSame(callerAxes, attrs.axes()),
                () -> assertNotSame(callerSteps, attrs.steps()),
                () -> assertThrows(
                        UnsupportedOperationException.class, () -> attrs.starts().set(0, 2L)),
                () -> assertThrows(
                        UnsupportedOperationException.class, () -> attrs.ends().set(0, 2L)),
                () -> assertThrows(
                        UnsupportedOperationException.class, () -> attrs.axes().set(0, 2)),
                () -> assertThrows(
                        UnsupportedOperationException.class, () -> attrs.steps().set(0, 3L)));
    }

    @Test
    void rejectsNullContainersInExactComponentOrder() {
        assertNullFailure(
                () -> new SliceAttrs(null, null, null, null), "starts");
        assertNullFailure(
                () -> new SliceAttrs(List.of(), null, null, null), "ends");
        assertNullFailure(
                () -> new SliceAttrs(List.of(), List.of(), null, null), "axes");
        assertNullFailure(
                () -> new SliceAttrs(List.of(), List.of(), List.of(), null), "steps");
    }

    @Test
    void rejectsMismatchedSizesBeforeInspectingElements() {
        String expected = "starts, ends, axes, and steps must have matching sizes";

        assertAll(
                () -> assertIllegalFailure(
                        () -> new SliceAttrs(List.of(), List.of(0L), List.of(), List.of()),
                        expected),
                () -> assertIllegalFailure(
                        () -> new SliceAttrs(List.of(0L), List.of(), List.of(0), List.of(1L)),
                        expected),
                () -> assertIllegalFailure(
                        () -> new SliceAttrs(List.of(0L), List.of(1L), List.of(), List.of(1L)),
                        expected),
                () -> assertIllegalFailure(
                        () -> new SliceAttrs(List.of(0L), List.of(1L), List.of(0), List.of()),
                        expected),
                () -> assertIllegalFailure(
                        () -> new SliceAttrs(
                                Arrays.asList((Long) null),
                                List.of(),
                                List.of(),
                                List.of()),
                        expected));
    }

    @Test
    void rejectsNullElementsWithExactIndexedMessagesAndEntryOrder() {
        assertNullFailure(
                () -> new SliceAttrs(
                        Arrays.asList((Long) null), List.of(1L), List.of(0), List.of(1L)),
                "starts[0]");
        assertNullFailure(
                () -> new SliceAttrs(
                        List.of(0L), Arrays.asList((Long) null), List.of(0), List.of(1L)),
                "ends[0]");
        assertNullFailure(
                () -> new SliceAttrs(
                        List.of(0L), List.of(1L), Arrays.asList((Integer) null), List.of(1L)),
                "axes[0]");
        assertNullFailure(
                () -> new SliceAttrs(
                        List.of(0L), List.of(1L), List.of(0), Arrays.asList((Long) null)),
                "steps[0]");
        assertNullFailure(
                () -> new SliceAttrs(
                        List.of(0L, 1L),
                        List.of(1L, 2L),
                        List.of(0, 1),
                        Arrays.asList(1L, null)),
                "steps[1]");
    }

    @Test
    void rejectsCoordinatesAxesDuplicatesAndStepsWithExactMessages() {
        assertIllegalFailure(
                () -> new SliceAttrs(List.of(-1L), List.of(1L), List.of(0), List.of(1L)),
                "starts[0] must be non-negative: -1");
        assertIllegalFailure(
                () -> new SliceAttrs(List.of(0L), List.of(-2L), List.of(0), List.of(1L)),
                "ends[0] must be non-negative: -2");
        assertIllegalFailure(
                () -> new SliceAttrs(List.of(0L), List.of(1L), List.of(-3), List.of(1L)),
                "axes[0] must be non-negative: -3");
        assertIllegalFailure(
                () -> new SliceAttrs(
                        List.of(0L, 1L),
                        List.of(2L, 3L),
                        List.of(2, 2),
                        List.of(1L, 1L)),
                "axes contains duplicate axis 2 at index 1");
        assertIllegalFailure(
                () -> new SliceAttrs(List.of(0L), List.of(1L), List.of(0), List.of(0L)),
                "steps[0] must be positive: 0");
        assertIllegalFailure(
                () -> new SliceAttrs(List.of(0L), List.of(1L), List.of(0), List.of(-4L)),
                "steps[0] must be positive: -4");
    }

    @Test
    void reportsTheFirstFailureInTheSpecifiedElementValidationOrder() {
        assertNullFailure(
                () -> new SliceAttrs(
                        List.of(-1L),
                        List.of(-1L),
                        List.of(-1),
                        Arrays.asList((Long) null)),
                "steps[0]");
        assertIllegalFailure(
                () -> new SliceAttrs(List.of(-1L), List.of(-1L), List.of(-1), List.of(0L)),
                "starts[0] must be non-negative: -1");
        assertIllegalFailure(
                () -> new SliceAttrs(List.of(0L), List.of(-1L), List.of(-1), List.of(0L)),
                "ends[0] must be non-negative: -1");
        assertIllegalFailure(
                () -> new SliceAttrs(List.of(0L), List.of(1L), List.of(-1), List.of(0L)),
                "axes[0] must be non-negative: -1");
        assertIllegalFailure(
                () -> new SliceAttrs(
                        List.of(0L, 0L),
                        List.of(1L, 1L),
                        List.of(0, 0),
                        List.of(1L, 0L)),
                "axes contains duplicate axis 0 at index 1");
        assertIllegalFailure(
                () -> new SliceAttrs(
                        List.of(0L, -1L),
                        List.of(1L, 1L),
                        List.of(0, 1),
                        List.of(0L, 1L)),
                "steps[0] must be positive: 0");
    }

    @Test
    void acceptsExtremeValuesAndEveryNonNegativeStartEndRelationship() {
        SliceAttrs extremes = new SliceAttrs(
                List.of(Long.MAX_VALUE),
                List.of(Long.MAX_VALUE),
                List.of(Integer.MAX_VALUE),
                List.of(Long.MAX_VALUE));
        SliceAttrs equal = new SliceAttrs(List.of(4L), List.of(4L), List.of(0), List.of(1L));
        SliceAttrs descending =
                new SliceAttrs(List.of(7L), List.of(2L), List.of(1), List.of(3L));

        assertAll(
                () -> assertEquals(List.of(Long.MAX_VALUE), extremes.starts()),
                () -> assertEquals(List.of(Long.MAX_VALUE), extremes.ends()),
                () -> assertEquals(List.of(Integer.MAX_VALUE), extremes.axes()),
                () -> assertEquals(List.of(Long.MAX_VALUE), extremes.steps()),
                () -> assertEquals(List.of(4L), equal.starts()),
                () -> assertEquals(List.of(4L), equal.ends()),
                () -> assertEquals(List.of(7L), descending.starts()),
                () -> assertEquals(List.of(2L), descending.ends()));
    }

    @Test
    void usesOrderedRecordValueSemanticsAndDiagnosticText() {
        SliceAttrs attrs = new SliceAttrs(
                List.of(0L, 1L), List.of(3L, 6L), List.of(0, 1), List.of(1L, 2L));
        SliceAttrs equal = new SliceAttrs(
                List.of(0L, 1L), List.of(3L, 6L), List.of(0, 1), List.of(1L, 2L));
        SliceAttrs reordered = new SliceAttrs(
                List.of(1L, 0L), List.of(6L, 3L), List.of(1, 0), List.of(2L, 1L));
        SliceAttrs differentStep = new SliceAttrs(
                List.of(0L, 1L), List.of(3L, 6L), List.of(0, 1), List.of(1L, 3L));

        assertAll(
                () -> assertEquals(attrs, equal),
                () -> assertEquals(attrs.hashCode(), equal.hashCode()),
                () -> assertNotEquals(attrs, reordered),
                () -> assertNotEquals(attrs, differentStep),
                () -> assertEquals(
                        "SliceAttrs[starts=[0, 1], ends=[3, 6], axes=[0, 1], steps=[1, 2]]",
                        attrs.toString()));
    }

    @Test
    void composesExactlyAndRepresentsSingleAxisConvenienceWithOneStepOneEntry() {
        SliceAttrs attrs = new SliceAttrs(
                List.of(0L, 1L), List.of(3L, 6L), List.of(0, 1), List.of(1L, 2L));
        Operation operation = new Operation(SliceKind.SLICE, attrs);
        long fromInclusive = 2L;
        long toExclusive = 7L;
        int normalizedAxis = 3;
        SliceAttrs singleAxis = new SliceAttrs(
                List.of(fromInclusive),
                List.of(toExclusive),
                List.of(normalizedAxis),
                List.of(1L));
        Operation singleAxisOperation = new Operation(SliceKind.SLICE, singleAxis);

        assertAll(
                () -> assertSame(SliceKind.SLICE, operation.kind()),
                () -> assertSame(attrs, operation.attrs()),
                () -> assertSame(SliceKind.SLICE, singleAxisOperation.kind()),
                () -> assertSame(singleAxis, singleAxisOperation.attrs()),
                () -> assertEquals(List.of(2L), singleAxis.starts()),
                () -> assertEquals(List.of(7L), singleAxis.ends()),
                () -> assertEquals(List.of(3), singleAxis.axes()),
                () -> assertEquals(List.of(1L), singleAxis.steps()));
    }

    @Test
    void attributesContainOnlyParallelJavaBaseState() {
        List<String> componentTypes = Arrays.stream(SliceAttrs.class.getRecordComponents())
                .map(component -> component.getGenericType().getTypeName())
                .toList();

        assertAll(
                () -> assertEquals(
                        List.of(
                                "java.util.List<java.lang.Long>",
                                "java.util.List<java.lang.Long>",
                                "java.util.List<java.lang.Integer>",
                                "java.util.List<java.lang.Long>"),
                        componentTypes),
                () -> assertFalse(componentTypes.stream()
                        .anyMatch(SliceSemanticsTest::isForbiddenComponentType)));
    }

    private static void assertNullFailure(
            org.junit.jupiter.api.function.Executable construction, String expectedMessage) {
        NullPointerException failure = assertThrows(NullPointerException.class, construction);

        assertEquals(expectedMessage, failure.getMessage());
    }

    private static void assertIllegalFailure(
            org.junit.jupiter.api.function.Executable construction, String expectedMessage) {
        IllegalArgumentException failure =
                assertThrows(IllegalArgumentException.class, construction);

        assertEquals(expectedMessage, failure.getMessage());
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
                || name.contains("Shape")
                || name.contains("layout.Layout")
                || name.contains("storage")
                || name.contains("graph")
                || name.contains("compiler")
                || name.contains("planning")
                || name.contains("prepare")
                || name.contains("runtime")
                || name.contains("backend")
                || name.contains("training");
    }
}
