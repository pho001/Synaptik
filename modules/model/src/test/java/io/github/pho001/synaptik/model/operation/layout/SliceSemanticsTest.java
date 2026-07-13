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
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class SliceSemanticsTest {
    @Test
    void declaresExactlyExtractionThenFunctionalUpdateKinds() {
        OperationKind slice = SliceKind.SLICE;
        OperationKind update = SliceKind.SLICE_UPDATE;

        assertAll(
                () -> assertArrayEquals(
                        new SliceKind[] {SliceKind.SLICE, SliceKind.SLICE_UPDATE},
                        SliceKind.values()),
                () -> assertEquals("SLICE", slice.name()),
                () -> assertEquals("SLICE_UPDATE", update.name()),
                () -> assertSame(SliceKind.SLICE, SliceKind.valueOf("SLICE")),
                () -> assertSame(
                        SliceKind.SLICE_UPDATE, SliceKind.valueOf("SLICE_UPDATE")),
                () -> assertInstanceOf(OperationKind.class, slice),
                () -> assertInstanceOf(OperationKind.class, update),
                () -> assertThrows(
                        IllegalArgumentException.class, () -> SliceKind.valueOf("FLIP")),
                () -> assertThrows(
                        IllegalArgumentException.class, () -> SliceKind.valueOf("CROP")),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> SliceKind.valueOf("SLICE_BACKWARD")));
    }

    @Test
    void exposesOnlyTheExactEnumAndRecordShapes() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(SliceKind.class);
        var components = SliceAttrs.class.getRecordComponents();
        var constructors = SliceAttrs.class.getDeclaredConstructors();
        var fields = SliceAttrs.class.getDeclaredFields();

        assertAll(
                () -> assertTrue(Modifier.isPublic(SliceAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(SliceAttrs.class.getModifiers())),
                () -> assertTrue(SliceAttrs.class.isRecord()),
                () -> assertEquals(List.of(OperationAttrs.class),
                        Arrays.asList(SliceAttrs.class.getInterfaces())),
                () -> assertEquals(List.of("starts", "lengths", "axes", "steps"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(List.of(List.class, List.class, List.class, List.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(List.of(List.class, List.class, List.class, List.class),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(List.of("starts", "lengths", "axes", "steps"),
                        Arrays.stream(fields).map(field -> field.getName()).toList()),
                () -> assertTrue(Arrays.stream(fields).allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers())
                                && !Modifier.isStatic(field.getModifiers()))),
                () -> assertEquals(
                        List.of(
                                "axes():java.util.List",
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "lengths():java.util.List",
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
    void cropAttributesExposeOnlyExactTwoShapeComponentsAndExplicitAccessors() {
        var components = CropToShapeAttrs.class.getRecordComponents();
        var constructors = CropToShapeAttrs.class.getDeclaredConstructors();
        var fields = CropToShapeAttrs.class.getDeclaredFields();

        assertAll(
                () -> assertTrue(Modifier.isPublic(CropToShapeAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(CropToShapeAttrs.class.getModifiers())),
                () -> assertTrue(CropToShapeAttrs.class.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(CropToShapeAttrs.class.getInterfaces())),
                () -> assertEquals(
                        List.of("targetShape", "prefixShape"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(
                        List.of(Shape.class, Shape.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        List.of(Shape.class, Shape.class),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(
                        List.of("targetShape", "prefixShape"),
                        Arrays.stream(fields).map(field -> field.getName()).toList()),
                () -> assertTrue(Arrays.stream(fields).allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers())
                                && !Modifier.isStatic(field.getModifiers()))),
                () -> assertEquals(
                        List.of(
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "prefixShape():io.github.pho001.synaptik.model.shape.Shape",
                                "targetShape():io.github.pho001.synaptik.model.shape.Shape",
                                "toString():java.lang.String"),
                        Arrays.stream(CropToShapeAttrs.class.getDeclaredMethods())
                                .map(SliceSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, CropToShapeAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void cropAttributesValidateNullsRetainExactShapesAndUseRecordValueSemantics() {
        Shape target = Shape.ofDimensions(new DynamicDimension("N"));
        Shape prefix = Shape.of(1);
        CropToShapeAttrs attrs = new CropToShapeAttrs(target, prefix);
        CropToShapeAttrs equal = new CropToShapeAttrs(target, prefix);
        CropToShapeAttrs different = new CropToShapeAttrs(target, Shape.of(0));

        assertAll(
                () -> assertEquals(
                        "targetShape",
                        assertThrows(
                                        NullPointerException.class,
                                        () -> new CropToShapeAttrs(null, null))
                                .getMessage()),
                () -> assertEquals(
                        "prefixShape",
                        assertThrows(
                                        NullPointerException.class,
                                        () -> new CropToShapeAttrs(target, null))
                                .getMessage()),
                () -> assertSame(target, attrs.targetShape()),
                () -> assertSame(prefix, attrs.prefixShape()),
                () -> assertEquals(attrs, equal),
                () -> assertEquals(attrs.hashCode(), equal.hashCode()),
                () -> assertNotEquals(attrs, different),
                () -> assertEquals(
                        "CropToShapeAttrs[targetShape=Shape[N], prefixShape=Shape[1]]",
                        attrs.toString()));
    }

    @Test
    void acceptsIdentityPositiveNegativeAndExtremeSignedSequences() {
        SliceAttrs identity = new SliceAttrs(List.of(), List.of(), List.of(), List.of());
        SliceAttrs general = new SliceAttrs(
                List.of(1L, 4L), List.of(2L, 5L), List.of(0, 1), List.of(2L, -1L));
        SliceAttrs minimumStep = new SliceAttrs(
                List.of(Long.MAX_VALUE), List.of(1L), List.of(Integer.MAX_VALUE),
                List.of(Long.MIN_VALUE));

        assertAll(
                () -> assertEquals(List.of(), identity.starts()),
                () -> assertEquals(List.of(), identity.lengths()),
                () -> assertEquals(List.of(1L, 4L), general.starts()),
                () -> assertEquals(List.of(2L, 5L), general.lengths()),
                () -> assertEquals(List.of(0, 1), general.axes()),
                () -> assertEquals(List.of(2L, -1L), general.steps()),
                () -> assertEquals(List.of(Long.MIN_VALUE), minimumStep.steps()));
    }

    @Test
    void snapshotsAllCallerListsAndExposesImmutableValues() {
        ArrayList<Long> starts = new ArrayList<>(List.of(1L, 4L));
        ArrayList<Long> lengths = new ArrayList<>(List.of(2L, 5L));
        ArrayList<Integer> axes = new ArrayList<>(List.of(0, 1));
        ArrayList<Long> steps = new ArrayList<>(List.of(2L, -1L));
        SliceAttrs attrs = new SliceAttrs(starts, lengths, axes, steps);

        starts.set(0, 2L);
        lengths.set(0, 1L);
        axes.set(0, 2);
        steps.set(0, 3L);

        assertAll(
                () -> assertEquals(List.of(1L, 4L), attrs.starts()),
                () -> assertEquals(List.of(2L, 5L), attrs.lengths()),
                () -> assertEquals(List.of(0, 1), attrs.axes()),
                () -> assertEquals(List.of(2L, -1L), attrs.steps()),
                () -> assertNotSame(starts, attrs.starts()),
                () -> assertNotSame(lengths, attrs.lengths()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> attrs.starts().set(0, 2L)),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> attrs.lengths().set(0, 2L)));
    }

    @Test
    void rejectsContainersSizesAndElementsInExactOrder() {
        assertNullFailure(() -> new SliceAttrs(null, null, null, null), "starts");
        assertNullFailure(() -> new SliceAttrs(List.of(), null, null, null), "lengths");
        assertNullFailure(() -> new SliceAttrs(List.of(), List.of(), null, null), "axes");
        assertNullFailure(() -> new SliceAttrs(List.of(), List.of(), List.of(), null), "steps");

        String sizes = "starts, lengths, axes, and steps must have matching sizes";
        assertIllegalFailure(
                () -> new SliceAttrs(List.of(), List.of(0L), List.of(), List.of()), sizes);
        assertNullFailure(() -> new SliceAttrs(
                Arrays.asList((Long) null), List.of(1L), List.of(0), List.of(1L)),
                "starts[0]");
        assertNullFailure(() -> new SliceAttrs(
                List.of(0L), Arrays.asList((Long) null), List.of(0), List.of(1L)),
                "lengths[0]");
        assertNullFailure(() -> new SliceAttrs(
                List.of(0L), List.of(1L), Arrays.asList((Integer) null), List.of(1L)),
                "axes[0]");
        assertNullFailure(() -> new SliceAttrs(
                List.of(0L), List.of(1L), List.of(0), Arrays.asList((Long) null)),
                "steps[0]");
    }

    @Test
    void enforcesCoordinateAxisStepAndCanonicalEmptyContractsInOrder() {
        assertIllegalFailure(
                () -> new SliceAttrs(List.of(-1L), List.of(-1L), List.of(-1), List.of(0L)),
                "starts[0] must be non-negative: -1");
        assertIllegalFailure(
                () -> new SliceAttrs(List.of(0L), List.of(-1L), List.of(-1), List.of(0L)),
                "lengths[0] must be non-negative: -1");
        assertIllegalFailure(
                () -> new SliceAttrs(List.of(0L), List.of(1L), List.of(-1), List.of(0L)),
                "axes[0] must be non-negative: -1");
        assertIllegalFailure(
                () -> new SliceAttrs(
                        List.of(0L, 0L), List.of(1L, 1L), List.of(0, 0), List.of(1L, 0L)),
                "axes contains duplicate axis 0 at index 1");
        assertIllegalFailure(
                () -> new SliceAttrs(List.of(0L), List.of(1L), List.of(0), List.of(0L)),
                "steps[0] must be non-zero: 0");
        assertIllegalFailure(
                () -> new SliceAttrs(List.of(2L), List.of(0L), List.of(0), List.of(-1L)),
                "starts[0] must be zero when lengths[0] is zero: 2");
        assertIllegalFailure(
                () -> new SliceAttrs(List.of(1L), List.of(3L), List.of(0), List.of(-1L)),
                "last slice coordinate at index 0 must be non-negative: -1");
    }

    @Test
    void propagatesCheckedLastCoordinateOverflow() {
        assertThrows(ArithmeticException.class, () -> new SliceAttrs(
                List.of(Long.MAX_VALUE), List.of(2L), List.of(0), List.of(1L)));
        assertThrows(ArithmeticException.class, () -> new SliceAttrs(
                List.of(0L), List.of(Long.MAX_VALUE), List.of(0), List.of(Long.MAX_VALUE)));
    }

    @Test
    void usesOrderedRecordValueSemanticsAndComposesExactly() {
        SliceAttrs attrs = new SliceAttrs(
                List.of(1L, 4L), List.of(2L, 5L), List.of(0, 1), List.of(2L, -1L));
        SliceAttrs equal = new SliceAttrs(
                List.of(1L, 4L), List.of(2L, 5L), List.of(0, 1), List.of(2L, -1L));
        SliceAttrs reordered = new SliceAttrs(
                List.of(4L, 1L), List.of(5L, 2L), List.of(1, 0), List.of(-1L, 2L));
        Operation operation = new Operation(SliceKind.SLICE, attrs);

        assertAll(
                () -> assertEquals(attrs, equal),
                () -> assertEquals(attrs.hashCode(), equal.hashCode()),
                () -> assertNotEquals(attrs, reordered),
                () -> assertSame(SliceKind.SLICE, operation.kind()),
                () -> assertSame(attrs, operation.attrs()),
                () -> assertEquals(
                        "SliceAttrs[starts=[1, 4], lengths=[2, 5], axes=[0, 1], steps=[2, -1]]",
                        attrs.toString()));
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
        assertEquals(expectedMessage,
                assertThrows(NullPointerException.class, construction).getMessage());
    }

    private static void assertIllegalFailure(
            org.junit.jupiter.api.function.Executable construction, String expectedMessage) {
        assertEquals(expectedMessage,
                assertThrows(IllegalArgumentException.class, construction).getMessage());
    }

    private static String methodSignature(java.lang.reflect.Method method) {
        String parameters = Arrays.stream(method.getParameterTypes())
                .map(Class::getName)
                .collect(java.util.stream.Collectors.joining(","));
        return method.getName() + "(" + parameters + "):" + method.getReturnType().getName();
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
