package io.github.pho001.synaptik.model.operation.layout;

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

class TensorCompositionSemanticsTest {
    @Test
    void declaresExactlyTheThreeOrderedCompositionKinds() {
        OperationKind concat = TensorCompositionKind.CONCAT;
        OperationKind stack = TensorCompositionKind.STACK;
        OperationKind unstack = TensorCompositionKind.UNSTACK;

        assertAll(
                () -> assertArrayEquals(
                        new TensorCompositionKind[] {
                            TensorCompositionKind.CONCAT,
                            TensorCompositionKind.STACK,
                            TensorCompositionKind.UNSTACK
                        },
                        TensorCompositionKind.values()),
                () -> assertEquals("CONCAT", concat.name()),
                () -> assertEquals("STACK", stack.name()),
                () -> assertEquals("UNSTACK", unstack.name()),
                () -> assertSame(
                        TensorCompositionKind.CONCAT,
                        TensorCompositionKind.valueOf("CONCAT")),
                () -> assertSame(
                        TensorCompositionKind.STACK,
                        TensorCompositionKind.valueOf("STACK")),
                () -> assertSame(
                        TensorCompositionKind.UNSTACK,
                        TensorCompositionKind.valueOf("UNSTACK")),
                () -> assertInstanceOf(OperationKind.class, concat),
                () -> assertNotEquals(concat, stack),
                () -> assertNotEquals(stack, unstack));
    }

    @Test
    void exposesOnlyTheExactEnumShape() {
        var constructors = TensorCompositionKind.class.getDeclaredConstructors();
        var fields = TensorCompositionKind.class.getDeclaredFields();
        var methods = TensorCompositionKind.class.getDeclaredMethods();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.layout",
                        TensorCompositionKind.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(TensorCompositionKind.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(TensorCompositionKind.class.getModifiers())),
                () -> assertTrue(TensorCompositionKind.class.isEnum()),
                () -> assertEquals(
                        List.of(OperationKind.class),
                        Arrays.asList(TensorCompositionKind.class.getInterfaces())),
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
                                "valueOf(java.lang.String):io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind",
                                "values():[Lio.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;"),
                        Arrays.stream(methods)
                                .filter(method -> !method.isSynthetic())
                                .map(TensorCompositionSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertTrue(Arrays.stream(methods)
                        .filter(method -> !Modifier.isStatic(method.getModifiers()))
                        .findAny()
                        .isEmpty()),
                () -> assertEquals(0, TensorCompositionKind.class.getDeclaredClasses().length),
                () -> assertSame(
                        TensorCompositionKind.class, TensorCompositionKind.CONCAT.getClass()),
                () -> assertSame(
                        TensorCompositionKind.class, TensorCompositionKind.STACK.getClass()),
                () -> assertSame(
                        TensorCompositionKind.class, TensorCompositionKind.UNSTACK.getClass()));
    }

    @Test
    void exposesOnlyTheExactRecordShapes() {
        assertRecordShape(
                CompositionAxisAttrs.class,
                List.of("axis"),
                List.of(int.class),
                List.of(
                        "axis():int",
                        "equals(java.lang.Object):boolean",
                        "hashCode():int",
                        "toString():java.lang.String"));
        assertRecordShape(
                UnstackOutputAttrs.class,
                List.of("axis", "outputIndex"),
                List.of(int.class, int.class),
                List.of(
                        "axis():int",
                        "equals(java.lang.Object):boolean",
                        "hashCode():int",
                        "outputIndex():int",
                        "toString():java.lang.String"));
    }

    @Test
    void retainsEveryNonNegativeCompositionAxis() {
        for (int axis : new int[] {0, 1, 37, Integer.MAX_VALUE}) {
            assertEquals(axis, new CompositionAxisAttrs(axis).axis());
        }
    }

    @Test
    void rejectsEveryRepresentativeNegativeCompositionAxisWithExactMessage() {
        for (int axis : new int[] {-1, -2, -37, Integer.MIN_VALUE}) {
            assertIllegalFailure(
                    () -> new CompositionAxisAttrs(axis),
                    "axis must be non-negative: " + axis);
        }
    }

    @Test
    void retainsEveryNonNegativeUnstackBoundary() {
        int[][] values = {
            {0, 0},
            {1, 2},
            {37, 91},
            {Integer.MAX_VALUE, Integer.MAX_VALUE}
        };

        for (int[] value : values) {
            var attrs = new UnstackOutputAttrs(value[0], value[1]);
            assertAll(
                    () -> assertEquals(value[0], attrs.axis()),
                    () -> assertEquals(value[1], attrs.outputIndex()));
        }
    }

    @Test
    void validatesUnstackComponentsInExactOrderWithExactMessages() {
        assertAll(
                () -> assertIllegalFailure(
                        () -> new UnstackOutputAttrs(-1, -2),
                        "axis must be non-negative: -1"),
                () -> assertIllegalFailure(
                        () -> new UnstackOutputAttrs(Integer.MIN_VALUE, -1),
                        "axis must be non-negative: " + Integer.MIN_VALUE),
                () -> assertIllegalFailure(
                        () -> new UnstackOutputAttrs(0, -1),
                        "outputIndex must be non-negative: -1"),
                () -> assertIllegalFailure(
                        () -> new UnstackOutputAttrs(1, Integer.MIN_VALUE),
                        "outputIndex must be non-negative: " + Integer.MIN_VALUE));
    }

    @Test
    void usesGeneratedRecordValueSemanticsAndDiagnosticText() {
        var axisAttrs = new CompositionAxisAttrs(2);
        var equalAxisAttrs = new CompositionAxisAttrs(2);
        var otherAxisAttrs = new CompositionAxisAttrs(1);
        var outputAttrs = new UnstackOutputAttrs(2, 1);
        var equalOutputAttrs = new UnstackOutputAttrs(2, 1);
        var otherAxisOutputAttrs = new UnstackOutputAttrs(1, 1);
        var otherIndexOutputAttrs = new UnstackOutputAttrs(2, 0);

        assertAll(
                () -> assertEquals(axisAttrs, equalAxisAttrs),
                () -> assertEquals(axisAttrs.hashCode(), equalAxisAttrs.hashCode()),
                () -> assertNotEquals(axisAttrs, otherAxisAttrs),
                () -> assertEquals("CompositionAxisAttrs[axis=2]", axisAttrs.toString()),
                () -> assertEquals(outputAttrs, equalOutputAttrs),
                () -> assertEquals(outputAttrs.hashCode(), equalOutputAttrs.hashCode()),
                () -> assertNotEquals(outputAttrs, otherAxisOutputAttrs),
                () -> assertNotEquals(outputAttrs, otherIndexOutputAttrs),
                () -> assertEquals(
                        "UnstackOutputAttrs[axis=2, outputIndex=1]", outputAttrs.toString()));
    }

    @Test
    void composesExactKindsWithExactAttributeReferences() {
        CompositionAxisAttrs axisAttrs = new CompositionAxisAttrs(1);
        UnstackOutputAttrs outputAttrs = new UnstackOutputAttrs(1, 2);
        Operation concat = new Operation(TensorCompositionKind.CONCAT, axisAttrs);
        Operation stack = new Operation(TensorCompositionKind.STACK, axisAttrs);
        Operation unstack = new Operation(TensorCompositionKind.UNSTACK, outputAttrs);

        assertAll(
                () -> assertSame(TensorCompositionKind.CONCAT, concat.kind()),
                () -> assertSame(axisAttrs, concat.attrs()),
                () -> assertSame(TensorCompositionKind.STACK, stack.kind()),
                () -> assertSame(axisAttrs, stack.attrs()),
                () -> assertSame(TensorCompositionKind.UNSTACK, unstack.kind()),
                () -> assertSame(outputAttrs, unstack.attrs()),
                () -> assertNotEquals(concat, stack),
                () -> assertNotEquals(stack, unstack));
    }

    @Test
    void distinctIndexedUnstackOutputsRemainSemanticallyDistinctWithoutGroupingState() {
        Operation first = new Operation(
                TensorCompositionKind.UNSTACK, new UnstackOutputAttrs(1, 0));
        Operation second = new Operation(
                TensorCompositionKind.UNSTACK, new UnstackOutputAttrs(1, 1));
        Operation third = new Operation(
                TensorCompositionKind.UNSTACK, new UnstackOutputAttrs(1, 2));

        assertAll(
                () -> assertNotEquals(first, second),
                () -> assertNotEquals(second, third),
                () -> assertNotEquals(first, third),
                () -> assertEquals(
                        List.of("int", "int"),
                        Arrays.stream(UnstackOutputAttrs.class.getRecordComponents())
                                .map(component -> component.getType().getName())
                                .toList()));
    }

    @Test
    void attributesContainOnlySpecifiedPrimitiveStateAndNoForbiddenDependencies() {
        List<String> axisComponentTypes = Arrays.stream(
                        CompositionAxisAttrs.class.getRecordComponents())
                .map(component -> component.getType().getName())
                .toList();
        List<String> outputComponentTypes = Arrays.stream(
                        UnstackOutputAttrs.class.getRecordComponents())
                .map(component -> component.getType().getName())
                .toList();

        assertAll(
                () -> assertEquals(List.of("int"), axisComponentTypes),
                () -> assertEquals(List.of("int", "int"), outputComponentTypes),
                () -> assertFalse(axisComponentTypes.stream()
                        .anyMatch(TensorCompositionSemanticsTest::isForbiddenComponentType)),
                () -> assertFalse(outputComponentTypes.stream()
                        .anyMatch(TensorCompositionSemanticsTest::isForbiddenComponentType)));
    }

    private static void assertRecordShape(
            Class<?> recordClass,
            List<String> componentNames,
            List<Class<?>> componentTypes,
            List<String> expectedMethods) {
        var components = recordClass.getRecordComponents();
        var constructors = recordClass.getDeclaredConstructors();
        var fields = recordClass.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.layout",
                        recordClass.getPackageName()),
                () -> assertTrue(Modifier.isPublic(recordClass.getModifiers())),
                () -> assertTrue(Modifier.isFinal(recordClass.getModifiers())),
                () -> assertTrue(recordClass.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(recordClass.getInterfaces())),
                () -> assertEquals(
                        componentNames,
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(
                        componentTypes,
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        componentTypes, Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(
                        componentNames,
                        Arrays.stream(fields).map(field -> field.getName()).toList()),
                () -> assertTrue(Arrays.stream(fields).allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers())
                                && !Modifier.isStatic(field.getModifiers()))),
                () -> assertEquals(
                        expectedMethods,
                        Arrays.stream(recordClass.getDeclaredMethods())
                                .map(TensorCompositionSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, recordClass.getDeclaredClasses().length));
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
                || name.contains("DataType")
                || name.contains("layout.Layout")
                || name.contains("provenance")
                || name.contains("graph")
                || name.contains("compiler")
                || name.contains("planning")
                || name.contains("prepare")
                || name.contains("runtime")
                || name.contains("backend")
                || name.contains("onnx")
                || name.contains("training");
    }
}
