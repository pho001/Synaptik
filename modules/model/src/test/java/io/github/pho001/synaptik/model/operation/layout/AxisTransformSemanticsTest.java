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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class AxisTransformSemanticsTest {
    @Test
    void declaresExactlyTheThreeOrderedKindsWithoutTranspose() {
        OperationKind permute = AxisTransformKind.PERMUTE;
        OperationKind expandDims = AxisTransformKind.EXPAND_DIMS;
        OperationKind squeeze = AxisTransformKind.SQUEEZE;

        assertAll(
                () -> assertArrayEquals(
                        new AxisTransformKind[] {
                            AxisTransformKind.PERMUTE,
                            AxisTransformKind.EXPAND_DIMS,
                            AxisTransformKind.SQUEEZE
                        },
                        AxisTransformKind.values()),
                () -> assertEquals("PERMUTE", permute.name()),
                () -> assertEquals("EXPAND_DIMS", expandDims.name()),
                () -> assertEquals("SQUEEZE", squeeze.name()),
                () -> assertSame(AxisTransformKind.PERMUTE, AxisTransformKind.valueOf("PERMUTE")),
                () -> assertSame(
                        AxisTransformKind.EXPAND_DIMS,
                        AxisTransformKind.valueOf("EXPAND_DIMS")),
                () -> assertSame(AxisTransformKind.SQUEEZE, AxisTransformKind.valueOf("SQUEEZE")),
                () -> assertInstanceOf(OperationKind.class, permute),
                () -> assertNotEquals(permute, expandDims),
                () -> assertNotEquals(expandDims, squeeze),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> AxisTransformKind.valueOf("TRANSPOSE")));
    }

    @Test
    void exposesOnlyTheExactEnumShape() {
        var constructors = AxisTransformKind.class.getDeclaredConstructors();
        var fields = AxisTransformKind.class.getDeclaredFields();
        var methods = AxisTransformKind.class.getDeclaredMethods();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.layout",
                        AxisTransformKind.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(AxisTransformKind.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(AxisTransformKind.class.getModifiers())),
                () -> assertTrue(AxisTransformKind.class.isEnum()),
                () -> assertEquals(
                        List.of(OperationKind.class),
                        Arrays.asList(AxisTransformKind.class.getInterfaces())),
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
                                "valueOf(java.lang.String):io.github.pho001.synaptik.model.operation.layout.AxisTransformKind",
                                "values():[Lio.github.pho001.synaptik.model.operation.layout.AxisTransformKind;"),
                        Arrays.stream(methods)
                                .filter(method -> !method.isSynthetic())
                                .map(AxisTransformSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertTrue(Arrays.stream(methods)
                        .filter(method -> !Modifier.isStatic(method.getModifiers()))
                        .findAny()
                        .isEmpty()),
                () -> assertEquals(0, AxisTransformKind.class.getDeclaredClasses().length),
                () -> assertSame(AxisTransformKind.class, AxisTransformKind.PERMUTE.getClass()),
                () -> assertSame(
                        AxisTransformKind.class, AxisTransformKind.EXPAND_DIMS.getClass()),
                () -> assertSame(AxisTransformKind.class, AxisTransformKind.SQUEEZE.getClass()));
    }

    @Test
    void exposesOnlyTheTwoExactRecordShapes() {
        assertRecordShape(PermutationAttrs.class, "axes", List.class);
        assertRecordShape(AxisTransformAttrs.class, "axis", int.class);

        assertAll(
                () -> assertEquals(
                        List.of(
                                "axes():java.util.List",
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "toString():java.lang.String"),
                        Arrays.stream(PermutationAttrs.class.getDeclaredMethods())
                                .map(AxisTransformSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(
                        List.of(
                                "axis():int",
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "toString():java.lang.String"),
                        Arrays.stream(AxisTransformAttrs.class.getDeclaredMethods())
                                .map(AxisTransformSemanticsTest::methodSignature)
                                .sorted()
                                .toList()));
    }

    @Test
    void acceptsScalarIdentityTransposeAndGeneralOutputToInputPermutations() {
        PermutationAttrs scalar = new PermutationAttrs(List.of());
        PermutationAttrs identity = new PermutationAttrs(List.of(0, 1, 2));
        PermutationAttrs transpose = new PermutationAttrs(List.of(1, 0));
        PermutationAttrs general = new PermutationAttrs(List.of(1, 0, 2));
        List<String> inputAxes = List.of("rows", "columns", "channels");

        assertAll(
                () -> assertEquals(List.of(), scalar.axes()),
                () -> assertEquals(List.of(0, 1, 2), identity.axes()),
                () -> assertEquals(List.of(1, 0), transpose.axes()),
                () -> assertEquals(List.of(1, 0, 2), general.axes()),
                () -> assertEquals(
                        List.of("columns", "rows", "channels"),
                        general.axes().stream().map(inputAxes::get).toList()));
    }

    @Test
    void snapshotsTheCallerListOnceAndExposesAnImmutableValue() {
        ArrayList<Integer> callerAxes = new ArrayList<>(List.of(2, 0, 1));
        PermutationAttrs attrs = new PermutationAttrs(callerAxes);

        callerAxes.set(0, 0);
        callerAxes.add(3);

        assertAll(
                () -> assertEquals(List.of(2, 0, 1), attrs.axes()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> attrs.axes().set(0, 0)),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> attrs.axes().add(3)));
    }

    @Test
    void rejectsNullPermutationStateWithExactIndexedMessages() {
        NullPointerException nullList =
                assertThrows(NullPointerException.class, () -> new PermutationAttrs(null));
        NullPointerException nullFirst = assertThrows(
                NullPointerException.class,
                () -> new PermutationAttrs(Arrays.asList(null, 0)));
        NullPointerException nullLater = assertThrows(
                NullPointerException.class,
                () -> new PermutationAttrs(Arrays.asList(0, null)));

        assertAll(
                () -> assertEquals("axes", nullList.getMessage()),
                () -> assertEquals("axes[0]", nullFirst.getMessage()),
                () -> assertEquals("axes[1]", nullLater.getMessage()));
    }

    @Test
    void rejectsNegativeOutOfRangeAndDuplicateAxesWithExactMessages() {
        assertPermutationFailure(
                List.of(-1), "axes[0] must be non-negative: -1");
        assertPermutationFailure(
                List.of(0, -2), "axes[1] must be non-negative: -2");
        assertPermutationFailure(
                List.of(1), "axes[0] must be less than permutation rank 1: 1");
        assertPermutationFailure(
                List.of(0, 2), "axes[1] must be less than permutation rank 2: 2");
        assertPermutationFailure(
                List.of(0, 0), "axes contains duplicate axis 0 at index 1");
        assertPermutationFailure(
                List.of(2, 0, 2), "axes contains duplicate axis 2 at index 2");
    }

    @Test
    void reportsTheFirstFailureInTheSpecifiedValidationOrder() {
        NullPointerException nullBeforeLaterNegative = assertThrows(
                NullPointerException.class,
                () -> new PermutationAttrs(Arrays.asList(null, -1)));
        IllegalArgumentException negativeBeforeLaterNull = assertThrows(
                IllegalArgumentException.class,
                () -> new PermutationAttrs(Arrays.asList(-1, null)));
        IllegalArgumentException rangeBeforeLaterNull = assertThrows(
                IllegalArgumentException.class,
                () -> new PermutationAttrs(Arrays.asList(2, null)));
        IllegalArgumentException duplicateBeforeLaterNegative = assertThrows(
                IllegalArgumentException.class,
                () -> new PermutationAttrs(List.of(0, 0, -1)));

        assertAll(
                () -> assertEquals("axes[0]", nullBeforeLaterNegative.getMessage()),
                () -> assertEquals(
                        "axes[0] must be non-negative: -1",
                        negativeBeforeLaterNull.getMessage()),
                () -> assertEquals(
                        "axes[0] must be less than permutation rank 2: 2",
                        rangeBeforeLaterNull.getMessage()),
                () -> assertEquals(
                        "axes contains duplicate axis 0 at index 1",
                        duplicateBeforeLaterNegative.getMessage()));
    }

    @Test
    void acceptsAllRepresentativeNonNegativeAxesAndRejectsNegativesExactly() {
        for (int axis : new int[] {0, 1, 37, Integer.MAX_VALUE}) {
            assertEquals(axis, new AxisTransformAttrs(axis).axis());
        }

        for (int axis : new int[] {-1, -2, -37, Integer.MIN_VALUE}) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new AxisTransformAttrs(axis));

            assertEquals("axis must be non-negative: " + axis, failure.getMessage());
        }
    }

    @Test
    void usesRecordValueSemanticsAndExactTypedOperationComposition() {
        PermutationAttrs permutation = new PermutationAttrs(List.of(1, 0));
        PermutationAttrs equalPermutation = new PermutationAttrs(List.of(1, 0));
        PermutationAttrs differentPermutation = new PermutationAttrs(List.of(0, 1));
        AxisTransformAttrs axis = new AxisTransformAttrs(1);
        AxisTransformAttrs equalAxis = new AxisTransformAttrs(1);
        AxisTransformAttrs differentAxis = new AxisTransformAttrs(0);
        Operation permute = new Operation(AxisTransformKind.PERMUTE, permutation);
        Operation expandDims = new Operation(AxisTransformKind.EXPAND_DIMS, axis);
        Operation squeeze = new Operation(AxisTransformKind.SQUEEZE, axis);

        assertAll(
                () -> assertEquals(permutation, equalPermutation),
                () -> assertEquals(permutation.hashCode(), equalPermutation.hashCode()),
                () -> assertNotEquals(permutation, differentPermutation),
                () -> assertEquals("PermutationAttrs[axes=[1, 0]]", permutation.toString()),
                () -> assertEquals(axis, equalAxis),
                () -> assertEquals(axis.hashCode(), equalAxis.hashCode()),
                () -> assertNotEquals(axis, differentAxis),
                () -> assertEquals("AxisTransformAttrs[axis=1]", axis.toString()),
                () -> assertSame(AxisTransformKind.PERMUTE, permute.kind()),
                () -> assertSame(permutation, permute.attrs()),
                () -> assertSame(AxisTransformKind.EXPAND_DIMS, expandDims.kind()),
                () -> assertSame(axis, expandDims.attrs()),
                () -> assertSame(AxisTransformKind.SQUEEZE, squeeze.kind()),
                () -> assertSame(axis, squeeze.attrs()),
                () -> assertNotEquals(expandDims, squeeze));
    }

    @Test
    void attributesContainOnlyTheirIntrinsicJavaBaseState() {
        List<String> permutationComponentTypes = Arrays.stream(
                        PermutationAttrs.class.getRecordComponents())
                .map(component -> component.getGenericType().getTypeName())
                .toList();
        List<String> axisComponentTypes = Arrays.stream(AxisTransformAttrs.class.getRecordComponents())
                .map(component -> component.getType().getName())
                .toList();

        assertAll(
                () -> assertEquals(
                        List.of("java.util.List<java.lang.Integer>"),
                        permutationComponentTypes),
                () -> assertEquals(List.of("int"), axisComponentTypes),
                () -> assertFalse(permutationComponentTypes.stream().anyMatch(
                        AxisTransformSemanticsTest::isForbiddenComponentType)),
                () -> assertFalse(axisComponentTypes.stream().anyMatch(
                        AxisTransformSemanticsTest::isForbiddenComponentType)));
    }

    private static void assertRecordShape(
            Class<?> recordType, String componentName, Class<?> componentType) {
        var components = recordType.getRecordComponents();
        var constructors = recordType.getDeclaredConstructors();
        var fields = recordType.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.layout",
                        recordType.getPackageName()),
                () -> assertTrue(Modifier.isPublic(recordType.getModifiers())),
                () -> assertTrue(Modifier.isFinal(recordType.getModifiers())),
                () -> assertTrue(recordType.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(recordType.getInterfaces())),
                () -> assertEquals(1, components.length),
                () -> assertEquals(componentName, components[0].getName()),
                () -> assertEquals(componentType, components[0].getType()),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        List.of(componentType),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(
                        List.of(componentName),
                        Arrays.stream(fields).map(field -> field.getName()).toList()),
                () -> assertTrue(Arrays.stream(fields).allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers())
                                && !Modifier.isStatic(field.getModifiers()))),
                () -> assertEquals(0, recordType.getDeclaredClasses().length));
    }

    private static void assertPermutationFailure(List<Integer> axes, String expectedMessage) {
        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> new PermutationAttrs(axes));

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
