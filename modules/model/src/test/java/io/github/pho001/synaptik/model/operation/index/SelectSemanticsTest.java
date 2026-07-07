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
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.UnstackOutputAttrs;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class SelectSemanticsTest {
    @Test
    void declaresExactlyTheScalarSelectKind() {
        OperationKind kind = SelectKind.SELECT;

        assertAll(
                () -> assertArrayEquals(
                        new SelectKind[] {SelectKind.SELECT}, SelectKind.values()),
                () -> assertEquals("SELECT", kind.name()),
                () -> assertEquals("SELECT", kind.toString()),
                () -> assertSame(SelectKind.SELECT, SelectKind.valueOf("SELECT")),
                () -> assertInstanceOf(OperationKind.class, kind));
    }

    @Test
    void exposesOnlyTheExactEnumShape() {
        var constructors = SelectKind.class.getDeclaredConstructors();
        var fields = SelectKind.class.getDeclaredFields();
        var methods = SelectKind.class.getDeclaredMethods();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.index",
                        SelectKind.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(SelectKind.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(SelectKind.class.getModifiers())),
                () -> assertTrue(SelectKind.class.isEnum()),
                () -> assertEquals(
                        List.of(OperationKind.class),
                        Arrays.asList(SelectKind.class.getInterfaces())),
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
                                "valueOf(java.lang.String):io.github.pho001.synaptik.model.operation.index.SelectKind",
                                "values():[Lio.github.pho001.synaptik.model.operation.index.SelectKind;"),
                        Arrays.stream(methods)
                                .filter(method -> !method.isSynthetic())
                                .map(SelectSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertTrue(Arrays.stream(methods)
                        .filter(method -> !Modifier.isStatic(method.getModifiers()))
                        .findAny()
                        .isEmpty()),
                () -> assertEquals(0, SelectKind.class.getDeclaredClasses().length),
                () -> assertSame(SelectKind.class, SelectKind.SELECT.getClass()));
    }

    @Test
    void exposesOnlyTheExactAttributesRecordShape() {
        var components = SelectAttrs.class.getRecordComponents();
        var constructors = SelectAttrs.class.getDeclaredConstructors();
        var fields = SelectAttrs.class.getDeclaredFields();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.index",
                        SelectAttrs.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(SelectAttrs.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(SelectAttrs.class.getModifiers())),
                () -> assertTrue(SelectAttrs.class.isRecord()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(SelectAttrs.class.getInterfaces())),
                () -> assertEquals(
                        List.of("axis", "index"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(
                        List.of(int.class, long.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(
                        List.of(int.class, long.class),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(
                        List.of("axis", "index"),
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
                                "index():long",
                                "toString():java.lang.String"),
                        Arrays.stream(SelectAttrs.class.getDeclaredMethods())
                                .map(SelectSemanticsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, SelectAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void retainsZeroOrdinaryAndPrimitiveMaximumValuesUnchanged() {
        int[] axes = {0, 1, 37, Integer.MAX_VALUE};
        long[] indices = {0L, 2L, 91L, Long.MAX_VALUE};

        for (int position = 0; position < axes.length; position++) {
            var attrs = new SelectAttrs(axes[position], indices[position]);
            int expectedAxis = axes[position];
            long expectedIndex = indices[position];
            assertAll(
                    () -> assertEquals(expectedAxis, attrs.axis()),
                    () -> assertEquals(expectedIndex, attrs.index()));
        }
    }

    @Test
    void validatesAxisThenIndexWithExactTypesAndMessages() {
        assertAll(
                () -> assertIllegalFailure(
                        () -> new SelectAttrs(-1, -2L),
                        "axis must be non-negative: -1"),
                () -> assertIllegalFailure(
                        () -> new SelectAttrs(Integer.MIN_VALUE, Long.MIN_VALUE),
                        "axis must be non-negative: " + Integer.MIN_VALUE),
                () -> assertIllegalFailure(
                        () -> new SelectAttrs(0, -1L),
                        "index must be non-negative: -1"),
                () -> assertIllegalFailure(
                        () -> new SelectAttrs(Integer.MAX_VALUE, Long.MIN_VALUE),
                        "index must be non-negative: " + Long.MIN_VALUE));
    }

    @Test
    void usesGeneratedRecordValueSemanticsAndDiagnosticText() {
        var attrs = new SelectAttrs(1, 2L);
        var equal = new SelectAttrs(1, 2L);
        var otherAxis = new SelectAttrs(0, 2L);
        var otherIndex = new SelectAttrs(1, 1L);

        assertAll(
                () -> assertEquals(attrs, equal),
                () -> assertEquals(attrs.hashCode(), equal.hashCode()),
                () -> assertNotEquals(attrs, otherAxis),
                () -> assertNotEquals(attrs, otherIndex),
                () -> assertEquals("SelectAttrs[axis=1, index=2]", attrs.toString()));
    }

    @Test
    void composesSelectWithTheExactAttributesReference() {
        SelectAttrs attrs = new SelectAttrs(1, 2L);
        Operation operation = new Operation(SelectKind.SELECT, attrs);

        assertAll(
                () -> assertSame(SelectKind.SELECT, operation.kind()),
                () -> assertSame(attrs, operation.attrs()),
                () -> assertEquals(
                        new Operation(SelectKind.SELECT, new SelectAttrs(1, 2L)), operation));
    }

    @Test
    void remainsDistinctFromWhereUnstackAndSliceSemantics() {
        Operation select = new Operation(SelectKind.SELECT, new SelectAttrs(1, 2L));
        Operation where = new Operation(
                WhereSelectionKind.WHERE,
                io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE);
        Operation unstack = new Operation(
                TensorCompositionKind.UNSTACK, new UnstackOutputAttrs(1, 2));
        Operation slice = new Operation(
                SliceKind.SLICE,
                new io.github.pho001.synaptik.model.operation.layout.SliceAttrs(
                        List.of(2L), List.of(3L), List.of(1), List.of(1L)));

        assertAll(
                () -> assertNotEquals(select, where),
                () -> assertNotEquals(select, unstack),
                () -> assertNotEquals(select, slice),
                () -> assertNotEquals(SelectKind.SELECT, WhereSelectionKind.WHERE),
                () -> assertNotEquals(SelectKind.SELECT, TensorCompositionKind.UNSTACK),
                () -> assertNotEquals(SelectKind.SELECT, SliceKind.SLICE));
    }

    @Test
    void attributesContainOnlyPrimitiveStateWithoutCrossLayerTypes() {
        List<String> componentTypes = Arrays.stream(SelectAttrs.class.getRecordComponents())
                .map(component -> component.getType().getName())
                .toList();

        assertAll(
                () -> assertEquals(List.of("int", "long"), componentTypes),
                () -> assertFalse(componentTypes.stream().anyMatch(name ->
                        name.contains("Tensor")
                                || name.contains("Shape")
                                || name.contains("DataType")
                                || name.contains("layout")
                                || name.contains("provenance")
                                || name.contains("graph")
                                || name.contains("compiler")
                                || name.contains("planning")
                                || name.contains("prepare")
                                || name.contains("runtime")
                                || name.contains("backend"))));
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
}
