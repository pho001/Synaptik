package io.github.pho001.synaptik.model.operation.elementwise.selection;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class WhereSelectionKindTest {
    @Test
    void declaresExactlyTheRequiredVocabulary() {
        WhereSelectionKind[] values = WhereSelectionKind.values();

        assertAll(
                () -> assertArrayEquals(
                        new WhereSelectionKind[] {WhereSelectionKind.WHERE}, values),
                () -> assertEquals(
                        List.of("WHERE"),
                        Arrays.stream(values).map(WhereSelectionKind::name).toList()));
    }

    @Test
    void implementsOperationKindThroughInheritedEnumBehavior() {
        OperationKind kind = WhereSelectionKind.WHERE;

        assertAll(
                () -> assertInstanceOf(OperationKind.class, kind),
                () -> assertEquals("WHERE", kind.name()),
                () -> assertEquals("WHERE", kind.toString()),
                () -> assertSame(WhereSelectionKind.WHERE, WhereSelectionKind.valueOf("WHERE")),
                () -> assertEquals(WhereSelectionKind.WHERE, WhereSelectionKind.WHERE),
                () -> assertEquals(
                        WhereSelectionKind.WHERE.hashCode(),
                        WhereSelectionKind.WHERE.hashCode()));
    }

    @Test
    void exposesOnlyTheExactEnumShapeWithoutArityState() {
        var constructors = WhereSelectionKind.class.getDeclaredConstructors();
        var fields = WhereSelectionKind.class.getDeclaredFields();
        var methods = WhereSelectionKind.class.getDeclaredMethods();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.elementwise.selection",
                        WhereSelectionKind.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(WhereSelectionKind.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(WhereSelectionKind.class.getModifiers())),
                () -> assertTrue(WhereSelectionKind.class.isEnum()),
                () -> assertEquals(
                        List.of(OperationKind.class),
                        Arrays.asList(WhereSelectionKind.class.getInterfaces())),
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
                                "valueOf(java.lang.String):io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind",
                                "values():[Lio.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;"),
                        Arrays.stream(methods)
                                .filter(method -> !method.isSynthetic())
                                .map(WhereSelectionKindTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertTrue(Arrays.stream(methods)
                        .filter(method -> !Modifier.isStatic(method.getModifiers()))
                        .findAny()
                        .isEmpty()),
                () -> assertEquals(0, WhereSelectionKind.class.getDeclaredClasses().length),
                () -> assertTrue(Arrays.stream(WhereSelectionKind.values())
                        .allMatch(value -> value.getClass() == WhereSelectionKind.class)));
    }

    @Test
    void composesWithTheCanonicalNoAttributesValue() {
        Operation operation =
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE);

        assertAll(
                () -> assertSame(WhereSelectionKind.WHERE, operation.kind()),
                () -> assertSame(NoOperationAttrs.INSTANCE, operation.attrs()));
    }

    @Test
    void keepsEqualDiagnosticNamesTypedByTheirKindFamily() {
        OperationKind selection = WhereSelectionKind.WHERE;
        OperationKind other = OtherKind.WHERE;

        assertAll(
                () -> assertEquals(selection.name(), other.name()),
                () -> assertNotEquals(selection, other),
                () -> assertNotEquals(
                        new Operation(selection, NoOperationAttrs.INSTANCE),
                        new Operation(other, NoOperationAttrs.INSTANCE)));
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

    private enum OtherKind implements OperationKind {
        WHERE
    }
}
