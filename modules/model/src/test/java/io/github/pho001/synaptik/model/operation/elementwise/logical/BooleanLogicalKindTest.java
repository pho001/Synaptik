package io.github.pho001.synaptik.model.operation.elementwise.logical;

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

class BooleanLogicalKindTest {
    @Test
    void declaresExactlyTheRequiredVocabularyInOrder() {
        BooleanLogicalKind[] values = BooleanLogicalKind.values();

        assertAll(
                () -> assertArrayEquals(
                        new BooleanLogicalKind[] {
                            BooleanLogicalKind.AND,
                            BooleanLogicalKind.OR,
                            BooleanLogicalKind.NOT
                        },
                        values),
                () -> assertEquals(
                        List.of("AND", "OR", "NOT"),
                        Arrays.stream(values).map(BooleanLogicalKind::name).toList()));
    }

    @Test
    void implementsOperationKindThroughInheritedEnumBehavior() {
        OperationKind kind = BooleanLogicalKind.AND;

        assertAll(
                () -> assertInstanceOf(OperationKind.class, kind),
                () -> assertEquals("AND", kind.name()),
                () -> assertEquals("AND", kind.toString()),
                () -> assertSame(BooleanLogicalKind.AND, BooleanLogicalKind.valueOf("AND")),
                () -> assertEquals(BooleanLogicalKind.AND, BooleanLogicalKind.AND),
                () -> assertEquals(
                        BooleanLogicalKind.AND.hashCode(), BooleanLogicalKind.AND.hashCode()),
                () -> assertNotEquals(BooleanLogicalKind.AND, BooleanLogicalKind.OR));
    }

    @Test
    void exposesOnlyTheExactEnumShapeWithoutArityState() {
        var constructors = BooleanLogicalKind.class.getDeclaredConstructors();
        var fields = BooleanLogicalKind.class.getDeclaredFields();
        var methods = BooleanLogicalKind.class.getDeclaredMethods();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.elementwise.logical",
                        BooleanLogicalKind.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(BooleanLogicalKind.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(BooleanLogicalKind.class.getModifiers())),
                () -> assertTrue(BooleanLogicalKind.class.isEnum()),
                () -> assertEquals(
                        List.of(OperationKind.class),
                        Arrays.asList(BooleanLogicalKind.class.getInterfaces())),
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
                                "valueOf(java.lang.String):io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind",
                                "values():[Lio.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;"),
                        Arrays.stream(methods)
                                .filter(method -> !method.isSynthetic())
                                .map(BooleanLogicalKindTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertTrue(Arrays.stream(methods)
                        .filter(method -> !Modifier.isStatic(method.getModifiers()))
                        .findAny()
                        .isEmpty()),
                () -> assertEquals(0, BooleanLogicalKind.class.getDeclaredClasses().length),
                () -> assertTrue(Arrays.stream(BooleanLogicalKind.values())
                        .allMatch(value -> value.getClass() == BooleanLogicalKind.class)));
    }

    @Test
    void composesEveryKindWithTheCanonicalNoAttributesValue() {
        for (BooleanLogicalKind kind : BooleanLogicalKind.values()) {
            Operation operation = new Operation(kind, NoOperationAttrs.INSTANCE);

            assertAll(
                    () -> assertSame(kind, operation.kind()),
                    () -> assertSame(NoOperationAttrs.INSTANCE, operation.attrs()));
        }
    }

    @Test
    void keepsEqualDiagnosticNamesTypedByTheirKindFamily() {
        OperationKind logical = BooleanLogicalKind.AND;
        OperationKind other = OtherKind.AND;

        assertAll(
                () -> assertEquals(logical.name(), other.name()),
                () -> assertNotEquals(logical, other),
                () -> assertNotEquals(
                        new Operation(logical, NoOperationAttrs.INSTANCE),
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
        AND
    }
}
