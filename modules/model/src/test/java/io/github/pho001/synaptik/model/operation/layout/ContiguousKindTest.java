package io.github.pho001.synaptik.model.operation.layout;

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

class ContiguousKindTest {
    @Test
    void declaresExactlyTheRequiredVocabularyInOrder() {
        ContiguousKind[] values = ContiguousKind.values();

        assertAll(
                () -> assertArrayEquals(
                        new ContiguousKind[] {ContiguousKind.CONTIGUOUS}, values),
                () -> assertEquals(
                        List.of("CONTIGUOUS"),
                        Arrays.stream(values).map(ContiguousKind::name).toList()));
    }

    @Test
    void implementsOperationKindThroughInheritedEnumBehavior() {
        OperationKind kind = ContiguousKind.CONTIGUOUS;

        assertAll(
                () -> assertInstanceOf(OperationKind.class, kind),
                () -> assertEquals("CONTIGUOUS", kind.name()),
                () -> assertEquals("CONTIGUOUS", kind.toString()),
                () -> assertSame(
                        ContiguousKind.CONTIGUOUS,
                        ContiguousKind.valueOf("CONTIGUOUS")),
                () -> assertEquals(ContiguousKind.CONTIGUOUS, ContiguousKind.CONTIGUOUS),
                () -> assertEquals(
                        ContiguousKind.CONTIGUOUS.hashCode(),
                        ContiguousKind.CONTIGUOUS.hashCode()));
    }

    @Test
    void exposesOnlyTheExactParameterlessEnumShape() {
        var constructors = ContiguousKind.class.getDeclaredConstructors();
        var fields = ContiguousKind.class.getDeclaredFields();
        var methods = ContiguousKind.class.getDeclaredMethods();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.model.operation.layout",
                        ContiguousKind.class.getPackageName()),
                () -> assertTrue(Modifier.isPublic(ContiguousKind.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(ContiguousKind.class.getModifiers())),
                () -> assertTrue(ContiguousKind.class.isEnum()),
                () -> assertEquals(
                        List.of(OperationKind.class),
                        Arrays.asList(ContiguousKind.class.getInterfaces())),
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
                                "valueOf(java.lang.String):io.github.pho001.synaptik.model.operation.layout.ContiguousKind",
                                "values():[Lio.github.pho001.synaptik.model.operation.layout.ContiguousKind;"),
                        Arrays.stream(methods)
                                .filter(method -> !method.isSynthetic())
                                .map(ContiguousKindTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertTrue(Arrays.stream(methods)
                        .filter(method -> !Modifier.isStatic(method.getModifiers()))
                        .findAny()
                        .isEmpty()),
                () -> assertEquals(0, ContiguousKind.class.getDeclaredClasses().length),
                () -> assertSame(
                        ContiguousKind.class,
                        ContiguousKind.CONTIGUOUS.getClass()));
    }

    @Test
    void composesWithTheCanonicalNoAttributesValue() {
        Operation operation =
                new Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE);

        assertAll(
                () -> assertSame(ContiguousKind.CONTIGUOUS, operation.kind()),
                () -> assertSame(NoOperationAttrs.INSTANCE, operation.attrs()));
    }

    @Test
    void keepsEqualDiagnosticNamesTypedByTheirKindFamily() {
        OperationKind contiguous = ContiguousKind.CONTIGUOUS;
        OperationKind other = OtherKind.CONTIGUOUS;

        assertAll(
                () -> assertEquals(contiguous.name(), other.name()),
                () -> assertNotEquals(contiguous, other),
                () -> assertNotEquals(
                        new Operation(contiguous, NoOperationAttrs.INSTANCE),
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
        CONTIGUOUS
    }
}
