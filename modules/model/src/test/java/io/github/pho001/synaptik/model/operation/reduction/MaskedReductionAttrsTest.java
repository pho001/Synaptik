package io.github.pho001.synaptik.model.operation.reduction;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class MaskedReductionAttrsTest {
    @Test
    void exposesExactlyOneAxisComponentAndNoMappingSurface() {
        Class<?> type = MaskedReductionAttrs.class;
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
                () -> assertEquals(1, components.length),
                () -> assertEquals("axis", components[0].getName()),
                () -> assertSame(int.class, components[0].getType()),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertEquals(
                        List.of(int.class), Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertEquals(1, fields.length),
                () -> assertEquals("axis", fields[0].getName()),
                () -> assertTrue(Modifier.isPrivate(fields[0].getModifiers())),
                () -> assertTrue(Modifier.isFinal(fields[0].getModifiers())),
                () -> assertEquals(
                        List.of(
                                "axis():int",
                                "equals(java.lang.Object):boolean",
                                "hashCode():int",
                                "toString():java.lang.String"),
                        Arrays.stream(type.getDeclaredMethods())
                                .map(MaskedReductionAttrsTest::methodSignature)
                                .sorted()
                                .toList()),
                () -> assertEquals(0, type.getDeclaredClasses().length));
    }

    @Test
    void acceptsEveryNonNegativeAxisBoundary() {
        assertAll(
                () -> assertEquals(0, new MaskedReductionAttrs(0).axis()),
                () -> assertEquals(1, new MaskedReductionAttrs(1).axis()),
                () -> assertEquals(
                        Integer.MAX_VALUE,
                        new MaskedReductionAttrs(Integer.MAX_VALUE).axis()));
    }

    @Test
    void rejectsEveryNegativeAxisWithTheExactMessage() {
        for (int axis : new int[] {-1, -37, Integer.MIN_VALUE}) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new MaskedReductionAttrs(axis));
            assertEquals("axis must be non-negative: " + axis, failure.getMessage());
        }
    }

    @Test
    void usesGeneratedValueSemanticsAndDiagnosticText() {
        var attrs = new MaskedReductionAttrs(1);
        var equal = new MaskedReductionAttrs(1);
        var different = new MaskedReductionAttrs(2);

        assertAll(
                () -> assertEquals(attrs, equal),
                () -> assertEquals(attrs.hashCode(), equal.hashCode()),
                () -> assertNotEquals(attrs, different),
                () -> assertEquals("MaskedReductionAttrs[axis=1]", attrs.toString()));
    }

    @Test
    void composesSumAndMeanWithTheExactAttributesReference() {
        var attrs = new MaskedReductionAttrs(1);
        var sum = new Operation(AggregateReductionKind.SUM, attrs);
        var mean = new Operation(AggregateReductionKind.MEAN, attrs);

        assertAll(
                () -> assertSame(AggregateReductionKind.SUM, sum.kind()),
                () -> assertSame(attrs, sum.attrs()),
                () -> assertSame(AggregateReductionKind.MEAN, mean.kind()),
                () -> assertSame(attrs, mean.attrs()));
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
