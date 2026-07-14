package io.github.pho001.synaptik.trace.id;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TraceModelCorrelationIdTest {
    private static final Class<?>[] ID_TYPES = {
        TraceNodeId.class, TraceValueId.class, TraceTensorId.class
    };

    @Test
    void eachIdentifierHasTheExactPublicRecordShape() throws ReflectiveOperationException {
        for (Class<?> type : ID_TYPES) {
            var components = type.getRecordComponents();
            Constructor<?>[] constructors = type.getDeclaredConstructors();

            assertAll(
                    type.getSimpleName(),
                    () -> assertEquals("io.github.pho001.synaptik.trace.id", type.getPackageName()),
                    () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                    () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                    () -> assertTrue(type.isRecord()),
                    () -> assertEquals(1, components.length),
                    () -> assertEquals("value", components[0].getName()),
                    () -> assertEquals(long.class, components[0].getType()),
                    () -> assertEquals(1, type.getDeclaredFields().length),
                    () -> assertEquals("value", type.getDeclaredFields()[0].getName()),
                    () -> assertEquals(long.class, type.getDeclaredFields()[0].getType()),
                    () -> assertEquals(1, constructors.length),
                    () -> assertArrayEquals(
                            new Class<?>[] {long.class}, constructors[0].getParameterTypes()),
                    () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                    () -> assertEquals(
                            long.class, type.getDeclaredMethod("value").getReturnType()),
                    () -> assertEquals(0, type.getInterfaces().length),
                    () -> assertEquals(0, type.getDeclaredClasses().length),
                    () -> assertFalse(Serializable.class.isAssignableFrom(type)),
                    () -> assertEquals(
                            Set.of("equals", "hashCode", "toString", "value"),
                            Arrays.stream(type.getDeclaredMethods())
                                    .filter(method -> Modifier.isPublic(method.getModifiers()))
                                    .map(method -> method.getName())
                                    .collect(Collectors.toSet())));
        }
    }

    @Test
    void acceptsAndRetainsEveryNonNegativeBoundary() {
        assertAll(
                () -> assertEquals(0L, new TraceNodeId(0L).value()),
                () -> assertEquals(1L, new TraceNodeId(1L).value()),
                () -> assertEquals(Long.MAX_VALUE, new TraceNodeId(Long.MAX_VALUE).value()),
                () -> assertEquals(0L, new TraceValueId(0L).value()),
                () -> assertEquals(1L, new TraceValueId(1L).value()),
                () -> assertEquals(Long.MAX_VALUE, new TraceValueId(Long.MAX_VALUE).value()),
                () -> assertEquals(0L, new TraceTensorId(0L).value()),
                () -> assertEquals(1L, new TraceTensorId(1L).value()),
                () -> assertEquals(Long.MAX_VALUE, new TraceTensorId(Long.MAX_VALUE).value()));
    }

    @Test
    void rejectsNegativeValuesWithTheExactMessage() {
        for (long invalid : new long[] {-1L, Long.MIN_VALUE}) {
            assertAll(
                    () -> assertExactFailure(() -> new TraceNodeId(invalid)),
                    () -> assertExactFailure(() -> new TraceValueId(invalid)),
                    () -> assertExactFailure(() -> new TraceTensorId(invalid)));
        }
    }

    @Test
    void preservesNominalSeparationAndOrdinaryRecordValueBehavior() {
        TraceNodeId node = new TraceNodeId(42L);
        TraceValueId value = new TraceValueId(42L);
        TraceTensorId tensor = new TraceTensorId(42L);

        assertAll(
                () -> assertNotEquals(node, value),
                () -> assertNotEquals(node, tensor),
                () -> assertNotEquals(value, tensor),
                () -> assertEquals(node, new TraceNodeId(42L)),
                () -> assertEquals(node.hashCode(), new TraceNodeId(42L).hashCode()),
                () -> assertNotEquals(node, new TraceNodeId(43L)),
                () -> assertEquals("TraceNodeId[value=42]", node.toString()),
                () -> assertEquals(value, new TraceValueId(42L)),
                () -> assertEquals(value.hashCode(), new TraceValueId(42L).hashCode()),
                () -> assertNotEquals(value, new TraceValueId(43L)),
                () -> assertEquals("TraceValueId[value=42]", value.toString()),
                () -> assertEquals(tensor, new TraceTensorId(42L)),
                () -> assertEquals(tensor.hashCode(), new TraceTensorId(42L).hashCode()),
                () -> assertNotEquals(tensor, new TraceTensorId(43L)),
                () -> assertEquals("TraceTensorId[value=42]", tensor.toString()));
    }

    private static void assertExactFailure(Runnable construction) {
        IllegalArgumentException failure =
                assertThrows(IllegalArgumentException.class, construction::run);

        assertEquals("value must be non-negative", failure.getMessage());
    }
}
