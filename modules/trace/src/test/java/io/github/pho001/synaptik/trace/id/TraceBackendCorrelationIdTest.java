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

class TraceBackendCorrelationIdTest {
    private static final Class<?>[] ID_TYPES = {
        TraceBackendId.class,
        TraceDeviceId.class,
        TracePreparedUnitId.class,
        TraceInvocationId.class
    };

    @Test
    void eachIdentifierHasTheExactPublicRecordShape() throws ReflectiveOperationException {
        for (Class<?> type : ID_TYPES) {
            var components = type.getRecordComponents();
            var fields = type.getDeclaredFields();
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
                    () -> assertEquals(1, fields.length),
                    () -> assertEquals("value", fields[0].getName()),
                    () -> assertEquals(long.class, fields[0].getType()),
                    () -> assertTrue(Modifier.isPrivate(fields[0].getModifiers())),
                    () -> assertTrue(Modifier.isFinal(fields[0].getModifiers())),
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
                () -> assertEquals(0L, new TraceBackendId(0L).value()),
                () -> assertEquals(Long.MAX_VALUE, new TraceBackendId(Long.MAX_VALUE).value()),
                () -> assertEquals(0L, new TraceDeviceId(0L).value()),
                () -> assertEquals(Long.MAX_VALUE, new TraceDeviceId(Long.MAX_VALUE).value()),
                () -> assertEquals(0L, new TracePreparedUnitId(0L).value()),
                () -> assertEquals(Long.MAX_VALUE, new TracePreparedUnitId(Long.MAX_VALUE).value()),
                () -> assertEquals(0L, new TraceInvocationId(0L).value()),
                () -> assertEquals(Long.MAX_VALUE, new TraceInvocationId(Long.MAX_VALUE).value()));
    }

    @Test
    void rejectsEveryNegativeBoundaryWithTheExactMessage() {
        for (long invalid : new long[] {-1L, Long.MIN_VALUE}) {
            assertAll(
                    () -> assertExactFailure(() -> new TraceBackendId(invalid)),
                    () -> assertExactFailure(() -> new TraceDeviceId(invalid)),
                    () -> assertExactFailure(() -> new TracePreparedUnitId(invalid)),
                    () -> assertExactFailure(() -> new TraceInvocationId(invalid)));
        }
    }

    @Test
    void preservesNominalSeparationAndOrdinaryRecordValueBehavior() {
        Object[] identifiers = {
            new TraceBackendId(42L),
            new TraceDeviceId(42L),
            new TracePreparedUnitId(42L),
            new TraceInvocationId(42L)
        };

        for (int left = 0; left < identifiers.length; left++) {
            for (int right = left + 1; right < identifiers.length; right++) {
                assertNotEquals(identifiers[left], identifiers[right]);
            }
        }

        TraceBackendId backend = new TraceBackendId(42L);
        assertAll(
                () -> assertEquals(backend, new TraceBackendId(42L)),
                () -> assertEquals(backend.hashCode(), new TraceBackendId(42L).hashCode()),
                () -> assertNotEquals(backend, new TraceBackendId(43L)),
                () -> assertEquals("TraceBackendId[value=42]", backend.toString()));
    }

    private static void assertExactFailure(Runnable construction) {
        IllegalArgumentException failure =
                assertThrows(IllegalArgumentException.class, construction::run);
        assertEquals("value must be non-negative", failure.getMessage());
    }
}
