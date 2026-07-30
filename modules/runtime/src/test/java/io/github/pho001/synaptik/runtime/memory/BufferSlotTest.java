package io.github.pho001.synaptik.runtime.memory;

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

class BufferSlotTest {
    @Test
    void hasTheExactPublicRecordShape() throws ReflectiveOperationException {
        var type = BufferSlot.class;
        var components = type.getRecordComponents();
        Constructor<?>[] constructors = type.getDeclaredConstructors();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.runtime.memory", type.getPackageName()),
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
                () -> assertEquals(long.class, type.getDeclaredMethod("value").getReturnType()),
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

    @Test
    void acceptsAndRetainsEveryNonNegativeBoundary() {
        assertAll(
                () -> assertEquals(0L, new BufferSlot(0L).value()),
                () -> assertEquals(1L, new BufferSlot(1L).value()),
                () -> assertEquals(Long.MAX_VALUE, new BufferSlot(Long.MAX_VALUE).value()));
    }

    @Test
    void rejectsNegativeValuesWithTheExactFailure() {
        for (long invalid : new long[] {-1L, Long.MIN_VALUE}) {
            IllegalArgumentException failure =
                    assertThrows(IllegalArgumentException.class, () -> new BufferSlot(invalid));

            assertEquals("value must be non-negative", failure.getMessage());
        }
    }

    @Test
    void usesOrdinaryRecordValueAndDiagnosticSemantics() {
        BufferSlot first = new BufferSlot(42L);
        BufferSlot equal = new BufferSlot(42L);
        BufferSlot different = new BufferSlot(43L);

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, different),
                () -> assertEquals("BufferSlot[value=42]", first.toString()));
    }
}
