package io.github.pho001.synaptik.trace;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.Serializable;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TraceEventIdTest {
    @Test
    void hasExactPublicRecordShape() {
        var components = TraceEventId.class.getRecordComponents();

        assertAll(
                () -> assertTrue(Modifier.isPublic(TraceEventId.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(TraceEventId.class.getModifiers())),
                () -> assertTrue(TraceEventId.class.isRecord()),
                () -> assertEquals(1, components.length),
                () -> assertEquals("value", components[0].getName()),
                () -> assertEquals(long.class, components[0].getType()),
                () -> assertEquals(long.class,
                        TraceEventId.class.getDeclaredMethod("value").getReturnType()),
                () -> assertEquals(0, TraceEventId.class.getInterfaces().length),
                () -> assertEquals(0, TraceEventId.class.getDeclaredClasses().length),
                () -> assertFalse(Serializable.class.isAssignableFrom(TraceEventId.class)),
                () -> assertEquals(
                        Set.of("equals", "hashCode", "toString", "value"),
                        Arrays.stream(TraceEventId.class.getDeclaredMethods())
                                .filter(method -> Modifier.isPublic(method.getModifiers()))
                                .map(method -> method.getName())
                                .collect(Collectors.toSet())));
    }

    @Test
    void acceptsAndRetainsEveryNonNegativeBoundary() {
        assertAll(
                () -> assertEquals(0L, new TraceEventId(0L).value()),
                () -> assertEquals(1L, new TraceEventId(1L).value()),
                () -> assertEquals(Long.MAX_VALUE, new TraceEventId(Long.MAX_VALUE).value()));
    }

    @Test
    void rejectsNegativeValuesWithExactMessage() {
        for (long invalid : new long[] {-1L, Long.MIN_VALUE}) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class, () -> new TraceEventId(invalid));

            assertEquals("value must be non-negative", failure.getMessage());
        }
    }

    @Test
    void retainsOrdinaryRecordValueSemantics() {
        TraceEventId first = new TraceEventId(42L);
        TraceEventId equal = new TraceEventId(42L);
        TraceEventId different = new TraceEventId(43L);

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, different),
                () -> assertEquals("TraceEventId[value=42]", first.toString()));
    }

}
