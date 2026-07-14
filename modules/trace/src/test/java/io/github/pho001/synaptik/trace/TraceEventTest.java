package io.github.pho001.synaptik.trace;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.Serializable;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TraceEventTest {
    @Test
    void phaseHasExactLifecycleConstantsInOrder() {
        assertArrayEquals(
                new TracePhase[] {TracePhase.COMPILE, TracePhase.PREPARE, TracePhase.RUN},
                TracePhase.values());
        assertEnumShape(TracePhase.class, Set.of("COMPILE", "PREPARE", "RUN"));
    }

    @Test
    void levelHasExactConstantsInDetailToSeverityOrder() {
        assertArrayEquals(
                new TraceLevel[] {
                    TraceLevel.TRACE,
                    TraceLevel.DEBUG,
                    TraceLevel.INFO,
                    TraceLevel.WARN,
                    TraceLevel.ERROR
                },
                TraceLevel.values());
        assertEnumShape(
                TraceLevel.class, Set.of("TRACE", "DEBUG", "INFO", "WARN", "ERROR"));
    }

    @Test
    void payloadIsPublicOpenMethodFreeRootMarker() {
        assertAll(
                () -> assertTrue(TracePayload.class.isInterface()),
                () -> assertTrue(Modifier.isPublic(TracePayload.class.getModifiers())),
                () -> assertFalse(TracePayload.class.isSealed()),
                () -> assertEquals(0, TracePayload.class.getInterfaces().length),
                () -> assertEquals(0, TracePayload.class.getDeclaredMethods().length),
                () -> assertEquals(0, TracePayload.class.getDeclaredFields().length),
                () -> assertEquals(0, TracePayload.class.getDeclaredClasses().length),
                () -> assertFalse(Serializable.class.isAssignableFrom(TracePayload.class)));
    }

    @Test
    void eventHasExactPublicGenericRecordShape() throws ReflectiveOperationException {
        var components = TraceEvent.class.getRecordComponents();
        var typeParameter = TraceEvent.class.getTypeParameters()[0];

        assertAll(
                () -> assertTrue(Modifier.isPublic(TraceEvent.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(TraceEvent.class.getModifiers())),
                () -> assertTrue(TraceEvent.class.isRecord()),
                () -> assertEquals(1, TraceEvent.class.getTypeParameters().length),
                () -> assertEquals("T", typeParameter.getName()),
                () -> assertArrayEquals(
                        new java.lang.reflect.Type[] {TracePayload.class},
                        typeParameter.getBounds()),
                () -> assertEquals(5, components.length),
                () -> assertArrayEquals(
                        new String[] {"id", "phase", "level", "monotonicNanos", "payload"},
                        Arrays.stream(components).map(component -> component.getName()).toArray()),
                () -> assertArrayEquals(
                        new Class<?>[] {
                            TraceEventId.class,
                            TracePhase.class,
                            TraceLevel.class,
                            long.class,
                            TracePayload.class
                        },
                        Arrays.stream(components).map(component -> component.getType()).toArray()),
                () -> assertEquals(TraceEventId.class,
                        TraceEvent.class.getDeclaredMethod("id").getReturnType()),
                () -> assertEquals(TracePhase.class,
                        TraceEvent.class.getDeclaredMethod("phase").getReturnType()),
                () -> assertEquals(TraceLevel.class,
                        TraceEvent.class.getDeclaredMethod("level").getReturnType()),
                () -> assertEquals(long.class,
                        TraceEvent.class.getDeclaredMethod("monotonicNanos").getReturnType()),
                () -> assertEquals("T",
                        TraceEvent.class.getDeclaredMethod("payload").getGenericReturnType()
                                .getTypeName()),
                () -> assertFalse(Serializable.class.isAssignableFrom(TraceEvent.class)),
                () -> assertEquals(0, TraceEvent.class.getInterfaces().length),
                () -> assertEquals(0, TraceEvent.class.getDeclaredClasses().length),
                () -> assertEquals(
                        Set.of(
                                "equals",
                                "hashCode",
                                "toString",
                                "id",
                                "phase",
                                "level",
                                "monotonicNanos",
                                "payload"),
                        Arrays.stream(TraceEvent.class.getDeclaredMethods())
                                .filter(method -> Modifier.isPublic(method.getModifiers()))
                                .map(method -> method.getName())
                                .collect(Collectors.toSet())));

        ParameterizedType eventType = (ParameterizedType) TypedEventHolder.class
                .getDeclaredField("event").getGenericType();
        assertEquals(SamplePayload.class, eventType.getActualTypeArguments()[0]);
    }

    @Test
    void retainsExactReferencesAndGenericPayloadType() {
        TraceEventId id = new TraceEventId(7L);
        TracePhase phase = TracePhase.PREPARE;
        TraceLevel level = TraceLevel.DEBUG;
        SamplePayload payload = new SamplePayload("route selected");

        TraceEvent<SamplePayload> event = new TraceEvent<>(id, phase, level, 91L, payload);

        assertAll(
                () -> assertSame(id, event.id()),
                () -> assertSame(phase, event.phase()),
                () -> assertSame(level, event.level()),
                () -> assertEquals(91L, event.monotonicNanos()),
                () -> assertSame(payload, event.payload()),
                () -> assertEquals("route selected", event.payload().detail()));
    }

    @Test
    void nullChecksReferencesInExactComponentOrderAndWithExactMessages() {
        TraceEventId id = new TraceEventId(1L);
        SamplePayload payload = new SamplePayload("payload");

        assertNullMessage("id", () -> new TraceEvent<>(
                null, null, null, 0L, null));
        assertNullMessage("phase", () -> new TraceEvent<>(
                id, null, null, 0L, null));
        assertNullMessage("level", () -> new TraceEvent<>(
                id, TracePhase.COMPILE, null, 0L, null));
        assertNullMessage("payload", () -> new TraceEvent<>(
                id, TracePhase.COMPILE, TraceLevel.INFO, 0L, null));

        new TraceEvent<>(id, TracePhase.COMPILE, TraceLevel.INFO, 0L, payload);
    }

    @Test
    void preservesEveryTimestampBoundaryUnchanged() {
        TraceEventId id = new TraceEventId(1L);
        SamplePayload payload = new SamplePayload("payload");

        for (long timestamp : new long[] {Long.MIN_VALUE, -1L, 0L, 1L, Long.MAX_VALUE}) {
            TraceEvent<SamplePayload> event = new TraceEvent<>(
                    id, TracePhase.RUN, TraceLevel.TRACE, timestamp, payload);

            assertEquals(timestamp, event.monotonicNanos());
        }
    }

    @Test
    void retainsOrdinaryRecordValueSemantics() {
        TraceEvent<SamplePayload> first = event(10L, new SamplePayload("same"));
        TraceEvent<SamplePayload> equal = event(10L, new SamplePayload("same"));
        TraceEvent<SamplePayload> differentTime = event(11L, new SamplePayload("same"));
        TraceEvent<SamplePayload> differentPayload = event(10L, new SamplePayload("different"));

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, differentTime),
                () -> assertNotEquals(first, differentPayload),
                () -> assertTrue(first.toString().contains("TraceEvent")),
                () -> assertTrue(first.toString().contains("monotonicNanos=10")));
    }

    private static TraceEvent<SamplePayload> event(long time, SamplePayload payload) {
        return new TraceEvent<>(
                new TraceEventId(3L), TracePhase.RUN, TraceLevel.WARN, time, payload);
    }

    private static void assertNullMessage(String message, ThrowingConstructor constructor) {
        NullPointerException failure = assertThrows(NullPointerException.class, constructor::run);
        assertEquals(message, failure.getMessage());
    }

    private static <E extends Enum<E>> void assertEnumShape(
            Class<E> enumType, Set<String> expectedConstants) {
        Set<String> publicFields = Arrays.stream(enumType.getDeclaredFields())
                .filter(field -> Modifier.isPublic(field.getModifiers()))
                .map(field -> field.getName())
                .collect(Collectors.toSet());
        Set<String> publicMethods = Arrays.stream(enumType.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(method -> method.getName())
                .collect(Collectors.toSet());

        assertAll(
                () -> assertTrue(Modifier.isPublic(enumType.getModifiers())),
                () -> assertTrue(Modifier.isFinal(enumType.getModifiers())),
                () -> assertEquals(expectedConstants, publicFields),
                () -> assertEquals(Set.of("valueOf", "values"), publicMethods),
                () -> assertEquals(0, enumType.getInterfaces().length),
                () -> assertEquals(0, enumType.getDeclaredClasses().length));
    }

    private record SamplePayload(String detail) implements TracePayload {
    }

    private static final class TypedEventHolder {
        private TraceEvent<SamplePayload> event;
    }

    @FunctionalInterface
    private interface ThrowingConstructor {
        void run();
    }
}
