package io.github.pho001.synaptik.runtime.run;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.io.InputStream;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class BoundPublicationTest {
    @Test
    void exposesExactFinalSurfaceWithNoPublicOrProtectedConstructor() {
        Class<BoundPublication> type = BoundPublication.class;
        var constructor = type.getDeclaredConstructors()[0];
        var fields = Arrays.stream(type.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers())).toList();

        assertAll(
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertEquals(1, type.getDeclaredConstructors().length),
                () -> assertFalse(Modifier.isPublic(constructor.getModifiers())),
                () -> assertFalse(Modifier.isProtected(constructor.getModifiers())),
                () -> assertEquals(
                        List.of("runState", "publication", "representation", "published"),
                        fields.stream().map(field -> field.getName()).toList()),
                () -> assertEquals(
                        List.of("isPublished", "publication", "publish", "representation", "runState"),
                        Arrays.stream(type.getDeclaredMethods())
                                .map(method -> method.getName()).sorted().toList()),
                () -> assertTrue(Modifier.isPublic(type.getDeclaredMethod("publish").getModifiers())),
                () -> assertTrue(Modifier.isPublic(type.getDeclaredMethod("isPublished").getModifiers())));
    }

    @Test
    void invalidRepresentationFailsWithoutFlagOrStateMutationThenCanPublishOnce() {
        PreparedMemoryPlan plan = plan();
        RunState state = state(plan, owned());
        BoundPublication bound = new PreparedPublication(plan, 0, 0, 0).bind(state);

        assertFailure(IllegalStateException.class,
                "published buffer representation is invalid", bound::publish);
        assertAll(
                () -> assertFalse(bound.isPublished()),
                () -> assertFalse(state.isBufferRepresentationValid(0, 0)));

        state.setBufferRepresentationValid(0, 0, true);
        bound.publish();

        assertAll(
                () -> assertTrue(bound.isPublished()),
                () -> assertTrue(state.isBufferRepresentationValid(0, 0)),
                () -> assertFailure(IllegalStateException.class,
                        "publication is already complete", bound::publish));
    }

    @Test
    void closedStateWinsAndLocalFlagRemainsInspectable() {
        PreparedMemoryPlan plan = plan();
        RunState unpublishedState = state(plan, borrowed());
        BoundPublication unpublished =
                new PreparedPublication(plan, 0, 0, 0).bind(unpublishedState);
        unpublishedState.close();
        RunState publishedState = state(plan, borrowed());
        BoundPublication published =
                new PreparedPublication(plan, 0, 0, 0).bind(publishedState);
        published.publish();
        publishedState.close();

        assertAll(
                () -> assertFailure(IllegalStateException.class,
                        "run state is closed", unpublished::publish),
                () -> assertFailure(IllegalStateException.class,
                        "run state is closed", published::publish),
                () -> assertFalse(unpublished.isPublished()),
                () -> assertTrue(published.isPublished()));
    }

    @Test
    void distinctResultOccurrencesMayAliasOneExactRepresentation() {
        PreparedMemoryPlan plan = plan();
        TestBuffer representation = new TestBuffer();
        RunState state = state(plan, borrowed(representation));
        BoundPublication first = new PreparedPublication(plan, 0, 0, 0).bind(state);
        BoundPublication second = new PreparedPublication(plan, 0, 0, 1).bind(state);
        first.publish();
        second.publish();

        assertAll(
                () -> assertSame(representation, first.representation()),
                () -> assertSame(representation, second.representation()),
                () -> assertTrue(first.isPublished()),
                () -> assertTrue(second.isPublished()));
    }

    @Test
    void hotPublicationBytecodeContainsNoRepresentationLookupOrForbiddenMechanism()
            throws Exception {
        String compiled = classBytes(BoundPublication.class);
        assertAll(
                () -> assertFalse(compiled.contains("bufferRepresentation\u0001")),
                () -> assertFalse(compiled.contains("BufferRepresentationBinding")),
                () -> assertFalse(compiled.contains("java/util/Map")),
                () -> assertFalse(compiled.contains("java/lang/reflect")),
                () -> assertFalse(compiled.contains("java/util/ServiceLoader")),
                () -> assertFalse(compiled.contains("java/lang/Integer")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/compiler")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/model")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/prepare")),
                () -> assertFalse(AutoCloseable.class.isAssignableFrom(BoundPublication.class)));
    }

    private static String classBytes(Class<?> type) throws Exception {
        String resource = "/" + type.getName().replace('.', '/') + ".class";
        try (InputStream stream = type.getResourceAsStream(resource)) {
            return new String(stream.readAllBytes(), StandardCharsets.ISO_8859_1);
        }
    }

    private static PreparedMemoryPlan plan() {
        return new PreparedMemoryPlan(
                List.of(new PreparedMemoryPlan.BufferEntry(new BufferSlot(0), 4, 1)), List.of());
    }

    private static RunState state(
            PreparedMemoryPlan plan, BufferRepresentationBinding... bindings) {
        return new RunState(plan, List.of(List.of(bindings)), List.of());
    }

    private static BufferRepresentationBinding borrowed() {
        return borrowed(new TestBuffer());
    }

    private static BufferRepresentationBinding borrowed(BufferRepresentation representation) {
        return new BufferRepresentationBinding(representation, RunResourceOwnership.BORROWED);
    }

    private static BufferRepresentationBinding owned() {
        return new BufferRepresentationBinding(new TestBuffer(), RunResourceOwnership.RUN_OWNED);
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> type, String message, Runnable action) {
        T failure = assertThrows(type, action::run);
        assertEquals(message, failure.getMessage());
    }

    private static final class TestBuffer implements BufferRepresentation {
        @Override
        public void close() {}
    }
}
