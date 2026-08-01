package io.github.pho001.synaptik.runtime.run;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;

class PreparedPublicationTest {
    @Test
    void exposesExactImmutablePublicSurface() {
        Class<PreparedPublication> type = PreparedPublication.class;
        var constructor = type.getDeclaredConstructors()[0];
        var fields = Arrays.stream(type.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();

        assertAll(
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertFalse(type.isRecord()),
                () -> assertEquals(1, type.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPublic(constructor.getModifiers())),
                () -> assertArrayEquals(
                        new Class<?>[] {PreparedMemoryPlan.class, int.class, int.class, int.class},
                        constructor.getParameterTypes()),
                () -> assertEquals(
                        List.of("memoryPlan", "bufferIndex", "representationIndex", "resultIndex"),
                        fields.stream().map(field -> field.getName()).toList()),
                () -> assertTrue(fields.stream().allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(
                        List.of("bind", "bufferIndex", "memoryPlan", "representationIndex", "resultIndex"),
                        Arrays.stream(type.getDeclaredMethods())
                                .map(method -> method.getName()).sorted().toList()),
                () -> assertTrue(Arrays.stream(type.getDeclaredMethods())
                        .allMatch(method -> Modifier.isPublic(method.getModifiers()))));
    }

    @Test
    void constructionValidatesInExactOrderAndRetainsCoordinates() {
        PreparedMemoryPlan empty = plan(0);
        PreparedMemoryPlan plan = plan(2);
        PreparedPublication publication = new PreparedPublication(plan, 1, 3, 4);

        assertAll(
                () -> assertFailure(NullPointerException.class, "memoryPlan",
                        () -> new PreparedPublication(null, -1, -1, -1)),
                () -> assertFailure(IllegalArgumentException.class,
                        "bufferIndex must be non-negative",
                        () -> new PreparedPublication(plan, -1, -1, -1)),
                () -> assertFailure(IllegalArgumentException.class,
                        "bufferIndex out of prepared-plan range: 0",
                        () -> new PreparedPublication(empty, 0, -1, -1)),
                () -> assertFailure(IllegalArgumentException.class,
                        "representationIndex must be non-negative",
                        () -> new PreparedPublication(plan, 0, -1, -1)),
                () -> assertFailure(IllegalArgumentException.class,
                        "resultIndex must be non-negative",
                        () -> new PreparedPublication(plan, 0, 0, -1)),
                () -> assertSame(plan, publication.memoryPlan()),
                () -> assertEquals(1, publication.bufferIndex()),
                () -> assertEquals(3, publication.representationIndex()),
                () -> assertEquals(4, publication.resultIndex()));
    }

    @Test
    void bindValidatesInExactOrderAndRetainsExactStateRecipeAndRepresentation() {
        PreparedMemoryPlan plan = plan(1);
        PreparedMemoryPlan equalPlan = plan(1);
        TestBuffer first = new TestBuffer();
        TestBuffer second = new TestBuffer();
        RunState state = state(plan, borrowed(first), owned(second));
        RunState foreign = state(equalPlan, borrowed(new TestBuffer()));
        RunState closed = state(plan, borrowed(new TestBuffer()));
        closed.close();
        PreparedPublication publication = new PreparedPublication(plan, 0, 1, 2);
        BoundPublication bound = publication.bind(state);

        assertAll(
                () -> assertFailure(NullPointerException.class, "runState",
                        () -> publication.bind(null)),
                () -> assertFailure(IllegalStateException.class, "run state is closed",
                        () -> publication.bind(closed)),
                () -> assertFailure(IllegalArgumentException.class,
                        "run state memory plan does not match prepared publication memory plan",
                        () -> publication.bind(foreign)),
                () -> assertFailure(IllegalArgumentException.class,
                        "representationIndex out of run-state range: 2",
                        () -> new PreparedPublication(plan, 0, 2, 0).bind(state)),
                () -> assertSame(state, bound.runState()),
                () -> assertSame(publication, bound.publication()),
                () -> assertSame(second, bound.representation()),
                () -> assertFalse(bound.isPublished()),
                () -> assertFalse(state.isBufferRepresentationValid(0, 1)));
    }

    @Test
    void immutableRecipeBindsIsolatedOccurrencesForSeparateRunsConcurrently() throws Exception {
        PreparedMemoryPlan plan = plan(1);
        PreparedPublication publication = new PreparedPublication(plan, 0, 0, 0);
        RunState first = state(plan, borrowed(new TestBuffer()));
        RunState second = state(plan, borrowed(new TestBuffer()));

        try (var executor = Executors.newFixedThreadPool(2)) {
            BoundPublication firstBound = executor.submit(() -> publication.bind(first)).get();
            BoundPublication secondBound = executor.submit(() -> publication.bind(second)).get();
            assertAll(
                    () -> assertNotSame(firstBound, secondBound),
                    () -> assertSame(first, firstBound.runState()),
                    () -> assertSame(second, secondBound.runState()),
                    () -> assertNotSame(firstBound.representation(), secondBound.representation()));
        }
    }

    private static PreparedMemoryPlan plan(int count) {
        var entries = new java.util.ArrayList<PreparedMemoryPlan.BufferEntry>();
        for (int index = 0; index < count; index++) {
            entries.add(new PreparedMemoryPlan.BufferEntry(new BufferSlot(index), 4, 1));
        }
        return new PreparedMemoryPlan(entries, List.of());
    }

    private static RunState state(
            PreparedMemoryPlan plan, BufferRepresentationBinding... bindings) {
        return new RunState(plan, List.of(List.of(bindings)), List.of());
    }

    private static BufferRepresentationBinding borrowed(BufferRepresentation representation) {
        return new BufferRepresentationBinding(representation, RunResourceOwnership.BORROWED);
    }

    private static BufferRepresentationBinding owned(BufferRepresentation representation) {
        return new BufferRepresentationBinding(representation, RunResourceOwnership.RUN_OWNED);
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
