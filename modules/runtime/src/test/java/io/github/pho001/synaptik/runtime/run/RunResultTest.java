package io.github.pho001.synaptik.runtime.run;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class RunResultTest {
    @Test
    void exposesExactResultAndLifecycleOnlySurface() {
        Class<RunResult> type = RunResult.class;
        var constructor = type.getDeclaredConstructors()[0];
        var publicationsType = (ParameterizedType) constructor.getGenericParameterTypes()[1];
        var fields = Arrays.stream(type.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers())).toList();

        assertAll(
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertArrayEquals(new Class<?>[] {AutoCloseable.class}, type.getInterfaces()),
                () -> assertEquals(1, type.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPublic(constructor.getModifiers())),
                () -> assertArrayEquals(
                        new Class<?>[] {RunState.class, List.class},
                        constructor.getParameterTypes()),
                () -> assertEquals(List.class, publicationsType.getRawType()),
                () -> assertArrayEquals(
                        new java.lang.reflect.Type[] {BoundPublication.class},
                        publicationsType.getActualTypeArguments()),
                () -> assertEquals(
                        List.of("runState", "representations"),
                        fields.stream().map(field -> field.getName()).toList()),
                () -> assertEquals(BufferRepresentation[].class, fields.get(1).getType()),
                () -> assertTrue(fields.stream().allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(
                        List.of("close", "isClosed", "resultCount"),
                        Arrays.stream(type.getDeclaredMethods())
                                .map(method -> method.getName()).sorted().toList()),
                () -> assertTrue(Arrays.stream(type.getDeclaredMethods())
                        .allMatch(method -> Modifier.isPublic(method.getModifiers()))));
    }

    @Test
    void constructorValidatesTopLevelInputsAndStateBeforeListElements() {
        PreparedMemoryPlan plan = plan(1);
        RunState open = state(plan, borrowed(new TrackingBuffer()));
        RunState closed = state(plan, borrowed(new TrackingBuffer()));
        closed.close();
        var nullElement = new ArrayList<BoundPublication>();
        nullElement.add(null);

        assertAll(
                () -> assertFailure(NullPointerException.class, "runState",
                        () -> new RunResult(null, null)),
                () -> assertFailure(NullPointerException.class, "publications",
                        () -> new RunResult(open, null)),
                () -> assertFailure(IllegalStateException.class, "run state is closed",
                        () -> new RunResult(closed, nullElement)),
                () -> assertFailure(NullPointerException.class, "publications[0]",
                        () -> new RunResult(open, nullElement)),
                () -> assertFalse(open.isClosed()));
    }

    @Test
    void constructorValidatesEachOccurrenceInExactOrderWithoutTakingCleanup() {
        PreparedMemoryPlan plan = plan(1);
        TrackingBuffer firstBuffer = new TrackingBuffer();
        TrackingBuffer otherBuffer = new TrackingBuffer();
        RunState state = state(plan, borrowed(firstBuffer));
        RunState other = state(plan, borrowed(otherBuffer));
        BoundPublication foreign = publication(other, 0);
        foreign.publish();
        BoundPublication wrongIndex = publication(state, 1);
        wrongIndex.publish();
        BoundPublication unpublished = publication(state, 0);

        assertAll(
                () -> assertFailure(IllegalArgumentException.class,
                        "publications[0] does not belong to supplied run state",
                        () -> new RunResult(state, List.of(foreign))),
                () -> assertFailure(IllegalArgumentException.class,
                        "publications[0] result index does not match encounter order",
                        () -> new RunResult(state, List.of(wrongIndex))),
                () -> assertFailure(IllegalStateException.class,
                        "publications[0] is not published",
                        () -> new RunResult(state, List.of(unpublished))),
                () -> assertFalse(state.isClosed()),
                () -> assertEquals(0, firstBuffer.closeCount));
    }

    @Test
    void acceptsEmptyResultAndDelegatesIdempotentWholeStateCleanup() {
        PreparedMemoryPlan plan = plan(1);
        TrackingBuffer owned = new TrackingBuffer();
        RunState state = state(plan, owned(owned));
        RunResult result = new RunResult(state, List.of());

        assertAll(
                () -> assertEquals(0, result.resultCount()),
                () -> assertFalse(result.isClosed()),
                () -> assertEquals(0, owned.closeCount));
        result.close();
        assertAll(
                () -> assertTrue(result.isClosed()),
                () -> assertTrue(state.isClosed()),
                () -> assertEquals(0, result.resultCount()),
                () -> assertEquals(1, owned.closeCount),
                () -> assertDoesNotThrow(result::close),
                () -> assertEquals(1, owned.closeCount));
    }

    @Test
    void snapshotsDensePublishedOccurrencesIncludingExactAliasesWithoutRetainingList() {
        PreparedMemoryPlan plan = plan(1);
        TrackingBuffer representation = new TrackingBuffer();
        RunState state = state(plan, borrowed(representation));
        BoundPublication first = publication(state, 0);
        BoundPublication alias = publication(state, 1);
        first.publish();
        alias.publish();
        var supplied = new ArrayList<>(List.of(first, alias));

        RunResult result = new RunResult(state, supplied);
        supplied.clear();

        assertAll(
                () -> assertEquals(2, result.resultCount()),
                () -> assertFalse(result.isClosed()),
                () -> assertSame(first.representation(), alias.representation()));
        result.close();
        assertEquals(0, representation.closeCount);
    }

    @Test
    void partialPublicationCannotConstructResultAndStateRemainsRunnerOwned() {
        PreparedMemoryPlan plan = plan(1);
        TrackingBuffer owned = new TrackingBuffer();
        RunState state = state(plan, owned(owned));
        state.setBufferRepresentationValid(0, 0, true);
        BoundPublication first = publication(state, 0);
        BoundPublication second = publication(state, 1);
        first.publish();

        assertFailure(IllegalStateException.class,
                "publications[1] is not published",
                () -> new RunResult(state, List.of(first, second)));
        assertAll(
                () -> assertTrue(first.isPublished()),
                () -> assertFalse(second.isPublished()),
                () -> assertFalse(state.isClosed()),
                () -> assertEquals(0, owned.closeCount));
        state.close();
        assertEquals(1, owned.closeCount);
    }

    @Test
    void closePropagatesExactExistingCleanupFailureAndRemainsClosed() {
        RuntimeException failure = new RuntimeException("cleanup failed");
        PreparedMemoryPlan plan = plan(1);
        TrackingBuffer owned = new TrackingBuffer(failure);
        RunState state = state(plan, owned(owned));
        RunResult result = new RunResult(state, List.of());

        RuntimeException observed = assertThrows(RuntimeException.class, result::close);
        assertAll(
                () -> assertSame(failure, observed),
                () -> assertTrue(result.isClosed()),
                () -> assertEquals(1, owned.closeCount),
                () -> assertDoesNotThrow(result::close),
                () -> assertEquals(1, owned.closeCount));
    }

    private static BoundPublication publication(RunState state, int resultIndex) {
        return new PreparedPublication(state.memoryPlan(), 0, 0, resultIndex).bind(state);
    }

    private static PreparedMemoryPlan plan(int count) {
        var entries = new ArrayList<PreparedMemoryPlan.BufferEntry>();
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

    private static final class TrackingBuffer implements BufferRepresentation {
        private final RuntimeException failure;
        private int closeCount;

        private TrackingBuffer() {
            this(null);
        }

        private TrackingBuffer(RuntimeException failure) {
            this.failure = failure;
        }

        @Override
        public void close() {
            closeCount++;
            if (failure != null) {
                throw failure;
            }
        }
    }
}
