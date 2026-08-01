package io.github.pho001.synaptik.runtime.execution;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class BoundBufferTransferTest {
    @Test
    void exposesExactAbstractTemplateAndAssociationSurface() throws ReflectiveOperationException {
        Class<BoundBufferTransfer> type = BoundBufferTransfer.class;
        var fields = Arrays.stream(type.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();
        var constructor = type.getDeclaredConstructors()[0];
        Method execute = type.getDeclaredMethod("execute");
        Method hook = type.getDeclaredMethod("executeTransfer");

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.runtime.execution", type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(type.getModifiers())),
                () -> assertFalse(type.isInterface()),
                () -> assertEquals(Object.class, type.getSuperclass()),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertTrue(Modifier.isProtected(constructor.getModifiers())),
                () -> assertArrayEquals(
                        new Class<?>[] {RunState.class, int.class, int.class, int.class},
                        constructor.getParameterTypes()),
                () -> assertEquals(
                        List.of(
                                "runState",
                                "bufferIndex",
                                "sourceRepresentationIndex",
                                "destinationRepresentationIndex"),
                        fields.stream().map(field -> field.getName()).toList()),
                () -> assertTrue(fields.stream().allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertTrue(Modifier.isPublic(execute.getModifiers())),
                () -> assertTrue(Modifier.isFinal(execute.getModifiers())),
                () -> assertTrue(Modifier.isProtected(hook.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(hook.getModifiers())),
                () -> assertEquals(
                        List.of(
                                "bufferIndex",
                                "destinationRepresentationIndex",
                                "execute",
                                "executeTransfer",
                                "runState",
                                "sourceRepresentationIndex"),
                        declaredMethodNames(type)),
                () -> assertTrue(Arrays.stream(type.getDeclaredMethods())
                        .filter(method -> !method.equals(execute) && !method.equals(hook))
                        .allMatch(method -> !Modifier.isPublic(method.getModifiers())
                                && !Modifier.isProtected(method.getModifiers()))));
    }

    @Test
    void constructorValidatesInExactOrderWithoutWorkOrValidityMutation() {
        PreparedMemoryPlan plan = plan(1);
        RunState state = state(plan, owned(), owned());
        TestBound valid = new TestBound(state, 0, 0, 1);
        RunState closed = state(plan, owned(), owned());
        closed.close();

        assertAll(
                () -> assertFailure(NullPointerException.class, "runState",
                        () -> new TestBound(null, -1, -1, -1)),
                () -> assertFailure(IllegalStateException.class, "run state is closed",
                        () -> new TestBound(closed, -1, -1, -1)),
                () -> assertFailure(IndexOutOfBoundsException.class,
                        "bufferIndex out of range: -1", () -> new TestBound(state, -1, -1, -1)),
                () -> assertFailure(IndexOutOfBoundsException.class,
                        "bufferIndex out of range: 1", () -> new TestBound(state, 1, -1, -1)),
                () -> assertFailure(IndexOutOfBoundsException.class,
                        "representationIndex out of range: -1",
                        () -> new TestBound(state, 0, -1, 2)),
                () -> assertFailure(IndexOutOfBoundsException.class,
                        "representationIndex out of range: 2",
                        () -> new TestBound(state, 0, 0, 2)),
                () -> assertFailure(IllegalArgumentException.class,
                        "sourceRepresentationIndex and destinationRepresentationIndex must be "
                                + "distinct",
                        () -> new TestBound(state, 0, 1, 1)),
                () -> assertEquals(0, valid.calls),
                () -> assertFalse(state.isBufferRepresentationValid(0, 0)),
                () -> assertFalse(state.isBufferRepresentationValid(0, 1)));
    }

    @Test
    void validDestinationIsNoOpWithoutValidSourceOrBackendCall() {
        PreparedMemoryPlan plan = plan(1);
        RunState state = state(plan, owned(), borrowed());
        TestBound bound = new TestBound(state, 0, 0, 1);

        bound.execute();

        assertAll(
                () -> assertFalse(state.isBufferRepresentationValid(0, 0)),
                () -> assertTrue(state.isBufferRepresentationValid(0, 1)),
                () -> assertEquals(0, bound.calls));
    }

    @Test
    void invalidSourceFailsBeforeBackendWorkAndChangesNothing() {
        PreparedMemoryPlan plan = plan(1);
        RunState state = state(plan, owned(), owned(), borrowed());
        TestBound bound = new TestBound(state, 0, 0, 1);

        assertFailure(IllegalStateException.class,
                "source buffer representation is invalid", bound::execute);
        assertAll(
                () -> assertFalse(state.isBufferRepresentationValid(0, 0)),
                () -> assertFalse(state.isBufferRepresentationValid(0, 1)),
                () -> assertTrue(state.isBufferRepresentationValid(0, 2)),
                () -> assertEquals(0, bound.calls));
    }

    @Test
    void successCallsBackendOnceAndMarksOnlyDestinationValid() {
        PreparedMemoryPlan plan = plan(1);
        TestBuffer source = new TestBuffer();
        TestBuffer destination = new TestBuffer();
        RunState state = state(plan, borrowed(source), owned(destination), borrowed());
        DirectBound bound = new DirectBound(state, source, destination);

        bound.execute();
        bound.execute();

        assertAll(
                () -> assertSame(source, bound.source),
                () -> assertSame(destination, bound.destination),
                () -> assertEquals(1, source.reads),
                () -> assertEquals(1, destination.writes),
                () -> assertTrue(state.isBufferRepresentationValid(0, 0)),
                () -> assertTrue(state.isBufferRepresentationValid(0, 1)),
                () -> assertTrue(state.isBufferRepresentationValid(0, 2)));
    }

    @Test
    void runtimeExceptionAndErrorPropagateExactlyAndPreserveFullValidityMatrix() {
        PreparedMemoryPlan plan = plan(1);
        RuntimeException runtimeFailure = new RuntimeException("copy failed");
        AssertionError errorFailure = new AssertionError("device failed");
        RunState runtimeState = state(plan, borrowed(), owned(), borrowed());
        RunState errorState = state(plan, borrowed(), owned(), borrowed());
        TestBound runtimeBound = new TestBound(runtimeState, 0, 0, 1);
        TestBound errorBound = new TestBound(errorState, 0, 0, 1);
        runtimeBound.failure = runtimeFailure;
        errorBound.failure = errorFailure;

        RuntimeException observedRuntime =
                assertThrows(RuntimeException.class, runtimeBound::execute);
        AssertionError observedError = assertThrows(AssertionError.class, errorBound::execute);

        assertAll(
                () -> assertSame(runtimeFailure, observedRuntime),
                () -> assertSame(errorFailure, observedError),
                () -> assertEquals(1, runtimeBound.calls),
                () -> assertEquals(1, errorBound.calls),
                () -> assertValidity(runtimeState, true, false, true),
                () -> assertValidity(errorState, true, false, true));
    }

    @Test
    void closedStateRejectsBeforeBackendWork() {
        PreparedMemoryPlan plan = plan(1);
        RunState state = state(plan, borrowed(), owned());
        TestBound bound = new TestBound(state, 0, 0, 1);
        state.close();

        assertFailure(IllegalStateException.class, "run state is closed", bound::execute);
        assertEquals(0, bound.calls);
    }

    @Test
    void compiledHotActionContainsOnlyValidityTransitionAndBackendHook() throws Exception {
        String compiled = classBytes(BoundBufferTransfer.class);
        assertAll(
                () -> assertFalse(compiled.contains("bufferRepresentation\u0001")),
                () -> assertFalse(compiled.contains("BufferRepresentationBinding")),
                () -> assertFalse(compiled.contains("BufferRepresentation;")),
                () -> assertFalse(compiled.contains("java/util/Map")),
                () -> assertFalse(compiled.contains("java/lang/reflect")),
                () -> assertFalse(compiled.contains("java/util/ServiceLoader")),
                () -> assertFalse(compiled.contains("java/lang/Integer")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/prepare")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/planning")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/compiler")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/model")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/engine")),
                () -> assertFalse(AutoCloseable.class.isAssignableFrom(BoundBufferTransfer.class)));
    }

    private static List<String> declaredMethodNames(Class<?> type) {
        return Arrays.stream(type.getDeclaredMethods()).map(Method::getName).sorted().toList();
    }

    private static String classBytes(Class<?> type) throws Exception {
        String resource = "/" + type.getName().replace('.', '/') + ".class";
        try (InputStream stream = type.getResourceAsStream(resource)) {
            return new String(stream.readAllBytes(), StandardCharsets.ISO_8859_1);
        }
    }

    private static PreparedMemoryPlan plan(int count) {
        return new PreparedMemoryPlan(
                List.of(new PreparedMemoryPlan.BufferEntry(new BufferSlot(0), 4, 1)),
                List.of());
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
        return owned(new TestBuffer());
    }

    private static BufferRepresentationBinding owned(BufferRepresentation representation) {
        return new BufferRepresentationBinding(representation, RunResourceOwnership.RUN_OWNED);
    }

    private static void assertValidity(RunState state, boolean... expected) {
        for (int index = 0; index < expected.length; index++) {
            assertEquals(expected[index], state.isBufferRepresentationValid(0, index));
        }
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> failureType, String message, Runnable action) {
        T failure = assertThrows(failureType, action::run);
        assertEquals(message, failure.getMessage());
    }

    private static class TestBound extends BoundBufferTransfer {
        private int calls;
        private Throwable failure;

        private TestBound(RunState state, int buffer, int source, int destination) {
            super(state, buffer, source, destination);
        }

        @Override
        protected void executeTransfer() {
            calls++;
            if (failure instanceof RuntimeException runtimeFailure) {
                throw runtimeFailure;
            }
            if (failure instanceof Error errorFailure) {
                throw errorFailure;
            }
        }
    }

    private static final class DirectBound extends BoundBufferTransfer {
        private final TestBuffer source;
        private final TestBuffer destination;

        private DirectBound(RunState state, TestBuffer source, TestBuffer destination) {
            super(state, 0, 0, 1);
            this.source = source;
            this.destination = destination;
        }

        @Override
        protected void executeTransfer() {
            source.reads++;
            destination.writes++;
        }
    }

    private static class TestBuffer implements BufferRepresentation {
        private int reads;
        private int writes;

        @Override
        public void close() {}
    }
}
