package io.github.pho001.synaptik.runtime.execution;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.ref.WeakReference;
import java.lang.reflect.Modifier;
import java.util.List;
import org.junit.jupiter.api.Test;

class BoundInvocationTest {
    @Test
    void exposesExactAbstractLifecycleTemplate() throws Exception {
        int modifiers = BoundInvocation.class.getModifiers();
        var constructors = BoundInvocation.class.getDeclaredConstructors();
        var constructor = constructors[0];
        var execute = BoundInvocation.class.getDeclaredMethod("execute");
        var executeBound = BoundInvocation.class.getDeclaredMethod("executeBound");
        var runStateField = BoundInvocation.class.getDeclaredField("runState");

        assertAll(
                () -> assertTrue(Modifier.isPublic(modifiers)),
                () -> assertTrue(Modifier.isAbstract(modifiers)),
                () -> assertEquals(Object.class, BoundInvocation.class.getSuperclass()),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isProtected(constructor.getModifiers())),
                () -> assertEquals(RunState.class, constructor.getParameterTypes()[0]),
                () -> assertEquals(0, constructor.getExceptionTypes().length),
                () -> assertEquals(void.class, execute.getReturnType()),
                () -> assertTrue(Modifier.isPublic(execute.getModifiers())),
                () -> assertTrue(Modifier.isFinal(execute.getModifiers())),
                () -> assertEquals(0, execute.getParameterCount()),
                () -> assertEquals(0, execute.getExceptionTypes().length),
                () -> assertEquals(void.class, executeBound.getReturnType()),
                () -> assertTrue(Modifier.isProtected(executeBound.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(executeBound.getModifiers())),
                () -> assertEquals(0, executeBound.getParameterCount()),
                () -> assertTrue(Modifier.isPrivate(runStateField.getModifiers())),
                () -> assertTrue(Modifier.isFinal(runStateField.getModifiers())),
                () -> assertEquals(RunState.class, runStateField.getType()),
                () -> assertEquals(0, BoundInvocation.class.getDeclaredClasses().length),
                () -> assertEquals(0, BoundInvocation.class.getInterfaces().length));
    }

    @Test
    void constructionRequiresOneExactOpenRunState() {
        RunState closed = emptyState();
        closed.close();

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "runState",
                        () -> new TrackingInvocation(null)),
                () -> assertFailure(
                        IllegalStateException.class,
                        "run state is closed",
                        () -> new TrackingInvocation(closed)));
    }

    @Test
    void invocationStronglyRetainsTheExactRunStateWithoutOwningIt() {
        RunState state = emptyState();
        WeakReference<RunState> weakState = new WeakReference<>(state);
        TrackingInvocation invocation = new TrackingInvocation(state);

        assertAll(
                () -> assertSame(state, invocation.runState()),
                () -> assertSame(state, weakState.get()),
                () -> assertEquals(0, invocation.executeCount),
                () -> assertTrue(!state.isClosed()));
    }

    @Test
    void executePermitsSequentialCallsAndRejectsClosedStateBeforeBackendWork() {
        RunState state = emptyState();
        TrackingInvocation invocation = new TrackingInvocation(state);

        invocation.execute();
        invocation.execute();
        state.close();

        assertFailure(
                IllegalStateException.class,
                "run state is closed",
                invocation::execute);
        assertAll(
                () -> assertEquals(2, invocation.executeCount),
                () -> assertTrue(state.isClosed()));
    }

    @Test
    void executePropagatesExactUncheckedBackendFailureWithoutRetryOrWrapping() {
        RuntimeException runtimeFailure = new RuntimeException("runtime failure");
        Error error = new AssertionError("error");
        FailingInvocation runtimeInvocation =
                new FailingInvocation(emptyState(), runtimeFailure);
        FailingInvocation errorInvocation = new FailingInvocation(emptyState(), error);

        RuntimeException thrownRuntime =
                assertThrows(RuntimeException.class, runtimeInvocation::execute);
        Error thrownError = assertThrows(Error.class, errorInvocation::execute);

        assertAll(
                () -> assertSame(runtimeFailure, thrownRuntime),
                () -> assertSame(error, thrownError),
                () -> assertEquals(1, runtimeInvocation.executeCount),
                () -> assertEquals(1, errorInvocation.executeCount),
                () -> assertTrue(!runtimeInvocation.runState().isClosed()),
                () -> assertTrue(!errorInvocation.runState().isClosed()));
    }

    private static RunState emptyState() {
        PreparedMemoryPlan plan = new PreparedMemoryPlan(List.of(), List.of());
        return new RunState(plan, List.of(), List.of());
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> failureType, String message, Runnable action) {
        T failure = assertThrows(failureType, action::run);
        assertEquals(message, failure.getMessage());
    }

    private static final class TrackingInvocation extends BoundInvocation {
        private int executeCount;

        private TrackingInvocation(RunState runState) {
            super(runState);
        }

        @Override
        protected void executeBound() {
            executeCount++;
        }
    }

    private static final class FailingInvocation extends BoundInvocation {
        private final Throwable failure;
        private int executeCount;

        private FailingInvocation(RunState runState, Throwable failure) {
            super(runState);
            this.failure = failure;
        }

        @Override
        protected void executeBound() {
            executeCount++;
            if (failure instanceof RuntimeException runtimeFailure) {
                throw runtimeFailure;
            }
            if (failure instanceof Error error) {
                throw error;
            }
        }
    }
}
