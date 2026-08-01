package io.github.pho001.synaptik.runtime.run;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.runtime.execution.BoundBufferTransfer;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedBufferTransfer;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.CallerInput;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.CreatedBuffer;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;

class PreparedExecutionRunnerTest {
    @Test
    void exposesExactStatelessRunnerSurface() throws Exception {
        Class<PreparedExecutionRunner> type = PreparedExecutionRunner.class;
        var constructor = type.getDeclaredConstructor();
        var run = type.getDeclaredMethod("run", PreparedExecution.class, List.class);
        var inputs = (ParameterizedType) run.getGenericParameterTypes()[1];

        assertAll(
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertEquals(Object.class, type.getSuperclass()),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertEquals(1, type.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPublic(constructor.getModifiers())),
                () -> assertEquals(0, constructor.getParameterCount()),
                () -> assertEquals(RunResult.class, run.getReturnType()),
                () -> assertTrue(Modifier.isPublic(run.getModifiers())),
                () -> assertFalse(Modifier.isStatic(run.getModifiers())),
                () -> assertEquals(List.class, inputs.getRawType()),
                () -> assertArrayEquals(
                        new java.lang.reflect.Type[] {BufferRepresentation.class},
                        inputs.getActualTypeArguments()),
                () -> assertEquals(0, type.getDeclaredFields().length),
                () -> assertEquals(
                        List.of("run"),
                        Arrays.stream(type.getDeclaredMethods())
                                .filter(method -> Modifier.isPublic(method.getModifiers()))
                                .map(method -> method.getName())
                                .toList()),
                () -> assertTrue(
                        Arrays.stream(type.getDeclaredClasses())
                                .noneMatch(nested -> Modifier.isPublic(nested.getModifiers()))));
    }

    @Test
    void validatesTopLevelAndCreationAbsenceInExactOrder() {
        PreparedExecutionRunner runner = new PreparedExecutionRunner();
        PreparedMemoryPlan emptyPlan = plan(0);
        PreparedExecution empty = execution(emptyPlan, List.of());
        PreparedMemoryPlan nonEmptyPlan = plan(1);
        PreparedExecution nonEmpty = execution(nonEmptyPlan, List.of());
        TestBuffer caller = new TestBuffer();

        assertAll(
                () -> assertFailure(
                        NullPointerException.class, "execution", () -> runner.run(null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "callerInputs",
                        () -> runner.run(empty, null)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "non-empty prepared memory plan requires a representation creation occurrence",
                        () -> runner.run(nonEmpty, List.of(caller))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "callerInputs size must equal caller-input preparation count 0",
                        () -> runner.run(empty, List.of(caller))));
    }

    @Test
    void runsEmptyAndCreationOnlySchedulesAsZeroResultLeases() {
        PreparedExecutionRunner runner = new PreparedExecutionRunner();
        PreparedMemoryPlan emptyPlan = plan(0);
        RunResult empty = runner.run(execution(emptyPlan, List.of()), List.of());

        PreparedMemoryPlan one = plan(1);
        TestBuffer owned = new TestBuffer();
        PreparedRepresentationPlan creation =
                new PreparedRepresentationPlan(
                        one, List.of(List.of(new CreatedBuffer(() -> owned))), List.of());
        RunResult creationOnly =
                runner.run(
                        execution(
                                one,
                                List.of(new PreparedSchedule.RepresentationCreationStep(creation))),
                        List.of());

        assertAll(
                () -> assertEquals(0, empty.resultCount()),
                () -> assertFalse(empty.isClosed()),
                () -> assertEquals(0, creationOnly.resultCount()),
                () -> assertFalse(creationOnly.isClosed()),
                () -> assertEquals(0, owned.closeCount));
        empty.close();
        creationOnly.close();
        assertAll(
                () -> assertTrue(empty.isClosed()),
                () -> assertTrue(creationOnly.isClosed()),
                () -> assertEquals(1, owned.closeCount));
    }

    @Test
    void bindsEveryOccurrenceBeforeFirstActionAndCleansBindingFailure() {
        PreparedMemoryPlan plan = plan(1);
        TestBuffer owned = new TestBuffer();
        PreparedRepresentationPlan creation = createdPlan(plan, owned);
        AtomicInteger executions = new AtomicInteger();
        TestExecutable first =
                executable(
                        plan,
                        List.of(selection(0, 0)),
                        List.of(PreparedExecutable.BufferAccess.WRITE_ONLY),
                        state -> executions.incrementAndGet());
        TestExecutable incompatible =
                new TestExecutable(
                        plan,
                        List.of(selection(0, 0)),
                        List.of(PreparedExecutable.BufferAccess.READ_ONLY),
                        state -> {},
                        false);
        PreparedExecution execution =
                execution(
                        plan,
                        List.of(
                                new PreparedSchedule.RepresentationCreationStep(creation),
                                new PreparedSchedule.ExecutionStep(first),
                                new PreparedSchedule.ExecutionStep(incompatible)));

        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> new PreparedExecutionRunner().run(execution, List.of()));
        assertAll(
                () -> assertEquals(
                        "bufferSelections[0] is incompatible with prepared executable",
                        failure.getMessage()),
                () -> assertEquals(0, executions.get()),
                () -> assertEquals(1, first.bindCount),
                () -> assertEquals(0, first.executeCount),
                () -> assertEquals(1, owned.closeCount));
    }

    @Test
    void validatesReadsBeforeInvalidationThenPublishesExactSuccessfulWrites() {
        PreparedMemoryPlan plan = plan(1);
        TestBuffer caller = new TestBuffer();
        TestBuffer output = new TestBuffer();
        TestBuffer stale = new TestBuffer();
        PreparedRepresentationPlan creation =
                new PreparedRepresentationPlan(
                        plan,
                        List.of(
                                List.of(
                                        new CallerInput(),
                                        new CreatedBuffer(() -> output),
                                        new CreatedBuffer(() -> stale))),
                        List.of());
        TestExecutable executable =
                executable(
                        plan,
                        List.of(selection(0, 0), selection(0, 1), selection(0, 1)),
                        List.of(
                                PreparedExecutable.BufferAccess.READ_ONLY,
                                PreparedExecutable.BufferAccess.WRITE_ONLY,
                                PreparedExecutable.BufferAccess.WRITE_ONLY),
                        state -> {
                            assertAll(
                                    () -> assertFalse(state.isBufferRepresentationValid(0, 0)),
                                    () -> assertFalse(state.isBufferRepresentationValid(0, 1)),
                                    () -> assertFalse(state.isBufferRepresentationValid(0, 2)));
                        });
        PreparedPublication publication = new PreparedPublication(plan, 0, 1, 0);
        PreparedExecution execution =
                execution(
                        plan,
                        List.of(
                                new PreparedSchedule.RepresentationCreationStep(creation),
                                new PreparedSchedule.ExecutionStep(executable),
                                new PreparedSchedule.PublicationStep(publication)));

        RunResult result = new PreparedExecutionRunner().run(execution, List.of(caller));
        RunState state = executable.boundState;
        assertAll(
                () -> assertEquals(1, executable.executeCount),
                () -> assertFalse(state.isBufferRepresentationValid(0, 0)),
                () -> assertTrue(state.isBufferRepresentationValid(0, 1)),
                () -> assertFalse(state.isBufferRepresentationValid(0, 2)),
                () -> assertEquals(1, result.resultCount()),
                () -> assertEquals(0, caller.closeCount));
        result.close();
        assertAll(
                () -> assertEquals(0, caller.closeCount),
                () -> assertEquals(1, output.closeCount),
                () -> assertEquals(1, stale.closeCount));
    }

    @Test
    void readWriteOverlapReadsOldValueThenInvalidatesAndRevalidatesExactCopy() {
        PreparedMemoryPlan plan = plan(1);
        TestBuffer caller = new TestBuffer();
        PreparedRepresentationPlan creation =
                new PreparedRepresentationPlan(
                        plan, List.of(List.of(new CallerInput())), List.of());
        TestExecutable executable =
                executable(
                        plan,
                        List.of(selection(0, 0)),
                        List.of(PreparedExecutable.BufferAccess.READ_WRITE),
                        state -> assertFalse(state.isBufferRepresentationValid(0, 0)));
        PreparedExecution execution =
                execution(
                        plan,
                        List.of(
                                new PreparedSchedule.RepresentationCreationStep(creation),
                                new PreparedSchedule.ExecutionStep(executable)));

        RunResult result = new PreparedExecutionRunner().run(execution, List.of(caller));
        assertTrue(executable.boundState.isBufferRepresentationValid(0, 0));
        result.close();
        assertEquals(0, caller.closeCount);
    }

    @Test
    void reportsFirstRepeatedInvalidReadInOriginalSelectionOrderWithoutBackendWork() {
        PreparedMemoryPlan plan = plan(1);
        TestBuffer owned = new TestBuffer();
        TestExecutable executable =
                executable(
                        plan,
                        List.of(selection(0, 0), selection(0, 0)),
                        List.of(
                                PreparedExecutable.BufferAccess.READ_ONLY,
                                PreparedExecutable.BufferAccess.READ_ONLY),
                        state -> {});
        PreparedExecution execution =
                execution(
                        plan,
                        List.of(
                                new PreparedSchedule.RepresentationCreationStep(
                                        createdPlan(plan, owned)),
                                new PreparedSchedule.ExecutionStep(executable)));

        IllegalStateException failure =
                assertThrows(
                        IllegalStateException.class,
                        () -> new PreparedExecutionRunner().run(execution, List.of()));
        assertAll(
                () -> assertEquals(
                        "executable buffer selection 0 requires a valid input representation",
                        failure.getMessage()),
                () -> assertEquals(0, executable.executeCount),
                () -> assertEquals(1, owned.closeCount));
    }

    @Test
    void backendFailureLeavesAllOutputCopiesInvalidAndSuppressesCleanupFailure() {
        PreparedMemoryPlan plan = plan(1);
        RuntimeException actionFailure = new RuntimeException("action");
        RuntimeException cleanupFailure = new RuntimeException("cleanup");
        TestBuffer caller = new TestBuffer();
        TestBuffer owned = new TestBuffer(cleanupFailure);
        PreparedRepresentationPlan creation =
                new PreparedRepresentationPlan(
                        plan,
                        List.of(List.of(new CallerInput(), new CreatedBuffer(() -> owned))),
                        List.of());
        TestExecutable executable =
                executable(
                        plan,
                        List.of(selection(0, 1)),
                        List.of(PreparedExecutable.BufferAccess.WRITE_ONLY),
                        state -> {
                            throw actionFailure;
                        });
        PreparedExecution execution =
                execution(
                        plan,
                        List.of(
                                new PreparedSchedule.RepresentationCreationStep(creation),
                                new PreparedSchedule.ExecutionStep(executable)));

        RuntimeException observed =
                assertThrows(
                        RuntimeException.class,
                        () -> new PreparedExecutionRunner().run(execution, List.of(caller)));
        assertAll(
                () -> assertSame(actionFailure, observed),
                () -> assertArrayEquals(new Throwable[] {cleanupFailure}, observed.getSuppressed()),
                () -> assertFalse(executable.validityAtFailure[0]),
                () -> assertFalse(executable.validityAtFailure[1]),
                () -> assertTrue(executable.boundState.isClosed()),
                () -> assertEquals(0, caller.closeCount),
                () -> assertEquals(1, owned.closeCount));
    }

    @Test
    void traversesExecutionTransferAndAliasedPublicationInEncounterOrder() {
        PreparedMemoryPlan plan = plan(1);
        TestBuffer caller = new TestBuffer();
        TestBuffer destination = new TestBuffer();
        List<String> order = new ArrayList<>();
        PreparedRepresentationPlan creation =
                new PreparedRepresentationPlan(
                        plan,
                        List.of(List.of(new CallerInput(), new CreatedBuffer(() -> destination))),
                        List.of());
        TestExecutable executable =
                executable(
                        plan,
                        List.of(selection(0, 0)),
                        List.of(PreparedExecutable.BufferAccess.READ_WRITE),
                        state -> order.add("execute"));
        TestTransfer transfer = new TestTransfer(plan, order);
        PreparedExecution execution =
                execution(
                        plan,
                        List.of(
                                new PreparedSchedule.RepresentationCreationStep(creation),
                                new PreparedSchedule.ExecutionStep(executable),
                                new PreparedSchedule.BufferTransferStep(transfer),
                                new PreparedSchedule.PublicationStep(
                                        new PreparedPublication(plan, 0, 1, 0)),
                                new PreparedSchedule.PublicationStep(
                                        new PreparedPublication(plan, 0, 1, 1))));

        RunResult result = new PreparedExecutionRunner().run(execution, List.of(caller));
        assertAll(
                () -> assertEquals(List.of("execute", "transfer"), order),
                () -> assertEquals(2, result.resultCount()),
                () -> assertEquals(1, transfer.bindCount),
                () -> assertEquals(1, transfer.executeCount));
        result.close();
    }

    @Test
    void repeatedAndConcurrentCallsUseIsolatedStateAndResources() throws Exception {
        PreparedMemoryPlan plan = plan(1);
        AtomicInteger sequence = new AtomicInteger();
        List<TestBuffer> created = java.util.Collections.synchronizedList(new ArrayList<>());
        PreparedRepresentationPlan creation =
                new PreparedRepresentationPlan(
                        plan,
                        List.of(
                                List.of(
                                        new CreatedBuffer(
                                                () -> {
                                                    TestBuffer buffer = new TestBuffer();
                                                    created.add(buffer);
                                                    return buffer;
                                                }))),
                        List.of());
        ConcurrentExecutable executable = new ConcurrentExecutable(plan, sequence);
        PreparedExecution execution =
                execution(
                        plan,
                        List.of(
                                new PreparedSchedule.RepresentationCreationStep(creation),
                                new PreparedSchedule.ExecutionStep(executable)));
        PreparedExecutionRunner runner = new PreparedExecutionRunner();

        RunResult first;
        RunResult second;
        try (var pool = Executors.newFixedThreadPool(2)) {
            var firstRun = pool.submit(() -> runner.run(execution, List.of()));
            var secondRun = pool.submit(() -> runner.run(execution, List.of()));
            first = firstRun.get();
            second = secondRun.get();
        }
        assertAll(
                () -> assertEquals(2, sequence.get()),
                () -> assertEquals(2, created.size()),
                () -> assertFalse(first.isClosed()),
                () -> assertFalse(second.isClosed()));
        first.close();
        assertAll(
                () -> assertTrue(first.isClosed()),
                () -> assertFalse(second.isClosed()),
                () -> assertEquals(1, created.stream().mapToInt(buffer -> buffer.closeCount).sum()));
        second.close();
        assertEquals(2, created.stream().mapToInt(buffer -> buffer.closeCount).sum());
    }

    @Test
    void productionMechanismHasNoForbiddenImportsOrMutableRunnerState() throws Exception {
        String runnerBytes = classBytes(PreparedExecutionRunner.class);
        assertAll(
                () -> assertFalse(runnerBytes.contains("io/github/pho001/synaptik/model")),
                () -> assertFalse(runnerBytes.contains("io/github/pho001/synaptik/compiler")),
                () -> assertFalse(runnerBytes.contains("io/github/pho001/synaptik/planning")),
                () -> assertFalse(runnerBytes.contains("io/github/pho001/synaptik/prepare")),
                () -> assertFalse(runnerBytes.contains("io/github/pho001/synaptik/engine")),
                () -> assertFalse(runnerBytes.contains("io/github/pho001/synaptik/trace")),
                () -> assertFalse(runnerBytes.contains("java/lang/reflect")),
                () -> assertFalse(runnerBytes.contains("java/util/Map")),
                () -> assertFalse(runnerBytes.contains("java/util/Set")),
                () -> assertFalse(runnerBytes.contains("java/util/ServiceLoader")),
                () -> assertEquals(0, PreparedExecutionRunner.class.getDeclaredFields().length));
    }

    private static PreparedExecution execution(
            PreparedMemoryPlan plan, List<PreparedSchedule.Step> steps) {
        return new PreparedExecution(plan, new PreparedSchedule(plan, steps));
    }

    private static PreparedMemoryPlan plan(int bufferCount) {
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        for (int index = 0; index < bufferCount; index++) {
            buffers.add(new PreparedMemoryPlan.BufferEntry(new BufferSlot(index), 4, 1));
        }
        return new PreparedMemoryPlan(buffers, List.of());
    }

    private static PreparedRepresentationPlan createdPlan(
            PreparedMemoryPlan plan, TestBuffer buffer) {
        return new PreparedRepresentationPlan(
                plan, List.of(List.of(new CreatedBuffer(() -> buffer))), List.of());
    }

    private static PreparedExecutable.BufferSelection selection(
            int bufferIndex, int representationIndex) {
        return new PreparedExecutable.BufferSelection(bufferIndex, representationIndex);
    }

    private static TestExecutable executable(
            PreparedMemoryPlan plan,
            List<PreparedExecutable.BufferSelection> selections,
            List<PreparedExecutable.BufferAccess> accesses,
            Consumer<RunState> action) {
        return new TestExecutable(plan, selections, accesses, action, true);
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> type, String message, Runnable action) {
        T failure = assertThrows(type, action::run);
        assertEquals(message, failure.getMessage());
    }

    private static String classBytes(Class<?> type) throws Exception {
        try (var input = type.getResourceAsStream(type.getSimpleName() + ".class")) {
            return new String(input.readAllBytes(), java.nio.charset.StandardCharsets.ISO_8859_1);
        }
    }

    private static final class TestExecutable extends PreparedExecutable {
        private final Consumer<RunState> action;
        private final boolean compatible;
        private int bindCount;
        private int executeCount;
        private RunState boundState;
        private boolean[] validityAtFailure;

        private TestExecutable(
                PreparedMemoryPlan plan,
                List<PreparedExecutable.BufferSelection> selections,
                List<PreparedExecutable.BufferAccess> accesses,
                Consumer<RunState> action,
                boolean compatible) {
            super(plan, selections, List.of(), accesses);
            this.action = action;
            this.compatible = compatible;
        }

        @Override
        protected boolean acceptsBufferRepresentation(
                int selectionIndex, BufferRepresentation representation) {
            return compatible && representation instanceof TestBuffer;
        }

        @Override
        protected boolean acceptsWorkspaceRepresentation(
                int selectionIndex, WorkspaceRepresentation representation) {
            throw new AssertionError("no workspace selection");
        }

        @Override
        protected BoundInvocation bindCompatible(
                RunState runState,
                BufferRepresentation[] bufferRepresentations,
                WorkspaceRepresentation[] workspaceRepresentations) {
            bindCount++;
            boundState = runState;
            return new BoundInvocation(runState) {
                @Override
                protected void executeBound() {
                    executeCount++;
                    try {
                        action.accept(runState);
                    } catch (RuntimeException | Error failure) {
                        validityAtFailure = new boolean[runState.bufferRepresentationCount(0)];
                        for (int index = 0; index < validityAtFailure.length; index++) {
                            validityAtFailure[index] =
                                    runState.isBufferRepresentationValid(0, index);
                        }
                        throw failure;
                    }
                }
            };
        }
    }

    private static final class TestTransfer extends PreparedBufferTransfer {
        private final List<String> order;
        private int bindCount;
        private int executeCount;

        private TestTransfer(PreparedMemoryPlan plan, List<String> order) {
            super(plan, 0, 0, 1);
            this.order = order;
        }

        @Override
        protected boolean acceptsSourceBufferRepresentation(BufferRepresentation representation) {
            return representation instanceof TestBuffer;
        }

        @Override
        protected boolean acceptsDestinationBufferRepresentation(
                BufferRepresentation representation) {
            return representation instanceof TestBuffer;
        }

        @Override
        protected BoundBufferTransfer bindCompatible(
                RunState runState,
                BufferRepresentation sourceRepresentation,
                BufferRepresentation destinationRepresentation) {
            bindCount++;
            return new BoundBufferTransfer(runState, 0, 0, 1) {
                @Override
                protected void executeTransfer() {
                    executeCount++;
                    order.add("transfer");
                }
            };
        }
    }

    private static final class ConcurrentExecutable extends PreparedExecutable {
        private final AtomicInteger executions;

        private ConcurrentExecutable(PreparedMemoryPlan plan, AtomicInteger executions) {
            super(
                    plan,
                    List.of(selection(0, 0)),
                    List.of(),
                    List.of(PreparedExecutable.BufferAccess.WRITE_ONLY));
            this.executions = executions;
        }

        @Override
        protected boolean acceptsBufferRepresentation(
                int selectionIndex, BufferRepresentation representation) {
            return representation instanceof TestBuffer;
        }

        @Override
        protected boolean acceptsWorkspaceRepresentation(
                int selectionIndex, WorkspaceRepresentation representation) {
            throw new AssertionError("no workspace selection");
        }

        @Override
        protected BoundInvocation bindCompatible(
                RunState runState,
                BufferRepresentation[] bufferRepresentations,
                WorkspaceRepresentation[] workspaceRepresentations) {
            TestBuffer buffer = (TestBuffer) bufferRepresentations[0];
            return new BoundInvocation(runState) {
                @Override
                protected void executeBound() {
                    if (buffer.closeCount != 0) {
                        throw new AssertionError("isolated buffer closed before invocation");
                    }
                    executions.incrementAndGet();
                }
            };
        }
    }

    private static final class TestBuffer implements BufferRepresentation {
        private final RuntimeException closeFailure;
        private int closeCount;

        private TestBuffer() {
            this(null);
        }

        private TestBuffer(RuntimeException closeFailure) {
            this.closeFailure = closeFailure;
        }

        @Override
        public void close() {
            closeCount++;
            if (closeFailure != null) {
                throw closeFailure;
            }
        }
    }
}
