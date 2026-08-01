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
import io.github.pho001.synaptik.runtime.memory.WorkspaceSlot;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.BufferPreparation;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.CallerInput;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.CreatedBuffer;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class RunStateCreationTest {
    @Test
    void exposesExactPlanAndPackagePrivateCreationSurface() throws Exception {
        Class<PreparedRepresentationPlan> planType = PreparedRepresentationPlan.class;
        Class<RunStateCreation> creationType = RunStateCreation.class;
        Method create =
                creationType.getDeclaredMethod(
                        "create", PreparedRepresentationPlan.class, List.class);

        assertAll(
                () -> assertTrue(Modifier.isPublic(planType.getModifiers())),
                () -> assertTrue(Modifier.isFinal(planType.getModifiers())),
                () -> assertTrue(planType.isRecord()),
                () -> assertEquals(
                        List.of("memoryPlan", "bufferPreparations", "workspaceCreators"),
                        Arrays.stream(planType.getRecordComponents())
                                .map(component -> component.getName())
                                .toList()),
                () -> assertEquals(
                        List.of(
                                PreparedRepresentationPlan.BufferCreator.class,
                                BufferPreparation.class,
                                CallerInput.class,
                                CreatedBuffer.class,
                                PreparedRepresentationPlan.WorkspaceCreator.class),
                        Arrays.stream(planType.getDeclaredClasses())
                                .sorted((left, right) -> left.getSimpleName().compareTo(right.getSimpleName()))
                                .toList()),
                () -> assertTrue(BufferPreparation.class.isSealed()),
                () -> assertArrayEquals(
                        new Class<?>[] {CallerInput.class, CreatedBuffer.class},
                        BufferPreparation.class.getPermittedSubclasses()),
                () -> assertTrue(
                        PreparedRepresentationPlan.BufferCreator.class.isAnnotationPresent(
                                FunctionalInterface.class)),
                () -> assertTrue(
                        PreparedRepresentationPlan.WorkspaceCreator.class.isAnnotationPresent(
                                FunctionalInterface.class)),
                () -> assertTrue(Modifier.isFinal(creationType.getModifiers())),
                () -> assertFalse(Modifier.isPublic(creationType.getModifiers())),
                () -> assertEquals(1, creationType.getDeclaredConstructors().length),
                () -> assertTrue(
                        Modifier.isPrivate(
                                creationType.getDeclaredConstructors()[0].getModifiers())),
                () -> assertEquals(
                        1,
                        Arrays.stream(creationType.getDeclaredMethods())
                                .filter(method -> !Modifier.isPrivate(method.getModifiers()))
                                .count()),
                () -> assertTrue(Modifier.isStatic(create.getModifiers())),
                () -> assertFalse(Modifier.isPublic(create.getModifiers())),
                () -> assertEquals(RunState.class, create.getReturnType()));
    }

    @Test
    void compiledContractsContainNoForbiddenUpstreamOrDynamicMechanisms() throws Exception {
        String compiled =
                classBytes(PreparedRepresentationPlan.class)
                        + classBytes(BufferPreparation.class)
                        + classBytes(CallerInput.class)
                        + classBytes(CreatedBuffer.class)
                        + classBytes(PreparedRepresentationPlan.BufferCreator.class)
                        + classBytes(PreparedRepresentationPlan.WorkspaceCreator.class)
                        + classBytes(RunStateCreation.class)
                        + classBytes(RunState.class);

        assertAll(
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/prepare")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/planning")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/compiler")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/model")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/engine")),
                () -> assertFalse(compiled.contains("java/lang/reflect")),
                () -> assertFalse(compiled.contains("java/util/Map")),
                () -> assertFalse(compiled.contains("java/util/ServiceLoader")),
                () -> assertFalse(compiled.contains("java/util/concurrent")),
                () -> assertFalse(
                        Arrays.stream(RunStateCreation.class.getDeclaredMethods())
                                .anyMatch(method -> Modifier.isPublic(method.getModifiers()))));
    }

    @Test
    void planValidatesInExactOrderAndSnapshotsBothListLevelsWithoutCreating() {
        PreparedMemoryPlan plan = memoryPlan(1, 1);
        AtomicInteger callbacks = new AtomicInteger();
        var created = new CreatedBuffer(() -> {
            callbacks.incrementAndGet();
            return new TrackingBuffer("created", null, null);
        });
        var inner = new ArrayList<BufferPreparation>(List.of(new CallerInput(), created));
        var buffers = new ArrayList<List<BufferPreparation>>(List.of(inner));
        var workspaceCreators = new ArrayList<PreparedRepresentationPlan.WorkspaceCreator>();
        workspaceCreators.add(() -> {
            callbacks.incrementAndGet();
            return new TrackingWorkspace("workspace", null, null);
        });

        PreparedRepresentationPlan representationPlan =
                new PreparedRepresentationPlan(plan, buffers, workspaceCreators);
        inner.clear();
        buffers.clear();
        workspaceCreators.clear();

        var nullInner = new ArrayList<List<BufferPreparation>>();
        nullInner.add(null);
        var nullPreparation = new ArrayList<BufferPreparation>();
        nullPreparation.add(null);
        var nullCreator = new ArrayList<PreparedRepresentationPlan.WorkspaceCreator>();
        nullCreator.add(null);

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "memoryPlan",
                        () -> new PreparedRepresentationPlan(null, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "bufferPreparations",
                        () -> new PreparedRepresentationPlan(plan, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "workspaceCreators",
                        () -> new PreparedRepresentationPlan(plan, List.of(), null)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "bufferPreparations size must equal prepared buffer count 1",
                        () -> new PreparedRepresentationPlan(plan, List.of(), List.of())),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "workspaceCreators size must equal prepared workspace count 1",
                        () -> new PreparedRepresentationPlan(
                                plan, List.of(List.of(new CallerInput())), List.of())),
                () -> assertFailure(
                        NullPointerException.class,
                        "bufferPreparations[0]",
                        () -> new PreparedRepresentationPlan(
                                plan, nullInner, List.of(() -> new TrackingWorkspace("w", null, null)))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "bufferPreparations[0] must not be empty",
                        () -> new PreparedRepresentationPlan(
                                plan, List.of(List.of()), List.of(() -> new TrackingWorkspace("w", null, null)))),
                () -> assertFailure(
                        NullPointerException.class,
                        "bufferPreparations[0][0]",
                        () -> new PreparedRepresentationPlan(
                                plan, List.of(nullPreparation), List.of(() -> new TrackingWorkspace("w", null, null)))),
                () -> assertFailure(
                        NullPointerException.class,
                        "workspaceCreators[0]",
                        () -> new PreparedRepresentationPlan(
                                plan, List.of(List.of(new CallerInput())), nullCreator)),
                () -> assertFailure(
                        NullPointerException.class,
                        "creator",
                        () -> new CreatedBuffer(null)),
                () -> assertSame(plan, representationPlan.memoryPlan()),
                () -> assertEquals(2, representationPlan.bufferPreparations().getFirst().size()),
                () -> assertSame(created, representationPlan.bufferPreparations().getFirst().get(1)),
                () -> assertEquals(1, representationPlan.workspaceCreators().size()),
                () -> assertEquals(0, callbacks.get()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> representationPlan.bufferPreparations().clear()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> representationPlan.bufferPreparations().getFirst().clear()));
    }

    @Test
    void validatesEveryCallerBeforeInvokingAnyCreator() {
        AtomicInteger callbacks = new AtomicInteger();
        PreparedRepresentationPlan plan =
                new PreparedRepresentationPlan(
                        memoryPlan(2, 0),
                        List.of(
                                List.of(new CallerInput()),
                                List.of(new CallerInput(), new CreatedBuffer(() -> {
                                    callbacks.incrementAndGet();
                                    return new TrackingBuffer("created", null, null);
                                }))),
                        List.of());
        TrackingBuffer repeated = new TrackingBuffer("repeated", null, null);
        var withNull = new ArrayList<BufferRepresentation>();
        withNull.add(repeated);
        withNull.add(null);

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "representationPlan",
                        () -> RunStateCreation.create(null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "callerInputs",
                        () -> RunStateCreation.create(plan, null)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "callerInputs size must equal caller-input preparation count 2",
                        () -> RunStateCreation.create(plan, List.of(repeated))),
                () -> assertFailure(
                        NullPointerException.class,
                        "callerInputs[1]",
                        () -> RunStateCreation.create(plan, withNull)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "representation is already bound to this run",
                        () -> RunStateCreation.create(plan, List.of(repeated, repeated))),
                () -> assertEquals(0, callbacks.get()),
                () -> assertEquals(0, repeated.closeCount));
    }

    @Test
    void createsBuffersThenWorkspacesAndInitializesOwnershipValidityAndResidency() {
        List<String> creationOrder = new ArrayList<>();
        TrackingBuffer caller0 = new TrackingBuffer("caller0", null, null);
        TrackingBuffer caller1 = new TrackingBuffer("caller1", null, null);
        TrackingBuffer created00 = new TrackingBuffer("created00", null, null);
        TrackingBuffer created11 = new TrackingBuffer("created11", null, null);
        TrackingWorkspace workspace0 = new TrackingWorkspace("workspace0", null, null);
        PreparedRepresentationPlan plan =
                new PreparedRepresentationPlan(
                        memoryPlan(2, 1),
                        List.of(
                                List.of(
                                        new CreatedBuffer(() -> {
                                            creationOrder.add("created00");
                                            return created00;
                                        }),
                                        new CallerInput()),
                                List.of(
                                        new CallerInput(),
                                        new CreatedBuffer(() -> {
                                            creationOrder.add("created11");
                                            return created11;
                                        }))),
                        List.of(() -> {
                            creationOrder.add("workspace0");
                            return workspace0;
                        }));

        RunState state = RunStateCreation.create(plan, List.of(caller0, caller1));

        assertAll(
                () -> assertEquals(
                        List.of("created00", "created11", "workspace0"), creationOrder),
                () -> assertSame(plan.memoryPlan(), state.memoryPlan()),
                () -> assertSame(created00, state.bufferRepresentation(0, 0).representation()),
                () -> assertSame(caller0, state.bufferRepresentation(0, 1).representation()),
                () -> assertSame(caller1, state.bufferRepresentation(1, 0).representation()),
                () -> assertSame(created11, state.bufferRepresentation(1, 1).representation()),
                () -> assertEquals(
                        RunResourceOwnership.RUN_OWNED,
                        state.bufferRepresentation(0, 0).ownership()),
                () -> assertEquals(
                        RunResourceOwnership.BORROWED,
                        state.bufferRepresentation(0, 1).ownership()),
                () -> assertFalse(state.isBufferRepresentationValid(0, 0)),
                () -> assertTrue(state.isBufferRepresentationValid(0, 1)),
                () -> assertTrue(state.isBufferRepresentationValid(1, 0)),
                () -> assertFalse(state.isBufferRepresentationValid(1, 1)),
                () -> assertSame(workspace0, state.workspaceRepresentation(0)));
    }

    @Test
    void nullAndDuplicateCreatorResultsRollbackOnlyEarlierFreshResources() {
        List<String> closeOrder = new ArrayList<>();
        TrackingBuffer caller = new TrackingBuffer("caller", closeOrder, null);
        TrackingBuffer created = new TrackingBuffer("created", closeOrder, null);
        PreparedRepresentationPlan nullResult =
                new PreparedRepresentationPlan(
                        memoryPlan(1, 0),
                        List.of(List.of(
                                new CallerInput(),
                                new CreatedBuffer(() -> created),
                                new CreatedBuffer(() -> null))),
                        List.of());
        NullPointerException nullFailure =
                assertThrows(
                        NullPointerException.class,
                        () -> RunStateCreation.create(nullResult, List.of(caller)));

        DualRepresentation dual = new DualRepresentation("dual", closeOrder, null);
        PreparedRepresentationPlan duplicateWorkspace =
                new PreparedRepresentationPlan(
                        memoryPlan(1, 1),
                        List.of(List.of(new CreatedBuffer(() -> dual))),
                        List.of(() -> dual));
        IllegalArgumentException duplicateFailure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> RunStateCreation.create(duplicateWorkspace, List.of()));

        assertAll(
                () -> assertEquals(
                        "bufferPreparations[0][2] creator result", nullFailure.getMessage()),
                () -> assertEquals(1, created.closeCount),
                () -> assertEquals(0, caller.closeCount),
                () -> assertEquals(
                        "representation is already bound to this run",
                        duplicateFailure.getMessage()),
                () -> assertEquals(1, dual.closeCount),
                () -> assertEquals(List.of("created", "dual"), closeOrder));
    }

    @Test
    void rejectsCallerAliasesAndNullWorkspaceResultsWithExactDiagnostics() {
        TrackingBuffer caller = new TrackingBuffer("caller", null, null);
        PreparedRepresentationPlan callerAlias =
                new PreparedRepresentationPlan(
                        memoryPlan(1, 0),
                        List.of(List.of(
                                new CallerInput(), new CreatedBuffer(() -> caller))),
                        List.of());
        IllegalArgumentException aliasFailure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> RunStateCreation.create(callerAlias, List.of(caller)));

        TrackingBuffer created = new TrackingBuffer("created", null, null);
        PreparedRepresentationPlan nullWorkspace =
                new PreparedRepresentationPlan(
                        memoryPlan(1, 1),
                        List.of(List.of(new CreatedBuffer(() -> created))),
                        List.of(() -> null));
        NullPointerException nullFailure =
                assertThrows(
                        NullPointerException.class,
                        () -> RunStateCreation.create(nullWorkspace, List.of()));

        assertAll(
                () -> assertEquals(
                        "representation is already bound to this run", aliasFailure.getMessage()),
                () -> assertEquals(0, caller.closeCount),
                () -> assertEquals("workspaceCreators[0] result", nullFailure.getMessage()),
                () -> assertEquals(1, created.closeCount));
    }

    @Test
    void successfulClosureIsReverseCreationOrderAndNeverClosesBorrowedInputs() {
        List<String> closeOrder = new ArrayList<>();
        TrackingBuffer caller = new TrackingBuffer("caller", closeOrder, null);
        TrackingBuffer buffer0 = new TrackingBuffer("buffer0", closeOrder, null);
        TrackingBuffer buffer1 = new TrackingBuffer("buffer1", closeOrder, null);
        TrackingWorkspace workspace0 = new TrackingWorkspace("workspace0", closeOrder, null);
        TrackingWorkspace workspace1 = new TrackingWorkspace("workspace1", closeOrder, null);
        PreparedRepresentationPlan plan =
                new PreparedRepresentationPlan(
                        memoryPlan(2, 2),
                        List.of(
                                List.of(new CreatedBuffer(() -> buffer0), new CallerInput()),
                                List.of(new CreatedBuffer(() -> buffer1))),
                        List.of(() -> workspace0, () -> workspace1));

        RunState state = RunStateCreation.create(plan, List.of(caller));
        state.close();

        assertAll(
                () -> assertEquals(
                        List.of("workspace1", "workspace0", "buffer1", "buffer0"),
                        closeOrder),
                () -> assertEquals(0, caller.closeCount),
                () -> assertEquals(1, buffer0.closeCount),
                () -> assertEquals(1, buffer1.closeCount),
                () -> assertEquals(1, workspace0.closeCount),
                () -> assertEquals(1, workspace1.closeCount));
    }

    @Test
    void creatorFailureRollsBackInReverseAndSuppressesCleanupFailuresInEncounterOrder() {
        List<String> closeOrder = new ArrayList<>();
        RuntimeException original = new RuntimeException("creator");
        RuntimeException firstCleanup = new RuntimeException("second cleanup");
        AssertionError secondCleanup = new AssertionError("first cleanup");
        TrackingBuffer first = new TrackingBuffer("first", closeOrder, secondCleanup);
        TrackingBuffer second = new TrackingBuffer("second", closeOrder, firstCleanup);
        PreparedRepresentationPlan plan =
                new PreparedRepresentationPlan(
                        memoryPlan(1, 0),
                        List.of(List.of(
                                new CreatedBuffer(() -> first),
                                new CreatedBuffer(() -> second),
                                new CreatedBuffer(() -> {
                                    throw original;
                                }))),
                        List.of());

        RuntimeException thrown =
                assertThrows(
                        RuntimeException.class,
                        () -> RunStateCreation.create(plan, List.of()));

        assertAll(
                () -> assertSame(original, thrown),
                () -> assertEquals(List.of("second", "first"), closeOrder),
                () -> assertArrayEquals(
                        new Throwable[] {firstCleanup, secondCleanup}, thrown.getSuppressed()),
                () -> assertEquals(1, first.closeCount),
                () -> assertEquals(1, second.closeCount));
    }

    @Test
    void successfulStatesOwnFreshResultsAndKeepValidityIsolated() {
        AtomicInteger bufferIds = new AtomicInteger();
        AtomicInteger workspaceIds = new AtomicInteger();
        PreparedRepresentationPlan plan =
                new PreparedRepresentationPlan(
                        memoryPlan(1, 1),
                        List.of(List.of(
                                new CallerInput(),
                                new CreatedBuffer(() -> new TrackingBuffer(
                                        "buffer" + bufferIds.incrementAndGet(), null, null)))),
                        List.of(() -> new TrackingWorkspace(
                                "workspace" + workspaceIds.incrementAndGet(), null, null)));
        TrackingBuffer caller = new TrackingBuffer("caller", null, null);

        RunState first = RunStateCreation.create(plan, List.of(caller));
        RunState second = RunStateCreation.create(plan, List.of(caller));
        first.setBufferRepresentationValid(0, 1, true);

        assertAll(
                () -> assertNotSame(
                        first.bufferRepresentation(0, 1).representation(),
                        second.bufferRepresentation(0, 1).representation()),
                () -> assertNotSame(
                        first.workspaceRepresentation(0), second.workspaceRepresentation(0)),
                () -> assertTrue(first.isBufferRepresentationValid(0, 1)),
                () -> assertFalse(second.isBufferRepresentationValid(0, 1)),
                () -> assertTrue(first.isBufferRepresentationValid(0, 0)),
                () -> assertTrue(second.isBufferRepresentationValid(0, 0)));

        first.close();
        second.close();
        assertEquals(0, caller.closeCount);
    }

    private static PreparedMemoryPlan memoryPlan(int bufferCount, int workspaceCount) {
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        for (int index = 0; index < bufferCount; index++) {
            buffers.add(new PreparedMemoryPlan.BufferEntry(new BufferSlot(index), 0, 1));
        }
        var workspaces = new ArrayList<PreparedMemoryPlan.WorkspaceEntry>();
        for (int index = 0; index < workspaceCount; index++) {
            workspaces.add(new PreparedMemoryPlan.WorkspaceEntry(new WorkspaceSlot(index), 0, 1));
        }
        return new PreparedMemoryPlan(buffers, workspaces);
    }

    private static String classBytes(Class<?> type) throws Exception {
        String resource = "/" + type.getName().replace('.', '/') + ".class";
        try (InputStream stream = type.getResourceAsStream(resource)) {
            return new String(stream.readAllBytes(), StandardCharsets.ISO_8859_1);
        }
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> type, String message, Runnable action) {
        T failure = assertThrows(type, action::run);
        assertEquals(message, failure.getMessage());
    }

    private static class TrackingBuffer implements BufferRepresentation {
        private final String name;
        private final List<String> closeOrder;
        private final Throwable failure;
        private int closeCount;

        private TrackingBuffer(String name, List<String> closeOrder, Throwable failure) {
            this.name = name;
            this.closeOrder = closeOrder;
            this.failure = failure;
        }

        @Override
        public void close() {
            closeCount++;
            if (closeOrder != null) {
                closeOrder.add(name);
            }
            throwFailure(failure);
        }
    }

    private static class TrackingWorkspace implements WorkspaceRepresentation {
        private final String name;
        private final List<String> closeOrder;
        private final Throwable failure;
        private int closeCount;

        private TrackingWorkspace(String name, List<String> closeOrder, Throwable failure) {
            this.name = name;
            this.closeOrder = closeOrder;
            this.failure = failure;
        }

        @Override
        public void close() {
            closeCount++;
            if (closeOrder != null) {
                closeOrder.add(name);
            }
            throwFailure(failure);
        }
    }

    private static final class DualRepresentation
            implements BufferRepresentation, WorkspaceRepresentation {
        private final String name;
        private final List<String> closeOrder;
        private final Throwable failure;
        private int closeCount;

        private DualRepresentation(String name, List<String> closeOrder, Throwable failure) {
            this.name = name;
            this.closeOrder = closeOrder;
            this.failure = failure;
        }

        @Override
        public void close() {
            closeCount++;
            if (closeOrder != null) {
                closeOrder.add(name);
            }
            throwFailure(failure);
        }
    }

    private static void throwFailure(Throwable failure) {
        if (failure instanceof RuntimeException runtimeException) {
            throw runtimeException;
        }
        if (failure instanceof Error error) {
            throw error;
        }
    }
}
