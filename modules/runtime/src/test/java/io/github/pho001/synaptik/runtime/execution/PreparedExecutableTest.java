package io.github.pho001.synaptik.runtime.execution;

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
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;

class PreparedExecutableTest {
    @Test
    void exposesExactAbstractTemplateAndNestedSelectionSurface() throws Exception {
        int modifiers = PreparedExecutable.class.getModifiers();
        var constructor = PreparedExecutable.class.getDeclaredConstructors()[0];
        var memoryPlan = PreparedExecutable.class.getDeclaredMethod("memoryPlan");
        var bind = PreparedExecutable.class.getDeclaredMethod("bind", RunState.class);
        var acceptsBuffer =
                PreparedExecutable.class.getDeclaredMethod(
                        "acceptsBufferRepresentation",
                        int.class,
                        BufferRepresentation.class);
        var acceptsWorkspace =
                PreparedExecutable.class.getDeclaredMethod(
                        "acceptsWorkspaceRepresentation",
                        int.class,
                        WorkspaceRepresentation.class);
        var bindCompatible =
                PreparedExecutable.class.getDeclaredMethod(
                        "bindCompatible",
                        RunState.class,
                        BufferRepresentation[].class,
                        WorkspaceRepresentation[].class);

        assertAll(
                () -> assertTrue(Modifier.isPublic(modifiers)),
                () -> assertTrue(Modifier.isAbstract(modifiers)),
                () -> assertFalse(Modifier.isFinal(modifiers)),
                () -> assertEquals(Object.class, PreparedExecutable.class.getSuperclass()),
                () -> assertEquals(1, PreparedExecutable.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isProtected(constructor.getModifiers())),
                () -> assertArrayEquals(
                        new Class<?>[] {PreparedMemoryPlan.class, List.class, List.class},
                        constructor.getParameterTypes()),
                () -> assertConstructorGenericSurface(constructor.getGenericParameterTypes()),
                () -> assertEquals(PreparedMemoryPlan.class, memoryPlan.getReturnType()),
                () -> assertTrue(Modifier.isPublic(memoryPlan.getModifiers())),
                () -> assertTrue(Modifier.isFinal(memoryPlan.getModifiers())),
                () -> assertEquals(BoundInvocation.class, bind.getReturnType()),
                () -> assertTrue(Modifier.isPublic(bind.getModifiers())),
                () -> assertTrue(Modifier.isFinal(bind.getModifiers())),
                () -> assertTrue(Modifier.isProtected(acceptsBuffer.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(acceptsBuffer.getModifiers())),
                () -> assertTrue(Modifier.isProtected(acceptsWorkspace.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(acceptsWorkspace.getModifiers())),
                () -> assertTrue(Modifier.isProtected(bindCompatible.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(bindCompatible.getModifiers())),
                () -> assertEquals(0, constructor.getExceptionTypes().length),
                () -> assertEquals(0, bind.getExceptionTypes().length),
                () -> assertEquals(0, acceptsBuffer.getExceptionTypes().length),
                () -> assertEquals(0, acceptsWorkspace.getExceptionTypes().length),
                () -> assertEquals(0, bindCompatible.getExceptionTypes().length),
                () -> assertArrayEquals(
                        new Class<?>[] {
                            PreparedExecutable.BufferSelection.class,
                            PreparedExecutable.WorkspaceSelection.class
                        },
                        Arrays.stream(PreparedExecutable.class.getDeclaredClasses())
                                .sorted((left, right) -> left.getSimpleName().compareTo(right.getSimpleName()))
                                .toArray(Class<?>[]::new)),
                () -> assertPrivateFinalField(
                        PreparedExecutable.class, "memoryPlan", PreparedMemoryPlan.class),
                () -> assertPrivateFinalField(
                        PreparedExecutable.class,
                        "bufferSelections",
                        PreparedExecutable.BufferSelection[].class),
                () -> assertPrivateFinalField(
                        PreparedExecutable.class,
                        "workspaceSelections",
                        PreparedExecutable.WorkspaceSelection[].class),
                () -> assertSelectionRecord(
                        PreparedExecutable.BufferSelection.class,
                        new String[] {"bufferIndex", "representationIndex"},
                        new Class<?>[] {int.class, int.class}),
                () -> assertSelectionRecord(
                        PreparedExecutable.WorkspaceSelection.class,
                        new String[] {"workspaceIndex"},
                        new Class<?>[] {int.class}));
    }

    @Test
    void selectionRecordsValidateComponentsInDeclarationOrder() {
        assertAll(
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "bufferIndex must be non-negative",
                        () -> new PreparedExecutable.BufferSelection(-1, -1)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "representationIndex must be non-negative",
                        () -> new PreparedExecutable.BufferSelection(0, -1)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "workspaceIndex must be non-negative",
                        () -> new PreparedExecutable.WorkspaceSelection(-1)));

        var buffer = new PreparedExecutable.BufferSelection(Integer.MAX_VALUE, Integer.MAX_VALUE);
        var workspace = new PreparedExecutable.WorkspaceSelection(Integer.MAX_VALUE);
        assertAll(
                () -> assertEquals(Integer.MAX_VALUE, buffer.bufferIndex()),
                () -> assertEquals(Integer.MAX_VALUE, buffer.representationIndex()),
                () -> assertEquals(Integer.MAX_VALUE, workspace.workspaceIndex()));
    }

    @Test
    void constructionRequiresTopLevelInputsBeforeScanningSelections() {
        PreparedMemoryPlan empty = memoryPlan(0, 0);
        PreparedMemoryPlan oneEach = memoryPlan(1, 1);
        List<PreparedExecutable.BufferSelection> invalidBuffers =
                List.of(new PreparedExecutable.BufferSelection(1, 0));

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "memoryPlan",
                        () -> new TestExecutable(null, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "bufferSelections",
                        () -> new TestExecutable(empty, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "workspaceSelections",
                        () -> new TestExecutable(empty, List.of(), null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "workspaceSelections",
                        () -> new TestExecutable(oneEach, invalidBuffers, null)));
    }

    @Test
    void constructionValidatesBufferThenWorkspaceEntriesInSuppliedOrder() {
        PreparedMemoryPlan plan = memoryPlan(1, 1);
        var nullThenOutOfRange = new ArrayList<PreparedExecutable.BufferSelection>();
        nullThenOutOfRange.add(null);
        nullThenOutOfRange.add(new PreparedExecutable.BufferSelection(1, 0));
        var nullWorkspace = new ArrayList<PreparedExecutable.WorkspaceSelection>();
        nullWorkspace.add(null);

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "bufferSelections[0]",
                        () -> new TestExecutable(plan, nullThenOutOfRange, List.of())),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "bufferSelections[0].bufferIndex out of prepared-plan range: 1",
                        () ->
                                new TestExecutable(
                                        plan,
                                        List.of(new PreparedExecutable.BufferSelection(1, 0)),
                                        nullWorkspace)),
                () -> assertFailure(
                        NullPointerException.class,
                        "workspaceSelections[0]",
                        () -> new TestExecutable(plan, List.of(), nullWorkspace)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "workspaceSelections[0].workspaceIndex out of prepared-plan range: 1",
                        () ->
                                new TestExecutable(
                                        plan,
                                        List.of(),
                                        List.of(new PreparedExecutable.WorkspaceSelection(1)))));
    }

    @Test
    void constructionSnapshotsListStructureAndRetainsExactSelectionReferences() throws Exception {
        PreparedMemoryPlan plan = memoryPlan(2, 2);
        var firstBuffer = new PreparedExecutable.BufferSelection(1, 0);
        var secondBuffer = new PreparedExecutable.BufferSelection(0, 1);
        var firstWorkspace = new PreparedExecutable.WorkspaceSelection(1);
        var secondWorkspace = new PreparedExecutable.WorkspaceSelection(0);
        var suppliedBuffers = new ArrayList<>(List.of(firstBuffer, secondBuffer, firstBuffer));
        var suppliedWorkspaces =
                new ArrayList<>(List.of(firstWorkspace, secondWorkspace, firstWorkspace));

        TestExecutable executable = new TestExecutable(plan, suppliedBuffers, suppliedWorkspaces);
        suppliedBuffers.clear();
        suppliedWorkspaces.clear();

        var bufferField = PreparedExecutable.class.getDeclaredField("bufferSelections");
        var workspaceField = PreparedExecutable.class.getDeclaredField("workspaceSelections");
        bufferField.setAccessible(true);
        workspaceField.setAccessible(true);
        var copiedBuffers =
                (PreparedExecutable.BufferSelection[]) bufferField.get(executable);
        var copiedWorkspaces =
                (PreparedExecutable.WorkspaceSelection[]) workspaceField.get(executable);

        assertAll(
                () -> assertSame(plan, executable.memoryPlan()),
                () -> assertEquals(3, copiedBuffers.length),
                () -> assertSame(firstBuffer, copiedBuffers[0]),
                () -> assertSame(secondBuffer, copiedBuffers[1]),
                () -> assertSame(firstBuffer, copiedBuffers[2]),
                () -> assertEquals(3, copiedWorkspaces.length),
                () -> assertSame(firstWorkspace, copiedWorkspaces[0]),
                () -> assertSame(secondWorkspace, copiedWorkspaces[1]),
                () -> assertSame(firstWorkspace, copiedWorkspaces[2]));
    }

    @Test
    void emptyAndRepeatedSelectionsResolveInOriginalDenseOrder() {
        PreparedMemoryPlan emptyPlan = memoryPlan(0, 0);
        TestExecutable empty = new TestExecutable(emptyPlan, List.of(), List.of());
        RunState emptyState = new RunState(emptyPlan, List.of(), List.of());
        BoundInvocation emptyInvocation = empty.bind(emptyState);

        PreparedMemoryPlan plan = memoryPlan(2, 2);
        CompatibleBuffer buffer00 = new CompatibleBuffer("buffer00");
        CompatibleBuffer buffer01 = new CompatibleBuffer("buffer01");
        CompatibleBuffer buffer10 = new CompatibleBuffer("buffer10");
        CompatibleWorkspace workspace0 = new CompatibleWorkspace("workspace0");
        CompatibleWorkspace workspace1 = new CompatibleWorkspace("workspace1");
        RunState state =
                new RunState(
                        plan,
                        List.of(
                                List.of(owned(buffer00), owned(buffer01)),
                                List.of(owned(buffer10))),
                        List.of(workspace0, workspace1));
        TestExecutable executable =
                new TestExecutable(
                        plan,
                        List.of(
                                new PreparedExecutable.BufferSelection(1, 0),
                                new PreparedExecutable.BufferSelection(0, 1),
                                new PreparedExecutable.BufferSelection(1, 0)),
                        List.of(
                                new PreparedExecutable.WorkspaceSelection(1),
                                new PreparedExecutable.WorkspaceSelection(0),
                                new PreparedExecutable.WorkspaceSelection(1)));

        ThreeResourceInvocation invocation =
                (ThreeResourceInvocation) executable.bind(state);

        assertAll(
                () -> assertTrue(emptyInvocation instanceof EmptyInvocation),
                () -> assertSame(buffer10, invocation.buffer0),
                () -> assertSame(buffer01, invocation.buffer1),
                () -> assertSame(buffer10, invocation.buffer2),
                () -> assertSame(workspace1, invocation.workspace0),
                () -> assertSame(workspace0, invocation.workspace1),
                () -> assertSame(workspace1, invocation.workspace2),
                () -> assertEquals(List.of(0, 1, 2), executable.bufferChecks),
                () -> assertEquals(List.of(0, 1, 2), executable.workspaceChecks),
                () -> assertEquals(1, executable.bindCalls));
    }

    @Test
    void bindRejectsNullClosedAndEqualButDistinctPlansInExactOrder() {
        PreparedMemoryPlan plan = memoryPlan(0, 0);
        PreparedMemoryPlan equalPlan = memoryPlan(0, 0);
        TestExecutable executable = new TestExecutable(plan, List.of(), List.of());
        RunState mismatched = new RunState(equalPlan, List.of(), List.of());
        RunState closedMismatched = new RunState(equalPlan, List.of(), List.of());
        closedMismatched.close();

        assertAll(
                () -> assertEquals(plan, equalPlan),
                () -> assertNotSame(plan, equalPlan),
                () -> assertFailure(
                        NullPointerException.class, "runState", () -> executable.bind(null)),
                () -> assertFailure(
                        IllegalStateException.class,
                        "run state is closed",
                        () -> executable.bind(closedMismatched)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "run state memory plan does not match prepared executable memory plan",
                        () -> executable.bind(mismatched)));
    }

    @Test
    void bindRejectsFirstMissingRepresentationBeforeCompatibility() {
        PreparedMemoryPlan plan = memoryPlan(1, 0);
        CompatibleBuffer buffer = new CompatibleBuffer("only");
        RunState state = new RunState(plan, List.of(List.of(owned(buffer))), List.of());
        TestExecutable executable =
                new TestExecutable(
                        plan,
                        List.of(
                                new PreparedExecutable.BufferSelection(0, 1),
                                new PreparedExecutable.BufferSelection(0, 0)),
                        List.of());

        assertFailure(
                IllegalArgumentException.class,
                "bufferSelections[0].representationIndex out of run-state range: 1",
                () -> executable.bind(state));
        assertAll(
                () -> assertEquals(List.of(), executable.bufferChecks),
                () -> assertEquals(List.of(), executable.workspaceChecks),
                () -> assertEquals(0, executable.bindCalls));
    }

    @Test
    void bufferCompatibilityRunsExactlyOncePerResolvedSelectionAndStopsOnFirstFailure() {
        PreparedMemoryPlan plan = memoryPlan(1, 1);
        CompatibleBuffer first = new CompatibleBuffer("first");
        OtherBuffer second = new OtherBuffer();
        CompatibleWorkspace workspace = new CompatibleWorkspace("workspace");
        RunState state =
                new RunState(
                        plan,
                        List.of(List.of(owned(first), owned(second))),
                        List.of(workspace));
        TestExecutable executable =
                new TestExecutable(
                        plan,
                        List.of(
                                new PreparedExecutable.BufferSelection(0, 0),
                                new PreparedExecutable.BufferSelection(0, 1)),
                        List.of(new PreparedExecutable.WorkspaceSelection(0)));

        assertFailure(
                IllegalArgumentException.class,
                "bufferSelections[1] is incompatible with prepared executable",
                () -> executable.bind(state));
        assertAll(
                () -> assertEquals(List.of(0, 1), executable.bufferChecks),
                () -> assertEquals(List.of(), executable.workspaceChecks),
                () -> assertEquals(0, executable.bindCalls));
    }

    @Test
    void workspaceCompatibilityRunsAfterEveryBufferAndStopsOnFirstFailure() {
        PreparedMemoryPlan plan = memoryPlan(1, 2);
        CompatibleBuffer buffer = new CompatibleBuffer("buffer");
        CompatibleWorkspace first = new CompatibleWorkspace("first");
        OtherWorkspace second = new OtherWorkspace();
        RunState state =
                new RunState(
                        plan,
                        List.of(List.of(owned(buffer))),
                        List.of(first, second));
        TestExecutable executable =
                new TestExecutable(
                        plan,
                        List.of(new PreparedExecutable.BufferSelection(0, 0)),
                        List.of(
                                new PreparedExecutable.WorkspaceSelection(0),
                                new PreparedExecutable.WorkspaceSelection(1)));

        assertFailure(
                IllegalArgumentException.class,
                "workspaceSelections[1] is incompatible with prepared executable",
                () -> executable.bind(state));
        assertAll(
                () -> assertEquals(List.of(0), executable.bufferChecks),
                () -> assertEquals(List.of(0, 1), executable.workspaceChecks),
                () -> assertEquals(0, executable.bindCalls));
    }

    @Test
    void bindCompatibleReceivesFreshExactArraysAndReturnsDirectTypedAssociatedInvocation()
            throws Exception {
        PreparedMemoryPlan plan = memoryPlan(1, 1);
        CompatibleBuffer buffer = new CompatibleBuffer("buffer");
        CompatibleWorkspace workspace = new CompatibleWorkspace("workspace");
        RunState state =
                new RunState(
                        plan,
                        List.of(List.of(owned(buffer))),
                        List.of(workspace));
        TestExecutable executable =
                new TestExecutable(
                        plan,
                        List.of(new PreparedExecutable.BufferSelection(0, 0)),
                        List.of(new PreparedExecutable.WorkspaceSelection(0)));

        OneResourceInvocation first = (OneResourceInvocation) executable.bind(state);
        BufferRepresentation[] firstBuffers = executable.lastBuffers;
        WorkspaceRepresentation[] firstWorkspaces = executable.lastWorkspaces;
        OneResourceInvocation second = (OneResourceInvocation) executable.bind(state);

        assertAll(
                () -> assertSame(state, first.runState()),
                () -> assertSame(state, second.runState()),
                () -> assertSame(buffer, first.buffer),
                () -> assertSame(workspace, first.workspace),
                () -> assertArrayEquals(
                        new BufferRepresentation[] {buffer}, firstBuffers),
                () -> assertArrayEquals(
                        new WorkspaceRepresentation[] {workspace}, firstWorkspaces),
                () -> assertNotSame(firstBuffers, executable.lastBuffers),
                () -> assertNotSame(firstWorkspaces, executable.lastWorkspaces),
                () -> assertEquals(2, executable.bindCalls),
                () -> assertPrivateFinalField(
                        OneResourceInvocation.class, "buffer", CompatibleBuffer.class),
                () -> assertPrivateFinalField(
                        OneResourceInvocation.class,
                        "workspace",
                        CompatibleWorkspace.class),
                () ->
                        assertTrue(
                                Arrays.stream(OneResourceInvocation.class.getDeclaredFields())
                                        .noneMatch(
                                                field ->
                                                        field.getType().isArray()
                                                                || List.class.isAssignableFrom(
                                                                        field.getType()))));
    }

    @Test
    void bindRejectsNullOrForeignInvocationAfterOneCompatibleBindCall() {
        PreparedMemoryPlan plan = memoryPlan(0, 0);
        RunState supplied = new RunState(plan, List.of(), List.of());
        RunState foreign = new RunState(plan, List.of(), List.of());
        ReturningExecutable nullReturning =
                new ReturningExecutable(plan, runState -> null);
        ReturningExecutable foreignReturning =
                new ReturningExecutable(plan, runState -> new EmptyInvocation(foreign));

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "boundInvocation",
                        () -> nullReturning.bind(supplied)),
                () -> assertEquals(1, nullReturning.bindCalls),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "bound invocation does not belong to supplied run state",
                        () -> foreignReturning.bind(supplied)),
                () -> assertEquals(1, foreignReturning.bindCalls));
    }

    @Test
    void oneImmutableRecipeBindsConcurrentlyToDistinctIsolatedRunStates() throws Exception {
        PreparedMemoryPlan plan = memoryPlan(1, 1);
        ImmutableExecutable executable = new ImmutableExecutable(plan);
        CompatibleBuffer firstBuffer = new CompatibleBuffer("firstBuffer");
        CompatibleWorkspace firstWorkspace = new CompatibleWorkspace("firstWorkspace");
        CompatibleBuffer secondBuffer = new CompatibleBuffer("secondBuffer");
        CompatibleWorkspace secondWorkspace = new CompatibleWorkspace("secondWorkspace");
        RunState firstState =
                new RunState(
                        plan,
                        List.of(List.of(owned(firstBuffer))),
                        List.of(firstWorkspace));
        RunState secondState =
                new RunState(
                        plan,
                        List.of(List.of(owned(secondBuffer))),
                        List.of(secondWorkspace));

        OneResourceInvocation first;
        OneResourceInvocation second;
        try (var executor = Executors.newFixedThreadPool(2)) {
            var firstBind = executor.submit(() -> (OneResourceInvocation) executable.bind(firstState));
            var secondBind =
                    executor.submit(() -> (OneResourceInvocation) executable.bind(secondState));
            first = firstBind.get();
            second = secondBind.get();
        }

        first.execute();
        second.execute();
        firstState.close();

        assertAll(
                () -> assertSame(firstState, first.runState()),
                () -> assertSame(secondState, second.runState()),
                () -> assertSame(firstBuffer, first.buffer),
                () -> assertSame(firstWorkspace, first.workspace),
                () -> assertSame(secondBuffer, second.buffer),
                () -> assertSame(secondWorkspace, second.workspace),
                () -> assertEquals(1, firstBuffer.executeCount),
                () -> assertEquals(1, firstWorkspace.executeCount),
                () -> assertEquals(1, secondBuffer.executeCount),
                () -> assertEquals(1, secondWorkspace.executeCount),
                () -> assertTrue(firstState.isClosed()),
                () -> assertFalse(secondState.isClosed()),
                () -> assertEquals(1, firstBuffer.closeCount),
                () -> assertEquals(1, firstWorkspace.closeCount),
                () -> assertEquals(0, secondBuffer.closeCount),
                () -> assertEquals(0, secondWorkspace.closeCount));
    }

    private static PreparedMemoryPlan memoryPlan(int bufferCount, int workspaceCount) {
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        for (int index = 0; index < bufferCount; index++) {
            buffers.add(
                    new PreparedMemoryPlan.BufferEntry(
                            new BufferSlot(900L - index), index, 1L));
        }
        var workspaces = new ArrayList<PreparedMemoryPlan.WorkspaceEntry>();
        for (int index = 0; index < workspaceCount; index++) {
            workspaces.add(
                    new PreparedMemoryPlan.WorkspaceEntry(
                            new WorkspaceSlot(700L - index), index, 1L));
        }
        return new PreparedMemoryPlan(buffers, workspaces);
    }

    private static BufferRepresentationBinding owned(BufferRepresentation representation) {
        return new BufferRepresentationBinding(
                representation, RunResourceOwnership.RUN_OWNED);
    }

    private static void assertConstructorGenericSurface(Type[] parameterTypes) {
        assertEquals(PreparedMemoryPlan.class, parameterTypes[0]);
        ParameterizedType buffers = (ParameterizedType) parameterTypes[1];
        ParameterizedType workspaces = (ParameterizedType) parameterTypes[2];
        assertEquals(List.class, buffers.getRawType());
        assertArrayEquals(
                new Type[] {PreparedExecutable.BufferSelection.class},
                buffers.getActualTypeArguments());
        assertEquals(List.class, workspaces.getRawType());
        assertArrayEquals(
                new Type[] {PreparedExecutable.WorkspaceSelection.class},
                workspaces.getActualTypeArguments());
    }

    private static void assertPrivateFinalField(Class<?> owner, String name, Class<?> type)
            throws Exception {
        var field = owner.getDeclaredField(name);
        assertEquals(type, field.getType());
        assertTrue(Modifier.isPrivate(field.getModifiers()));
        assertTrue(Modifier.isFinal(field.getModifiers()));
    }

    private static void assertSelectionRecord(
            Class<?> type, String[] componentNames, Class<?>[] componentTypes) {
        assertTrue(type.isRecord());
        assertTrue(Modifier.isPublic(type.getModifiers()));
        assertTrue(Modifier.isStatic(type.getModifiers()));
        assertTrue(Modifier.isFinal(type.getModifiers()));
        assertArrayEquals(
                componentNames,
                Arrays.stream(type.getRecordComponents())
                        .map(component -> component.getName())
                        .toArray(String[]::new));
        assertArrayEquals(
                componentTypes,
                Arrays.stream(type.getRecordComponents())
                        .map(component -> component.getType())
                        .toArray(Class<?>[]::new));
        assertEquals(0, type.getInterfaces().length);
        assertEquals(0, type.getDeclaredClasses().length);
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> failureType, String message, Runnable action) {
        T failure = assertThrows(failureType, action::run);
        assertEquals(message, failure.getMessage());
    }

    private static final class TestExecutable extends PreparedExecutable {
        private final List<Integer> bufferChecks = new ArrayList<>();
        private final List<Integer> workspaceChecks = new ArrayList<>();
        private int bindCalls;
        private BufferRepresentation[] lastBuffers;
        private WorkspaceRepresentation[] lastWorkspaces;

        private TestExecutable(
                PreparedMemoryPlan memoryPlan,
                List<PreparedExecutable.BufferSelection> bufferSelections,
                List<PreparedExecutable.WorkspaceSelection> workspaceSelections) {
            super(memoryPlan, bufferSelections, workspaceSelections);
        }

        @Override
        protected boolean acceptsBufferRepresentation(
                int selectionIndex, BufferRepresentation representation) {
            bufferChecks.add(selectionIndex);
            return representation instanceof CompatibleBuffer;
        }

        @Override
        protected boolean acceptsWorkspaceRepresentation(
                int selectionIndex, WorkspaceRepresentation representation) {
            workspaceChecks.add(selectionIndex);
            return representation instanceof CompatibleWorkspace;
        }

        @Override
        protected BoundInvocation bindCompatible(
                RunState runState,
                BufferRepresentation[] bufferRepresentations,
                WorkspaceRepresentation[] workspaceRepresentations) {
            bindCalls++;
            lastBuffers = bufferRepresentations;
            lastWorkspaces = workspaceRepresentations;
            return switch (bufferRepresentations.length) {
                case 0 -> new EmptyInvocation(runState);
                case 1 ->
                        new OneResourceInvocation(
                                runState,
                                (CompatibleBuffer) bufferRepresentations[0],
                                (CompatibleWorkspace) workspaceRepresentations[0]);
                case 3 ->
                        new ThreeResourceInvocation(
                                runState,
                                (CompatibleBuffer) bufferRepresentations[0],
                                (CompatibleBuffer) bufferRepresentations[1],
                                (CompatibleBuffer) bufferRepresentations[2],
                                (CompatibleWorkspace) workspaceRepresentations[0],
                                (CompatibleWorkspace) workspaceRepresentations[1],
                                (CompatibleWorkspace) workspaceRepresentations[2]);
                default -> throw new AssertionError("unexpected test selection count");
            };
        }
    }

    private interface InvocationFactory {
        BoundInvocation create(RunState runState);
    }

    private static final class ReturningExecutable extends PreparedExecutable {
        private final InvocationFactory factory;
        private int bindCalls;

        private ReturningExecutable(PreparedMemoryPlan plan, InvocationFactory factory) {
            super(plan, List.of(), List.of());
            this.factory = factory;
        }

        @Override
        protected boolean acceptsBufferRepresentation(
                int selectionIndex, BufferRepresentation representation) {
            throw new AssertionError("no buffer selections");
        }

        @Override
        protected boolean acceptsWorkspaceRepresentation(
                int selectionIndex, WorkspaceRepresentation representation) {
            throw new AssertionError("no workspace selections");
        }

        @Override
        protected BoundInvocation bindCompatible(
                RunState runState,
                BufferRepresentation[] bufferRepresentations,
                WorkspaceRepresentation[] workspaceRepresentations) {
            bindCalls++;
            return factory.create(runState);
        }
    }

    private static final class ImmutableExecutable extends PreparedExecutable {
        private ImmutableExecutable(PreparedMemoryPlan plan) {
            super(
                    plan,
                    List.of(new PreparedExecutable.BufferSelection(0, 0)),
                    List.of(new PreparedExecutable.WorkspaceSelection(0)));
        }

        @Override
        protected boolean acceptsBufferRepresentation(
                int selectionIndex, BufferRepresentation representation) {
            return selectionIndex == 0 && representation instanceof CompatibleBuffer;
        }

        @Override
        protected boolean acceptsWorkspaceRepresentation(
                int selectionIndex, WorkspaceRepresentation representation) {
            return selectionIndex == 0 && representation instanceof CompatibleWorkspace;
        }

        @Override
        protected BoundInvocation bindCompatible(
                RunState runState,
                BufferRepresentation[] bufferRepresentations,
                WorkspaceRepresentation[] workspaceRepresentations) {
            return new OneResourceInvocation(
                    runState,
                    (CompatibleBuffer) bufferRepresentations[0],
                    (CompatibleWorkspace) workspaceRepresentations[0]);
        }
    }

    private static final class EmptyInvocation extends BoundInvocation {
        private EmptyInvocation(RunState runState) {
            super(runState);
        }

        @Override
        protected void executeBound() {}
    }

    private static final class OneResourceInvocation extends BoundInvocation {
        private final CompatibleBuffer buffer;
        private final CompatibleWorkspace workspace;

        private OneResourceInvocation(
                RunState runState,
                CompatibleBuffer buffer,
                CompatibleWorkspace workspace) {
            super(runState);
            this.buffer = buffer;
            this.workspace = workspace;
        }

        @Override
        protected void executeBound() {
            buffer.executeCount++;
            workspace.executeCount++;
        }
    }

    private static final class ThreeResourceInvocation extends BoundInvocation {
        private final CompatibleBuffer buffer0;
        private final CompatibleBuffer buffer1;
        private final CompatibleBuffer buffer2;
        private final CompatibleWorkspace workspace0;
        private final CompatibleWorkspace workspace1;
        private final CompatibleWorkspace workspace2;

        private ThreeResourceInvocation(
                RunState runState,
                CompatibleBuffer buffer0,
                CompatibleBuffer buffer1,
                CompatibleBuffer buffer2,
                CompatibleWorkspace workspace0,
                CompatibleWorkspace workspace1,
                CompatibleWorkspace workspace2) {
            super(runState);
            this.buffer0 = buffer0;
            this.buffer1 = buffer1;
            this.buffer2 = buffer2;
            this.workspace0 = workspace0;
            this.workspace1 = workspace1;
            this.workspace2 = workspace2;
        }

        @Override
        protected void executeBound() {}
    }

    private static final class CompatibleBuffer implements BufferRepresentation {
        private final String name;
        private int executeCount;
        private int closeCount;

        private CompatibleBuffer(String name) {
            this.name = name;
        }

        @Override
        public void close() {
            closeCount++;
        }

        @Override
        public String toString() {
            return name;
        }
    }

    private static final class OtherBuffer implements BufferRepresentation {
        @Override
        public void close() {}
    }

    private static final class CompatibleWorkspace implements WorkspaceRepresentation {
        private final String name;
        private int executeCount;
        private int closeCount;

        private CompatibleWorkspace(String name) {
            this.name = name;
        }

        @Override
        public void close() {
            closeCount++;
        }

        @Override
        public String toString() {
            return name;
        }
    }

    private static final class OtherWorkspace implements WorkspaceRepresentation {
        @Override
        public void close() {}
    }
}
