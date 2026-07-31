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
import io.github.pho001.synaptik.runtime.memory.WorkspaceSlot;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class RunStateTest {
    @Test
    void hasTheExactPublicSurfaceAndArrayBackedState() throws ReflectiveOperationException {
        var type = RunState.class;
        var constructors = type.getDeclaredConstructors();
        var fields = type.getDeclaredFields();
        var constructor = constructors[0];

        assertAll(
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertFalse(type.isRecord()),
                () -> assertArrayEquals(
                        new Class<?>[] {AutoCloseable.class}, type.getInterfaces()),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPublic(constructor.getModifiers())),
                () -> assertArrayEquals(
                        new Class<?>[] {PreparedMemoryPlan.class, List.class, List.class},
                        constructor.getParameterTypes()),
                () -> assertConstructorGenericSurface(constructor.getGenericParameterTypes()),
                () -> assertEquals(4, fields.length),
                () -> assertField(type, "memoryPlan", PreparedMemoryPlan.class, true),
                () -> assertField(
                        type,
                        "bufferBindings",
                        BufferRepresentationBinding[][].class,
                        true),
                () -> assertField(
                        type,
                        "workspaceRepresentations",
                        WorkspaceRepresentation[].class,
                        true),
                () -> assertField(type, "closed", boolean.class, false),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertEquals(
                        Set.of(
                                "memoryPlan",
                                "bufferSlotCount",
                                "bufferRepresentationCount",
                                "bufferRepresentation",
                                "workspaceSlotCount",
                                "workspaceRepresentation",
                                "isClosed",
                                "close"),
                        Arrays.stream(type.getDeclaredMethods())
                                .filter(method -> Modifier.isPublic(method.getModifiers()))
                                .map(method -> method.getName())
                                .collect(Collectors.toSet())));
    }

    @Test
    void validatesTopLevelInputsInExactOrder() {
        PreparedMemoryPlan emptyPlan = memoryPlan(0, 0);

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "memoryPlan",
                        () -> new RunState(null, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "bufferBindings",
                        () -> new RunState(emptyPlan, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "workspaceRepresentations",
                        () -> new RunState(emptyPlan, List.of(), null)));
    }

    @Test
    void validatesPlanSizedCountsBeforeContentsWithExactFailures() {
        PreparedMemoryPlan plan = memoryPlan(2, 1);
        var bufferWithNull = new ArrayList<List<BufferRepresentationBinding>>();
        bufferWithNull.add(null);
        var workspaceWithNull = new ArrayList<WorkspaceRepresentation>();
        workspaceWithNull.add(null);

        assertAll(
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "bufferBindings size must equal prepared buffer count 2",
                        () -> new RunState(plan, bufferWithNull, List.of())),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "workspaceRepresentations size must equal prepared workspace count 1",
                        () -> new RunState(plan, List.of(List.of(), List.of()), List.of())),
                () -> assertFailure(
                        NullPointerException.class,
                        "bufferBindings[0]",
                        () -> {
                            var buffers =
                                    new ArrayList<List<BufferRepresentationBinding>>();
                            buffers.add(null);
                            buffers.add(List.of());
                            new RunState(plan, buffers, workspaceWithNull);
                        }));
    }

    @Test
    void validatesBufferPositionsInExactOrderWithExactFailures() {
        PreparedMemoryPlan plan = memoryPlan(2, 0);
        TrackingBuffer representation = new TrackingBuffer("buffer", null, null);
        BufferRepresentationBinding binding = owned(representation);
        var nullBindingList = new ArrayList<BufferRepresentationBinding>();
        nullBindingList.add(null);

        assertAll(
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "bufferBindings[0] must not be empty",
                        () -> new RunState(plan, List.of(List.of(), List.of(binding)), List.of())),
                () -> assertFailure(
                        NullPointerException.class,
                        "bufferBindings[0][0]",
                        () -> new RunState(
                                plan, List.of(nullBindingList, List.of(binding)), List.of())),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "representation is already bound to this run",
                        () -> new RunState(
                                plan,
                                List.of(List.of(binding), List.of(owned(representation))),
                                List.of())));
    }

    @Test
    void validatesWorkspacesAfterAllBuffersAndAcrossBothIdentityDomains() {
        PreparedMemoryPlan plan = memoryPlan(1, 2);
        DualRepresentation repeated = new DualRepresentation();
        BufferRepresentationBinding buffer = owned(repeated);
        var nullWorkspaces = new ArrayList<WorkspaceRepresentation>();
        nullWorkspaces.add(null);
        nullWorkspaces.add(repeated);

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "workspaceRepresentations[0]",
                        () -> new RunState(plan, List.of(List.of(buffer)), nullWorkspaces)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "representation is already bound to this run",
                        () -> new RunState(
                                plan, List.of(List.of(buffer)), List.of(repeated, new DualRepresentation()))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "representation is already bound to this run",
                        () -> {
                            DualRepresentation workspace = new DualRepresentation();
                            new RunState(
                                    plan,
                                    List.of(List.of(owned(new DualRepresentation()))),
                                    List.of(workspace, workspace));
                        }));
    }

    @Test
    void constructionFailureTransfersNoOwnershipAndClosesNothing() {
        PreparedMemoryPlan plan = memoryPlan(1, 1);
        DualRepresentation duplicate = new DualRepresentation();
        BufferRepresentationBinding binding = owned(duplicate);

        assertFailure(
                IllegalArgumentException.class,
                "representation is already bound to this run",
                () -> new RunState(plan, List.of(List.of(binding)), List.of(duplicate)));

        assertEquals(0, duplicate.closeCount);
    }

    @Test
    void snapshotsOnlyListStructureAndRetainsEveryExactReferenceInDensePlanOrder() {
        PreparedMemoryPlan plan = memoryPlan(2, 2);
        TrackingBuffer buffer00 = new TrackingBuffer("buffer00", null, null);
        TrackingBuffer buffer01 = new TrackingBuffer("buffer01", null, null);
        TrackingBuffer buffer10 = new TrackingBuffer("buffer10", null, null);
        BufferRepresentationBinding binding00 = borrowed(buffer00);
        BufferRepresentationBinding binding01 = owned(buffer01);
        BufferRepresentationBinding binding10 = owned(buffer10);
        TrackingWorkspace workspace0 = new TrackingWorkspace("workspace0", null, null, null);
        TrackingWorkspace workspace1 = new TrackingWorkspace("workspace1", null, null, null);
        var inner0 = new ArrayList<>(List.of(binding00, binding01));
        var inner1 = new ArrayList<>(List.of(binding10));
        var suppliedBuffers = new ArrayList<List<BufferRepresentationBinding>>();
        suppliedBuffers.add(inner0);
        suppliedBuffers.add(inner1);
        var suppliedWorkspaces = new ArrayList<WorkspaceRepresentation>();
        suppliedWorkspaces.add(workspace0);
        suppliedWorkspaces.add(workspace1);

        RunState state = new RunState(plan, suppliedBuffers, suppliedWorkspaces);
        inner0.clear();
        inner1.clear();
        suppliedBuffers.clear();
        suppliedWorkspaces.clear();

        assertAll(
                () -> assertSame(plan, state.memoryPlan()),
                () -> assertEquals(2, state.bufferSlotCount()),
                () -> assertEquals(2, state.bufferRepresentationCount(0)),
                () -> assertEquals(1, state.bufferRepresentationCount(1)),
                () -> assertSame(binding00, state.bufferRepresentation(0, 0)),
                () -> assertSame(binding01, state.bufferRepresentation(0, 1)),
                () -> assertSame(binding10, state.bufferRepresentation(1, 0)),
                () -> assertSame(buffer00, state.bufferRepresentation(0, 0).representation()),
                () -> assertSame(workspace0, state.workspaceRepresentation(0)),
                () -> assertSame(workspace1, state.workspaceRepresentation(1)),
                () -> assertEquals(2, state.workspaceSlotCount()),
                () -> assertEquals(100L, state.memoryPlan().buffers().get(0).slot().value()),
                () -> assertEquals(99L, state.memoryPlan().buffers().get(1).slot().value()),
                () -> assertEquals(700L, state.memoryPlan().workspaces().get(0).slot().value()));
    }

    @Test
    void invalidIndicesUseExactMessagesAndDensePositionBoundaries() {
        RunState state =
                new RunState(
                        memoryPlan(1, 1),
                        List.of(
                                List.of(
                                        owned(new TrackingBuffer("buffer", null, null)))),
                        List.of(new TrackingWorkspace("workspace", null, null, null)));

        assertAll(
                () -> assertFailure(
                        IndexOutOfBoundsException.class,
                        "bufferIndex out of range: -1",
                        () -> state.bufferRepresentationCount(-1)),
                () -> assertFailure(
                        IndexOutOfBoundsException.class,
                        "bufferIndex out of range: 1",
                        () -> state.bufferRepresentation(1, 0)),
                () -> assertFailure(
                        IndexOutOfBoundsException.class,
                        "representationIndex out of range: -1",
                        () -> state.bufferRepresentation(0, -1)),
                () -> assertFailure(
                        IndexOutOfBoundsException.class,
                        "representationIndex out of range: 1",
                        () -> state.bufferRepresentation(0, 1)),
                () -> assertFailure(
                        IndexOutOfBoundsException.class,
                        "workspaceIndex out of range: -1",
                        () -> state.workspaceRepresentation(-1)),
                () -> assertFailure(
                        IndexOutOfBoundsException.class,
                        "workspaceIndex out of range: 1",
                        () -> state.workspaceRepresentation(1)));
    }

    @Test
    void closeMarksClosedFirstAndCleansOwnedResourcesInExactReverseOrder() {
        List<String> closeOrder = new ArrayList<>();
        RunState[] holder = new RunState[1];
        TrackingBuffer buffer00 = new TrackingBuffer("buffer00", closeOrder, null);
        TrackingBuffer borrowed01 = new TrackingBuffer("borrowed01", closeOrder, null);
        TrackingBuffer buffer10 = new TrackingBuffer("buffer10", closeOrder, null);
        TrackingBuffer buffer11 = new TrackingBuffer("buffer11", closeOrder, null);
        TrackingWorkspace workspace0 =
                new TrackingWorkspace("workspace0", closeOrder, null, null);
        TrackingWorkspace workspace1 =
                new TrackingWorkspace(
                        "workspace1",
                        closeOrder,
                        () -> assertTrue(holder[0].isClosed()),
                        null);
        holder[0] =
                new RunState(
                        memoryPlan(2, 2),
                        List.of(
                                List.of(owned(buffer00), borrowed(borrowed01)),
                                List.of(owned(buffer10), owned(buffer11))),
                        List.of(workspace0, workspace1));

        holder[0].close();

        assertAll(
                () -> assertTrue(holder[0].isClosed()),
                () -> assertEquals(
                        List.of("workspace1", "workspace0", "buffer11", "buffer10", "buffer00"),
                        closeOrder),
                () -> assertEquals(1, workspace1.closeCount),
                () -> assertEquals(1, workspace0.closeCount),
                () -> assertEquals(1, buffer11.closeCount),
                () -> assertEquals(1, buffer10.closeCount),
                () -> assertEquals(0, borrowed01.closeCount),
                () -> assertEquals(1, buffer00.closeCount));
    }

    @Test
    void closePreservesRuntimeExceptionAndErrorFailuresInEncounterOrder() {
        List<String> closeOrder = new ArrayList<>();
        RuntimeException first = new RuntimeException("workspace1");
        AssertionError second = new AssertionError("workspace0");
        RuntimeException third = new RuntimeException("buffer1");
        TrackingBuffer buffer0 = new TrackingBuffer("buffer0", closeOrder, null);
        TrackingBuffer buffer1 = new TrackingBuffer("buffer1", closeOrder, third);
        TrackingWorkspace workspace0 =
                new TrackingWorkspace("workspace0", closeOrder, null, second);
        TrackingWorkspace workspace1 =
                new TrackingWorkspace("workspace1", closeOrder, null, first);
        RunState state =
                new RunState(
                        memoryPlan(2, 2),
                        List.of(List.of(owned(buffer0)), List.of(owned(buffer1))),
                        List.of(workspace0, workspace1));

        RuntimeException thrown = assertThrows(RuntimeException.class, state::close);

        assertAll(
                () -> assertSame(first, thrown),
                () -> assertArrayEquals(new Throwable[] {second, third}, thrown.getSuppressed()),
                () -> assertEquals(
                        List.of("workspace1", "workspace0", "buffer1", "buffer0"), closeOrder),
                () -> assertTrue(state.isClosed()),
                () -> assertDoesNotThrow(state::close),
                () -> assertEquals(1, workspace1.closeCount),
                () -> assertEquals(1, workspace0.closeCount),
                () -> assertEquals(1, buffer1.closeCount),
                () -> assertEquals(1, buffer0.closeCount));
    }

    @Test
    void closeRethrowsAnErrorWhenItIsTheFirstFailure() {
        AssertionError first = new AssertionError("workspace");
        RuntimeException second = new RuntimeException("buffer");
        RunState state =
                new RunState(
                        memoryPlan(1, 1),
                        List.of(
                                List.of(
                                        owned(new TrackingBuffer("buffer", null, second)))),
                        List.of(new TrackingWorkspace("workspace", null, null, first)));

        AssertionError thrown = assertThrows(AssertionError.class, state::close);

        assertAll(
                () -> assertSame(first, thrown),
                () -> assertArrayEquals(new Throwable[] {second}, thrown.getSuppressed()));
    }

    @Test
    void representationAccessClosesFirstWhilePlanAndCountsRemainInspectable() {
        PreparedMemoryPlan plan = memoryPlan(1, 1);
        RunState state =
                new RunState(
                        plan,
                        List.of(
                                List.of(
                                        owned(new TrackingBuffer("buffer", null, null)))),
                        List.of(new TrackingWorkspace("workspace", null, null, null)));
        state.close();

        assertAll(
                () -> assertFailure(
                        IllegalStateException.class,
                        "run state is closed",
                        () -> state.bufferRepresentation(-1, -1)),
                () -> assertFailure(
                        IllegalStateException.class,
                        "run state is closed",
                        () -> state.workspaceRepresentation(-1)),
                () -> assertSame(plan, state.memoryPlan()),
                () -> assertEquals(1, state.bufferSlotCount()),
                () -> assertEquals(1, state.bufferRepresentationCount(0)),
                () -> assertEquals(1, state.workspaceSlotCount()),
                () -> assertTrue(state.isClosed()),
                () -> assertDoesNotThrow(state::close));
    }

    @Test
    void separateStatesShareOnlyThePlanAndCloseIsolatedRunOwnedResourcesConcurrently()
            throws Exception {
        PreparedMemoryPlan plan = memoryPlan(2, 1);
        TrackingBuffer sharedBorrowed = new TrackingBuffer("borrowed", null, null);
        TrackingBuffer firstOwned0 = new TrackingBuffer("first0", null, null);
        TrackingBuffer firstOwned1 = new TrackingBuffer("first1", null, null);
        TrackingWorkspace firstWorkspace =
                new TrackingWorkspace("firstWorkspace", null, null, null);
        TrackingBuffer secondOwned0 = new TrackingBuffer("second0", null, null);
        TrackingBuffer secondOwned1 = new TrackingBuffer("second1", null, null);
        TrackingWorkspace secondWorkspace =
                new TrackingWorkspace("secondWorkspace", null, null, null);
        RunState first =
                new RunState(
                        plan,
                        List.of(
                                List.of(borrowed(sharedBorrowed), owned(firstOwned0)),
                                List.of(owned(firstOwned1))),
                        List.of(firstWorkspace));
        RunState second =
                new RunState(
                        plan,
                        List.of(
                                List.of(borrowed(sharedBorrowed), owned(secondOwned0)),
                                List.of(owned(secondOwned1))),
                        List.of(secondWorkspace));

        try (var executor = Executors.newFixedThreadPool(2)) {
            var firstClose = executor.submit(first::close);
            var secondClose = executor.submit(second::close);
            firstClose.get();
            secondClose.get();
        }

        assertAll(
                () -> assertSame(plan, first.memoryPlan()),
                () -> assertSame(plan, second.memoryPlan()),
                () -> assertTrue(first.isClosed()),
                () -> assertTrue(second.isClosed()),
                () -> assertEquals(0, sharedBorrowed.closeCount),
                () -> assertEquals(1, firstOwned0.closeCount),
                () -> assertEquals(1, firstOwned1.closeCount),
                () -> assertEquals(1, firstWorkspace.closeCount),
                () -> assertEquals(1, secondOwned0.closeCount),
                () -> assertEquals(1, secondOwned1.closeCount),
                () -> assertEquals(1, secondWorkspace.closeCount));
    }

    @Test
    void emptyPlanHasAnIndependentIdempotentLifecycle() {
        PreparedMemoryPlan plan = memoryPlan(0, 0);
        RunState state = new RunState(plan, List.of(), List.of());

        assertAll(
                () -> assertSame(plan, state.memoryPlan()),
                () -> assertEquals(0, state.bufferSlotCount()),
                () -> assertEquals(0, state.workspaceSlotCount()),
                () -> assertFalse(state.isClosed()));

        assertAll(
                () -> assertDoesNotThrow(state::close),
                () -> assertDoesNotThrow(state::close),
                () -> assertTrue(state.isClosed()));
    }

    private static PreparedMemoryPlan memoryPlan(int bufferCount, int workspaceCount) {
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        for (int index = 0; index < bufferCount; index++) {
            buffers.add(
                    new PreparedMemoryPlan.BufferEntry(
                            new BufferSlot(100L - index), index, 1L));
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

    private static BufferRepresentationBinding borrowed(BufferRepresentation representation) {
        return new BufferRepresentationBinding(
                representation, RunResourceOwnership.BORROWED);
    }

    private static void assertConstructorGenericSurface(Type[] parameterTypes) {
        assertEquals(PreparedMemoryPlan.class, parameterTypes[0]);
        var buffers = (ParameterizedType) parameterTypes[1];
        assertEquals(List.class, buffers.getRawType());
        var innerList = (ParameterizedType) buffers.getActualTypeArguments()[0];
        assertEquals(List.class, innerList.getRawType());
        assertArrayEquals(
                new Type[] {BufferRepresentationBinding.class},
                innerList.getActualTypeArguments());
        var workspaces = (ParameterizedType) parameterTypes[2];
        assertEquals(List.class, workspaces.getRawType());
        assertArrayEquals(
                new Type[] {WorkspaceRepresentation.class},
                workspaces.getActualTypeArguments());
    }

    private static void assertField(
            Class<?> owner, String name, Class<?> fieldType, boolean finalField)
            throws ReflectiveOperationException {
        var field = owner.getDeclaredField(name);
        assertEquals(fieldType, field.getType());
        assertTrue(Modifier.isPrivate(field.getModifiers()));
        assertEquals(finalField, Modifier.isFinal(field.getModifiers()));
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> failureType, String message, Runnable action) {
        T failure = assertThrows(failureType, action::run);
        assertEquals(message, failure.getMessage());
    }

    private static final class TrackingBuffer implements BufferRepresentation {
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

    private static final class TrackingWorkspace implements WorkspaceRepresentation {
        private final String name;
        private final List<String> closeOrder;
        private final Runnable closeAction;
        private final Throwable failure;
        private int closeCount;

        private TrackingWorkspace(
                String name,
                List<String> closeOrder,
                Runnable closeAction,
                Throwable failure) {
            this.name = name;
            this.closeOrder = closeOrder;
            this.closeAction = closeAction;
            this.failure = failure;
        }

        @Override
        public void close() {
            closeCount++;
            if (closeOrder != null) {
                closeOrder.add(name);
            }
            if (closeAction != null) {
                closeAction.run();
            }
            throwFailure(failure);
        }
    }

    private static final class DualRepresentation
            implements BufferRepresentation, WorkspaceRepresentation {
        private int closeCount;

        @Override
        public void close() {
            closeCount++;
        }
    }

    private static void throwFailure(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            throw runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
    }
}
