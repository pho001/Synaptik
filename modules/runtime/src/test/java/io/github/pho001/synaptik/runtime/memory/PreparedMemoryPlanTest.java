package io.github.pho001.synaptik.runtime.memory;

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
import java.lang.reflect.RecordComponent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class PreparedMemoryPlanTest {
    @Test
    void hasTheExactTopLevelPublicRecordShape() throws ReflectiveOperationException {
        Class<PreparedMemoryPlan> type = PreparedMemoryPlan.class;
        RecordComponent[] components = type.getRecordComponents();
        Class<?>[] nestedTypes = type.getDeclaredClasses();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.runtime.memory", type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () -> assertEquals(2, components.length),
                () -> assertEquals("buffers", components[0].getName()),
                () -> assertEquals(List.class, components[0].getType()),
                () -> assertGenericListComponent(components[0], PreparedMemoryPlan.BufferEntry.class),
                () -> assertEquals("workspaces", components[1].getName()),
                () -> assertEquals(List.class, components[1].getType()),
                () -> assertGenericListComponent(
                        components[1], PreparedMemoryPlan.WorkspaceEntry.class),
                () -> assertEquals(2, type.getDeclaredFields().length),
                () -> assertEquals(1, type.getDeclaredConstructors().length),
                () -> assertArrayEquals(
                        new Class<?>[] {List.class, List.class},
                        type.getDeclaredConstructors()[0].getParameterTypes()),
                () -> assertTrue(
                        Modifier.isPublic(type.getDeclaredConstructors()[0].getModifiers())),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertFalse(Serializable.class.isAssignableFrom(type)),
                () -> assertEquals(
                        Set.of(
                                PreparedMemoryPlan.BufferEntry.class,
                                PreparedMemoryPlan.WorkspaceEntry.class),
                        Set.of(nestedTypes)),
                () -> assertEquals(
                        Set.of("buffers", "workspaces", "equals", "hashCode", "toString"),
                        publicMethodNames(type)));
    }

    @Test
    void hasTheExactTwoNestedPublicRecordShapes() throws ReflectiveOperationException {
        assertEntryRecordShape(
                PreparedMemoryPlan.BufferEntry.class, BufferSlot.class, "BufferEntry");
        assertEntryRecordShape(
                PreparedMemoryPlan.WorkspaceEntry.class, WorkspaceSlot.class, "WorkspaceEntry");
    }

    @Test
    void entryRecordsRetainExactReferencesAndValidBoundaries() {
        BufferSlot bufferSlot = new BufferSlot(Long.MAX_VALUE);
        WorkspaceSlot workspaceSlot = new WorkspaceSlot(Long.MAX_VALUE);
        PreparedMemoryPlan.BufferEntry buffer =
                new PreparedMemoryPlan.BufferEntry(bufferSlot, Long.MAX_VALUE, 1L << 62);
        PreparedMemoryPlan.WorkspaceEntry workspace =
                new PreparedMemoryPlan.WorkspaceEntry(workspaceSlot, 0L, 1L);

        assertAll(
                () -> assertSame(bufferSlot, buffer.slot()),
                () -> assertEquals(Long.MAX_VALUE, buffer.byteSize()),
                () -> assertEquals(1L << 62, buffer.byteAlignment()),
                () -> assertSame(workspaceSlot, workspace.slot()),
                () -> assertEquals(0L, workspace.byteSize()),
                () -> assertEquals(1L, workspace.byteAlignment()));
    }

    @Test
    void entryRecordsAcceptEveryPositivePowerOfTwoLongAlignment() {
        BufferSlot bufferSlot = new BufferSlot(0L);
        WorkspaceSlot workspaceSlot = new WorkspaceSlot(0L);

        for (int shift = 0; shift <= 62; shift++) {
            long alignment = 1L << shift;
            assertAll(
                    () -> assertEquals(
                            alignment,
                            new PreparedMemoryPlan.BufferEntry(bufferSlot, 0L, alignment)
                                    .byteAlignment()),
                    () -> assertEquals(
                            alignment,
                            new PreparedMemoryPlan.WorkspaceEntry(
                                            workspaceSlot, 0L, alignment)
                                    .byteAlignment()));
        }
    }

    @Test
    void bufferEntryRejectsComponentsInExactOrderWithExactFailures() {
        BufferSlot slot = new BufferSlot(0L);

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "slot",
                        () -> new PreparedMemoryPlan.BufferEntry(null, -1L, 3L)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "byteSize must be non-negative",
                        () -> new PreparedMemoryPlan.BufferEntry(slot, -1L, 3L)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "byteAlignment must be a positive power of two",
                        () -> new PreparedMemoryPlan.BufferEntry(slot, 0L, 3L)));
    }

    @Test
    void workspaceEntryRejectsComponentsInExactOrderWithExactFailures() {
        WorkspaceSlot slot = new WorkspaceSlot(0L);

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "slot",
                        () -> new PreparedMemoryPlan.WorkspaceEntry(null, -1L, 3L)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "byteSize must be non-negative",
                        () -> new PreparedMemoryPlan.WorkspaceEntry(slot, -1L, 3L)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "byteAlignment must be a positive power of two",
                        () -> new PreparedMemoryPlan.WorkspaceEntry(slot, 0L, 3L)));
    }

    @Test
    void entryRecordsRejectEveryInvalidAlignmentBoundary() {
        BufferSlot bufferSlot = new BufferSlot(0L);
        WorkspaceSlot workspaceSlot = new WorkspaceSlot(0L);

        for (long invalid :
                new long[] {Long.MIN_VALUE, -2L, -1L, 0L, 3L, 6L, Long.MAX_VALUE}) {
            assertAll(
                    () -> assertFailure(
                            IllegalArgumentException.class,
                            "byteAlignment must be a positive power of two",
                            () -> new PreparedMemoryPlan.BufferEntry(bufferSlot, 0L, invalid)),
                    () -> assertFailure(
                            IllegalArgumentException.class,
                            "byteAlignment must be a positive power of two",
                            () -> new PreparedMemoryPlan.WorkspaceEntry(
                                    workspaceSlot, 0L, invalid)));
        }
    }

    @Test
    void acceptsACompletelyEmptyPlan() {
        PreparedMemoryPlan plan = new PreparedMemoryPlan(List.of(), List.of());

        assertAll(
                () -> assertTrue(plan.buffers().isEmpty()),
                () -> assertTrue(plan.workspaces().isEmpty()));
    }

    @Test
    void validatesTopLevelListsBeforeScanningEntries() {
        var bufferWithNull = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        bufferWithNull.add(null);

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "buffers",
                        () -> new PreparedMemoryPlan(null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "workspaces",
                        () -> new PreparedMemoryPlan(bufferWithNull, null)));
    }

    @Test
    void rejectsTheFirstNullEntryInSuppliedOrderWithExactFailures() {
        PreparedMemoryPlan.BufferEntry buffer =
                new PreparedMemoryPlan.BufferEntry(new BufferSlot(0L), 0L, 1L);
        PreparedMemoryPlan.WorkspaceEntry workspace =
                new PreparedMemoryPlan.WorkspaceEntry(new WorkspaceSlot(0L), 0L, 1L);
        List<PreparedMemoryPlan.BufferEntry> buffers = new ArrayList<>();
        buffers.add(buffer);
        buffers.add(null);
        List<PreparedMemoryPlan.WorkspaceEntry> workspaces = new ArrayList<>();
        workspaces.add(workspace);
        workspaces.add(null);

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "buffers[1]",
                        () -> new PreparedMemoryPlan(buffers, List.of())),
                () -> assertFailure(
                        NullPointerException.class,
                        "workspaces[1]",
                        () -> new PreparedMemoryPlan(List.of(), workspaces)));
    }

    @Test
    void rejectsTheFirstLaterDuplicateInEachSeparateDomainWithExactFailures() {
        BufferSlot firstBufferSlot = new BufferSlot(9L);
        WorkspaceSlot firstWorkspaceSlot = new WorkspaceSlot(7L);

        var buffers =
                List.of(
                        new PreparedMemoryPlan.BufferEntry(firstBufferSlot, 1L, 1L),
                        new PreparedMemoryPlan.BufferEntry(new BufferSlot(3L), 1L, 1L),
                        new PreparedMemoryPlan.BufferEntry(new BufferSlot(9L), 2L, 2L),
                        new PreparedMemoryPlan.BufferEntry(new BufferSlot(9L), 4L, 4L));
        var workspaces =
                List.of(
                        new PreparedMemoryPlan.WorkspaceEntry(firstWorkspaceSlot, 1L, 1L),
                        new PreparedMemoryPlan.WorkspaceEntry(new WorkspaceSlot(7L), 2L, 2L));

        assertAll(
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "buffers[2].slot duplicates BufferSlot[value=9]",
                        () -> new PreparedMemoryPlan(buffers, List.of())),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "workspaces[1].slot duplicates WorkspaceSlot[value=7]",
                        () -> new PreparedMemoryPlan(List.of(), workspaces)));
    }

    @Test
    void bufferValidationCompletesBeforeWorkspaceEntryValidation() {
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        buffers.add(null);
        var workspaces = new ArrayList<PreparedMemoryPlan.WorkspaceEntry>();
        workspaces.add(null);

        assertFailure(
                NullPointerException.class,
                "buffers[0]",
                () -> new PreparedMemoryPlan(buffers, workspaces));
    }

    @Test
    void snapshotsSuppliedOrderAndRetainsExactEntryAndSlotReferences() {
        BufferSlot bufferSlot0 = new BufferSlot(3L);
        BufferSlot bufferSlot1 = new BufferSlot(1L);
        WorkspaceSlot workspaceSlot0 = new WorkspaceSlot(8L);
        WorkspaceSlot workspaceSlot1 = new WorkspaceSlot(2L);
        var buffer0 = new PreparedMemoryPlan.BufferEntry(bufferSlot0, 30L, 2L);
        var buffer1 = new PreparedMemoryPlan.BufferEntry(bufferSlot1, 10L, 1L);
        var workspace0 = new PreparedMemoryPlan.WorkspaceEntry(workspaceSlot0, 80L, 8L);
        var workspace1 = new PreparedMemoryPlan.WorkspaceEntry(workspaceSlot1, 20L, 4L);
        var suppliedBuffers = new ArrayList<>(List.of(buffer0, buffer1));
        var suppliedWorkspaces = new ArrayList<>(List.of(workspace0, workspace1));

        PreparedMemoryPlan plan = new PreparedMemoryPlan(suppliedBuffers, suppliedWorkspaces);
        suppliedBuffers.clear();
        suppliedWorkspaces.clear();

        assertAll(
                () -> assertEquals(List.of(buffer0, buffer1), plan.buffers()),
                () -> assertEquals(List.of(workspace0, workspace1), plan.workspaces()),
                () -> assertSame(buffer0, plan.buffers().get(0)),
                () -> assertSame(buffer1, plan.buffers().get(1)),
                () -> assertSame(bufferSlot0, plan.buffers().get(0).slot()),
                () -> assertSame(bufferSlot1, plan.buffers().get(1).slot()),
                () -> assertSame(workspace0, plan.workspaces().get(0)),
                () -> assertSame(workspace1, plan.workspaces().get(1)),
                () -> assertSame(workspaceSlot0, plan.workspaces().get(0).slot()),
                () -> assertSame(workspaceSlot1, plan.workspaces().get(1).slot()));
    }

    @Test
    void exposesImmutableSnapshots() {
        PreparedMemoryPlan plan =
                new PreparedMemoryPlan(
                        List.of(
                                new PreparedMemoryPlan.BufferEntry(
                                        new BufferSlot(0L), 0L, 1L)),
                        List.of(
                                new PreparedMemoryPlan.WorkspaceEntry(
                                        new WorkspaceSlot(0L), 0L, 1L)));

        assertAll(
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> plan.buffers().add(plan.buffers().getFirst())),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> plan.workspaces().add(plan.workspaces().getFirst())));
    }

    @Test
    void permitsEqualNumericValuesAcrossSeparateSlotDomains() {
        PreparedMemoryPlan plan =
                new PreparedMemoryPlan(
                        List.of(
                                new PreparedMemoryPlan.BufferEntry(
                                        new BufferSlot(4L), 16L, 4L)),
                        List.of(
                                new PreparedMemoryPlan.WorkspaceEntry(
                                        new WorkspaceSlot(4L), 16L, 4L)));

        assertAll(
                () -> assertEquals(4L, plan.buffers().getFirst().slot().value()),
                () -> assertEquals(4L, plan.workspaces().getFirst().slot().value()));
    }

    @Test
    void usesOrdinaryRecordValueHashingAndDiagnosticSemantics() {
        PreparedMemoryPlan.BufferEntry buffer =
                new PreparedMemoryPlan.BufferEntry(new BufferSlot(2L), 16L, 8L);
        PreparedMemoryPlan.WorkspaceEntry workspace =
                new PreparedMemoryPlan.WorkspaceEntry(new WorkspaceSlot(3L), 32L, 16L);
        PreparedMemoryPlan first = new PreparedMemoryPlan(List.of(buffer), List.of(workspace));
        PreparedMemoryPlan equal =
                new PreparedMemoryPlan(
                        List.of(
                                new PreparedMemoryPlan.BufferEntry(
                                        new BufferSlot(2L), 16L, 8L)),
                        List.of(
                                new PreparedMemoryPlan.WorkspaceEntry(
                                        new WorkspaceSlot(3L), 32L, 16L)));
        PreparedMemoryPlan different = new PreparedMemoryPlan(List.of(), List.of(workspace));

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, different),
                () -> assertEquals(
                        "BufferEntry[slot=BufferSlot[value=2], byteSize=16, byteAlignment=8]",
                        buffer.toString()),
                () -> assertEquals(
                        "WorkspaceEntry[slot=WorkspaceSlot[value=3], byteSize=32, "
                                + "byteAlignment=16]",
                        workspace.toString()),
                () -> assertEquals(
                        "PreparedMemoryPlan[buffers=["
                                + buffer
                                + "], workspaces=["
                                + workspace
                                + "]]",
                        first.toString()));
    }

    private static void assertGenericListComponent(
            RecordComponent component, Class<?> entryType) {
        ParameterizedType genericType = (ParameterizedType) component.getGenericType();
        assertEquals(List.class, genericType.getRawType());
        assertArrayEquals(new Object[] {entryType}, genericType.getActualTypeArguments());
    }

    private static void assertEntryRecordShape(
            Class<?> type, Class<?> slotType, String simpleName)
            throws ReflectiveOperationException {
        RecordComponent[] components = type.getRecordComponents();

        assertAll(
                () -> assertEquals(simpleName, type.getSimpleName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isStatic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () -> assertEquals(3, components.length),
                () -> assertEquals("slot", components[0].getName()),
                () -> assertEquals(slotType, components[0].getType()),
                () -> assertEquals("byteSize", components[1].getName()),
                () -> assertEquals(long.class, components[1].getType()),
                () -> assertEquals("byteAlignment", components[2].getName()),
                () -> assertEquals(long.class, components[2].getType()),
                () -> assertEquals(3, type.getDeclaredFields().length),
                () -> assertEquals(1, type.getDeclaredConstructors().length),
                () -> assertArrayEquals(
                        new Class<?>[] {slotType, long.class, long.class},
                        type.getDeclaredConstructors()[0].getParameterTypes()),
                () -> assertTrue(
                        Modifier.isPublic(type.getDeclaredConstructors()[0].getModifiers())),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertFalse(Serializable.class.isAssignableFrom(type)),
                () -> assertEquals(
                        Set.of(
                                "slot",
                                "byteSize",
                                "byteAlignment",
                                "equals",
                                "hashCode",
                                "toString"),
                        publicMethodNames(type)));
    }

    private static Set<String> publicMethodNames(Class<?> type) {
        return Arrays.stream(type.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(method -> method.getName())
                .collect(Collectors.toSet());
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> failureType, String message, Runnable construction) {
        T failure = assertThrows(failureType, construction::run);
        assertEquals(message, failure.getMessage());
    }
}
