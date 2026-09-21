package io.github.pho001.synaptik.prepare;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedResource;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class FinalizationPublicShapeTest {
    @Test
    void exposesThePlannedPublicRootTypes() {
        assertAll(
                () -> assertPublicSealedInterface(PreparationResourceAssignment.class),
                () -> assertPublicRecord(BackendPartitionFinalization.class),
                () -> assertPublicInterface(BackendPartitionFinalizer.class),
                () -> assertPublicRecord(BackendPartitionFinalizationResult.class),
                () -> assertPublicRecord(PreparedPartition.class),
                () -> assertFalse(Modifier.isPublic(
                        BackendPartitionFinalizationHandoff.class.getModifiers())),
                () -> assertEquals(
                        BackendPreparationPlan.class,
                        BackendPartitionFinalization.class
                                .getTypeParameters()[0]
                                .getBounds()[0]),
                () -> assertEquals(
                        BackendPreparationPlan.class,
                        BackendPartitionFinalizer.class
                                .getTypeParameters()[0]
                                .getBounds()[0]));
    }

    @Test
    void locksAssignmentAndInternalHandoffRecordShapes() {
        assertAll(
                () -> assertEquals(
                        java.util.Set.of(
                                PreparationResourceAssignment.Buffer.class,
                                PreparationResourceAssignment.Workspace.class),
                        java.util.Set.of(PreparationResourceAssignment.class.getPermittedSubclasses())),
                () -> assertRecordComponents(
                        PreparationResourceAssignment.Buffer.class,
                        "requirement", "slot", "planIndex"),
                () -> assertRecordComponents(
                        PreparationResourceAssignment.Workspace.class,
                        "requirement", "slot", "planIndex"),
                () -> assertRecordComponents(
                        BackendPartitionFinalization.class,
                        "analysis", "memoryPlan", "assignments"),
                () -> assertRecordComponents(
                        BackendPartitionFinalizationResult.class, "executable", "resources"),
                () -> assertRecordComponents(
                        PreparedPartition.class, "partition", "executable"),
                () -> assertRecordComponents(
                        BackendPartitionFinalizationHandoff.Entry.class,
                        "context", "analysis", "finalizer"),
                () -> assertRecordComponents(
                        BackendPartitionFinalizationHandoff.Result.class,
                        "memoryPlan", "partitions", "bufferAssignments", "resources"),
                () -> assertTrue(Arrays.stream(
                                BackendPartitionFinalizationHandoff.class.getDeclaredMethods())
                        .anyMatch(method -> method.getName().equals("finalizePartitions")
                                && Modifier.isStatic(method.getModifiers())
                                && !Modifier.isPublic(method.getModifiers()))),
                () -> assertTrue(BackendPartitionFinalization.class
                                .getRecordComponents()[2]
                                .getGenericType()
                        instanceof ParameterizedType));
    }

    @Test
    void finalizationResultSnapshotsResourcesAndValidatesNullsInOrder() {
        PreparedExecutable executable = new TestExecutable();
        PreparedResource resource = () -> {};
        var supplied = new ArrayList<>(List.of(resource));
        var result = new BackendPartitionFinalizationResult(executable, supplied);
        supplied.clear();

        assertAll(
                () -> assertSame(executable, result.executable()),
                () -> assertEquals(List.of(resource), result.resources()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> result.resources().clear()),
                () -> assertTrue(new BackendPartitionFinalizationResult(executable)
                        .resources().isEmpty()),
                () -> assertEquals("executable", assertThrows(NullPointerException.class,
                        () -> new BackendPartitionFinalizationResult(null, null)).getMessage()),
                () -> assertEquals("resources", assertThrows(NullPointerException.class,
                        () -> new BackendPartitionFinalizationResult(executable, null)).getMessage()),
                () -> assertEquals("resources[0]", assertThrows(NullPointerException.class,
                        () -> new BackendPartitionFinalizationResult(
                                executable, Arrays.asList((PreparedResource) null))).getMessage()));
    }

    private static final class TestExecutable extends PreparedExecutable {
        private TestExecutable() {
            super(new PreparedMemoryPlan(List.of(), List.of()), List.of(), List.of());
        }

        @Override
        protected boolean acceptsBufferRepresentation(
                int selectionIndex, BufferRepresentation representation) {
            return true;
        }

        @Override
        protected boolean acceptsWorkspaceRepresentation(
                int selectionIndex, WorkspaceRepresentation representation) {
            return true;
        }

        @Override
        protected BoundInvocation bindCompatible(
                RunState state,
                BufferRepresentation[] buffers,
                WorkspaceRepresentation[] workspaces) {
            throw new UnsupportedOperationException();
        }
    }

    private static void assertPublicRecord(Class<?> type) {
        assertTrue(Modifier.isPublic(type.getModifiers()));
        assertTrue(type.isRecord());
    }

    private static void assertPublicInterface(Class<?> type) {
        assertTrue(Modifier.isPublic(type.getModifiers()));
        assertTrue(type.isInterface());
    }

    private static void assertPublicSealedInterface(Class<?> type) {
        assertPublicInterface(type);
        assertTrue(type.isSealed());
    }

    private static void assertRecordComponents(Class<?> type, String... names) {
        assertTrue(type.isRecord());
        assertEquals(
                java.util.List.of(names),
                Arrays.stream(type.getRecordComponents()).map(component -> component.getName()).toList());
    }
}
