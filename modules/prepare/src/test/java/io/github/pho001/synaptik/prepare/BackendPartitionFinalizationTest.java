package io.github.pho001.synaptik.prepare;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.memory.WorkspaceSlot;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class BackendPartitionFinalizationTest {
    @Test
    void assignmentValuesValidateInDeclarationOrder() {
        var buffer = new PreparationResourceRequirement.Buffer(new ValueId(0), 4, 4);
        var workspace = new PreparationResourceRequirement.Workspace(0, 8, 8);
        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "requirement",
                        () -> new PreparationResourceAssignment.Buffer(null, null, -1)),
                () -> assertFailure(
                        NullPointerException.class,
                        "slot",
                        () -> new PreparationResourceAssignment.Buffer(buffer, null, -1)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "planIndex must be non-negative",
                        () -> new PreparationResourceAssignment.Buffer(buffer, new BufferSlot(0), -1)),
                () -> assertFailure(
                        NullPointerException.class,
                        "requirement",
                        () -> new PreparationResourceAssignment.Workspace(null, null, -1)),
                () -> assertFailure(
                        NullPointerException.class,
                        "slot",
                        () -> new PreparationResourceAssignment.Workspace(workspace, null, -1)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "planIndex must be non-negative",
                        () -> new PreparationResourceAssignment.Workspace(
                                workspace, new WorkspaceSlot(0), -1)));
    }

    @Test
    void finalizationRetainsExactReferencesAndSnapshotsAssignments() {
        Fixture fixture = fixture();
        var supplied = new ArrayList<>(fixture.assignments);
        var finalization = new BackendPartitionFinalization<>(
                fixture.analysis, fixture.memoryPlan, supplied);
        supplied.clear();

        assertAll(
                () -> assertSame(fixture.analysis, finalization.analysis()),
                () -> assertSame(fixture.memoryPlan, finalization.memoryPlan()),
                () -> assertEquals(fixture.assignments, finalization.assignments()),
                () -> assertNotSame(supplied, finalization.assignments()),
                () -> assertSame(
                        fixture.assignments.getFirst(), finalization.assignments().getFirst()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> finalization.assignments().clear()));
    }

    @Test
    void finalizationRejectsCoverageSourceIndexSlotAndGeometryFailuresExactly() {
        Fixture fixture = fixture();
        var foreignBuffer = new PreparationResourceRequirement.Buffer(new ValueId(0), 4, 4);
        var foreignSlot = new BufferSlot(9);
        var shortPlan = new PreparedMemoryPlan(
                List.of(new PreparedMemoryPlan.BufferEntry(fixture.bufferSlot, 3, 4)),
                fixture.memoryPlan.workspaces());
        var wrongWorkspacePlan = new PreparedMemoryPlan(
                fixture.memoryPlan.buffers(),
                List.of(new PreparedMemoryPlan.WorkspaceEntry(fixture.workspaceSlot, 9, 8)));
        var assignmentsWithNull = new ArrayList<PreparationResourceAssignment>();
        assignmentsWithNull.add(null);
        assignmentsWithNull.add(fixture.assignments.get(1));

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "analysis",
                        () -> new BackendPartitionFinalization<>(null, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "memoryPlan",
                        () -> new BackendPartitionFinalization<>(fixture.analysis, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "assignments",
                        () -> new BackendPartitionFinalization<>(
                                fixture.analysis, fixture.memoryPlan, null)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "assignments size must equal analysis requirement count 2",
                        () -> new BackendPartitionFinalization<>(
                                fixture.analysis, fixture.memoryPlan, List.of())),
                () -> assertFailure(
                        NullPointerException.class,
                        "assignments[0]",
                        () -> new BackendPartitionFinalization<>(
                                fixture.analysis, fixture.memoryPlan, assignmentsWithNull)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "assignments[0].requirement does not match analysis.requirements[0]",
                        () -> new BackendPartitionFinalization<>(
                                fixture.analysis,
                                fixture.memoryPlan,
                                List.of(
                                        new PreparationResourceAssignment.Buffer(
                                                foreignBuffer, fixture.bufferSlot, 0),
                                        fixture.assignments.get(1)))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "assignments[0] buffer planIndex out of range: 1",
                        () -> new BackendPartitionFinalization<>(
                                fixture.analysis,
                                fixture.memoryPlan,
                                List.of(
                                        new PreparationResourceAssignment.Buffer(
                                                fixture.bufferRequirement, fixture.bufferSlot, 1),
                                        fixture.assignments.get(1)))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "assignments[0] buffer slot does not match memoryPlan.buffers[0]",
                        () -> new BackendPartitionFinalization<>(
                                fixture.analysis,
                                fixture.memoryPlan,
                                List.of(
                                        new PreparationResourceAssignment.Buffer(
                                                fixture.bufferRequirement, foreignSlot, 0),
                                        fixture.assignments.get(1)))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "assignments[0] buffer geometry does not satisfy requirement",
                        () -> new BackendPartitionFinalization<>(
                                fixture.analysis, shortPlan, fixture.assignments)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "assignments[1] workspace planIndex out of range: 1",
                        () -> new BackendPartitionFinalization<>(
                                fixture.analysis,
                                fixture.memoryPlan,
                                List.of(
                                        fixture.assignments.get(0),
                                        new PreparationResourceAssignment.Workspace(
                                                fixture.workspaceRequirement,
                                                fixture.workspaceSlot,
                                                1)))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "assignments[1] workspace slot does not match memoryPlan.workspaces[0]",
                        () -> new BackendPartitionFinalization<>(
                                fixture.analysis,
                                fixture.memoryPlan,
                                List.of(
                                        fixture.assignments.get(0),
                                        new PreparationResourceAssignment.Workspace(
                                                fixture.workspaceRequirement,
                                                new WorkspaceSlot(9),
                                                0)))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "assignments[1] workspace geometry does not match requirement",
                        () -> new BackendPartitionFinalization<>(
                                fixture.analysis, wrongWorkspacePlan, fixture.assignments)));
    }

    @Test
    void preparedPartitionValidatesAndRetainsItsTwoExactReferences() {
        Fixture fixture = fixture();
        PreparedExecutable executable = new TestExecutable(fixture.memoryPlan);
        PreparedPartition prepared = new PreparedPartition(fixture.partition, executable);
        assertAll(
                () -> assertSame(fixture.partition, prepared.partition()),
                () -> assertSame(executable, prepared.executable()),
                () -> assertFailure(
                        NullPointerException.class,
                        "partition",
                        () -> new PreparedPartition(null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "executable",
                        () -> new PreparedPartition(fixture.partition, null)));
    }

    private static Fixture fixture() {
        PlannedPartition partition =
                new PlannedPartition(new BackendId("cpu"), List.of(new NodeId(0)));
        var bufferRequirement =
                new PreparationResourceRequirement.Buffer(new ValueId(0), 4, 4);
        var workspaceRequirement =
                new PreparationResourceRequirement.Workspace(0, 8, 8);
        var analysis = new BackendPartitionAnalysis<>(
                partition,
                new FakePlan("route"),
                List.of(bufferRequirement, workspaceRequirement));
        var bufferSlot = new BufferSlot(0);
        var workspaceSlot = new WorkspaceSlot(0);
        var memoryPlan = new PreparedMemoryPlan(
                List.of(new PreparedMemoryPlan.BufferEntry(bufferSlot, 4, 4)),
                List.of(new PreparedMemoryPlan.WorkspaceEntry(workspaceSlot, 8, 8)));
        List<PreparationResourceAssignment> assignments = List.of(
                new PreparationResourceAssignment.Buffer(bufferRequirement, bufferSlot, 0),
                new PreparationResourceAssignment.Workspace(workspaceRequirement, workspaceSlot, 0));
        return new Fixture(
                partition,
                analysis,
                bufferRequirement,
                workspaceRequirement,
                bufferSlot,
                workspaceSlot,
                memoryPlan,
                assignments);
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> type, String message, org.junit.jupiter.api.function.Executable executable) {
        assertEquals(message, assertThrows(type, executable).getMessage());
    }

    private record FakePlan(String route) implements BackendPreparationPlan {}

    private record Fixture(
            PlannedPartition partition,
            BackendPartitionAnalysis<FakePlan> analysis,
            PreparationResourceRequirement.Buffer bufferRequirement,
            PreparationResourceRequirement.Workspace workspaceRequirement,
            BufferSlot bufferSlot,
            WorkspaceSlot workspaceSlot,
            PreparedMemoryPlan memoryPlan,
            List<PreparationResourceAssignment> assignments) {}

    private static final class TestExecutable extends PreparedExecutable {
        private TestExecutable(PreparedMemoryPlan memoryPlan) {
            super(memoryPlan, List.of(), List.of());
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
                RunState runState,
                BufferRepresentation[] bufferRepresentations,
                WorkspaceRepresentation[] workspaceRepresentations) {
            throw new UnsupportedOperationException();
        }
    }
}
