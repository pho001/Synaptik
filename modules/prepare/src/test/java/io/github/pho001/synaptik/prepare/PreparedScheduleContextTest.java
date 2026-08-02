package io.github.pho001.synaptik.prepare;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class PreparedScheduleContextTest {
    @Test
    void snapshotsListsAndRetainsExactCompleteAssociations() {
        GraphPreparationTest.Fixture fixture = GraphPreparationTest.fixture();
        PreparedMemoryPlan memoryPlan = memoryPlan(4);
        var partitions = new ArrayList<>(List.of(
                new PreparedPartition(
                        fixture.partitions().get(0), new TestExecutable(memoryPlan)),
                new PreparedPartition(
                        fixture.partitions().get(1), new TestExecutable(memoryPlan))));
        var assignments = new ArrayList<>(assignments(memoryPlan));

        PreparedScheduleContext context = new PreparedScheduleContext(
                fixture.artifacts(), memoryPlan, partitions, assignments);
        partitions.clear();
        assignments.clear();

        assertAll(
                () -> assertSame(fixture.artifacts(), context.artifacts()),
                () -> assertSame(memoryPlan, context.memoryPlan()),
                () -> assertEquals(2, context.partitions().size()),
                () -> assertEquals(4, context.bufferAssignments().size()),
                () -> assertSame(
                        fixture.partitions().get(0), context.partitions().get(0).partition()),
                () -> assertSame(
                        memoryPlan.buffers().get(2).slot(),
                        context.bufferAssignments().get(2).slot()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> context.partitions().clear()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> context.bufferAssignments().clear()));
    }

    @Test
    void validatesComponentsCoverageOrderSlotsUniquenessAndGraphMembership() {
        GraphPreparationTest.Fixture fixture = GraphPreparationTest.fixture();
        PreparedMemoryPlan memoryPlan = memoryPlan(4);
        List<PreparedPartition> partitions = List.of(
                new PreparedPartition(
                        fixture.partitions().get(0), new TestExecutable(memoryPlan)),
                new PreparedPartition(
                        fixture.partitions().get(1), new TestExecutable(memoryPlan)));
        List<PreparedBufferAssignment> assignments = assignments(memoryPlan);
        PreparedMemoryPlan foreignPlan = memoryPlan(4);

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "artifacts",
                        () -> new PreparedScheduleContext(null, null, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "memoryPlan",
                        () -> new PreparedScheduleContext(
                                fixture.artifacts(), null, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "partitions",
                        () -> new PreparedScheduleContext(
                                fixture.artifacts(), memoryPlan, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "bufferAssignments",
                        () -> new PreparedScheduleContext(
                                fixture.artifacts(), memoryPlan, List.of(), null)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "partitions size must equal compile partition count 2",
                        () -> new PreparedScheduleContext(
                                fixture.artifacts(), memoryPlan, List.of(), assignments)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "partitions[0] does not retain artifacts.partitions[0]",
                        () -> new PreparedScheduleContext(
                                fixture.artifacts(),
                                memoryPlan,
                                List.of(partitions.get(1), partitions.get(0)),
                                assignments)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "partitions[0].executable does not retain memoryPlan",
                        () -> new PreparedScheduleContext(
                                fixture.artifacts(),
                                memoryPlan,
                                List.of(
                                        new PreparedPartition(
                                                fixture.partitions().get(0),
                                                new TestExecutable(foreignPlan)),
                                        partitions.get(1)),
                                assignments)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "bufferAssignments[1].planIndex must equal 1",
                        () -> new PreparedScheduleContext(
                                fixture.artifacts(),
                                memoryPlan,
                                partitions,
                                List.of(
                                        assignments.get(0),
                                        new PreparedBufferAssignment(
                                                assignments.get(1).valueId(),
                                                assignments.get(1).slot(),
                                                2),
                                        assignments.get(2),
                                        assignments.get(3)))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "bufferAssignments[1].valueId duplicates ValueId[value=0]",
                        () -> new PreparedScheduleContext(
                                fixture.artifacts(),
                                memoryPlan,
                                partitions,
                                List.of(
                                        assignments.get(0),
                                        new PreparedBufferAssignment(
                                                new ValueId(0),
                                                assignments.get(1).slot(),
                                                1),
                                        assignments.get(2),
                                        assignments.get(3)))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "bufferAssignments[3].valueId is absent from artifacts.graph: ValueId[value=99]",
                        () -> new PreparedScheduleContext(
                                fixture.artifacts(),
                                memoryPlan,
                                partitions,
                                List.of(
                                        assignments.get(0),
                                        assignments.get(1),
                                        assignments.get(2),
                                        new PreparedBufferAssignment(
                                                new ValueId(99),
                                                assignments.get(3).slot(),
                                                3)))));
    }

    private static PreparedMemoryPlan memoryPlan(int count) {
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        for (int index = 0; index < count; index++) {
            buffers.add(new PreparedMemoryPlan.BufferEntry(new BufferSlot(index), 24, 8));
        }
        return new PreparedMemoryPlan(buffers, List.of());
    }

    private static List<PreparedBufferAssignment> assignments(PreparedMemoryPlan plan) {
        var assignments = new ArrayList<PreparedBufferAssignment>();
        for (int index = 0; index < plan.buffers().size(); index++) {
            assignments.add(new PreparedBufferAssignment(
                    new ValueId(index), plan.buffers().get(index).slot(), index));
        }
        return List.copyOf(assignments);
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> type, String message, org.junit.jupiter.api.function.Executable action) {
        assertEquals(message, assertThrows(type, action).getMessage());
    }

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
