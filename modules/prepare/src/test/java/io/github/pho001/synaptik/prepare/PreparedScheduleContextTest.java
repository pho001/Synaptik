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
import java.util.LinkedHashMap;
import java.util.List;
import org.junit.jupiter.api.Test;

class PreparedScheduleContextTest {
    @Test
    void snapshotsStableFactsAndRetainsExactPreparedAssociations() {
        var fixture = GraphPreparationTest.fixture();
        PreparedMemoryPlan memoryPlan = memoryPlan(4);
        var partitions = new ArrayList<>(List.of(
                new PreparedPartition(fixture.partitions().get(0), new TestExecutable(memoryPlan)),
                new PreparedPartition(fixture.partitions().get(1), new TestExecutable(memoryPlan))));
        var assignments = new ArrayList<>(assignments(memoryPlan));
        var constants = constants(fixture);

        PreparedScheduleContext context = context(fixture, memoryPlan, partitions, assignments);
        partitions.clear();
        assignments.clear();
        constants.clear();

        assertAll(
                () -> assertSame(fixture.partitions().getFirst(),
                        context.plannedPartitions().getFirst()),
                () -> assertSame(fixture.artifacts().graph().values().getFirst(),
                        context.graphValues().getFirst()),
                () -> assertSame(memoryPlan, context.memoryPlan()),
                () -> assertEquals(2, context.partitions().size()),
                () -> assertEquals(4, context.bufferAssignments().size()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> context.graphValues().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> context.constants().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> context.partitions().clear()));
    }

    @Test
    void rejectsWrongPartitionAssociationAndForeignGraphAssignment() {
        var fixture = GraphPreparationTest.fixture();
        PreparedMemoryPlan memoryPlan = memoryPlan(4);
        List<PreparedPartition> partitions = List.of(
                new PreparedPartition(fixture.partitions().get(0), new TestExecutable(memoryPlan)),
                new PreparedPartition(fixture.partitions().get(1), new TestExecutable(memoryPlan)));
        List<PreparedBufferAssignment> assignments = assignments(memoryPlan);

        assertAll(
                () -> assertEquals("plannedPartitions", assertThrows(NullPointerException.class,
                        () -> new PreparedScheduleContext(null, null, null, null, null,
                                null, null, null)).getMessage()),
                () -> assertEquals(
                        "partitions[0] does not retain plannedPartitions[0]",
                        assertThrows(IllegalArgumentException.class, () -> context(
                                fixture, memoryPlan,
                                List.of(partitions.get(1), partitions.get(0)), assignments))
                                .getMessage()),
                () -> assertEquals(
                        "bufferAssignments[3].valueId is absent from graphValues: ValueId[value=99]",
                        assertThrows(IllegalArgumentException.class, () -> context(
                                fixture, memoryPlan, partitions,
                                List.of(assignments.get(0), assignments.get(1), assignments.get(2),
                                        new PreparedBufferAssignment(new ValueId(99),
                                                assignments.get(3).slot(), 3))))
                                .getMessage()));
    }

    private static PreparedScheduleContext context(
            GraphPreparationTest.Fixture fixture,
            PreparedMemoryPlan memoryPlan,
            List<PreparedPartition> partitions,
            List<PreparedBufferAssignment> assignments) {
        var publications = new ArrayList<ValueId>();
        fixture.artifacts().publication().forwardBindings().forEach(binding ->
                publications.add(binding.valueId()));
        fixture.artifacts().publication().gradientBindings().forEach(binding ->
                publications.add(binding.valueId()));
        return new PreparedScheduleContext(fixture.artifacts().partitions(),
                fixture.artifacts().graph().values(),
                fixture.artifacts().constants().bindableInputs(), constants(fixture), publications,
                memoryPlan, partitions, assignments);
    }

    private static LinkedHashMap<ValueId, io.github.pho001.synaptik.model.datatype.ScalarValue>
            constants(GraphPreparationTest.Fixture fixture) {
        var constants = new LinkedHashMap<ValueId,
                io.github.pho001.synaptik.model.datatype.ScalarValue>();
        fixture.artifacts().constants().constantSources().forEach(source ->
                constants.put(source.valueId(), source.value()));
        return constants;
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

    private static final class TestExecutable extends PreparedExecutable {
        private TestExecutable(PreparedMemoryPlan memoryPlan) {
            super(memoryPlan, List.of(), List.of());
        }
        @Override protected boolean acceptsBufferRepresentation(
                int selectionIndex, BufferRepresentation representation) { return true; }
        @Override protected boolean acceptsWorkspaceRepresentation(
                int selectionIndex, WorkspaceRepresentation representation) { return true; }
        @Override protected BoundInvocation bindCompatible(RunState runState,
                BufferRepresentation[] bufferRepresentations,
                WorkspaceRepresentation[] workspaceRepresentations) {
            throw new UnsupportedOperationException();
        }
    }
}
