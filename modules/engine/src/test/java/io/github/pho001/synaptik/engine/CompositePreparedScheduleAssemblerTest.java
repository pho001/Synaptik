package io.github.pho001.synaptik.engine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.PreparedBufferAssignment;
import io.github.pho001.synaptik.prepare.PreparedPartition;
import io.github.pho001.synaptik.prepare.PreparedScheduleAssembler;
import io.github.pho001.synaptik.prepare.PreparedScheduleContext;
import io.github.pho001.synaptik.prepare.PreparedScheduleContribution;
import io.github.pho001.synaptik.prepare.PreparedScheduleContributor;
import io.github.pho001.synaptik.runtime.execution.BoundBufferTransfer;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedBufferTransfer;
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Behavioral contract for owner-indexed physical contributions and deterministic fan-out order. */
final class CompositePreparedScheduleAssemblerTest {
    @Test
    void mapsAndExecutesRepeatedInterleavedOwnersWithOneTransferPerDestination() {
        BackendId first = new BackendId("first");
        BackendId second = new BackendId("second");
        BackendId third = new BackendId("third");
        PlannedPartition producer = new PlannedPartition(first, List.of(new NodeId(1)));
        PlannedPartition consumerOne = new PlannedPartition(second, List.of(new NodeId(2)));
        PlannedPartition consumerTwo = new PlannedPartition(third, List.of(new NodeId(3)));
        PlannedPartition repeatedConsumer =
                new PlannedPartition(second, List.of(new NodeId(4)));
        ValueId valueId = new ValueId(10);
        Shape shape = Shape.of(2);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        GraphValue value = new GraphValue(valueId, descriptor);
        LogicalMemoryRequirement requirement = new LogicalMemoryRequirement(
                valueId,
                descriptor,
                Optional.of(producer),
                List.of(consumerOne, consumerTwo, repeatedConsumer),
                true);
        PreparedMemoryPlan memoryPlan = new PreparedMemoryPlan(
                List.of(new PreparedMemoryPlan.BufferEntry(new BufferSlot(0), 8L, 4L)),
                List.of());
        PreparedBufferAssignment assignment = new PreparedBufferAssignment(
                valueId, memoryPlan.buffers().getFirst().slot(), 0,
                List.of(first, second, third));
        var events = new ArrayList<String>();
        var partitions = List.of(
                new PreparedPartition(
                        producer, new EmptyExecutable(memoryPlan, events, "execute-first")),
                new PreparedPartition(
                        consumerOne, new EmptyExecutable(memoryPlan, events, "execute-second")),
                new PreparedPartition(
                        consumerTwo, new EmptyExecutable(memoryPlan, events, "execute-third")),
                new PreparedPartition(
                        repeatedConsumer,
                        new EmptyExecutable(memoryPlan, events, "execute-second-repeat")));
        PreparedScheduleContext context = new PreparedScheduleContext(
                List.of(producer, consumerOne, consumerTwo, repeatedConsumer),
                List.of(value),
                List.of(requirement),
                List.of(),
                Map.of(),
                List.of(valueId),
                memoryPlan,
                partitions,
                List.of(assignment),
                List.of());

        var adapters = new LinkedHashMap<BackendId, EngineBackendComposition>();
        adapters.put(first, new TestAdapter(first, true, events));
        adapters.put(second, new TestAdapter(second, false, events));
        adapters.put(third, new TestAdapter(third, false, events));
        PreparedSchedule schedule = new CompositePreparedScheduleAssembler(adapters).assemble(context);

        var creation = assertInstanceOf(
                PreparedSchedule.RepresentationCreationStep.class, schedule.steps().get(0));
        assertEquals(3, creation.representationPlan().bufferPreparations().getFirst().size());
        assertSame(partitions.get(0).executable(), assertInstanceOf(
                PreparedSchedule.ExecutionStep.class, schedule.steps().get(1)).executable());
        PreparedBufferTransfer firstTransfer = assertInstanceOf(
                PreparedSchedule.BufferTransferStep.class, schedule.steps().get(2)).transfer();
        assertEquals(0, firstTransfer.sourceRepresentationIndex());
        assertEquals(1, firstTransfer.destinationRepresentationIndex());
        assertSame(partitions.get(1).executable(), assertInstanceOf(
                PreparedSchedule.ExecutionStep.class, schedule.steps().get(3)).executable());
        PreparedBufferTransfer secondTransfer = assertInstanceOf(
                PreparedSchedule.BufferTransferStep.class, schedule.steps().get(4)).transfer();
        assertEquals(0, secondTransfer.sourceRepresentationIndex());
        assertEquals(2, secondTransfer.destinationRepresentationIndex());
        assertSame(partitions.get(2).executable(), assertInstanceOf(
                PreparedSchedule.ExecutionStep.class, schedule.steps().get(5)).executable());
        assertSame(partitions.get(3).executable(), assertInstanceOf(
                PreparedSchedule.ExecutionStep.class, schedule.steps().get(6)).executable());
        PreparedSchedule.PublicationStep publication = assertInstanceOf(
                PreparedSchedule.PublicationStep.class, schedule.steps().get(7));
        assertEquals(0, publication.publication().representationIndex());

        try (var execution =
                        new io.github.pho001.synaptik.runtime.execution.PreparedExecution(
                                memoryPlan, schedule);
                var result = new PreparedExecutionRunner().run(execution, List.of())) {
            assertEquals(1, result.resultCount());
            assertEquals(
                    List.of(
                            "execute-first",
                            "transfer-first-second",
                            "execute-second",
                            "transfer-first-third",
                            "execute-third",
                            "execute-second-repeat"),
                    events);
        }
    }

    private static final class TestAdapter implements EngineBackendComposition {
        private final BackendId owner;
        private final boolean initialized;
        private final List<String> events;

        private TestAdapter(
                BackendId owner, boolean initialized, List<String> events) {
            this.owner = owner;
            this.initialized = initialized;
            this.events = events;
        }

        @Override
        public List<BackendCapabilityProvider> capabilityProviders() {
            return List.of(new BackendCapabilityProvider() {
                @Override public BackendId backendId() { return owner; }
                @Override public boolean supports(OperationCapabilityQuery query) { return true; }
            });
        }

        @Override
        public List<BackendAvailabilitySnapshot> availabilitySnapshots() {
            return List.of(new BackendAvailabilitySnapshot(owner, Map.of()));
        }

        @Override
        public PartitionPreparation<?, ?> partitionPreparation() {
            throw new AssertionError("analysis is outside this assembler test");
        }

        @Override
        public PreparedScheduleContributor scheduleContributor() {
            return new PreparedScheduleContributor() {
                @Override public BackendId backendId() { return owner; }
                @Override public PreparedScheduleContribution contribute(
                        PreparedScheduleContext context) {
                    PreparedBufferAssignment assignment = context.bufferAssignments().getFirst();
                    int representationIndex = assignment.representationOwners().indexOf(owner);
                    PreparedRepresentationPlan.BufferPreparation preparation = initialized
                            ? new PreparedRepresentationPlan.InitializedBuffer(TestBuffer::new)
                            : new PreparedRepresentationPlan.CreatedBuffer(TestBuffer::new);
                    return new PreparedScheduleContribution(
                            owner,
                            List.of(new PreparedScheduleContribution.Buffer(
                                    assignment, representationIndex, preparation)),
                            List.of());
                }
            };
        }

        @Override
        public PreparedScheduleAssembler scheduleAssembler() {
            throw new AssertionError("producerless assembly is outside this test");
        }

        @Override
        public boolean supportsTransferTo(
                EngineBackendComposition destination, TensorDescriptor descriptor) {
            return true;
        }

        @Override
        public PreparedBufferTransfer prepareTransferTo(
                EngineBackendComposition destination,
                PreparedMemoryPlan memoryPlan,
                int bufferIndex,
                int sourceRepresentationIndex,
                int destinationRepresentationIndex,
                TensorDescriptor descriptor) {
            TestAdapter target = (TestAdapter) destination;
            return new TestTransfer(
                    memoryPlan,
                    bufferIndex,
                    sourceRepresentationIndex,
                    destinationRepresentationIndex,
                    events,
                    "transfer-" + owner.value() + "-" + target.owner.value());
        }

        @Override
        public BufferRepresentation borrow(HostTensorStorage storage) {
            throw new AssertionError("borrowing is outside this test");
        }

        @Override
        public byte[] copyToCanonicalHostBytes(
                BufferRepresentation representation,
                TensorDescriptor descriptor,
                long maximumBytes) {
            throw new AssertionError("materialization is outside this test");
        }

        @Override public void close() {}
    }

    private static final class TestTransfer extends PreparedBufferTransfer {
        private final List<String> events;
        private final String event;
        private TestTransfer(
                PreparedMemoryPlan memoryPlan,
                int bufferIndex,
                int sourceRepresentationIndex,
                int destinationRepresentationIndex,
                List<String> events,
                String event) {
            super(memoryPlan, bufferIndex, sourceRepresentationIndex,
                    destinationRepresentationIndex);
            this.events = events;
            this.event = event;
        }

        @Override protected boolean acceptsSourceBufferRepresentation(
                BufferRepresentation representation) { return true; }
        @Override protected boolean acceptsDestinationBufferRepresentation(
                BufferRepresentation representation) { return true; }
        @Override protected BoundBufferTransfer bindCompatible(
                RunState runState,
                BufferRepresentation sourceRepresentation,
                BufferRepresentation destinationRepresentation) {
            return new BoundBufferTransfer(
                    runState,
                    bufferIndex(),
                    sourceRepresentationIndex(),
                    destinationRepresentationIndex()) {
                @Override protected void executeTransfer() { events.add(event); }
            };
        }
    }

    private static final class EmptyExecutable extends PreparedExecutable {
        private final List<String> events;
        private final String event;

        private EmptyExecutable(
                PreparedMemoryPlan memoryPlan, List<String> events, String event) {
            super(memoryPlan, List.of(), List.of());
            this.events = events;
            this.event = event;
        }
        @Override protected boolean acceptsBufferRepresentation(
                int selectionIndex, BufferRepresentation representation) { return false; }
        @Override protected boolean acceptsWorkspaceRepresentation(
                int selectionIndex, WorkspaceRepresentation representation) { return false; }
        @Override protected BoundInvocation bindCompatible(
                RunState runState,
                BufferRepresentation[] bufferRepresentations,
                WorkspaceRepresentation[] workspaceRepresentations) {
            return new BoundInvocation(runState) {
                @Override protected void executeBound() { events.add(event); }
            };
        }
    }

    private static final class TestBuffer implements BufferRepresentation {
        @Override public void close() {}
    }
}
