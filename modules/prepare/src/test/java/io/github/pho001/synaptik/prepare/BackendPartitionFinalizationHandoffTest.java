package io.github.pho001.synaptik.prepare;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class BackendPartitionFinalizationHandoffTest {
    @Test
    void acceptsAnEmptyCompleteSetAndSnapshotsItsResult() {
        var result = BackendPartitionFinalizationHandoff.finalizePartitions(List.of(), List.of());
        assertAll(
                () -> assertTrue(result.memoryPlan().buffers().isEmpty()),
                () -> assertTrue(result.memoryPlan().workspaces().isEmpty()),
                () -> assertTrue(result.partitions().isEmpty()),
                () -> assertThrows(
                        UnsupportedOperationException.class, () -> result.partitions().clear()));
    }

    @Test
    void assignsDeterministicallyAndFinalizesInPartitionOrder() {
        Fixture fixture = fixture();
        var suppliedPartitions = new ArrayList<>(fixture.partitions);
        var suppliedEntries = new ArrayList<>(fixture.entries);

        var result = BackendPartitionFinalizationHandoff.finalizePartitions(
                suppliedPartitions, suppliedEntries);
        suppliedPartitions.clear();
        suppliedEntries.clear();

        PreparedMemoryPlan plan = result.memoryPlan();
        BackendPartitionFinalization<FakePlan> first = fixture.firstFinalizer.seen.getFirst();
        BackendPartitionFinalization<FakePlan> second = fixture.secondFinalizer.seen.getFirst();
        var sharedFirst = (PreparationResourceAssignment.Buffer) first.assignments().get(0);
        var firstWorkspace = (PreparationResourceAssignment.Workspace) first.assignments().get(1);
        var sharedSecond = (PreparationResourceAssignment.Buffer) second.assignments().get(1);
        var secondWorkspace = (PreparationResourceAssignment.Workspace) second.assignments().get(0);

        assertAll(
                () -> assertEquals(List.of("first", "second"), fixture.callOrder),
                () -> assertEquals(1, fixture.firstFinalizer.backendIdCalls),
                () -> assertEquals(1, fixture.secondFinalizer.backendIdCalls),
                () -> assertEquals(3, plan.buffers().size()),
                () -> assertEquals(2, plan.workspaces().size()),
                () -> assertEquals(0, plan.buffers().get(0).slot().value()),
                () -> assertEquals(10, plan.buffers().get(0).byteSize()),
                () -> assertEquals(16, plan.buffers().get(0).byteAlignment()),
                () -> assertEquals(1, plan.buffers().get(1).slot().value()),
                () -> assertEquals(2, plan.buffers().get(2).slot().value()),
                () -> assertSame(sharedFirst.slot(), sharedSecond.slot()),
                () -> assertEquals(0, sharedFirst.planIndex()),
                () -> assertEquals(0, sharedSecond.planIndex()),
                () -> assertNotSame(firstWorkspace.slot(), secondWorkspace.slot()),
                () -> assertEquals(0, firstWorkspace.planIndex()),
                () -> assertEquals(1, secondWorkspace.planIndex()),
                () -> assertEquals(8, plan.workspaces().get(0).byteSize()),
                () -> assertEquals(16, plan.workspaces().get(1).byteSize()),
                () -> assertSame(plan, first.memoryPlan()),
                () -> assertSame(plan, second.memoryPlan()),
                () -> assertSame(fixture.partitions.get(0), result.partitions().get(0).partition()),
                () -> assertSame(fixture.partitions.get(1), result.partitions().get(1).partition()),
                () -> assertSame(plan, result.partitions().get(0).executable().memoryPlan()),
                () -> assertThrows(
                        UnsupportedOperationException.class, () -> result.partitions().clear()));
    }

    @Test
    void validatesCompleteCoverageOwnershipAndSourcesBeforeCallingAnyFinalizer() {
        Fixture fixture = fixture();
        PlannedPartition equalDuplicate = new PlannedPartition(
                fixture.partitions.getFirst().owner(),
                fixture.partitions.getFirst().nodeIds());
        var entriesWithNull = new ArrayList<BackendPartitionFinalizationHandoff.Entry<?, ?>>(
                Arrays.asList(fixture.entries.getFirst(), null));
        var wrongPartitionAnalysis = new BackendPartitionAnalysis<>(
                fixture.partitions.get(1),
                new FakePlan("wrong-partition"),
                fixture.analyses.getFirst().requirements());
        var wrongAnalysisEntry = new BackendPartitionFinalizationHandoff.Entry<>(
                fixture.entries.getFirst().context(),
                wrongPartitionAnalysis,
                fixture.firstFinalizer);
        var wrongBackendFinalizer =
                new FakeFinalizer(new BackendId("gpu"), "wrong", fixture.callOrder);
        var wrongBackendEntry = new BackendPartitionFinalizationHandoff.Entry<>(
                fixture.entries.getFirst().context(),
                fixture.analyses.getFirst(),
                wrongBackendFinalizer);
        var nullBackendFinalizer = new FakeFinalizer(null, "null", fixture.callOrder);
        var nullBackendEntry = new BackendPartitionFinalizationHandoff.Entry<>(
                fixture.entries.getFirst().context(),
                fixture.analyses.getFirst(),
                nullBackendFinalizer);
        var absentAnalysis = new BackendPartitionAnalysis<>(
                fixture.partitions.getFirst(),
                new FakePlan("absent"),
                List.of(new PreparationResourceRequirement.Buffer(new ValueId(99), 1, 1)));
        var absentEntry = new BackendPartitionFinalizationHandoff.Entry<>(
                fixture.entries.getFirst().context(), absentAnalysis, fixture.firstFinalizer);

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "partitions",
                        () -> BackendPartitionFinalizationHandoff.finalizePartitions(null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "entries",
                        () -> BackendPartitionFinalizationHandoff.finalizePartitions(List.of(), null)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "partitions[1] duplicates " + equalDuplicate,
                        () -> BackendPartitionFinalizationHandoff.finalizePartitions(
                                List.of(fixture.partitions.getFirst(), equalDuplicate),
                                fixture.entries)),
                () -> assertFailure(
                        NullPointerException.class,
                        "entries[1]",
                        () -> BackendPartitionFinalizationHandoff.finalizePartitions(
                                fixture.partitions, entriesWithNull)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "entries size must equal partitions size 2",
                        () -> BackendPartitionFinalizationHandoff.finalizePartitions(
                                fixture.partitions, List.of(fixture.entries.getFirst()))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "entries[0].context.partition does not match partitions[0]",
                        () -> BackendPartitionFinalizationHandoff.finalizePartitions(
                                fixture.partitions,
                                List.of(fixture.entries.get(1), fixture.entries.get(0)))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "entries[0].analysis.partition does not match context partition",
                        () -> BackendPartitionFinalizationHandoff.finalizePartitions(
                                fixture.partitions,
                                List.of(wrongAnalysisEntry, fixture.entries.get(1)))),
                () -> assertFailure(
                        NullPointerException.class,
                        "entries[0].finalizer.backendId",
                        () -> BackendPartitionFinalizationHandoff.finalizePartitions(
                                fixture.partitions,
                                List.of(nullBackendEntry, fixture.entries.get(1)))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "entries[0].finalizer backendId BackendId[value=gpu] does not match partition owner BackendId[value=cpu]",
                        () -> BackendPartitionFinalizationHandoff.finalizePartitions(
                                fixture.partitions,
                                List.of(wrongBackendEntry, fixture.entries.get(1)))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "entries[0].requirements[0] buffer valueId is absent from context.values: ValueId[value=99]",
                        () -> BackendPartitionFinalizationHandoff.finalizePartitions(
                                fixture.partitions,
                                List.of(absentEntry, fixture.entries.get(1)))),
                () -> assertTrue(fixture.callOrder.isEmpty()));
    }

    @Test
    void rejectsForeignRepeatedSourceReferencesBeforeFinalization() {
        Fixture fixture = fixture();
        GraphValue foreignShared = new GraphValue(
                fixture.sharedValue.id(), fixture.sharedValue.descriptor());
        LogicalMemoryRequirement foreignLogical = new LogicalMemoryRequirement(
                fixture.sharedLogical.valueId(),
                fixture.sharedLogical.descriptor(),
                fixture.sharedLogical.producerPartition(),
                fixture.sharedLogical.consumerPartitions(),
                fixture.sharedLogical.graphOutput());
        PrepareContext<FakeInputs> foreignValueContext = context(
                fixture.partitions.get(1),
                fixture.nodes.get(1),
                List.of(foreignShared, fixture.outputValues.get(1)),
                List.of(fixture.sharedLogical, fixture.outputLogical.get(1)));
        PrepareContext<FakeInputs> foreignLogicalContext = context(
                fixture.partitions.get(1),
                fixture.nodes.get(1),
                List.of(fixture.sharedValue, fixture.outputValues.get(1)),
                List.of(foreignLogical, fixture.outputLogical.get(1)));

        var foreignValueEntry = new BackendPartitionFinalizationHandoff.Entry<>(
                foreignValueContext, fixture.analyses.get(1), fixture.secondFinalizer);
        var foreignLogicalEntry = new BackendPartitionFinalizationHandoff.Entry<>(
                foreignLogicalContext, fixture.analyses.get(1), fixture.secondFinalizer);

        assertAll(
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "entries[1].requirements[1] projected value reference does not match first declaration for "
                                + fixture.sharedValue.id(),
                        () -> BackendPartitionFinalizationHandoff.finalizePartitions(
                                fixture.partitions,
                                List.of(fixture.entries.getFirst(), foreignValueEntry))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "entries[1].requirements[1] logical requirement reference does not match first declaration for "
                                + fixture.sharedValue.id(),
                        () -> BackendPartitionFinalizationHandoff.finalizePartitions(
                                fixture.partitions,
                                List.of(fixture.entries.getFirst(), foreignLogicalEntry))),
                () -> assertTrue(fixture.callOrder.isEmpty()));
    }

    @Test
    void stopsAtTheFirstFinalizerFailureAndRejectsNullOrForeignPlanResults() {
        Fixture failingFixture = fixture();
        RuntimeException failure = new RuntimeException("boom");
        failingFixture.firstFinalizer.failure = failure;
        assertSame(
                failure,
                assertThrows(
                        RuntimeException.class,
                        () -> BackendPartitionFinalizationHandoff.finalizePartitions(
                                failingFixture.partitions, failingFixture.entries)));
        assertAll(
                () -> assertEquals(List.of("first"), failingFixture.callOrder),
                () -> assertTrue(failingFixture.secondFinalizer.seen.isEmpty()));

        Fixture nullFixture = fixture();
        nullFixture.firstFinalizer.returnNull = true;
        assertFailure(
                NullPointerException.class,
                "entries[0].finalizer returned null",
                () -> BackendPartitionFinalizationHandoff.finalizePartitions(
                        nullFixture.partitions, nullFixture.entries));

        Fixture foreignFixture = fixture();
        foreignFixture.firstFinalizer.foreignPlan = true;
        assertFailure(
                IllegalArgumentException.class,
                "entries[0] executable memory plan does not match assigned memory plan",
                () -> BackendPartitionFinalizationHandoff.finalizePartitions(
                        foreignFixture.partitions, foreignFixture.entries));
    }

    private static Fixture fixture() {
        BackendId cpu = new BackendId("cpu");
        CompiledNode firstNode = node(10, new ValueId(0), new ValueId(1));
        CompiledNode secondNode = node(11, new ValueId(0), new ValueId(2));
        PlannedPartition firstPartition = new PlannedPartition(cpu, List.of(firstNode.id()));
        PlannedPartition secondPartition = new PlannedPartition(cpu, List.of(secondNode.id()));
        List<PlannedPartition> partitions = List.of(firstPartition, secondPartition);

        GraphValue shared = new GraphValue(new ValueId(0), descriptor());
        GraphValue firstOutput = new GraphValue(new ValueId(1), descriptor());
        GraphValue secondOutput = new GraphValue(new ValueId(2), descriptor());
        LogicalMemoryRequirement sharedLogical = new LogicalMemoryRequirement(
                shared.id(), shared.descriptor(), Optional.empty(), partitions, false);
        LogicalMemoryRequirement firstOutputLogical = new LogicalMemoryRequirement(
                firstOutput.id(), firstOutput.descriptor(), Optional.of(firstPartition), List.of(), true);
        LogicalMemoryRequirement secondOutputLogical = new LogicalMemoryRequirement(
                secondOutput.id(), secondOutput.descriptor(), Optional.of(secondPartition), List.of(), true);

        PrepareContext<FakeInputs> firstContext = context(
                firstPartition,
                firstNode,
                List.of(shared, firstOutput),
                List.of(sharedLogical, firstOutputLogical));
        PrepareContext<FakeInputs> secondContext = context(
                secondPartition,
                secondNode,
                List.of(shared, secondOutput),
                List.of(sharedLogical, secondOutputLogical));

        var firstRequirements = List.<PreparationResourceRequirement>of(
                new PreparationResourceRequirement.Buffer(shared.id(), 4, 4),
                new PreparationResourceRequirement.Workspace(7, 8, 8),
                new PreparationResourceRequirement.Buffer(firstOutput.id(), 12, 4));
        var secondRequirements = List.<PreparationResourceRequirement>of(
                new PreparationResourceRequirement.Workspace(7, 16, 4),
                new PreparationResourceRequirement.Buffer(shared.id(), 10, 16),
                new PreparationResourceRequirement.Buffer(secondOutput.id(), 20, 8));
        BackendPartitionAnalysis<FakePlan> firstAnalysis = new BackendPartitionAnalysis<>(
                firstPartition, new FakePlan("first-route"), firstRequirements);
        BackendPartitionAnalysis<FakePlan> secondAnalysis = new BackendPartitionAnalysis<>(
                secondPartition, new FakePlan("second-route"), secondRequirements);

        var callOrder = new ArrayList<String>();
        var firstFinalizer = new FakeFinalizer(cpu, "first", callOrder);
        var secondFinalizer = new FakeFinalizer(cpu, "second", callOrder);
        List<BackendPartitionFinalizationHandoff.Entry<?, ?>> entries = List.of(
                new BackendPartitionFinalizationHandoff.Entry<>(
                        firstContext, firstAnalysis, firstFinalizer),
                new BackendPartitionFinalizationHandoff.Entry<>(
                        secondContext, secondAnalysis, secondFinalizer));
        return new Fixture(
                partitions,
                List.of(firstNode, secondNode),
                List.of(firstOutput, secondOutput),
                List.of(firstOutputLogical, secondOutputLogical),
                shared,
                sharedLogical,
                List.of(firstAnalysis, secondAnalysis),
                entries,
                firstFinalizer,
                secondFinalizer,
                callOrder);
    }

    private static PrepareContext<FakeInputs> context(
            PlannedPartition partition,
            CompiledNode node,
            List<GraphValue> values,
            List<LogicalMemoryRequirement> requirements) {
        return new PrepareContext<>(
                partition,
                List.of(node),
                values,
                requirements,
                Map.of(),
                new FakeInputs("target"));
    }

    private static CompiledNode node(long id, ValueId input, ValueId output) {
        return new CompiledNode(
                new NodeId(id),
                new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE),
                List.of(input),
                List.of(output));
    }

    private static TensorDescriptor descriptor() {
        return new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false);
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> type, String message, org.junit.jupiter.api.function.Executable executable) {
        assertEquals(message, assertThrows(type, executable).getMessage());
    }

    private record FakeInputs(String target) implements BackendAnalysisInputs {}

    private record FakePlan(String route) implements BackendPreparationPlan {}

    private static final class FakeFinalizer implements BackendPartitionFinalizer<FakePlan> {
        private final BackendId backendId;
        private final String name;
        private final List<String> callOrder;
        private final List<BackendPartitionFinalization<FakePlan>> seen = new ArrayList<>();
        private RuntimeException failure;
        private boolean returnNull;
        private boolean foreignPlan;
        private int backendIdCalls;

        private FakeFinalizer(BackendId backendId, String name, List<String> callOrder) {
            this.backendId = backendId;
            this.name = name;
            this.callOrder = callOrder;
        }

        @Override
        public BackendId backendId() {
            backendIdCalls++;
            return backendId;
        }

        @Override
        public PreparedExecutable finalizePartition(
                BackendPartitionFinalization<FakePlan> finalization) {
            callOrder.add(name);
            seen.add(finalization);
            if (failure != null) {
                throw failure;
            }
            if (returnNull) {
                return null;
            }
            PreparedMemoryPlan plan = foreignPlan
                    ? new PreparedMemoryPlan(
                            finalization.memoryPlan().buffers(),
                            finalization.memoryPlan().workspaces())
                    : finalization.memoryPlan();
            return new TestExecutable(plan);
        }
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

    private record Fixture(
            List<PlannedPartition> partitions,
            List<CompiledNode> nodes,
            List<GraphValue> outputValues,
            List<LogicalMemoryRequirement> outputLogical,
            GraphValue sharedValue,
            LogicalMemoryRequirement sharedLogical,
            List<BackendPartitionAnalysis<FakePlan>> analyses,
            List<BackendPartitionFinalizationHandoff.Entry<?, ?>> entries,
            FakeFinalizer firstFinalizer,
            FakeFinalizer secondFinalizer,
            List<String> callOrder) {}

    private enum SampleKind implements OperationKind {
        SAMPLE;

        private static final List<OperationSignature> SIGNATURES = List.of(
                new OperationSignature(
                        NoOperationAttrs.class, 0, Integer.MAX_VALUE, 1, Integer.MAX_VALUE));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
