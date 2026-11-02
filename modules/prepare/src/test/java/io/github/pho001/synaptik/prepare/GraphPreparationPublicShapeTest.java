package io.github.pho001.synaptik.prepare;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class GraphPreparationPublicShapeTest {
    @Test
    void locksRecordComponentsAndBackendGenericAssociations() {
        assertAll(
                () -> assertRecordComponents(
                        PartitionPreparation.class, "backendInputs", "preparer", "finalizer"),
                () -> assertRecordComponents(
                        PreparedBufferAssignment.class,
                        "valueId", "slot", "planIndex", "representationOwners"),
                () -> assertRecordComponents(
                        PreparedScheduleContext.class,
                        "plannedPartitions", "graphValues", "logicalMemoryRequirements",
                        "bindableInputValueIds", "constants", "publicationValueIds", "memoryPlan",
                        "partitions", "bufferAssignments", "workspaceAssignments"),
                () -> assertEquals(
                        BackendAnalysisInputs.class,
                        PartitionPreparation.class.getTypeParameters()[0].getBounds()[0]),
                () -> assertEquals(
                        BackendPreparationPlan.class,
                        PartitionPreparation.class.getTypeParameters()[1].getBounds()[0]),
                () -> assertTrue(
                        PartitionPreparation.class.getRecordComponents()[1].getGenericType()
                                instanceof ParameterizedType),
                () -> assertTrue(
                        PartitionPreparation.class.getRecordComponents()[2].getGenericType()
                                instanceof ParameterizedType));
    }

    @Test
    void runtimeInitializedOriginIsTheOnlySealedFamilyExtension() {
        Class<?> preparation =
                io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan
                        .BufferPreparation.class;
        assertAll(
                () -> assertEquals(
                        List.of(
                                io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan
                                        .CallerInput.class,
                                io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan
                                        .CreatedBuffer.class,
                                io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan
                                        .InitializedBuffer.class),
                        List.of(preparation.getPermittedSubclasses())),
                () -> assertRecordComponents(
                        io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan
                                .InitializedBuffer.class,
                        "creator"),
                () -> assertFalse(Modifier.isAbstract(
                        io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan
                                .InitializedBuffer.class.getModifiers())));
    }

    @Test
    void newValuesValidateComponentsInDeclarationOrder() {
        var inputs = new FakeInputs();
        var preparer = new FakePreparer();
        var finalizer = new FakeFinalizer();
        var slot = new BufferSlot(0);

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "backendInputs",
                        () -> new PartitionPreparation<>(null, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "preparer",
                        () -> new PartitionPreparation<>(inputs, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "finalizer",
                        () -> new PartitionPreparation<>(inputs, preparer, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "valueId",
                        () -> new PreparedBufferAssignment(null, null, -1, java.util.List.of(new io.github.pho001.synaptik.backend.contract.BackendId("test")))),
                () -> assertFailure(
                        NullPointerException.class,
                        "slot",
                        () -> new PreparedBufferAssignment(new ValueId(0), null, -1, java.util.List.of(new io.github.pho001.synaptik.backend.contract.BackendId("test")))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "planIndex must be non-negative",
                        () -> new PreparedBufferAssignment(new ValueId(0), slot, -1, java.util.List.of(new io.github.pho001.synaptik.backend.contract.BackendId("test")))));
    }

    @Test
    void producerlessResourceRetainsRecipesAndRejectsInvalidAssociations() {
        GraphValue value = new GraphValue(new ValueId(10), resolvedDescriptor());
        LogicalMemoryRequirement valid = requirement(value, Optional.empty(), List.of(), true);
        var owner = new io.github.pho001.synaptik.backend.contract.BackendId("test");
        var size = new PreparedMemoryPlan.FixedByteSize(0);
        PlannedPartition partition = new PlannedPartition(
                owner, List.of(new io.github.pho001.synaptik.model.graph.NodeId(1)));
        assertAll(
                () -> assertFailure(NullPointerException.class, "value",
                        () -> new ProducerlessPublishedValueResource(null, valid, owner, size, 8)),
                () -> assertFailure(NullPointerException.class, "logicalRequirement",
                        () -> new ProducerlessPublishedValueResource(value, null, owner, size, 8)),
                () -> assertFailure(NullPointerException.class, "owner",
                        () -> new ProducerlessPublishedValueResource(value, valid, null, size, 8)),
                () -> assertFailure(NullPointerException.class, "size",
                        () -> new ProducerlessPublishedValueResource(value, valid, owner, null, 8)),
                () -> assertFailure(IllegalArgumentException.class,
                        "logicalRequirement.valueId must match value.id",
                        () -> new ProducerlessPublishedValueResource(value,
                                new LogicalMemoryRequirement(new ValueId(11), value.descriptor(),
                                        Optional.empty(), List.of(), true), owner, size, 8)),
                () -> assertFailure(IllegalArgumentException.class,
                        "logicalRequirement.descriptor must match value.descriptor",
                        () -> new ProducerlessPublishedValueResource(value,
                                new LogicalMemoryRequirement(value.id(), resolvedDescriptor(2),
                                        Optional.empty(), List.of(), true), owner, size, 8)),
                () -> assertFailure(IllegalArgumentException.class,
                        "logicalRequirement must be producerless",
                        () -> new ProducerlessPublishedValueResource(value,
                                requirement(value, Optional.of(partition), List.of(), true),
                                owner, size, 8)),
                () -> assertFailure(IllegalArgumentException.class,
                        "logicalRequirement must be consumerless",
                        () -> new ProducerlessPublishedValueResource(value,
                                requirement(value, Optional.empty(), List.of(partition), true),
                                owner, size, 8)),
                () -> assertFailure(IllegalArgumentException.class,
                        "logicalRequirement must require graph output",
                        () -> new ProducerlessPublishedValueResource(value,
                                requirement(value, Optional.empty(), List.of(), false),
                                owner, size, 8)));
        for (long alignment : new long[] {0, -1, 3}) {
            assertFailure(IllegalArgumentException.class,
                    "byteAlignment must be a positive power of two",
                    () -> new ProducerlessPublishedValueResource(value, valid, owner, size, alignment));
        }
        var resource = new ProducerlessPublishedValueResource(value, valid, owner, size, 8);
        org.junit.jupiter.api.Assertions.assertSame(value, resource.value());
        org.junit.jupiter.api.Assertions.assertSame(valid, resource.logicalRequirement());
        org.junit.jupiter.api.Assertions.assertSame(owner, resource.owner());
        org.junit.jupiter.api.Assertions.assertSame(size, resource.size());
    }

    private static LogicalMemoryRequirement requirement(
            GraphValue value,
            Optional<PlannedPartition> producer,
            List<PlannedPartition> consumers,
            boolean graphOutput) {
        return new LogicalMemoryRequirement(
                value.id(), value.descriptor(), producer, consumers, graphOutput);
    }

    private static TensorDescriptor resolvedDescriptor() {
        return resolvedDescriptor(1);
    }

    private static TensorDescriptor resolvedDescriptor(long extent) {
        Shape shape = Shape.of(extent);
        return new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
    }

    private static void assertRecordComponents(Class<?> type, String... names) {
        assertTrue(type.isRecord());
        assertEquals(
                List.of(names),
                Arrays.stream(type.getRecordComponents())
                        .map(component -> component.getName())
                        .toList());
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> type, String message, org.junit.jupiter.api.function.Executable action) {
        assertEquals(message, org.junit.jupiter.api.Assertions.assertThrows(type, action).getMessage());
    }

    private record FakeInputs() implements BackendAnalysisInputs {}

    private record FakePlan() implements BackendPreparationPlan {}

    private static final class FakePreparer
            implements io.github.pho001.synaptik.prepare.analysis.BackendPartitionPreparer<
                    FakeInputs, FakePlan> {
        @Override
        public io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis<FakePlan>
                analyze(io.github.pho001.synaptik.prepare.analysis.PrepareContext<FakeInputs> context) {
            throw new UnsupportedOperationException();
        }
    }

    private static final class FakeFinalizer implements BackendPartitionFinalizer<FakePlan> {
        @Override
        public io.github.pho001.synaptik.backend.contract.BackendId backendId() {
            throw new UnsupportedOperationException();
        }

        @Override
        public BackendPartitionFinalizationResult finalizePartition(
                BackendPartitionFinalization<FakePlan> finalization) {
            throw new UnsupportedOperationException();
        }
    }
}
