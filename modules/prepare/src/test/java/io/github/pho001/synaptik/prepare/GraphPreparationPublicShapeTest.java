package io.github.pho001.synaptik.prepare;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class GraphPreparationPublicShapeTest {
    @Test
    void exposesExactlyThePlannedRootOrchestrationTypes() throws Exception {
        Method prepare = GraphPreparation.class.getDeclaredMethod(
                "prepare", CompileArtifacts.class, List.class, PreparedScheduleAssembler.class);
        Method prepareWithProducerlessResources = GraphPreparation.class.getDeclaredMethod(
                "prepare",
                CompileArtifacts.class,
                List.class,
                List.class,
                PreparedScheduleAssembler.class);
        Method assemble = PreparedScheduleAssembler.class.getDeclaredMethod(
                "assemble", PreparedScheduleContext.class);

        assertAll(
                () -> assertPublicRecord(PartitionPreparation.class),
                () -> assertPublicRecord(PreparedBufferAssignment.class),
                () -> assertPublicRecord(PreparedScheduleContext.class),
                () -> assertPublicRecord(ProducerlessPublishedConstantResource.class),
                () -> assertTrue(Modifier.isPublic(PreparedScheduleAssembler.class.getModifiers())),
                () -> assertTrue(PreparedScheduleAssembler.class.isInterface()),
                () -> assertTrue(PreparedScheduleAssembler.class.isAnnotationPresent(
                        FunctionalInterface.class)),
                () -> assertTrue(Modifier.isPublic(GraphPreparation.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(GraphPreparation.class.getModifiers())),
                () -> assertEquals(1, GraphPreparation.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(
                        GraphPreparation.class.getDeclaredConstructors()[0].getModifiers())),
                () -> assertEquals(0, GraphPreparation.class.getDeclaredFields().length),
                () -> assertEquals(2, Arrays.stream(GraphPreparation.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .count()),
                () -> assertTrue(Modifier.isPublic(prepare.getModifiers())),
                () -> assertTrue(Modifier.isStatic(prepare.getModifiers())),
                () -> assertEquals(PreparedExecution.class, prepare.getReturnType()),
                () -> assertTrue(Modifier.isPublic(
                        prepareWithProducerlessResources.getModifiers())),
                () -> assertTrue(Modifier.isStatic(
                        prepareWithProducerlessResources.getModifiers())),
                () -> assertEquals(
                        PreparedExecution.class,
                        prepareWithProducerlessResources.getReturnType()),
                () -> assertEquals(1, Arrays.stream(
                                PreparedScheduleAssembler.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .count()),
                () -> assertEquals(assemble, PreparedScheduleAssembler.class.getDeclaredMethods()[0]));
    }

    @Test
    void locksRecordComponentsAndBackendGenericAssociations() {
        assertAll(
                () -> assertRecordComponents(
                        PartitionPreparation.class, "backendInputs", "preparer", "finalizer"),
                () -> assertRecordComponents(
                        PreparedBufferAssignment.class, "valueId", "slot", "planIndex"),
                () -> assertRecordComponents(
                        PreparedScheduleContext.class,
                        "artifacts", "memoryPlan", "partitions", "bufferAssignments"),
                () -> assertRecordComponents(
                        ProducerlessPublishedConstantResource.class,
                        "value", "logicalRequirement", "byteSize", "byteAlignment"),
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
                        () -> new PreparedBufferAssignment(null, null, -1)),
                () -> assertFailure(
                        NullPointerException.class,
                        "slot",
                        () -> new PreparedBufferAssignment(new ValueId(0), null, -1)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "planIndex must be non-negative",
                        () -> new PreparedBufferAssignment(new ValueId(0), slot, -1)));
    }

    @Test
    void producerlessResourceHasExactSurfaceAndValidatesInInvariantOrder() {
        TensorDescriptor descriptor = resolvedDescriptor();
        GraphValue value = new GraphValue(new ValueId(10), descriptor);
        LogicalMemoryRequirement valid = requirement(value, Optional.empty(), List.of(), true);
        PlannedPartition partition = new PlannedPartition(
                new io.github.pho001.synaptik.backend.contract.BackendId("cpu"),
                List.of(new io.github.pho001.synaptik.model.graph.NodeId(1)));

        assertAll(
                () -> assertEquals(
                        7,
                        Arrays.stream(ProducerlessPublishedConstantResource.class.getDeclaredMethods())
                                .filter(method -> Modifier.isPublic(method.getModifiers()))
                                .count()),
                () -> assertEquals(
                        1,
                        ProducerlessPublishedConstantResource.class.getDeclaredConstructors().length),
                () -> assertFailure(
                        NullPointerException.class,
                        "value",
                        () -> new ProducerlessPublishedConstantResource(null, null, -1, 0)),
                () -> assertFailure(
                        NullPointerException.class,
                        "logicalRequirement",
                        () -> new ProducerlessPublishedConstantResource(value, null, -1, 0)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "logicalRequirement.valueId must match value.id",
                        () -> new ProducerlessPublishedConstantResource(
                                value,
                                new LogicalMemoryRequirement(
                                        new ValueId(11), descriptor, Optional.empty(), List.of(), true),
                                -1,
                                0)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "logicalRequirement.descriptor must match value.descriptor",
                        () -> new ProducerlessPublishedConstantResource(
                                value,
                                new LogicalMemoryRequirement(
                                        value.id(), resolvedDescriptor(2), Optional.empty(), List.of(), true),
                                -1,
                                0)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "logicalRequirement must be producerless",
                        () -> new ProducerlessPublishedConstantResource(
                                value, requirement(value, Optional.of(partition), List.of(), true), -1, 0)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "logicalRequirement must be consumerless",
                        () -> new ProducerlessPublishedConstantResource(
                                value, requirement(value, Optional.empty(), List.of(partition), true), -1, 0)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "logicalRequirement must require graph output",
                        () -> new ProducerlessPublishedConstantResource(
                                value, requirement(value, Optional.empty(), List.of(), false), -1, 0)),
                () -> {
                    TensorDescriptor dynamic = new TensorDescriptor(
                            DataType.FLOAT32,
                            Shape.ofDimensions(new DynamicDimension("N")),
                            Optional.empty(),
                            false);
                    GraphValue dynamicValue = new GraphValue(value.id(), dynamic);
                    assertFailure(
                            IllegalArgumentException.class,
                            "value descriptor shape must be fully static",
                            () -> new ProducerlessPublishedConstantResource(
                                    dynamicValue,
                                    requirement(dynamicValue, Optional.empty(), List.of(), true),
                                    -1,
                                    0));
                },
                () -> {
                    TensorDescriptor unresolved = new TensorDescriptor(
                            DataType.FLOAT32, Shape.of(1), Optional.empty(), false);
                    GraphValue unresolvedValue = new GraphValue(value.id(), unresolved);
                    assertFailure(
                            IllegalArgumentException.class,
                            "value descriptor layout must be resolved",
                            () -> new ProducerlessPublishedConstantResource(
                                    unresolvedValue,
                                    requirement(unresolvedValue, Optional.empty(), List.of(), true),
                                    -1,
                                    0));
                },
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "byteSize must be non-negative",
                        () -> new ProducerlessPublishedConstantResource(value, valid, -1, 0)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "byteAlignment must be a positive power of two",
                        () -> new ProducerlessPublishedConstantResource(value, valid, 0, 0)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "byteAlignment must be a positive power of two",
                        () -> new ProducerlessPublishedConstantResource(value, valid, 0, 3)),
                () -> {
                    ProducerlessPublishedConstantResource resource =
                            new ProducerlessPublishedConstantResource(value, valid, 0, 8);
                    assertSameReferences(value, valid, resource);
                });
    }

    private static void assertSameReferences(
            GraphValue value,
            LogicalMemoryRequirement requirement,
            ProducerlessPublishedConstantResource resource) {
        org.junit.jupiter.api.Assertions.assertSame(value, resource.value());
        org.junit.jupiter.api.Assertions.assertSame(requirement, resource.logicalRequirement());
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

    private static void assertPublicRecord(Class<?> type) {
        assertTrue(Modifier.isPublic(type.getModifiers()));
        assertTrue(type.isRecord());
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
