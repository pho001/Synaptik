package io.github.pho001.synaptik.prepare;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.model.graph.ValueId;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class GraphPreparationPublicShapeTest {
    @Test
    void exposesExactlyThePlannedRootOrchestrationTypes() throws Exception {
        Method prepare = GraphPreparation.class.getDeclaredMethod(
                "prepare", CompileArtifacts.class, List.class, PreparedScheduleAssembler.class);
        Method assemble = PreparedScheduleAssembler.class.getDeclaredMethod(
                "assemble", PreparedScheduleContext.class);

        assertAll(
                () -> assertPublicRecord(PartitionPreparation.class),
                () -> assertPublicRecord(PreparedBufferAssignment.class),
                () -> assertPublicRecord(PreparedScheduleContext.class),
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
                () -> assertEquals(1, Arrays.stream(GraphPreparation.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .count()),
                () -> assertTrue(Modifier.isPublic(prepare.getModifiers())),
                () -> assertTrue(Modifier.isStatic(prepare.getModifiers())),
                () -> assertEquals(PreparedExecution.class, prepare.getReturnType()),
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
        public io.github.pho001.synaptik.runtime.execution.PreparedExecutable finalizePartition(
                BackendPartitionFinalization<FakePlan> finalization) {
            throw new UnsupportedOperationException();
        }
    }
}
