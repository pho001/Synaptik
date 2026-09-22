package io.github.pho001.synaptik.backend.cpu.spi;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.cpu.CpuCompletePlanTuning;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.CompileConstantPlan;
import io.github.pho001.synaptik.compiler.CompileDiagnostics;
import io.github.pho001.synaptik.compiler.DerivativeGraphMetadata;
import io.github.pho001.synaptik.compiler.PublicationPlan;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.ForwardPublicationBinding;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorId;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanning;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import java.lang.reflect.Modifier;
import java.lang.reflect.Constructor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Distinct-package checks for the supported complete-plan collaboration. */
final class CpuCompletePlanTuningPublicTest {
    @Test
    void exposesOnlyTheTypedOpaqueSupportedShape() throws Exception {
        assertAll(
                () -> assertTrue(Modifier.isPublic(CpuCompletePlanTuning.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(CpuCompletePlanTuning.class.getModifiers())),
                () -> assertEquals(0, CpuCompletePlanTuning.class.getConstructors().length),
                () -> assertEquals(List.of(BackendTuningCandidateBatch.class),
                        List.of(CpuCompletePlanTuning.CandidateBatch.class.getInterfaces())),
                () -> assertEquals(List.of(BackendTuningDecision.class),
                        List.of(CpuCompletePlanTuning.SelectedDecision.class.getInterfaces())),
                () -> assertEquals(List.of("candidateHandoff", "candidateIdentity", "candidates",
                                "compatibility", "decodeCompatibleDecision", "encodeDecision",
                                "selectedDecision", "selectedPreparation", "trialPreparation"),
                        Arrays.stream(CpuCompletePlanTuning.class.getDeclaredMethods())
                                .filter(method -> Modifier.isPublic(method.getModifiers()))
                                .map(method -> method.getName()).sorted().toList()),
                () -> Arrays.stream(CpuCompletePlanTuning.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .forEach(method -> assertFalse(
                                method.toGenericString().contains(".compiler."))));
        for (Class<?> value : List.of(CpuCompletePlanTuning.CandidateBatch.class,
                CpuCompletePlanTuning.SelectedDecision.class,
                CpuCompletePlanTuning.Candidate.class,
                CpuCompletePlanTuning.Compatibility.class,
                CpuCompletePlanTuning.CandidateIdentity.class)) {
            assertEquals(0, value.getConstructors().length);
        }
        String source = Files.readString(Path.of("src/main/java/io/github/pho001/synaptik/backend/cpu/"
                + "CpuCompletePlanTuning.java"));
        assertAll(
                () -> assertFalse(source.contains("tools.tuning")),
                () -> assertFalse(source.contains("synaptik.engine")),
                () -> assertFalse(source.contains("synaptik.compiler")),
                () -> assertFalse(source.contains("java.io.")),
                () -> assertFalse(source.contains("java.nio.file")),
                () -> assertFalse(source.contains("reflect")));
    }

    @Test
    void completeAlternativesRoundTripAndPrepareFreshRecipes() {
        try (CpuBackendIntegration integration = CpuBackendIntegration.open()) {
            CpuCompletePlanTuning tuning = integration.completePlanTuning();
            CompileArtifacts artifacts = pointwiseArtifacts(0);
            var handoff = tuning.candidateHandoff(
                    context(integration, artifacts), Optional.empty()).orElseThrow();
            var batch = handoff.candidateBatch();
            var candidates = tuning.candidates(batch);
            assertAll(
                    () -> assertSame(tuning, integration.completePlanTuning()),
                    () -> assertSame(artifacts.partitions().getFirst(), handoff.partition()),
                    () -> assertTrue(handoff.selectedDecision().isEmpty()),
                    () -> assertTrue(candidates.size() >= 2),
                    () -> assertEquals(CpuCompletePlanTuning.ReuseScope.SESSION,
                            tuning.compatibility(batch).reuseScope()));

            byte[] compatibility = tuning.compatibility(batch).bytes();
            compatibility[0] ^= 1;
            assertFalse(Arrays.equals(compatibility, tuning.compatibility(batch).bytes()));
            var identities = candidates.stream().map(tuning::candidateIdentity).toList();
            assertEquals(identities.size(), identities.stream().distinct().count());
            var shiftedArtifacts = pointwiseArtifacts(10_000);
            var shifted = tuning.candidateHandoff(
                    context(integration, shiftedArtifacts), Optional.empty())
                    .orElseThrow().candidateBatch();
            assertAll(
                    () -> assertEquals(tuning.compatibility(batch),
                            tuning.compatibility(shifted)),
                    () -> assertEquals(identities, tuning.candidates(shifted).stream()
                            .map(tuning::candidateIdentity).toList()));

            var decision = tuning.selectedDecision(batch, candidates.getFirst());
            byte[] encoded = tuning.encodeDecision(decision);
            var decoded = tuning.decodeCompatibleDecision(batch, encoded).orElseThrow();
            assertEquals(decision, decoded);
            var first = tuning.trialPreparation(batch, candidates.getFirst());
            var second = tuning.selectedPreparation(batch, decoded);
            assertNotSame(first, second);

            byte[] corrupt = encoded.clone();
            corrupt[corrupt.length - 1] ^= 1;
            assertTrue(tuning.decodeCompatibleDecision(batch, corrupt).isEmpty());
            assertTrue(tuning.decodeCompatibleDecision(batch,
                    Arrays.copyOf(encoded, encoded.length - 1)).isEmpty());

            var otherBatch = tuning.candidateHandoff(
                    context(integration, artifacts), Optional.empty())
                    .orElseThrow().candidateBatch();
            assertThrows(IllegalArgumentException.class,
                    () -> tuning.selectedDecision(otherBatch, candidates.getFirst()));
            assertThrows(IllegalArgumentException.class,
                    () -> tuning.selectedPreparation(otherBatch, decision));
        }
    }

    @Test
    void directOnlyWorkIsNotTunableAndClosedUseFails() {
        CpuBackendIntegration integration = CpuBackendIntegration.open();
        CpuCompletePlanTuning tuning = integration.completePlanTuning();
        CompileArtifacts artifacts = compile(integration, leaf().contiguous());
        assertTrue(tuning.candidateHandoff(
                context(integration, artifacts), Optional.empty()).isEmpty());
        integration.close();
        assertAll(
                () -> assertThrows(IllegalStateException.class,
                        integration::completePlanTuning),
                () -> assertThrows(IllegalStateException.class,
                        () -> tuning.candidateHandoff(
                                context(integration, artifacts), Optional.empty())));
    }

    private static io.github.pho001.synaptik.prepare.analysis.PrepareContext<?> context(
            CpuBackendIntegration integration, CompileArtifacts artifacts) {
        var preparation = integration.partitionPreparation();
        return GraphPreparation.project(artifacts, artifacts.partitions().getFirst(),
                preparation.backendInputs());
    }

    private static Tensor leaf() {
        Shape shape = Shape.of(4);
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false));
    }

    private static CompileArtifacts compile(CpuBackendIntegration integration, Tensor output) {
        BackendCapabilityProvider captureProvider = new BackendCapabilityProvider() {
            @Override public io.github.pho001.synaptik.backend.contract.BackendId backendId() {
                return integration.capabilityProvider().backendId();
            }
            @Override public boolean supports(
                    io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery query) {
                return true;
            }
        };
        return GraphCompilationPort.compile(CompileMode.FORWARD_ONLY, List.of(output),
                Optional.empty(), GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(), List.of(captureProvider),
                List.of(integration.availabilitySnapshot()));
    }

    private static CompileArtifacts pointwiseArtifacts(int identityOffset) {
        Shape shape = Shape.of(4);
        TensorDescriptor descriptor = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var ids = java.util.stream.IntStream.range(0, 6)
                .mapToObj(index -> new ValueId(identityOffset + index)).toList();
        var nodes = List.of(
                new CompiledNode(new NodeId(identityOffset),
                        new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                        List.of(ids.get(0), ids.get(1)), List.of(ids.get(3))),
                new CompiledNode(new NodeId(identityOffset + 1),
                        new Operation(UnaryElementwiseKind.GELU, NoOperationAttrs.INSTANCE),
                        List.of(ids.get(3)), List.of(ids.get(4))),
                new CompiledNode(new NodeId(identityOffset + 2),
                        new Operation(BinaryArithmeticKind.MUL, NoOperationAttrs.INSTANCE),
                        List.of(ids.get(4), ids.get(2)), List.of(ids.get(5))));
        ValueId constantId = new ValueId(identityOffset + 6);
        TensorDescriptor constantDescriptor = new TensorDescriptor(DataType.FLOAT64,
                Shape.scalar(), Optional.of(LayoutDescriptor.contiguous(Shape.scalar())), false);
        var values = new java.util.ArrayList<GraphValue>();
        ids.forEach(id -> values.add(new GraphValue(id, descriptor)));
        values.add(new GraphValue(constantId, constantDescriptor));
        var phases = new LinkedHashMap<NodeId, GraphPhase>();
        nodes.forEach(node -> phases.put(node.id(), GraphPhase.FORWARD));
        var graph = new CompiledGraphModel(values, nodes,
                List.of(ids.get(0), ids.get(1), ids.get(2), constantId),
                List.of(ids.get(5), constantId), phases);
        var partition = new PlannedPartition(
                io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        List<PlannedPartition> partitions = List.of(partition);
        var publication = construct(PublicationPlan.class,
                new Class<?>[] {CompiledGraphModel.class, List.class, List.class}, graph,
                List.of(new ForwardPublicationBinding(new TensorId(100), ids.get(5)),
                        new ForwardPublicationBinding(new TensorId(101), constantId)), List.of());
        var bindable = java.util.stream.IntStream.range(0, 3)
                .mapToObj(index -> new CompileConstantPlan.BindableInput(
                        new TensorId(index), ids.get(index))).toList();
        var constants = construct(CompileConstantPlan.class,
                new Class<?>[] {List.class, List.class}, bindable,
                List.of(new CompileConstantPlan.ConstantSource(constantId,
                        ScalarValue.float64(2.0))));
        var diagnostics = construct(CompileDiagnostics.class,
                new Class<?>[] {List.class}, List.of());
        var derivativeOrders = new LinkedHashMap<NodeId, Integer>();
        nodes.forEach(node -> derivativeOrders.put(node.id(), 0));
        return new CompileArtifacts(CompileMode.FORWARD_ONLY, graph, partitions,
                LogicalMemoryPlanning.plan(graph, partitions), publication, constants,
                diagnostics, new DerivativeGraphMetadata(graph, derivativeOrders));
    }

    private static <T> T construct(Class<T> type, Class<?>[] parameterTypes,
            Object... arguments) {
        try {
            Constructor<T> constructor = type.getDeclaredConstructor(parameterTypes);
            constructor.setAccessible(true);
            return constructor.newInstance(arguments);
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
    }
}
