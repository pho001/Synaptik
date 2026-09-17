package io.github.pho001.synaptik.backend.cpu.spi;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.cpu.CpuLocalWorkloadTuning;
import io.github.pho001.synaptik.compiler.DerivativeGraphMetadata;
import io.github.pho001.synaptik.compiler.PublicationPlan;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.CompileConstantPlan;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanning;
import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Distinct-package checks for the supported opaque CPU tuning collaboration. */
final class CpuLocalWorkloadTuningPublicTest {
    @Test
    void exposesOnlyTheTypedOpaqueSupportedShape() throws Exception {
        assertAll(
                () -> assertTrue(Modifier.isPublic(CpuLocalWorkloadTuning.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(CpuLocalWorkloadTuning.class.getModifiers())),
                () -> assertEquals(0, CpuLocalWorkloadTuning.class.getConstructors().length),
                () -> assertEquals(List.of(BackendTuningCandidateBatch.class),
                        List.of(CpuLocalWorkloadTuning.CandidateBatch.class.getInterfaces())),
                () -> assertEquals(List.of(BackendTuningDecision.class),
                        List.of(CpuLocalWorkloadTuning.SelectedDecision.class.getInterfaces())));
        for (Class<?> value : List.of(CpuLocalWorkloadTuning.CandidateBatch.class,
                CpuLocalWorkloadTuning.SelectedDecision.class,
                CpuLocalWorkloadTuning.Candidate.class,
                CpuLocalWorkloadTuning.Compatibility.class,
                CpuLocalWorkloadTuning.CandidateIdentity.class)) {
            assertEquals(0, value.getConstructors().length);
        }
        String source = Files.readString(Path.of("src/main/java/io/github/pho001/synaptik/backend/cpu/"
                + "CpuLocalWorkloadTuning.java"));
        assertAll(
                () -> assertFalse(source.contains("tools.tuning")),
                () -> assertFalse(source.contains("synaptik.engine")),
                () -> assertFalse(source.contains("java.io.")),
                () -> assertFalse(source.contains("java.nio.file")),
                () -> assertFalse(source.contains("reflect")));
    }

    @Test
    void retainedCollaborationRejectsClosedUseAndValidNonTunableWork() {
        CpuBackendIntegration integration = CpuBackendIntegration.open();
        CpuLocalWorkloadTuning tuning = integration.localWorkloadTuning();
        CompileArtifacts artifacts = compile(integration, leaf(Shape.of(4)).contiguous());
        assertAll(
                () -> assertSame(tuning, integration.localWorkloadTuning()),
                () -> assertTrue(tuning.candidateHandoff(artifacts).isEmpty()));
        integration.close();
        integration.close();
        assertAll(
                () -> assertThrows(IllegalStateException.class,
                        integration::localWorkloadTuning),
                () -> assertThrows(IllegalStateException.class,
                        () -> tuning.candidateHandoff(artifacts)));
    }

    @Test
    void eligibleBatchRoundTripsAndPreparesExactCandidatesWhenProviderIsAvailable() {
        try (CpuBackendIntegration integration = CpuBackendIntegration.open()) {
            CpuLocalWorkloadTuning tuning = integration.localWorkloadTuning();
            CompileArtifacts artifacts = resolvedMatmulArtifacts(integration,
                    leaf(Shape.of(2, 3)).matmul(leaf(Shape.of(3, 2))));
            Optional<io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff<
                    CpuLocalWorkloadTuning.CandidateBatch,
                    CpuLocalWorkloadTuning.SelectedDecision>> maybe =
                    tuning.candidateHandoff(artifacts);
            if (maybe.isEmpty()) return; // portable-only installations are supported
            var handoff = maybe.orElseThrow();
            var batch = handoff.candidateBatch();
            var candidates = tuning.candidates(batch);
            assertTrue(candidates.size() >= 2);
            assertSame(artifacts.partitions().getFirst(), handoff.partition());
            assertTrue(handoff.selectedDecision().isEmpty());

            var compatibility = tuning.compatibility(batch);
            byte[] compatibilityBytes = compatibility.bytes();
            compatibilityBytes[0] ^= 1;
            assertFalse(Arrays.equals(compatibilityBytes, compatibility.bytes()));
            var firstIdentity = tuning.candidateIdentity(candidates.getFirst());
            byte[] identityBytes = firstIdentity.bytes();
            identityBytes[0] ^= 1;
            assertFalse(Arrays.equals(identityBytes, firstIdentity.bytes()));

            for (var candidate : candidates) {
                var decision = tuning.selectedDecision(batch, candidate);
                byte[] encoded = tuning.encodeDecision(decision);
                var decoded = tuning.decodeCompatibleDecision(batch, encoded).orElseThrow();
                assertEquals(decision, decoded);
                assertNotNull(tuning.prepareTrial(batch, candidate));
                assertNotNull(tuning.prepareSelected(batch, decoded));
                byte[] corrupt = encoded.clone();
                corrupt[corrupt.length - 1] ^= 1;
                assertTrue(tuning.decodeCompatibleDecision(batch, corrupt).isEmpty());
                assertTrue(tuning.decodeCompatibleDecision(batch,
                        Arrays.copyOf(encoded, encoded.length - 1)).isEmpty());
                assertTrue(tuning.decodeCompatibleDecision(batch,
                        Arrays.copyOf(encoded, encoded.length + 1)).isEmpty());
            }

            var secondBatch = tuning.candidateHandoff(artifacts).orElseThrow().candidateBatch();
            assertThrows(IllegalArgumentException.class,
                    () -> tuning.selectedDecision(secondBatch, candidates.getFirst()));
            var firstDecision = tuning.selectedDecision(batch, candidates.getFirst());
            assertThrows(IllegalArgumentException.class,
                    () -> tuning.prepareSelected(secondBatch, firstDecision));
        }
    }

    private static Tensor leaf(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false));
    }

    private static CompileArtifacts compile(CpuBackendIntegration integration, Tensor output) {
        return GraphCompilationPort.compile(CompileMode.FORWARD_ONLY, List.of(output),
                Optional.empty(), GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(), List.of(integration.capabilityProvider()),
                List.of(integration.availabilitySnapshot()));
    }

    private static CompileArtifacts resolvedMatmulArtifacts(
            CpuBackendIntegration integration, Tensor output) {
        BackendCapabilityProvider captureProvider = new BackendCapabilityProvider() {
            @Override public io.github.pho001.synaptik.backend.contract.BackendId backendId() {
                return CpuBackendIntegration.class.cast(integration).capabilityProvider().backendId();
            }
            @Override public boolean supports(
                    io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery query) {
                return true;
            }
        };
        CompileArtifacts base = GraphCompilationPort.compile(CompileMode.FORWARD_ONLY,
                List.of(output), Optional.empty(), GraphOptimizationConfig.disabled(),
                BackendIntent.unconstrained(), PartitionScoringConfig.neutral(),
                List.of(captureProvider), List.of(integration.availabilitySnapshot()));
        var outputIds = java.util.Set.copyOf(base.graph().outputs());
        var values = base.graph().values().stream().map(value -> {
            if (!outputIds.contains(value.id())) return value;
            var descriptor = value.descriptor();
            var resolved = new TensorDescriptor(descriptor.dataType(), descriptor.shape(),
                    Optional.of(LayoutDescriptor.contiguous(descriptor.shape())),
                    descriptor.requiresGrad());
            return new GraphValue(value.id(), resolved);
        }).toList();
        var graph = new CompiledGraphModel(values, base.graph().nodes(), base.graph().inputs(),
                base.graph().outputs(), base.graph().nodePhases());
        var publication = construct(PublicationPlan.class,
                new Class<?>[] {CompiledGraphModel.class, List.class, List.class}, graph,
                base.publication().forwardBindings(), base.publication().gradientBindings());
        var constants = construct(CompileConstantPlan.class,
                new Class<?>[] {List.class, List.class}, List.of(), graph.inputs().stream()
                        .map(id -> new CompileConstantPlan.ConstantSource(
                                id, ScalarValue.float32(1.0f))).toList());
        return new CompileArtifacts(base.mode(), graph, base.partitions(),
                LogicalMemoryPlanning.plan(graph, base.partitions()), publication, constants,
                base.diagnostics(), new DerivativeGraphMetadata(
                        graph, base.derivatives().derivativeOrderByNode()));
    }

    private static <T> T construct(
            Class<T> type, Class<?>[] parameterTypes, Object... arguments) {
        try {
            Constructor<T> constructor = type.getDeclaredConstructor(parameterTypes);
            constructor.setAccessible(true);
            return constructor.newInstance(arguments);
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
    }
}
