package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.DerivativeGraphMetadata;
import io.github.pho001.synaptik.compiler.PublicationPlan;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import io.github.pho001.synaptik.engine.CompiledGraph;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.ModelAutotuningPreparation;
import io.github.pho001.synaptik.engine.ModelAutotuningRequest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.reflect.Constructor;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Exercises conditional eligible CPU tuning and deterministic absent-handoff public fallback. */
final class EngineModelAutotuningIntegrationTest {
    @Test
    void eligibleCpuAlternativesCompleteBothPublicTuningPhases(@TempDir Path directory) {
        try (Engine engine = Engine.standard()) {
            Shape shape = Shape.of(2, 2);
            var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                    Optional.of(LayoutDescriptor.contiguous(shape)), false);
            var left = TensorFactory.fromFlatArray(
                    descriptor, Optional.empty(), new float[] {1, 2, 3, 4});
            var right = TensorFactory.fromFlatArray(
                    descriptor, Optional.empty(), new float[] {5, 6, 7, 8});
            CompiledGraph compiled;
            try (CpuBackendIntegration fixture = CpuBackendIntegration.open()) {
                CompileArtifacts artifacts = resolvedMatmulArtifacts(
                        fixture, left.matmul(right));
                org.junit.jupiter.api.Assumptions.assumeTrue(
                        hasLocalWorkloadAlternatives(fixture, artifacts),
                        "host CPU fixture has no eligible local-workload alternatives");
                compiled = construct(CompiledGraph.class,
                        new Class<?>[] {Engine.class, CompileArtifacts.class}, engine, artifacts);
            }
            var config = new ModelAutotuningConfig(
                    ModelAutotuningConfig.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                    new ModelAutotuningConfig.Budget(1, 16, 0, 1),
                    new ModelAutotuningConfig.RepresentativeProfileIdentity(
                            1, new byte[] {4}),
                    ModelAutotuningConfig.FallbackPolicy.REQUIRE_TUNED_RESULT,
                    directory.resolve("workload.bin"),
                    new ModelAutotuningConfig.CompletePlanBudget(64, 0, 1, 128, 16),
                    directory.resolve("model-plan.bin"));
            var request = new ModelAutotuningRequest(
                    config,
                    new ModelAutotuningRequest.ModelIdentity(1, new byte[] {3}),
                    List.of(left, right));

            ModelAutotuningPreparation preparation = engine.prepareTuned(compiled, request);

            assertEquals(ModelAutotuningPreparation.Outcome.TUNED, preparation.outcome());
            var evidence = preparation.evidence().orElseThrow();
            assertEquals(ModelAutotuningPreparation.Source.MEASURED,
                    evidence.completePlan().source());
            assertEquals(ModelAutotuningPreparation.ReuseScope.SESSION,
                    evidence.completePlan().compatibility().reuseScope());
            assertEquals(config.completePlanBudget(), evidence.completePlan().budget());
            org.junit.jupiter.api.Assertions.assertFalse(
                    java.nio.file.Files.exists(config.modelPlanCache()));
            try (var result = engine.run(
                    preparation.preparedExecution(), List.of(left, right))) {
                assertEquals(1, result.resultCount());
            }
        }
    }

    @Test
    void absentCpuHandoffFallsBackAndReturnedPreparationRuns() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Shape shape = Shape.of(2);
            var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                    Optional.of(LayoutDescriptor.contiguous(shape)), false);
            var input = TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                    new MemorySegmentStorage(DataType.FLOAT32, 2,
                            arena.allocate(8, Float.BYTES))));
            var compiled = engine.compile(List.of(input.contiguous()));
            var config = new ModelAutotuningConfig(
                    ModelAutotuningConfig.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                    new ModelAutotuningConfig.Budget(1, 4, 0, 1),
                    new ModelAutotuningConfig.RepresentativeProfileIdentity(1, new byte[] {2}),
                    ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC,
                    Path.of("unused-native-free-autotuning-cache.bin"),
                    new ModelAutotuningConfig.CompletePlanBudget(1, 0, 1, 1L, 0L),
                    Path.of("unused-model-plan-cache.bin"));
            var request = new ModelAutotuningRequest(config,
                    new ModelAutotuningRequest.ModelIdentity(1, new byte[] {1}), List.of(input));

            ModelAutotuningPreparation preparation = engine.prepareTuned(compiled, request);

            assertEquals(ModelAutotuningPreparation.Outcome.SAFE_HEURISTIC_FALLBACK,
                    preparation.outcome());
            assertEquals(Optional.empty(), preparation.evidence());
            assertSame(compiled, preparation.preparedExecution().compiledGraph());
            try (var result = engine.run(preparation.preparedExecution(), List.of(input))) {
                assertEquals(1, result.resultCount());
            }
        }
    }

    private static CompileArtifacts resolvedMatmulArtifacts(
            CpuBackendIntegration integration,
            Tensor output) {
        CompileArtifacts base = compileFixture(integration, output);
        var outputIds = java.util.Set.copyOf(base.graph().outputs());
        var values = base.graph().values().stream().map(value -> {
            if (!outputIds.contains(value.id())) return value;
            var descriptor = value.descriptor();
            return new GraphValue(value.id(), new TensorDescriptor(
                    descriptor.dataType(), descriptor.shape(),
                    Optional.of(LayoutDescriptor.contiguous(descriptor.shape())),
                    descriptor.requiresGrad()));
        }).toList();
        var graph = new CompiledGraphModel(values, base.graph().nodes(), base.graph().inputs(),
                base.graph().outputs(), base.graph().nodePhases());
        var publication = construct(PublicationPlan.class,
                new Class<?>[] {CompiledGraphModel.class, List.class, List.class}, graph,
                base.publication().forwardBindings(), base.publication().gradientBindings());
        try {
            List<?> partitions = (List<?>) CompileArtifacts.class.getMethod("partitions")
                    .invoke(base);
            Class<?> memoryPlanning = Class.forName(
                    "io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanning");
            Object memory = java.util.Arrays.stream(memoryPlanning.getMethods())
                    .filter(method -> method.getName().equals("plan")
                            && method.getParameterCount() == 2)
                    .findFirst().orElseThrow().invoke(null, graph, partitions);
            Constructor<?> constructor = CompileArtifacts.class.getConstructors()[0];
            return (CompileArtifacts) constructor.newInstance(
                    base.mode(), graph, partitions, memory, publication,
                    base.constants(), base.diagnostics(), new DerivativeGraphMetadata(
                            graph, base.derivatives().derivativeOrderByNode()));
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
    }

    private static CompileArtifacts compileFixture(
            CpuBackendIntegration integration,
            Tensor output) {
        try {
            Class<?> providerType = Class.forName(
                    "io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider");
            Object provider = Proxy.newProxyInstance(providerType.getClassLoader(),
                    new Class<?>[] {providerType}, (proxy, method, arguments) -> switch (
                            method.getName()) {
                        case "backendId" -> {
                            Object capabilityProvider = CpuBackendIntegration.class
                                    .getMethod("capabilityProvider").invoke(integration);
                            yield capabilityProvider.getClass().getMethod("backendId")
                                    .invoke(capabilityProvider);
                        }
                        case "supports" -> true;
                        case "toString" -> "resolved-matmul-fixture";
                        case "hashCode" -> System.identityHashCode(proxy);
                        case "equals" -> proxy == arguments[0];
                        default -> throw new UnsupportedOperationException(method.toString());
                    });
            Class<?> compilation = Class.forName(
                    "io.github.pho001.synaptik.compiler.GraphCompilationPort");
            var compile = java.util.Arrays.stream(compilation.getMethods())
                    .filter(method -> method.getName().equals("compile")
                            && method.getParameterCount() == 8)
                    .findFirst().orElseThrow();
            return (CompileArtifacts) compile.invoke(null,
                    CompileMode.FORWARD_ONLY, List.of(output), Optional.empty(),
                    GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(),
                    PartitionScoringConfig.neutral(), List.of(provider),
                    List.of(integration.availabilitySnapshot()));
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
    }

    private static boolean hasLocalWorkloadAlternatives(
            CpuBackendIntegration integration, CompileArtifacts artifacts) {
        try {
            Object tuning = CpuBackendIntegration.class.getMethod("localWorkloadTuning")
                    .invoke(integration);
            Object preparation = CpuBackendIntegration.class.getMethod("partitionPreparation")
                    .invoke(integration);
            Object inputs = preparation.getClass().getMethod("backendInputs")
                    .invoke(preparation);
            Class<?> inputsType = Class.forName(
                    "io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs");
            Class<?> preparationType = Class.forName(
                    "io.github.pho001.synaptik.prepare.GraphPreparation");
            Object partition = ((List<?>) CompileArtifacts.class.getMethod("partitions")
                    .invoke(artifacts)).getFirst();
            Object context = preparationType.getMethod("project", CompileArtifacts.class,
                            Class.forName(
                                    "io.github.pho001.synaptik.planning.partition.PlannedPartition"),
                            inputsType)
                    .invoke(null, artifacts, partition, inputs);
            Object maybeHandoff = tuning.getClass()
                    .getMethod("candidateHandoff", Class.forName(
                            "io.github.pho001.synaptik.prepare.analysis.PrepareContext"))
                    .invoke(tuning, context);
            Optional<?> handoff = (Optional<?>) maybeHandoff;
            if (handoff.isEmpty()) return false;
            Object batch = handoff.orElseThrow().getClass().getMethod("candidateBatch")
                    .invoke(handoff.orElseThrow());
            var candidatesMethod = java.util.Arrays.stream(tuning.getClass().getMethods())
                    .filter(method -> method.getName().equals("candidates")
                            && method.getParameterCount() == 1)
                    .findFirst().orElseThrow();
            return ((List<?>) candidatesMethod.invoke(tuning, batch)).size() >= 2;
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
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
