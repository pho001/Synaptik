package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.ModelAutotuningPreparation;
import io.github.pho001.synaptik.engine.ModelAutotuningRequest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** Exercises ordinary public CPU compilation, strict tuning, numerical execution, and fallback. */
final class EngineModelAutotuningIntegrationTest {
    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void ordinaryCpuMatmulTunesAndRunsExactValues(DataType dataType, @TempDir Path directory) {
        try (Engine engine = Engine.standard()) {
            Shape shape = Shape.of(2, 2);
            var descriptor = new TensorDescriptor(dataType, shape,
                    Optional.of(LayoutDescriptor.contiguous(shape)), false);
            Tensor left;
            Tensor right;
            if (dataType == DataType.FLOAT32) {
                left = TensorFactory.fromFlatArray(
                        descriptor, Optional.empty(), new float[] {1, 2, 3, 4});
                right = TensorFactory.fromFlatArray(
                        descriptor, Optional.empty(), new float[] {5, 6, 7, 8});
            } else {
                left = TensorFactory.fromFlatArray(
                        descriptor, Optional.empty(), new double[] {1, 2, 3, 4});
                right = TensorFactory.fromFlatArray(
                        descriptor, Optional.empty(), new double[] {5, 6, 7, 8});
            }
            var compiled = engine.compile(List.of(left.matmul(right)));
            var config = strictConfig(directory, dataType == DataType.FLOAT32 ? 16 : 32);
            var request = new ModelAutotuningRequest(config,
                    new ModelAutotuningRequest.ModelIdentity(1, new byte[] {3}),
                    List.of(left, right));

            var preparation = engine.prepareTuned(compiled, request);

            assertEquals(ModelAutotuningPreparation.Outcome.TUNED, preparation.outcome());
            assertSame(compiled, preparation.preparedExecution().compiledGraph());
            var evidence = preparation.evidence().orElseThrow();
            assertEquals(1, evidence.workloads().size());
            assertFalse(evidence.workloads().getFirst().candidates().isEmpty());
            assertEquals(ModelAutotuningPreparation.Source.MEASURED,
                    evidence.workloads().getFirst().source());
            assertEquals(ModelAutotuningPreparation.Source.MEASURED,
                    evidence.completePlan().source());
            assertEquals(ModelAutotuningPreparation.ReuseScope.SESSION,
                    evidence.completePlan().compatibility().reuseScope());
            assertEquals(config.completePlanBudget(), evidence.completePlan().budget());
            assertFalse(evidence.completePlan().candidates().isEmpty());
            assertFalse(Files.exists(config.modelPlanCache()));
            try (var prepared = preparation.preparedExecution();
                    var result = engine.run(prepared, List.of(left, right))) {
                assertEquals(1, result.resultCount());
                var bytes = result.materialize(result.publications().getFirst(),
                        dataType == DataType.FLOAT32 ? 16 : 32).bytes();
                if (dataType == DataType.FLOAT32) {
                    assertArrayEquals(new float[] {19, 22, 43, 50},
                            new float[] {bytes.getFloat(), bytes.getFloat(),
                                    bytes.getFloat(), bytes.getFloat()});
                } else {
                    assertArrayEquals(new double[] {19, 22, 43, 50},
                            new double[] {bytes.getDouble(), bytes.getDouble(),
                                    bytes.getDouble(), bytes.getDouble()});
                }
            }
        }
    }

    @Test
    void pointwiseWithoutLocalWorkloadsTunesCompletePlansAndRuns(@TempDir Path directory)
            throws Exception {
        try (Engine engine = Engine.standard()) {
            Shape shape = Shape.of(4);
            var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                    Optional.of(LayoutDescriptor.contiguous(shape)), false);
            var left = TensorFactory.fromFlatArray(
                    descriptor, Optional.empty(), new float[] {1, -2, 3, -4});
            var right = TensorFactory.fromFlatArray(
                    descriptor, Optional.empty(), new float[] {4, 1, -6, 2});
            var compiled = engine.compile(List.of(left.add(right).abs().neg()));
            var config = strictConfig(directory, 16);
            // A workload-cache directory would fail cache-file reads if the skipped tuner ran.
            Files.createDirectory(config.workloadCache());
            var request = new ModelAutotuningRequest(config,
                    new ModelAutotuningRequest.ModelIdentity(1, new byte[] {5}),
                    List.of(left, right));

            var preparation = engine.prepareTuned(compiled, request);

            assertEquals(ModelAutotuningPreparation.Outcome.TUNED, preparation.outcome());
            var evidence = preparation.evidence().orElseThrow();
            assertEquals(List.of(), evidence.workloads());
            assertSame(config.budget(), evidence.budget());
            assertEquals(ModelAutotuningPreparation.Source.MEASURED,
                    evidence.completePlan().source());
            assertTrue(evidence.completePlan().candidates().size() >= 2);
            assertEquals(ModelAutotuningPreparation.CorrectnessAction.REFERENCE_CAPTURED,
                    evidence.completePlan().candidates().getFirst().correctnessAction());
            assertTrue(evidence.completePlan().candidates().stream().skip(1).allMatch(candidate ->
                    candidate.correctnessAction() == ModelAutotuningPreparation.CorrectnessAction.MATCH));
            assertTrue(Files.isDirectory(config.workloadCache()));
            assertFalse(Files.exists(config.modelPlanCache()));
            try (var prepared = preparation.preparedExecution();
                    var result = engine.run(prepared, List.of(left, right))) {
                var bytes = result.materialize(result.publications().getFirst(), 16).bytes();
                assertArrayEquals(new float[] {-5, -1, -3, -2},
                        new float[] {bytes.getFloat(), bytes.getFloat(),
                                bytes.getFloat(), bytes.getFloat()});
            }
        }
    }

    @Test
    void absentCpuLocalWorkloadWithInsufficientCompleteBudgetFallsBackAndRuns() {
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
                    new ModelAutotuningConfig.CompletePlanBudget(4096, 0, 1, 1L, 0L),
                    Path.of("unused-model-plan-cache.bin"));
            var request = new ModelAutotuningRequest(config,
                    new ModelAutotuningRequest.ModelIdentity(1, new byte[] {1}), List.of(input));

            var preparation = engine.prepareTuned(compiled, request);

            assertEquals(ModelAutotuningPreparation.Outcome.SAFE_HEURISTIC_FALLBACK,
                    preparation.outcome());
            assertEquals(Optional.empty(), preparation.evidence());
            assertSame(compiled, preparation.preparedExecution().compiledGraph());
            try (var prepared = preparation.preparedExecution();
                    var result = engine.run(prepared, List.of(input))) {
                assertEquals(1, result.resultCount());
            }
        }
    }

    private static ModelAutotuningConfig strictConfig(Path directory, long correctnessBytes) {
        return new ModelAutotuningConfig(
                ModelAutotuningConfig.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                new ModelAutotuningConfig.Budget(1, 16, 0, 1),
                new ModelAutotuningConfig.RepresentativeProfileIdentity(1, new byte[] {4}),
                ModelAutotuningConfig.FallbackPolicy.REQUIRE_TUNED_RESULT,
                directory.resolve("workload.bin"),
                new ModelAutotuningConfig.CompletePlanBudget(4096, 0, 1, 8192, correctnessBytes),
                directory.resolve("model-plan.bin"));
    }
}
