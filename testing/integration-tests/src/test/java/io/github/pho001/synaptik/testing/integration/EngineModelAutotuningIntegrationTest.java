package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.ModelAutotuningPreparation;
import io.github.pho001.synaptik.engine.ModelAutotuningRequest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Exercises deterministic native-free absent-handoff fallback through the public API. */
final class EngineModelAutotuningIntegrationTest {
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
                    Path.of("unused-native-free-autotuning-cache.bin"));
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
}
