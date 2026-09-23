package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.EngineMixedOwnerTestAccess;
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
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Exercises real CPU/Metal composition, both transfer directions, reuse, and CPU-only tuning. */
final class EngineExplicitCompositionMetalIntegrationTest {
    @Test
    void realMetalLifecycleMixedOwnerTransfersAndCpuTuning(@TempDir Path directory) {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared()) {
        try (Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            builder.takeOwnership(CpuBackendIntegration.open());
            try (Engine engine = builder.build()) {
                Shape shape = Shape.of(2);
                TensorDescriptor descriptor = new TensorDescriptor(
                        DataType.FLOAT32,
                        shape,
                        Optional.of(LayoutDescriptor.contiguous(shape)),
                        false);
                Tensor input = nativeTensor(
                        descriptor, arena, 1.25f, -2.5f);

                var metalCompiled = engine.compile(List.of(input.neg()));
                try (var prepared = engine.prepare(metalCompiled)) {
                    for (int run = 0; run < 2; run++) {
                        try (var result = engine.run(prepared, List.of(input))) {
                            ByteBuffer canonical = result.materialize(
                                    result.publications().getFirst(), 8L).bytes();
                            assertEquals(Float.floatToRawIntBits(-1.25f),
                                    canonical.getInt());
                            assertEquals(Float.floatToRawIntBits(2.5f),
                                    canonical.getInt());
                        }
                    }
                }

                Tensor other = nativeTensor(descriptor, arena, 3.0f, 4.0f);
                assertMixedResult(
                        engine,
                        engine.compile(List.of(input.neg().add(other))),
                        List.of(input, other),
                        1.75f,
                        6.5f);
                assertMixedResult(
                        engine,
                        engine.compile(List.of(input.add(other).neg())),
                        List.of(input, other),
                        -4.25f,
                        -1.5f);
                assertMixedResult(
                        engine,
                        engine.compile(List.of(input.neg().add(input))),
                        List.of(input),
                        0.0f,
                        0.0f);
                assertMixedResult(
                        engine,
                        engine.compile(List.of(input.add(other.neg()))),
                        List.of(input, other),
                        -1.75f,
                        -6.5f);
                Tensor fanOut = input.add(other);
                assertMixedResult(
                        engine,
                        engine.compile(List.of(fanOut.neg().add(fanOut))),
                        List.of(input, other),
                        0.0f,
                        0.0f);

                Tensor cpuInput = TensorFactory.create(
                        descriptor,
                        Optional.empty(),
                        Optional.of(new MemorySegmentStorage(
                                DataType.FLOAT32, 2, arena.allocate(8, Float.BYTES))));
                var cpuCompiled = engine.compile(List.of(cpuInput.contiguous()));
                ModelAutotuningRequest cpuRequest = tuningRequest(
                        directory,
                        cpuInput,
                        ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC);
                ModelAutotuningPreparation preparation =
                        engine.prepareTuned(cpuCompiled, cpuRequest);
                assertEquals(ModelAutotuningPreparation.Outcome.SAFE_HEURISTIC_FALLBACK,
                        preparation.outcome());
                try (var tuned = preparation.preparedExecution();
                        var result = engine.run(tuned, List.of(cpuInput))) {
                    assertEquals(1, result.resultCount());
                }
                Tensor noStorage = TensorFactory.create(
                        descriptor, Optional.empty(), Optional.empty());
                var metalForTuning = engine.compile(List.of(noStorage.neg()));
                ModelAutotuningRequest metalRequest = tuningRequest(
                        directory, noStorage,
                        ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC);
                assertEquals("model autotuning requires a CPU-owned partition plan",
                        assertThrows(IllegalStateException.class,
                                () -> engine.prepareTuned(metalForTuning, metalRequest)).getMessage());
                Tensor capturedBase = input.add(other);
                Tensor metalPublication = capturedBase.neg();
                Tensor cpuPublication = metalPublication.add(capturedBase);
                assertCapturedAdaptersSurviveRegistryPoison(
                        engine,
                        engine.compile(List.of(metalPublication, cpuPublication)),
                        List.of(input, other));
            }
        }
        }
    }

    private static void assertMixedResult(
            Engine engine,
            io.github.pho001.synaptik.engine.CompiledGraph compiled,
            List<Tensor> inputs,
            float first,
            float second) {
        try (var prepared = engine.prepare(compiled)) {
            for (int run = 0; run < 2; run++) {
                try (var result = engine.run(prepared, inputs)) {
                    ByteBuffer canonical = result.materialize(
                            result.publications().getFirst(), 8L).bytes();
                    assertEquals(Float.floatToRawIntBits(first), canonical.getInt());
                    assertEquals(Float.floatToRawIntBits(second), canonical.getInt());
                }
            }
        }
    }

    private static void assertCapturedAdaptersSurviveRegistryPoison(
            Engine engine,
            io.github.pho001.synaptik.engine.CompiledGraph compiled,
            List<Tensor> inputs) {
        try (var prepared = engine.prepare(compiled)) {
            EngineMixedOwnerTestAccess.poisonBackendLookup(engine);
            for (int run = 0; run < 2; run++) {
                try (var result = engine.run(prepared, inputs)) {
                    assertEquals(2, result.resultCount());
                    assertCanonical(
                            result.materialize(result.publications().get(0), 8L).bytes(),
                            -4.25f,
                            -1.5f);
                    assertCanonical(
                            result.materialize(result.publications().get(1), 8L).bytes(),
                            0.0f,
                            0.0f);
                }
            }
        }
    }

    private static void assertCanonical(ByteBuffer canonical, float first, float second) {
        assertEquals(Float.floatToRawIntBits(first), canonical.getInt());
        assertEquals(Float.floatToRawIntBits(second), canonical.getInt());
    }

    private static Tensor nativeTensor(
            TensorDescriptor descriptor, Arena arena, float... values) {
        var segment = arena.allocate(
                Math.multiplyExact(values.length, Float.BYTES), Float.BYTES);
        for (int index = 0; index < values.length; index++) {
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, index, values[index]);
        }
        return TensorFactory.create(
                descriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(
                        DataType.FLOAT32, values.length, segment)));
    }

    private static ModelAutotuningRequest tuningRequest(
            Path directory,
            Tensor representative,
            ModelAutotuningConfig.FallbackPolicy fallbackPolicy) {
        var config = new ModelAutotuningConfig(
                ModelAutotuningConfig.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                new ModelAutotuningConfig.Budget(1, 4, 0, 1),
                new ModelAutotuningConfig.RepresentativeProfileIdentity(1, new byte[] {2}),
                fallbackPolicy,
                directory.resolve("workload.bin"),
                new ModelAutotuningConfig.CompletePlanBudget(1, 0, 1, 1L, 8L),
                directory.resolve("model-plan.bin"));
        return new ModelAutotuningRequest(
                config,
                new ModelAutotuningRequest.ModelIdentity(1, new byte[] {1}),
                List.of(representative));
    }

    private static Path configuredMetalLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        Assumptions.assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is required for real Metal integration");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        Assumptions.assumeTrue(Files.isRegularFile(library),
                "configured Metal library must exist: " + library);
        return library;
    }
}
