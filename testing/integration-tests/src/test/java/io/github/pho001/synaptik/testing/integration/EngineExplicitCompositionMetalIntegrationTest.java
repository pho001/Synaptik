package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.EngineMixedOwnerTestAccess;
import io.github.pho001.synaptik.engine.InferenceSession;
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

/** Exercises real CPU-free Metal and mixed CPU/Metal public Engine composition. */
final class EngineExplicitCompositionMetalIntegrationTest {
    @Test
    void cpuFreeMetalEngineRunsCallerBroadcastGraphAcrossIsolatedSessions() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                Shape matrixShape = Shape.of(2, 3);
                TensorDescriptor matrixDescriptor = new TensorDescriptor(
                        DataType.FLOAT32,
                        matrixShape,
                        Optional.of(LayoutDescriptor.contiguous(matrixShape)),
                        false);
                Tensor matrix = nativeTensor(
                        matrixDescriptor, arena, 3.0f, 4.0f, 6.0f, 3.0f, 4.0f, 6.0f);
                Shape rowShape = Shape.of(3);
                TensorDescriptor rowDescriptor = new TensorDescriptor(
                        DataType.FLOAT32,
                        rowShape,
                        Optional.of(LayoutDescriptor.contiguous(rowShape)),
                        false);
                Tensor row = nativeTensor(rowDescriptor, arena, 2.0f, 2.0f, 2.0f);
                Tensor negated = matrix.neg();
                Tensor zero = negated.add(matrix);
                Tensor broadcast = zero.add(row);
                Tensor subtracted = matrix.sub(broadcast);
                Tensor multiplied = subtracted.mul(row);
                Tensor divided = multiplied.div(row);
                Tensor reverseDivided = row.div(divided);
                var compiled = engine.compile(
                        List.of(zero, subtracted, divided, reverseDivided));
                assertEquals(2, compiled.inputs().size());

                InferenceSession first = engine.session(compiled);
                InferenceSession second = engine.session(compiled);
                try {
                    assertCpuFreeMetalBroadcastResult(first, matrix, row);
                    assertCpuFreeMetalBroadcastResult(first, matrix, row);
                    assertCpuFreeMetalBroadcastResult(second, matrix, row);

                    first.close();
                    assertTrue(first.isClosed());
                    assertEquals("prepared execution is closed",
                            assertThrows(IllegalStateException.class,
                                    () -> first.run(null)).getMessage());
                    assertCpuFreeMetalBroadcastResult(second, matrix, row);

                    second.close();
                    assertTrue(second.isClosed());
                    assertEquals("prepared execution is closed",
                            assertThrows(IllegalStateException.class,
                                    () -> second.run(null)).getMessage());
                } finally {
                    second.close();
                    first.close();
                }
            }
        }
    }


    @Test
    void cpuFreeMetalEngineMaterializesEveryAffineKindFromDenseDirectTargets() {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
            builder.takeOwnership(MetalBackendIntegration.open(
                    new MetalBackendConfiguration(library)));
            try (Engine engine = builder.build()) {
                int[] matrixBits = {
                        0x00000000, 0x80000000, 0x3f800000,
                        0xc0000000, 0x40600000, 0xc0880000
                };
                int[] rowBits = {0x3f800000, 0xc0000000, 0x40400000};
                int[] singletonBits = {
                        0x3e800000, 0xbe800000, 0x40800000,
                        0xc0a00000, 0x40c00000, 0xc0e00000
                };
                Tensor matrix = nativeTensorBits(
                        descriptor(Shape.of(2, 3)), arena, matrixBits);
                Tensor row = nativeTensorBits(
                        descriptor(Shape.of(1, 3)), arena, rowBits);
                Tensor singleton = nativeTensorBits(
                        descriptor(Shape.of(2, 1, 3)), arena, singletonBits);

                var compiled = engine.compile(List.of(
                        matrix.reshape(3, 2),
                        row.expand(2, 3),
                        matrix.permute(1, 0),
                        matrix.expandDims(1),
                        singleton.squeeze(1)));
                assertEquals(3, compiled.inputs().size());

                List<int[]> expected = List.of(
                        matrixBits,
                        new int[] {
                                rowBits[0], rowBits[1], rowBits[2],
                                rowBits[0], rowBits[1], rowBits[2]
                        },
                        new int[] {
                                matrixBits[0], matrixBits[3],
                                matrixBits[1], matrixBits[4],
                                matrixBits[2], matrixBits[5]
                        },
                        matrixBits,
                        singletonBits);
                try (InferenceSession first = engine.session(compiled);
                        InferenceSession second = engine.session(compiled)) {
                    assertAffineResults(first, List.of(matrix, row, singleton), expected);
                    assertAffineResults(first, List.of(matrix, row, singleton), expected);
                    assertAffineResults(second, List.of(matrix, row, singleton), expected);
                }
            }
        }
    }
    @Test
    void realMetalLifecycleMixedOwnerTransfersAndCpuTuning(@TempDir Path directory) {
        Path library = configuredMetalLibrary();
        try (Arena arena = Arena.ofShared();
                Engine.Builder builder = Engine.builder()) {
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
                try (var session = engine.session(metalCompiled)) {
                    for (int run = 0; run < 2; run++) {
                        try (var result = session.run(List.of(input))) {
                            ByteBuffer canonical = result.materialize(
                                    result.publications().getFirst(), 8L).bytes();
                            assertEquals(Float.floatToRawIntBits(-1.25f),
                                    canonical.getInt());
                            assertEquals(Float.floatToRawIntBits(2.5f),
                                    canonical.getInt());
                        }
                    }
                }

                assertMixedResult(
                        engine,
                        engine.compile(List.of(input.abs().neg())),
                        List.of(input),
                        -1.25f,
                        -2.5f);
                assertMixedResult(
                        engine,
                        engine.compile(List.of(input.neg().abs())),
                        List.of(input),
                        1.25f,
                        2.5f);
                assertMixedResult(
                        engine,
                        engine.compile(List.of(input.neg().abs().neg())),
                        List.of(input),
                        -1.25f,
                        -2.5f);

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
                Tensor capturedBase = input.abs();
                Tensor metalPublication = capturedBase.neg();
                Tensor cpuPublication = metalPublication.abs();
                assertCapturedAdaptersSurviveRegistryPoison(
                        engine,
                        engine.compile(List.of(metalPublication, cpuPublication)),
                        List.of(input));
            }
        }
    }

    private static void assertCpuFreeMetalBroadcastResult(
            InferenceSession session, Tensor matrix, Tensor row) {
        try (var result = session.run(List.of(matrix, row))) {
            assertEquals(4, result.resultCount());
            assertCanonical(
                    result.materialize(result.publications().get(0), 24L).bytes(),
                    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
            assertCanonical(
                    result.materialize(result.publications().get(1), 24L).bytes(),
                    1.0f, 2.0f, 4.0f, 1.0f, 2.0f, 4.0f);
            assertCanonical(
                    result.materialize(result.publications().get(2), 24L).bytes(),
                    1.0f, 2.0f, 4.0f, 1.0f, 2.0f, 4.0f);
            assertCanonical(
                    result.materialize(result.publications().get(3), 24L).bytes(),
                    2.0f, 1.0f, 0.5f, 2.0f, 1.0f, 0.5f);
        }
    }

    private static void assertMixedResult(
            Engine engine,
            io.github.pho001.synaptik.engine.CompiledGraph compiled,
            List<Tensor> inputs,
            float first,
            float second) {
        try (var session = engine.session(compiled)) {
            for (int run = 0; run < 2; run++) {
                try (var result = session.run(inputs)) {
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
        try (var session = engine.session(compiled)) {
            EngineMixedOwnerTestAccess.poisonBackendLookup(engine);
            for (int run = 0; run < 2; run++) {
                try (var result = session.run(inputs)) {
                    assertEquals(2, result.resultCount());
                    assertCanonical(
                            result.materialize(result.publications().get(0), 8L).bytes(),
                            -1.25f,
                            -2.5f);
                    assertCanonical(
                            result.materialize(result.publications().get(1), 8L).bytes(),
                            1.25f,
                            2.5f);
                }
            }
        }
    }


    private static void assertAffineResults(
            InferenceSession session, List<Tensor> inputs, List<int[]> expected) {
        try (var result = session.run(inputs)) {
            assertEquals(expected.size(), result.resultCount());
            for (int index = 0; index < expected.size(); index++) {
                ByteBuffer canonical = result.materialize(
                        result.publications().get(index),
                        Math.multiplyExact((long) expected.get(index).length, Float.BYTES)).bytes();
                for (int bits : expected.get(index)) {
                    assertEquals(bits, canonical.getInt());
                }
            }
        }
    }

    private static void assertCanonical(ByteBuffer canonical, float... expected) {
        for (float value : expected) {
            assertEquals(Float.floatToRawIntBits(value), canonical.getInt());
        }
    }


    private static TensorDescriptor descriptor(Shape shape) {
        return new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
    }

    private static Tensor nativeTensorBits(
            TensorDescriptor descriptor, Arena arena, int... bits) {
        var segment = arena.allocate(
                Math.multiplyExact(bits.length, Integer.BYTES), Integer.BYTES);
        for (int index = 0; index < bits.length; index++) {
            segment.setAtIndex(ValueLayout.JAVA_INT, index, bits[index]);
        }
        return TensorFactory.create(
                descriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(
                        DataType.FLOAT32, bits.length, segment)));
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
